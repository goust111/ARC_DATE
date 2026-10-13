from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    values = Counter(map(int, np.asarray(grid).ravel()))
    return int(values.most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors(r: int, c: int, connectivity: int) -> list[Coord]:
    if int(connectivity) == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not arr[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)

    if monochromatic:
        if colors is None:
            cols = colors_present(g, exclude={bg})
        else:
            cols = [int(c) for c in colors if int(c) != bg]
        out = []
        for color in cols:
            for coords in components_mask(g == int(color), connectivity=connectivity):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    if colors is None:
        mask = g != bg
    else:
        mask = (g != bg) & np.isin(g, np.array(colors, dtype=int))
    out = []
    for coords in components_mask(mask, connectivity=connectivity):
        out.append(
            {
                "color": None,
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = np.asarray(comp["coords"], dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch


def _bbox_border_coords(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = map(int, bbox)
    out: set[Coord] = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _touches_all_sides(coords: set[Coord], bbox: BBox) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return (
        any(r == r0 for r, _ in coords)
        and any(r == r1 for r, _ in coords)
        and any(c == c0 for _, c in coords)
        and any(c == c1 for _, c in coords)
    )


def _extract_frame_infos(patch: np.ndarray, bg: int) -> dict[int, dict] | None:
    g = np.asarray(patch, dtype=int)
    infos: dict[int, dict] = {}
    cols = colors_present(g, exclude={bg})
    if len(cols) < 2:
        return None

    for color in cols:
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            return None
        bbox = bbox_coords(coords)
        h = int(bbox[2] - bbox[0] + 1)
        w = int(bbox[3] - bbox[1] + 1)
        if h < 4 or w < 4:
            return None
        border = _bbox_border_coords(bbox)
        visible = {tuple(map(int, rc)) for rc in coords}
        if not visible.issubset(border):
            return None
        if not _touches_all_sides(visible, bbox):
            return None
        infos[int(color)] = {"bbox": bbox, "border": border, "visible": visible}
    return infos


def _pick_frame_and_query(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    assert len(comps) == 2, "expected exactly two foreground regions"

    frame_idx = None
    frame_patch = None
    for idx, comp in enumerate(comps):
        patch = _component_patch(grid, comp, bg)
        if _extract_frame_infos(patch, bg) is not None:
            assert frame_idx is None, "expected exactly one frame cluster"
            frame_idx = idx
            frame_patch = patch

    assert frame_idx is not None and frame_patch is not None
    query_patch = _component_patch(grid, comps[1 - frame_idx], bg)
    return frame_patch, query_patch


def _infer_mapping_from_frame_patch(frame_patch: np.ndarray, bg: int) -> dict[int, int]:
    infos = _extract_frame_infos(frame_patch, bg)
    assert infos is not None, "frame patch is not a valid rectangle-frame cluster"
    colors = sorted(infos)

    outgoing: dict[int, set[int]] = {color: set() for color in colors}
    for a, b in itertools.combinations(colors, 2):
        inter = infos[a]["border"] & infos[b]["border"]
        if not inter:
            continue
        visible_here = {int(frame_patch[r, c]) for r, c in inter}
        assert visible_here <= {a, b}, "intersections must not be covered by a third color"
        if a in visible_here:
            outgoing[b].add(a)
        if b in visible_here:
            outgoing[a].add(b)

    mapping = {}
    for src, targets in outgoing.items():
        assert len(targets) == 1, "each frame color must map to exactly one visible replacement color"
        mapping[int(src)] = int(next(iter(targets)))
    return mapping


def _has_cycle(mapping: dict[int, int]) -> bool:
    for start in mapping:
        seen = set()
        cur = start
        while cur not in seen:
            seen.add(cur)
            cur = mapping[cur]
        if cur == start and len(seen) > 1:
            return True
    return False


def _recolor_query_patch(query_patch: np.ndarray, mapping: dict[int, int], bg: int) -> np.ndarray:
    src = np.asarray(query_patch, dtype=int)
    out = src.copy()
    for color in colors_present(src, exclude={bg}):
        assert color in mapping, "every query color must appear in the frame-derived map"
        out[src == int(color)] = int(mapping[color])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Read the overlapping rectangle frames as a color-substitution table and recolor the query sprite.

    Input structure:
    - The background is the most common color.
    - One foreground region is a cluster of one-pixel-thick monochrome rectangle frames whose borders overlap.
      At each crossing only one of the two frame colors remains visible.
    - The other foreground region is a connected multicolor query sprite using only colors that appear among the frames.

    Rule:
    - Recover each frame from the bounding box of its visible color pixels.
    - At every border crossing, record a directed rule from the hidden frame color to the visible frame color.
      After combining all crossings, every query color has exactly one outgoing replacement target.
    - Recolor the query sprite by applying that color map exactly once and return the query sprite's tight crop.

    Key insight:
    - The frame cluster is not a stack to sort into one depth order; its local occlusions write a replacement dictionary,
      and cycles in that dictionary show why reading a single global layer order would be wrong.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_patch, query_patch = _pick_frame_and_query(g, bg)
    mapping = _infer_mapping_from_frame_patch(frame_patch, bg)
    return _recolor_query_patch(query_patch, mapping, bg)


FRAME_COLORS_RANGE = (3, 3)
FRAME_CANVAS_RANGE = (12, 15)
FRAME_SIZE_RANGE = (5, 10)
QUERY_BBOX_RANGE = (4, 7)
QUERY_FILL_RANGE = (7, 15)
GAP_RANGE = (3, 5)
BORDER_RANGE = (1, 2)
MAX_ATTEMPTS = 400


def _rand_bbox(h: int, w: int) -> BBox:
    box_h = random.randint(*FRAME_SIZE_RANGE)
    box_w = random.randint(*FRAME_SIZE_RANGE)
    box_h = min(box_h, h)
    box_w = min(box_w, w)
    r0 = random.randint(0, h - box_h)
    c0 = random.randint(0, w - box_w)
    return (r0, c0, r0 + box_h - 1, c0 + box_w - 1)


def _sample_rectangles(frame_h: int, frame_w: int, count: int) -> list[BBox] | None:
    for _ in range(300):
        bboxes = [_rand_bbox(frame_h, frame_w) for _ in range(count)]
        row_sides = [x for box in bboxes for x in (box[0], box[2])]
        col_sides = [x for box in bboxes for x in (box[1], box[3])]
        if len(set(row_sides)) != len(row_sides) or len(set(col_sides)) != len(col_sides):
            continue

        perims = [_bbox_border_coords(box) for box in bboxes]
        triple = set.intersection(*perims)
        if triple:
            continue

        ok = True
        for i, j in itertools.combinations(range(count), 2):
            if len(perims[i] & perims[j]) != 2:
                ok = False
                break
        if ok:
            return bboxes
    return None


def _render_frame_patch(
    frame_h: int,
    frame_w: int,
    colors: list[int],
    bboxes: list[BBox],
    mapping: dict[int, int],
    bg: int,
) -> np.ndarray | None:
    perims = {int(color): _bbox_border_coords(box) for color, box in zip(colors, bboxes, strict=True)}
    owners_by_coord: dict[Coord, list[int]] = {}
    for color, coords in perims.items():
        for coord in coords:
            owners_by_coord.setdefault(coord, []).append(int(color))

    patch = np.full((frame_h, frame_w), int(bg), dtype=int)
    for coord, owners in owners_by_coord.items():
        if len(owners) == 1:
            color = owners[0]
        elif len(owners) == 2:
            a, b = owners
            if mapping[a] == b and mapping[b] != a:
                color = b
            elif mapping[b] == a and mapping[a] != b:
                color = a
            else:
                return None
        else:
            return None
        patch[coord] = int(color)
    return patch


def _grow_query_mask(h: int, w: int, target: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors(r, c, 4):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if len(set(rows)) < 2 or len(set(cols)) < 2:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    rr, cc = np.where(mask)
    return mask[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1]


def _make_query_patch(colors: list[int], bg: int) -> np.ndarray | None:
    qh = random.randint(*QUERY_BBOX_RANGE)
    qw = random.randint(*QUERY_BBOX_RANGE)
    area = random.randint(max(len(colors) + 2, QUERY_FILL_RANGE[0]), min(qh * qw, QUERY_FILL_RANGE[1]))
    mask = _grow_query_mask(qh, qw, area)
    if mask is None:
        return None

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    patch = np.full(mask.shape, int(bg), dtype=int)
    for coord, color in zip(coords[: len(colors)], colors, strict=True):
        patch[coord] = int(color)
    for coord in coords[len(colors) :]:
        patch[coord] = int(random.choice(colors))

    used = set(colors_present(patch, exclude={bg}))
    if used != set(colors):
        return None
    return patch


def _place_regions(frame_patch: np.ndarray, query_patch: np.ndarray, bg: int) -> np.ndarray:
    fh, fw = frame_patch.shape
    qh, qw = query_patch.shape
    gap = random.randint(*GAP_RANGE)
    border = random.randint(*BORDER_RANGE)
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        h = max(fh, qh) + 2 * border
        w = fw + qw + gap + 2 * border
        grid = np.full((h, w), int(bg), dtype=int)
        frame_r = border + random.randint(0, h - 2 * border - fh)
        query_r = border + random.randint(0, h - 2 * border - qh)
        frame_c = border
        query_c = border + fw + gap
    else:
        h = fh + qh + gap + 2 * border
        w = max(fw, qw) + 2 * border
        grid = np.full((h, w), int(bg), dtype=int)
        frame_r = border
        query_r = border + fh + gap
        frame_c = border + random.randint(0, w - 2 * border - fw)
        query_c = border + random.randint(0, w - 2 * border - qw)

    grid[frame_r : frame_r + fh, frame_c : frame_c + fw] = frame_patch
    qmask = query_patch != int(bg)
    sub = grid[query_r : query_r + qh, query_c : query_c + qw]
    sub[qmask] = query_patch[qmask]
    grid[query_r : query_r + qh, query_c : query_c + qw] = sub
    return grid


def _independent_oracle(query_patch: np.ndarray, mapping: dict[int, int], bg: int) -> np.ndarray:
    src = np.asarray(query_patch, dtype=int)
    out = src.copy()
    for color, target in mapping.items():
        out[src == int(color)] = int(target)
    assert set(colors_present(out, exclude={bg})) == set(mapping)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    count = random.randint(*FRAME_COLORS_RANGE)
    assert count == 3
    frame_h = random.randint(*FRAME_CANVAS_RANGE)
    frame_w = random.randint(*FRAME_CANVAS_RANGE)

    palette = random.sample(range(10), count + 1)
    bg = int(palette[0])
    colors = [int(c) for c in palette[1:]]
    cycle = colors[:]
    random.shuffle(cycle)
    mapping = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}

    bboxes = _sample_rectangles(frame_h, frame_w, count)
    if bboxes is None:
        raise ValueError("failed to sample frame rectangles")

    frame_patch = _render_frame_patch(frame_h, frame_w, colors, bboxes, mapping, bg)
    if frame_patch is None:
        raise ValueError("failed to render frame patch")

    frame_info = _extract_frame_infos(frame_patch, bg)
    if frame_info is None:
        raise ValueError("rendered patch is not frame-like")
    observed = _infer_mapping_from_frame_patch(frame_patch, bg)
    if observed != mapping:
        raise ValueError("rendered frame patch does not encode the intended mapping")
    if not _has_cycle(observed):
        raise ValueError("mapping must block a single global layer order")

    query_patch = _make_query_patch(colors, bg)
    if query_patch is None:
        raise ValueError("failed to generate query sprite")
    oracle = _independent_oracle(query_patch, mapping, bg)

    inp = _place_regions(frame_patch, query_patch, bg)
    if max(inp.shape) > 30 or max(oracle.shape) > 30:
        raise ValueError("generated grid exceeds ARC size limit")
    return inp, oracle


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, oracle = _generate_case()
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, oracle):
            return inp
    raise RuntimeError("failed to generate valid input")
