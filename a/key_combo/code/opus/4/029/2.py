from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
OUTSIDE_COLORS = (BLUE, RED, YELLOW, ORANGE)


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(grid: np.ndarray, colors: tuple[int, ...] | list[int]) -> list[dict]:
    out: list[dict] = []
    for color in colors:
        for coords in components_mask(grid == int(color)):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(len(coords)),
                }
            )
    return out


def normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def detect_frame(grid: np.ndarray) -> dict:
    frames = []
    for comp in find_components(grid, [GREY]):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 == r0 or c1 == c0:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros_like(patch, dtype=bool)
        expected[-1, :] = True
        expected[:, -1] = True
        if np.array_equal(patch == GREY, expected):
            frames.append(comp)
    assert len(frames) == 1, "expected exactly one grey L-frame"
    return frames[0]


def template_info(grid: np.ndarray, frame_bbox: BBox) -> tuple[int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = frame_bbox
    interior = grid[r0:r1, c0:c1]
    mask = (interior != BLACK) & (interior != GREY)
    comps = components_mask(mask)
    assert len(comps) == 1, "expected one framed template object"
    coords = comps[0]
    colors = {int(interior[r, c]) for r, c in coords}
    assert len(colors) == 1
    return colors.pop(), normalized_shape(coords)


def outside_objects(grid: np.ndarray, frame_bbox: BBox) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    out = []
    for comp in find_components(grid, list(OUTSIDE_COLORS) + [GREEN]):
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        inside = (r0 <= rows) & (rows <= r1) & (c0 <= cols) & (cols <= c1)
        if np.any(inside):
            continue
        out.append(comp)
    return out


def flood_from_seeds(grid: np.ndarray, seed_coords: list[Coord], color: int) -> None:
    seen = np.zeros(grid.shape, dtype=bool)
    q: deque[Coord] = deque()
    for r, c in seed_coords:
        rr, cc = int(r), int(c)
        if not seen[rr, cc]:
            seen[rr, cc] = True
            q.append((rr, cc))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or seen[nr, nc] or int(grid[nr, nc]) != BLACK:
                continue
            seen[nr, nc] = True
            grid[nr, nc] = int(color)
            q.append((int(nr), int(nc)))


def classify_objects(grid: np.ndarray):
    frame = detect_frame(grid)
    frame_bbox = frame["bbox"]
    template_color, template_shape = template_info(grid, frame_bbox)
    objs = outside_objects(grid, frame_bbox)
    assert objs, "expected outside colored objects"
    matches = [normalized_shape(obj["coords"]) == template_shape for obj in objs]
    assert any(matches) and not all(matches), "need both matching and non-matching objects"
    return frame_bbox, template_color, objs, matches


def _transform(
    grid: np.ndarray,
    *,
    seed_indices: set[int],
    mismatch_to_grey: bool,
    do_flood: bool,
    erase_frame: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_bbox, template_color, objs, matches = classify_objects(g)
    out = g.copy()
    seeds: list[Coord] = []
    for i, obj in enumerate(objs):
        coords = obj["coords"]
        if i in seed_indices:
            paint_coords(out, coords, template_color)
            seeds.extend((int(r), int(c)) for r, c in coords)
        elif mismatch_to_grey:
            paint_coords(out, coords, GREY)
    if do_flood:
        flood_from_seeds(out, seeds, template_color)
    if erase_frame:
        r0, c0, r1, c1 = frame_bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the framed template shape to classify outside objects into flood seeds or grey blockers, then erase the frame.

    Input structure:
    - The black background contains one grey L-frame whose vertical leg is on the
      right side of its bbox and whose horizontal leg is on the bottom side.
    - Inside the L-frame interior there is one small 4-connected template object.
    - Outside the frame there are several small colored 4-connected objects and
      some separate grey wall segments on black background.

    Rule:
    - Extract the normalized binary shape of the framed template, ignoring its
      position.
    - Compare every outside non-grey object against that template shape.
    - Recolor matching outside objects to the template color so they become flood
      seeds, and recolor non-matching outside objects to grey so they remain as
      blockers.
    - Flood-fill from all seed pixels through 4-connected black cells only, then
      erase the entire L-frame bbox, including the original template.

    Key insight:
    - The template does not identify a single object to recolor; it partitions
      outside objects into two roles, and those roles determine the flood
      topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frame_bbox, _template_color, objs, matches = classify_objects(g)
    seed_indices = {i for i, matched in enumerate(matches) if matched}
    return _transform(
        g,
        seed_indices=seed_indices,
        mismatch_to_grey=True,
        do_flood=True,
        erase_frame=True,
    )


def _shape_is_connected(cells: tuple[Coord, ...]) -> bool:
    wanted = set(cells)
    stack = [next(iter(wanted))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if nxt in wanted and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == wanted


def _enumerate_shapes() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    seen = set()
    shapes = []
    for size in (3, 4, 5):
        for combo in itertools.combinations(cells, size):
            if not _shape_is_connected(combo):
                continue
            key = normalized_shape(np.array(combo, dtype=int))
            if key not in seen:
                seen.add(key)
                shapes.append(key)
    return shapes


SHAPES = _enumerate_shapes()


def shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> np.ndarray:
    coords = []
    for dr, dc in shape:
        rr, cc = int(r0 + dr), int(c0 + dc)
        grid[rr, cc] = int(color)
        coords.append((rr, cc))
    return np.array(coords, dtype=int)


def corner_frame_bbox(h: int, w: int, interior_h: int, interior_w: int, corner: str) -> BBox:
    fh = interior_h + 1
    fw = interior_w + 1
    if corner == "tl":
        return 0, 0, fh - 1, fw - 1
    if corner == "tr":
        return 0, w - fw, fh - 1, w - 1
    if corner == "bl":
        return h - fh, 0, h - 1, fw - 1
    if corner == "br":
        return h - fh, w - fw, h - 1, w - 1
    raise ValueError(corner)


def paint_frame(grid: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def corner_pocket_cells(h: int, w: int, corner: str, a: int, b: int) -> tuple[list[Coord], np.ndarray]:
    wall_cells: list[Coord] = []
    pocket = np.zeros((h, w), dtype=bool)
    if corner == "tl":
        for r in range(a):
            wall_cells.append((r, b))
        for c in range(b):
            wall_cells.append((a, c))
        pocket[:a, :b] = True
    elif corner == "tr":
        vc = w - b - 1
        for r in range(a):
            wall_cells.append((r, vc))
        for c in range(w - b, w):
            wall_cells.append((a, c))
        pocket[:a, w - b : w] = True
    elif corner == "bl":
        hr = h - a - 1
        for c in range(b):
            wall_cells.append((hr, c))
        for r in range(h - a, h):
            wall_cells.append((r, b))
        pocket[h - a : h, :b] = True
    elif corner == "br":
        hr = h - a - 1
        vc = w - b - 1
        for c in range(w - b, w):
            wall_cells.append((hr, c))
        for r in range(h - a, h):
            wall_cells.append((r, vc))
        pocket[h - a : h, w - b : w] = True
    else:
        raise ValueError(corner)
    return wall_cells, pocket


def place_shape_in_mask(
    shape: tuple[tuple[int, int], ...],
    allowed: np.ndarray,
    blocked: np.ndarray,
    object_cells: list[Coord],
    *,
    object_gap: int,
) -> tuple[int, int] | None:
    h, w = allowed.shape
    sh, sw = shape_dims(shape)
    candidates = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if not all(allowed[r, c] and not blocked[r, c] for r, c in coords):
                continue
            ok = True
            for r, c in coords:
                for rr, cc in object_cells:
                    if max(abs(r - rr), abs(c - cc)) <= object_gap:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    return random.choice(candidates) if candidates else None


def black_component_labels(grid: np.ndarray) -> np.ndarray:
    labels = np.zeros(grid.shape, dtype=int)
    comp_id = 0
    for coords in components_mask(grid == BLACK):
        comp_id += 1
        labels[coords[:, 0], coords[:, 1]] = comp_id
    return labels


def adjacent_black_components(grid: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> set[int]:
    ids = set()
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK and int(labels[nr, nc]) > 0:
                ids.add(int(labels[nr, nc]))
    return ids


def random_standalone_wall(blocked: np.ndarray) -> list[Coord] | None:
    h, w = blocked.shape
    candidates: list[list[Coord]] = []
    for orient in ("h", "v"):
        for length in range(2, 5):
            for r0 in range(h):
                for c0 in range(w):
                    if orient == "h":
                        cells = [(r0, c0 + k) for k in range(length)]
                    else:
                        cells = [(r0 + k, c0) for k in range(length)]
                    if not all(in_bounds((h, w), r, c) for r, c in cells):
                        continue
                    ok = True
                    for r, c in cells:
                        for rr in range(max(0, r - 1), min(h, r + 2)):
                            for cc in range(max(0, c - 1), min(w, c + 2)):
                                if blocked[rr, cc]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        candidates.append(cells)
    return random.choice(candidates) if candidates else None


def semantic_checks(inp: np.ndarray) -> bool:
    try:
        frame_bbox, template_color, objs, matches = classify_objects(inp)
    except Exception:
        return False
    if sum(matches) < 1 or sum(not m for m in matches) < 1:
        return False

    out = main(inp)
    if np.array_equal(out, inp):
        return False

    labels = black_component_labels(inp)
    if labels.max() < 2:
        return False

    match_adjacent = [adjacent_black_components(inp, obj["coords"], labels) for obj, m in zip(objs, matches) if m]
    if not match_adjacent or not set.intersection(*match_adjacent):
        return False

    actual_green = int(np.sum(out == template_color))
    if actual_green <= sum(obj["area"] for obj, m in zip(objs, matches) if m):
        return False

    if not np.any(out == GREY):
        return False
    if not np.any(out == BLACK):
        return False

    seed_indices = {i for i, m in enumerate(matches) if m}
    all_indices = set(range(len(objs)))
    alt_all = _transform(inp, seed_indices=all_indices, mismatch_to_grey=False, do_flood=True, erase_frame=True)
    alt_noflood = _transform(inp, seed_indices=seed_indices, mismatch_to_grey=True, do_flood=False, erase_frame=True)
    if np.array_equal(out, alt_all):
        return False
    if np.array_equal(out, alt_noflood):
        return False
    if int(np.sum(alt_all == template_color)) <= actual_green:
        return False

    r0, c0, r1, c1 = frame_bbox
    if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
        return False

    green_ratio = float(np.mean(out == template_color))
    if not (0.04 <= green_ratio <= 0.45):
        return False
    return True


def generate_input() -> np.ndarray:
    corners = ("tl", "tr", "bl", "br")
    for _ in range(500):
        h = random.randint(17, 24)
        w = random.randint(17, 24)
        grid = np.zeros((h, w), dtype=int)

        frame_corner = random.choice(corners)
        remaining = [c for c in corners if c != frame_corner]
        random.shuffle(remaining)
        match_corner, mismatch_corner = remaining[:2]

        interior_h = random.randint(3, 4)
        interior_w = random.randint(3, 4)
        frame_bbox = corner_frame_bbox(h, w, interior_h, interior_w, frame_corner)
        paint_frame(grid, frame_bbox)

        template_choices = [s for s in SHAPES if shape_dims(s)[0] <= interior_h and shape_dims(s)[1] <= interior_w]
        template_shape = random.choice(template_choices)
        th, tw = shape_dims(template_shape)
        tr0 = frame_bbox[0] + random.randint(0, interior_h - th)
        tc0 = frame_bbox[1] + random.randint(0, interior_w - tw)
        paint_shape(grid, template_shape, tr0, tc0, GREEN)

        blocked = grid != BLACK
        pocket_specs = {}
        for corner in (match_corner, mismatch_corner):
            a = random.randint(3, 4)
            b = random.randint(3, 4)
            wall_cells, pocket = corner_pocket_cells(h, w, corner, a, b)
            if any(blocked[r, c] for r, c in wall_cells):
                break
            for r, c in wall_cells:
                grid[r, c] = GREY
            blocked = grid != BLACK
            pocket_specs[corner] = pocket
        else:
            object_cells: list[Coord] = []

            match_allowed = pocket_specs[match_corner].copy()
            mismatch_allowed = pocket_specs[mismatch_corner].copy()
            match_pos = place_shape_in_mask(
                template_shape,
                match_allowed,
                blocked,
                object_cells,
                object_gap=2,
            )
            if match_pos is None:
                continue
            match_color = random.choice(OUTSIDE_COLORS)
            match_coords = paint_shape(grid, template_shape, match_pos[0], match_pos[1], match_color)
            object_cells.extend((int(r), int(c)) for r, c in match_coords)
            blocked = grid != BLACK

            mismatch_shapes = [s for s in SHAPES if s != template_shape]
            mismatch_shape = random.choice(
                [s for s in mismatch_shapes if shape_dims(s)[0] <= mismatch_allowed.shape[0] and shape_dims(s)[1] <= mismatch_allowed.shape[1]]
            )
            mismatch_pos = place_shape_in_mask(
                mismatch_shape,
                mismatch_allowed,
                blocked,
                object_cells,
                object_gap=2,
            )
            if mismatch_pos is None:
                continue
            mismatch_color = random.choice([c for c in OUTSIDE_COLORS if c != match_color] or list(OUTSIDE_COLORS))
            mismatch_coords = paint_shape(grid, mismatch_shape, mismatch_pos[0], mismatch_pos[1], mismatch_color)
            object_cells.extend((int(r), int(c)) for r, c in mismatch_coords)
            blocked = grid != BLACK

            extra_count = random.randint(1, 3)
            extra_shapes = random.sample([s for s in mismatch_shapes if s != mismatch_shape], k=min(extra_count, len([s for s in mismatch_shapes if s != mismatch_shape])))
            main_allowed = np.ones((h, w), dtype=bool)
            r0, c0, r1, c1 = frame_bbox
            main_allowed[r0 : r1 + 1, c0 : c1 + 1] = False
            main_allowed[pocket_specs[match_corner]] = False
            main_allowed[pocket_specs[mismatch_corner]] = False
            main_allowed[grid == GREY] = False

            success = True
            for shape in extra_shapes:
                pos = place_shape_in_mask(shape, main_allowed, blocked, object_cells, object_gap=2)
                if pos is None:
                    success = False
                    break
                color_choices = list(OUTSIDE_COLORS)
                color = random.choice(color_choices)
                coords = paint_shape(grid, shape, pos[0], pos[1], color)
                object_cells.extend((int(r), int(c)) for r, c in coords)
                blocked = grid != BLACK
                main_allowed[coords[:, 0], coords[:, 1]] = False
            if not success:
                continue

            for _extra_wall in range(random.randint(0, 2)):
                candidate = random_standalone_wall(blocked)
                if candidate is None:
                    break
                for r, c in candidate:
                    grid[r, c] = GREY
                blocked = grid != BLACK

            if semantic_checks(grid):
                return grid

    raise RuntimeError("failed to generate a valid template-match gated flood-fill puzzle")
