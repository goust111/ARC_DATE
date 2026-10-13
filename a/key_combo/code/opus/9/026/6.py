from __future__ import annotations

import hashlib
import random
from collections import Counter, deque

import numpy as np


FRAME_COLORS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 800
GRID_ROWS_RANGE = (20, 26)
GRID_COLS_RANGE = (20, 26)
INTERIOR_SIDE_RANGE = (3, 8)
FRAME_SIDE_RANGE = (5, 10)
SPRITE_PADDING = 1

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


SOLID_PATTERNS = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
]

OPEN_PATTERNS = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    return out


def _interior_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected = _frame_mask_from_bbox(mask.shape, (r0, c0, r1, c1))
    if not np.array_equal(mask, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    if src_len < 2 or dst_len < 2:
        raise AssertionError("sprite bbox and target interior must both span at least 2 cells on each axis")
    return int(round(int(i) * (dst_len - 1) / (src_len - 1)))


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke.astype(bool)
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    ph, pw = padded.shape

    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke.astype(bool) | holes


def _draw_segment(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        mask[r0, lo : hi + 1] = True
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        mask[lo : hi + 1, c0] = True
        return
    raise AssertionError("mapped adjacent sprite pixels must remain row- or column-aligned")


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = map(int, target_shape)
    assert src_h >= 2 and src_w >= 2, "sprite bbox must be at least 2x2"
    assert dst_h >= 2 and dst_w >= 2, "frame interior must be at least 2x2"
    coords = np.argwhere(sprite)
    assert coords.size > 0

    stroke = np.zeros((dst_h, dst_w), dtype=bool)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in coords:
        tr = _scaled_index(int(sr), src_h, dst_h)
        tc = _scaled_index(int(sc), src_w, dst_w)
        mapped[(int(sr), int(sc))] = (tr, tc)
        stroke[tr, tc] = True

    coord_set = {tuple(map(int, rc)) for rc in coords}
    for sr, sc in coord_set:
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) in coord_set:
                _draw_segment(stroke, mapped[(sr, sc)], mapped[(nr, nc)])

    return _fill_enclosed(stroke)


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    pairs: list[dict] = []
    for color in colors:
        comps = _components(g == color)
        frame_candidates: list[tuple[int, int, int, int]] = []
        sprite_candidates: list[np.ndarray] = []
        for comp in comps:
            comp_mask = np.zeros_like(g, dtype=bool)
            comp_mask[comp[:, 0], comp[:, 1]] = True
            frame_bbox = _rect_frame_bbox(comp_mask)
            if frame_bbox is not None:
                frame_candidates.append(frame_bbox)
            else:
                sprite_candidates.append(comp)
        assert len(frame_candidates) == 1, "each color must have exactly one hollow frame"
        assert len(sprite_candidates) == 1, "each color must have exactly one sprite"

        frame_bbox = frame_candidates[0]
        sprite_coords = sprite_candidates[0]
        sr0, sc0, sr1, sc1 = _bbox_from_coords(sprite_coords)
        sprite = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        sprite[sprite_coords[:, 0] - sr0, sprite_coords[:, 1] - sc0] = True
        assert sprite.shape[0] >= 2 and sprite.shape[1] >= 2, "sprite bbox must span both axes"
        ir0, ic0, ir1, ic1 = _interior_bbox(frame_bbox)
        assert ir1 > ir0 and ic1 > ic0, "frame interior must be at least 2x2"

        pairs.append(
            {
                "color": color,
                "frame_bbox": frame_bbox,
                "sprite_coords": sprite_coords,
                "sprite_bbox": (sr0, sc0, sr1, sc1),
                "sprite": sprite,
                "interior_bbox": (ir0, ic0, ir1, ic1),
                "interior_area": (ir1 - ir0 + 1) * (ic1 - ic0 + 1),
            }
        )
    return pairs


def _paint_masks(
    shape: tuple[int, int], pairs: list[dict]
) -> tuple[list[dict], np.ndarray]:
    border_union = np.zeros(shape, dtype=bool)
    for pair in pairs:
        border_union |= _frame_mask_from_bbox(shape, pair["frame_bbox"])

    painted: list[dict] = []
    for pair in pairs:
        ir0, ic0, ir1, ic1 = pair["interior_bbox"]
        stretched = _stretch_sprite(pair["sprite"], (ir1 - ir0 + 1, ic1 - ic0 + 1))
        mask = np.zeros(shape, dtype=bool)
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = stretched
        visible_mask = mask & ~border_union
        painted.append({**pair, "stretched": stretched, "paint_mask": mask, "visible_mask": visible_mask})
    return painted, border_union


def _render_from_pairs(grid: np.ndarray, pairs: list[dict], *, descending: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    painted, border_union = _paint_masks(g.shape, pairs)
    out = g.copy()
    for pair in painted:
        coords = pair["sprite_coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    order = sorted(painted, key=lambda item: item["interior_area"], reverse=descending)
    for pair in order:
        r0, c0, r1, c1 = pair["interior_bbox"]
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[pair["stretched"]] = int(pair["color"])

    original_borders = g * border_union.astype(int)
    out[border_union] = original_borders[border_union]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each colored sprite into its matching frame interior, then let larger frame interiors win any overlap.

    Input structure:
    - The background is black and the foreground uses several colors, each appearing in exactly two objects.
    - For each such color, one object is a one-pixel-thick hollow rectangular frame and the other is a small 4-connected sprite outside the frame.
    - Different frames may have overlapping interior rectangles, although all frame border pixels stay distinct.

    Rule:
    - Pair each sprite with the hollow frame of the same color.
    - For each pair, map the sprite bbox extremes to the frame-interior extremes, draw mapped row/column connections between adjacent sprite pixels, and fill any enclosed region.
    - Erase the original outside sprites, draw the stretched results in ascending order of frame-interior area, and finally preserve every frame border pixel unchanged so larger interiors overwrite smaller ones in shared interior cells.

    Key insight:
    - The sprite-to-frame transform is the same corner-anchored stretch-and-fill rule as the single-frame reference task, but the visible color in an overlap is decided only by comparing the frames' interior areas.
    """
    pairs = _extract_pairs(np.asarray(grid, dtype=int))
    areas = [pair["interior_area"] for pair in pairs]
    assert len(set(areas)) == len(areas), "frame interior areas must be pairwise distinct"
    return _render_from_pairs(np.asarray(grid, dtype=int), pairs, descending=False)


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _all_orientations(pattern: np.ndarray) -> list[np.ndarray]:
    seen: set[bytes] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(pattern, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = variant.astype(np.uint8).tobytes() + bytes(variant.shape)
            if key not in seen:
                seen.add(key)
                out.append(variant.astype(bool))
    return out


_SOLID_VARIANTS = [variant for base in SOLID_PATTERNS for variant in _all_orientations(base)]
_OPEN_VARIANTS = [variant for base in OPEN_PATTERNS for variant in _all_orientations(base)]


def _sample_distinct_interior_sizes() -> list[tuple[int, int]]:
    options = [(h, w) for h in range(INTERIOR_SIDE_RANGE[0], INTERIOR_SIDE_RANGE[1] + 1) for w in range(INTERIOR_SIDE_RANGE[0], INTERIOR_SIDE_RANGE[1] + 1)]
    while True:
        dims = random.sample(options, 3)
        areas = [h * w for h, w in dims]
        if len(set(areas)) != 3:
            continue
        return dims


def _interior_overlap_cells(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = _interior_bbox(a)
    br0, bc0, br1, bc1 = _interior_bbox(b)
    r0 = max(ar0, br0)
    c0 = max(ac0, bc0)
    r1 = min(ar1, br1)
    c1 = min(ac1, bc1)
    if r0 > r1 or c0 > c1:
        return set()
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _boxes_with_padding_disjoint(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0
) -> bool:
    return (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _candidate_positions(
    rows: int,
    cols: int,
    frame_size: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
    *,
    require_overlap_with: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int, int, int]]:
    fh, fw = frame_size
    out: list[tuple[int, int, int, int]] = []
    for r0 in range(rows - fh + 1):
        for c0 in range(cols - fw + 1):
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            border = _frame_mask_from_bbox((rows, cols), bbox)
            if any(np.any(border & _frame_mask_from_bbox((rows, cols), other)) for other in placed):
                continue
            if require_overlap_with is not None and not _interior_overlap_cells(bbox, require_overlap_with):
                continue
            out.append(bbox)
    return out


def _clean_overlap_pairs(
    shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]]
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    border_union = np.zeros(shape, dtype=bool)
    for bbox in bboxes:
        border_union |= _frame_mask_from_bbox(shape, bbox)

    overlaps: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            cells = {cell for cell in _interior_overlap_cells(bboxes[i], bboxes[j]) if not border_union[cell]}
            if cells:
                overlaps[(i, j)] = cells
    return overlaps


def _sample_frame_layout() -> tuple[int, int, list[tuple[int, int, int, int]], dict[tuple[int, int], set[tuple[int, int]]]] | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    interior_sizes = _sample_distinct_interior_sizes()
    frame_sizes = [(h + 2, w + 2) for h, w in interior_sizes]
    order = list(range(3))
    random.shuffle(order)
    shuffled_sizes = [frame_sizes[i] for i in order]

    placed: list[tuple[int, int, int, int]] = []
    fh0, fw0 = shuffled_sizes[0]
    center_r = max(0, (rows - fh0) // 2)
    center_c = max(0, (cols - fw0) // 2)
    first_options = [
        (r0, c0, r0 + fh0 - 1, c0 + fw0 - 1)
        for r0 in range(max(0, center_r - 3), min(rows - fh0, center_r + 3) + 1)
        for c0 in range(max(0, center_c - 3), min(cols - fw0, center_c + 3) + 1)
    ]
    if not first_options:
        return None
    placed.append(random.choice(first_options))

    second_options = _candidate_positions(rows, cols, shuffled_sizes[1], placed, require_overlap_with=placed[0])
    second_options = [bbox for bbox in second_options if 1 <= len(_interior_overlap_cells(bbox, placed[0])) <= 18]
    if not second_options:
        return None
    placed.append(random.choice(second_options))

    third_all = _candidate_positions(rows, cols, shuffled_sizes[2], placed)
    if not third_all:
        return None
    overlapping = []
    separated = []
    for bbox in third_all:
        overlaps = sum(bool(_interior_overlap_cells(bbox, other)) for other in placed)
        if overlaps:
            overlapping.append(bbox)
        else:
            separated.append(bbox)
    if random.random() < 0.6 and overlapping:
        placed.append(random.choice(overlapping))
    else:
        placed.append(random.choice(separated or overlapping))

    unshuffled: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * 3
    for idx, src in enumerate(order):
        unshuffled[src] = placed[idx]
    overlaps = _clean_overlap_pairs((rows, cols), unshuffled)
    if not overlaps:
        return None
    total_overlap = sum(len(cells) for cells in overlaps.values())
    total_area = sum(_bbox_area(_interior_bbox(bbox)) for bbox in unshuffled)
    if total_overlap > total_area // 3:
        return None
    return rows, cols, unshuffled, overlaps


def _sample_sprite_plan(
    bboxes: list[tuple[int, int, int, int]], overlaps: dict[tuple[int, int], set[tuple[int, int]]]
) -> list[np.ndarray] | None:
    best_pair = max(overlaps.items(), key=lambda item: len(item[1]))[0]
    patterns: list[np.ndarray | None] = [None, None, None]
    patterns[best_pair[0]] = random.choice(_SOLID_VARIANTS).copy()
    patterns[best_pair[1]] = random.choice(_SOLID_VARIANTS).copy()

    remaining = [idx for idx in range(3) if patterns[idx] is None]
    assert len(remaining) == 1
    patterns[remaining[0]] = random.choice(_OPEN_VARIANTS).copy()

    if random.random() < 0.4:
        swap_idx = random.choice(best_pair)
        patterns[swap_idx] = random.choice(_OPEN_VARIANTS).copy()

    result = [np.asarray(pattern, dtype=bool) for pattern in patterns]
    for idx, pattern in enumerate(result):
        ih = _bbox_area(_interior_bbox(bboxes[idx]))
        interior = _interior_bbox(bboxes[idx])
        stretched = _stretch_sprite(pattern, (interior[2] - interior[0] + 1, interior[3] - interior[1] + 1))
        if not np.any(stretched):
            return None
        if pattern.shape[0] < 2 or pattern.shape[1] < 2:
            return None
    return result


def _sprite_positions(
    rows: int,
    cols: int,
    frame_bboxes: list[tuple[int, int, int, int]],
    sprite_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    placed_rects = frame_bboxes[:]
    placements: list[tuple[int, int]] = []
    for shape in sprite_shapes:
        sh, sw = shape
        candidates = []
        for r0 in range(rows - sh + 1):
            for c0 in range(cols - sw + 1):
                rect = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if all(_boxes_with_padding_disjoint(rect, other, padding=SPRITE_PADDING) for other in placed_rects):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements.append((r0, c0))
        placed_rects.append((r0, c0, r0 + sh - 1, c0 + sw - 1))
    return placements


def _build_grid(
    rows: int,
    cols: int,
    frame_bboxes: list[tuple[int, int, int, int]],
    sprites: list[np.ndarray],
    sprite_positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for color, bbox in zip(FRAME_COLORS, frame_bboxes):
        mask = _frame_mask_from_bbox((rows, cols), bbox)
        grid[mask] = color
    for color, sprite, (r0, c0) in zip(FRAME_COLORS, sprites, sprite_positions):
        view = grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]]
        view[sprite] = color
    return grid


def _validate_example(grid: np.ndarray) -> bool:
    try:
        pairs = _extract_pairs(grid)
    except AssertionError:
        return False
    if len(pairs) != 3 or {pair["color"] for pair in pairs} != set(FRAME_COLORS):
        return False

    areas = [pair["interior_area"] for pair in pairs]
    if len(set(areas)) != 3:
        return False

    painted, border_union = _paint_masks(grid.shape, pairs)
    visible_masks = [pair["visible_mask"] for pair in painted]
    contest = np.zeros(grid.shape, dtype=int)
    for mask in visible_masks:
        contest += mask.astype(int)
    if not np.any(contest >= 2):
        return False

    nontrivial_count = 0
    for pair in painted:
        ir0, ic0, ir1, ic1 = pair["interior_bbox"]
        full = np.ones((ir1 - ir0 + 1, ic1 - ic0 + 1), dtype=bool)
        if not np.array_equal(pair["stretched"], full):
            nontrivial_count += 1
    if nontrivial_count == 0:
        return False

    out = main(grid)
    reversed_out = _render_from_pairs(grid, pairs, descending=True)
    if np.array_equal(out, reversed_out):
        return False

    for pair in pairs:
        border = _frame_mask_from_bbox(grid.shape, pair["frame_bbox"])
        if not np.array_equal(out[border], grid[border]):
            return False
        coords = pair["sprite_coords"]
        if np.any(out[coords[:, 0], coords[:, 1]] != 0):
            return False

    if _get_bg(grid) != 0:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    layout = _sample_frame_layout()
    if layout is None:
        return None
    rows, cols, frame_bboxes, overlaps = layout
    sprites = _sample_sprite_plan(frame_bboxes, overlaps)
    if sprites is None:
        return None
    positions = _sprite_positions(rows, cols, frame_bboxes, [sprite.shape for sprite in sprites])
    if positions is None:
        return None
    grid = _build_grid(rows, cols, frame_bboxes, sprites, positions)
    if not _validate_example(grid):
        return None
    out = main(grid)
    input_sig = _signature(grid)
    output_sig = _signature(out)
    if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_INPUT_SIGS.add(input_sig)
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
