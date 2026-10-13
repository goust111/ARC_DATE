from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
MAROON = 9
UPPER_CODE = GREY
LOWER_CODE = MAROON
LEGEND_H = 2
LEGEND_W = 3
SOURCE_POOL = (1, 2, 3, 4, 6, 7, 8)
SIZE_RETRIES = 200
PLACEMENT_RETRIES = 400
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_legends(grid: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    assert w >= LEGEND_W, "grid too narrow for legend"

    legend_mask = np.zeros_like(grid, dtype=bool)
    mapping: dict[int, tuple[int, int]] = {}
    r = 0
    while r <= h - LEGEND_H:
        block = grid[r : r + LEGEND_H, :LEGEND_W]
        if block.shape != (LEGEND_H, LEGEND_W):
            break
        src = int(block[0, 0])
        target = int(block[0, 1])
        code = int(block[1, 1])
        valid = (
            np.all(block != bg)
            and np.all(block[:, 0] == src)
            and np.all(block[0, 1:] == target)
            and np.all(block[1, 1:] == code)
            and src not in (bg, GREY, MAROON)
            and target not in (bg, GREY, MAROON)
            and src != target
            and code in (UPPER_CODE, LOWER_CODE)
            and not legend_mask[r : r + LEGEND_H, :LEGEND_W].any()
        )
        if valid:
            assert src not in mapping, "legend source colors must be unique"
            mapping[src] = (target, code)
            legend_mask[r : r + LEGEND_H, :LEGEND_W] = True
            r += LEGEND_H + 1
        else:
            r += 1

    assert 1 <= len(mapping) <= 2, "expected 1-2 left-edge legend blocks"
    assert not np.any((grid[:, :LEGEND_W] != bg) & ~legend_mask[:, :LEGEND_W]), "left-edge foreground must belong to legends"
    return mapping, legend_mask


def _split_component(
    coords: np.ndarray, code: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    height = r1 - r0 + 1
    assert height % 2 == 0, "sprite bbox height must be even"
    split = r0 + height // 2

    selected: list[tuple[int, int]] = []
    other: list[tuple[int, int]] = []
    for r, c in map(tuple, coords):
        if (code == UPPER_CODE and r < split) or (code == LOWER_CODE and r >= split):
            selected.append((int(r), int(c)))
        else:
            other.append((int(r), int(c)))

    assert selected and other, "both bbox halves must contain sprite pixels"
    return selected, other, (r0, c0, r1, c1)


def _solve_with_options(
    grid: np.ndarray,
    mapping: dict[int, tuple[int, int]],
    legend_mask: np.ndarray,
    *,
    use_opposite_half: bool = False,
    dilate_mode: str = "selected",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)
    h, w = g.shape

    for source, (target, code) in mapping.items():
        actual_code = LOWER_CODE if (use_opposite_half and code == UPPER_CODE) else UPPER_CODE if (use_opposite_half and code == LOWER_CODE) else code
        comps = _component_coords((g == source) & ~legend_mask)
        assert comps, "every legend source must label at least one sprite"
        for coords in comps:
            selected, _other, _bbox = _split_component(coords, actual_code)
            for r, c in selected:
                out[r, c] = target

            if dilate_mode == "none":
                seeds = []
            elif dilate_mode == "whole":
                seeds = [tuple(map(int, rc)) for rc in coords]
            else:
                seeds = selected

            additions: set[tuple[int, int]] = set()
            for r, c in seeds:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == bg:
                        additions.add((nr, nc))
            for nr, nc in additions:
                out[nr, nc] = target
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the legend-selected half of each sprite and dilate only that recolored half by one layer.

    Input structure:
    - The grid has a black background.
    - The left edge contains 1-2 full 2x3 legend blocks; in each block the left column is a source color, the top-right 1x2 strip repeats the target color, and the bottom-right 1x2 strip repeats the half code (grey=upper half, maroon=lower half).
    - To the right are 2-3 separated 4-connected monochrome sprites whose colors match the legend source colors, and every sprite has an even-height tight bounding box.

    Rule:
    - Read each legend block as source color -> (target color, chosen half).
    - For every sprite of that source color, split its tight bounding box across the horizontal midpoint into upper and lower halves.
    - Recolor only the legend-selected half of the sprite to the target color.
    - Then dilate exactly that recolored half by one 4-neighbor layer onto background cells only, without overwriting the untouched half or the legends.

    Key insight:
    - The left legend carries two coupled instructions: one color maps to the repaint target, while the grey/maroon code decides which bbox half changes, and only those changed pixels seed the outward growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    mapping, legend_mask = _parse_legends(g, bg)
    return _solve_with_options(g, mapping, legend_mask, dilate_mode="selected")


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _random_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(SIZE_RETRIES):
        h = rng.choice((4, 6))
        w = rng.choice((3, 4, 5))
        widths = [rng.randint(1, min(3, w)) for _ in range(h)]
        intervals: list[tuple[int, int]] = [(0, widths[0] - 1)]

        ok = True
        for row in range(1, h - 1):
            width = widths[row]
            prev_l, prev_r = intervals[-1]
            left_lo = max(0, prev_l - width + 1)
            left_hi = min(prev_r, w - width)
            if left_lo > left_hi:
                ok = False
                break
            left = rng.randint(left_lo, left_hi)
            intervals.append((left, left + width - 1))
        if not ok:
            continue

        last_width = widths[-1]
        last_interval = (w - last_width, w - 1)
        if not _intervals_overlap(intervals[-1], last_interval):
            continue
        intervals.append(last_interval)

        mask = np.zeros((h, w), dtype=bool)
        for r, (l, rr) in enumerate(intervals):
            mask[r, l : rr + 1] = True

        area = int(mask.sum())
        if not (6 <= area <= 12):
            continue
        if area == h * w:
            continue
        if np.all(mask == mask[0]):
            continue
        if np.all(mask == mask[:, [0]]):
            continue

        coords = np.argwhere(mask)
        if _bbox_from_coords(coords) != (0, 0, h - 1, w - 1):
            continue
        top = mask[: h // 2]
        bottom = mask[h // 2 :]
        if not top.any() or not bottom.any():
            continue
        if len(_component_coords(mask)) != 1:
            continue
        return mask
    return None


def _box_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return row_gap >= gap or col_gap >= gap


def _dilation_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    out: set[tuple[int, int]] = set()
    for r, c in map(tuple, np.argwhere(mask)):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((nr, nc))
    return out


def _place_objects(
    rng: random.Random, shape_masks: list[np.ndarray], grid_shape: tuple[int, int]
) -> list[tuple[np.ndarray, int, int]] | None:
    h, w = grid_shape
    placed: list[tuple[np.ndarray, int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for mask in shape_masks:
        mh, mw = mask.shape
        if mw + 7 > w or mh + 2 > h:
            return None
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, h - mh - 1)
            c0 = rng.randint(7, w - mw - 1)
            box = (r0, c0, r0 + mh - 1, c0 + mw - 1)
            if any(not _box_separated(box, other, 3) for other in boxes):
                continue
            placed.append((mask, r0, c0))
            boxes.append(box)
            break
        else:
            return None
    return placed


def _build_grid(
    rng: random.Random,
) -> tuple[np.ndarray, dict[int, tuple[int, int]], np.ndarray] | None:
    h = rng.randint(15, 22)
    w = rng.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)

    n_legends = 1 if rng.random() < 0.45 else 2
    total_objects = 2 if rng.random() < 0.45 else 3

    colors = list(SOURCE_POOL)
    rng.shuffle(colors)
    sources = colors[:n_legends]
    targets = colors[n_legends : 2 * n_legends]

    codes = [rng.choice((UPPER_CODE, LOWER_CODE)) for _ in range(n_legends)]
    if n_legends == 2 and rng.random() < 0.7:
        codes = [UPPER_CODE, LOWER_CODE]
        rng.shuffle(codes)

    mapping = {src: (dst, code) for src, dst, code in zip(sources, targets, codes)}

    legend_rows: list[int]
    if n_legends == 1:
        legend_rows = [rng.randint(1, h - LEGEND_H - 1)]
    else:
        top = rng.randint(1, h - 6)
        gap = rng.randint(2, 4)
        second = top + LEGEND_H + gap
        if second + LEGEND_H >= h:
            return None
        legend_rows = [top, second]

    legend_mask = np.zeros_like(grid, dtype=bool)
    for row, src in zip(legend_rows, sources):
        dst, code = mapping[src]
        block = np.array([[src, dst, dst], [src, code, code]], dtype=int)
        grid[row : row + LEGEND_H, :LEGEND_W] = block
        legend_mask[row : row + LEGEND_H, :LEGEND_W] = True

    counts = [1] * n_legends
    for _ in range(total_objects - n_legends):
        counts[rng.randrange(n_legends)] += 1

    object_specs: list[tuple[int, np.ndarray]] = []
    for src, count in zip(sources, counts):
        for _ in range(count):
            mask = _random_mask(rng)
            if mask is None:
                return None
            object_specs.append((src, mask))
    rng.shuffle(object_specs)

    placements = _place_objects(rng, [mask for _src, mask in object_specs], (h, w))
    if placements is None:
        return None

    info = list(zip(object_specs, placements))
    for (src, mask), (_mask_copy, r0, c0) in info:
        mh, mw = mask.shape
        grid[r0 : r0 + mh, c0 : c0 + mw][mask] = src

    if np.any(grid[:, :LEGEND_W][~legend_mask[:, :LEGEND_W]] != BG):
        return None
    return grid, mapping, legend_mask


def _validate_case(grid: np.ndarray, mapping: dict[int, tuple[int, int]], legend_mask: np.ndarray) -> bool:
    try:
        expected = main(grid)
    except Exception:
        return False

    if np.array_equal(expected, grid):
        return False

    parsed_mapping, parsed_legend_mask = _parse_legends(grid, _get_bg(grid))
    if parsed_mapping != mapping or not np.array_equal(parsed_legend_mask, legend_mask):
        return False

    if not np.array_equal(expected, _solve_with_options(grid, mapping, legend_mask, dilate_mode="selected")):
        return False

    recolor_only = _solve_with_options(grid, mapping, legend_mask, dilate_mode="none")
    opposite_half = _solve_with_options(grid, mapping, legend_mask, use_opposite_half=True, dilate_mode="selected")
    whole_dilate = _solve_with_options(grid, mapping, legend_mask, dilate_mode="whole")
    if np.array_equal(expected, recolor_only):
        return False
    if np.array_equal(expected, opposite_half):
        return False
    if np.array_equal(expected, whole_dilate):
        return False

    changed = expected != grid
    if not changed.any():
        return False

    added_cells = (grid == BG) & (expected != BG)
    if not added_cells.any():
        return False

    legend_count = len(mapping)
    if not (1 <= legend_count <= 2):
        return False

    codes_seen = {code for _target, code in mapping.values()}
    if not codes_seen.issubset({UPPER_CODE, LOWER_CODE}):
        return False

    for source, (target, code) in mapping.items():
        comps = _component_coords((grid == source) & ~legend_mask)
        if not comps:
            return False
        for coords in comps:
            selected, other, bbox = _split_component(coords, code)
            if not selected or not other:
                return False

            selected_mask = np.zeros_like(grid, dtype=bool)
            for r, c in selected:
                selected_mask[r, c] = True
            ring = _dilation_cells(selected_mask)
            ring = {(r, c) for r, c in ring if grid[r, c] == BG}
            if not ring:
                return False
            if any(expected[r, c] != target for r, c in ring):
                return False

            r0, c0, r1, c1 = bbox
            if (r1 - r0 + 1) % 2 != 0:
                return False

    object_boxes = []
    for source in mapping:
        for coords in _component_coords((grid == source) & ~legend_mask):
            object_boxes.append(_bbox_from_coords(coords))
    if not (2 <= len(object_boxes) <= 3):
        return False
    for i, box_a in enumerate(object_boxes):
        if box_a[1] < 7:
            return False
        for box_b in object_boxes[i + 1 :]:
            if not _box_separated(box_a, box_b, 3):
                return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        built = _build_grid(rng)
        if built is None:
            continue
        grid, mapping, legend_mask = built
        if _validate_case(grid, mapping, legend_mask):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
