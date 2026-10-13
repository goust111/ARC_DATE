from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CONTOUR = 7
MAROON = 9
PAIR_MAP = {
    1: 4,
    4: 1,
    2: 3,
    3: 2,
    5: 6,
    6: 5,
    7: 8,
    8: 7,
}
CONTROL_PALETTE = [1, 2, 3, 4, 5, 6]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = np.zeros_like(mask, dtype=bool)
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        comp[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _draw_rect_outline(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _strict_interior_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _full_rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _count_contour_crossings(grid: np.ndarray, col: int, stop_row: int) -> int:
    col_mask = grid[:stop_row, col] == CONTOUR
    if stop_row <= 0:
        return 0
    starts = col_mask & np.concatenate(([True], ~col_mask[:-1]))
    return int(starts.sum())


def _swap_color(color: int) -> int:
    assert color in PAIR_MAP, f"unsupported control color: {color}"
    return PAIR_MAP[color]


def _fill_color(control_color: int, crossings: int) -> int:
    if crossings == 1:
        return int(control_color)
    if crossings == 2:
        return _swap_color(int(control_color))
    if crossings == 3:
        return MAROON
    raise AssertionError(f"unexpected contour crossing count: {crossings}")


def _control_pixels(grid: np.ndarray) -> list[tuple[int, int]]:
    cols = np.flatnonzero(grid[0] != BG)
    return [(int(c), int(grid[0, c])) for c in cols]


def _region_ray(region: np.ndarray, control_cols: list[int]) -> int:
    region_cols = np.flatnonzero(region.any(axis=0))
    assert region_cols.size > 0
    candidates = [c for c in control_cols if region[:, c].any()]
    assert candidates, "every enclosed region must be crossed by a control ray"
    center = 0.5 * (int(region_cols.min()) + int(region_cols.max()))
    scored = sorted((abs(c - center), c) for c in candidates)
    if len(scored) >= 2:
        assert scored[0][0] != scored[1][0], "generator must avoid center-distance ties"
    return int(scored[0][1])


def _enclosed_regions(grid: np.ndarray) -> list[np.ndarray]:
    bg_components = _component_masks(grid == BG)
    return [comp for comp in bg_components if not _touches_edge(comp)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each orange-enclosed region using the chosen vertical ray's contour-crossing depth.

    Input structure:
    - The grid has black background, orange contour loops, and 2-3 distinct control pixels on the top row.
    - Orange pixels form one-pixel closed contour lines that partition background into exterior and enclosed 4-connected regions.
    - Each enclosed region is crossed by at least one control column, and some wider regions may be crossed by multiple control rays.

    Rule:
    - Find every background component that does not touch the grid edge.
    - For each enclosed region, consider the top-row control rays whose columns actually pass through that region; if several do, choose the one whose column is closest to the region's horizontal center.
    - Count how many separate orange contour runs that chosen ray crosses before first entering the region.
    - Crossing count 1 fills the region with the control pixel's color, count 2 fills it with that color's fixed swap-pair partner, and count 3 fills it maroon; contours and controls stay unchanged.

    Key insight:
    The fill color is encoded by ray depth through nested contours rather than by region size or position, so the same ray can color successive enclosed regions differently as it descends.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == CONTOUR), "expected orange contour pixels"

    controls = _control_pixels(g)
    assert 2 <= len(controls) <= 3, "expected 2-3 top-row control pixels"
    control_cols = [c for c, _ in controls]
    control_colors = {c: color for c, color in controls}
    assert len(set(control_colors.values())) == len(controls), "control colors must be distinct"

    out = g.copy()
    for region in _enclosed_regions(g):
        col = _region_ray(region, control_cols)
        first_row = int(np.flatnonzero(region[:, col])[0])
        crossings = _count_contour_crossings(g, col, first_row)
        out[region] = _fill_color(control_colors[col], crossings)
    return out


def _sample_control_columns(width: int, n_controls: int) -> list[int] | None:
    if n_controls == 2:
        gap = random.randint(4, 6)
        low = 4
        high = width - gap - 5
        if low > high:
            return None
        c0 = random.randint(low, high)
        return [c0, c0 + gap]

    gap1 = random.randint(4, 6)
    gap2 = random.randint(4, 6)
    total = gap1 + gap2
    low = 4
    high = width - total - 5
    if low > high:
        return None
    c0 = random.randint(low, high)
    return [c0, c0 + gap1, c0 + gap1 + gap2]


def _main_stack_rects(
    shape: tuple[int, int],
    left_col: int,
    right_col: int,
    prefer_right: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = shape
    top = random.randint(2, 4)
    outer_h = random.randint(11, min(15, h - top - 2))
    r0 = top
    r1 = r0 + outer_h - 1

    small_pad = random.randint(1, 2)
    large_pad = random.randint(5, 6)
    if prefer_right:
        c0 = left_col - small_pad
        c1 = right_col + large_pad
    else:
        c0 = left_col - large_pad
        c1 = right_col + small_pad

    if c0 < 1 or c1 > w - 2 or c1 - c0 + 1 < 11:
        return None

    outer = (r0, c0, r1, c1)
    middle = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    inner = (r0 + 4, c0 + 4, r1 - 4, c1 - 4)
    if inner[2] - inner[0] < 2 or inner[3] - inner[1] < 2:
        return None
    return outer, middle, inner


def _side_stack_rects(
    shape: tuple[int, int],
    main_outer: tuple[int, int, int, int],
    control_col: int,
    depth: int,
    on_left: bool,
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    need_width = 7 if depth == 2 else 5
    half = need_width // 2
    if on_left:
        max_c1 = main_outer[1] - 3
        c0 = control_col - half
        c1 = control_col + half
        if c0 < 1:
            c1 += 1 - c0
            c0 = 1
        if c1 > max_c1:
            return None
    else:
        min_c0 = main_outer[3] + 3
        c0 = control_col - half
        c1 = control_col + half
        if c1 > w - 2:
            c0 -= c1 - (w - 2)
            c1 = w - 2
        if c0 < min_c0:
            return None

    outer_h = random.randint(7 if depth == 2 else 5, min(11, h - 4))
    top_low = 2
    top_high = h - outer_h - 2
    if top_low > top_high:
        return None
    r0 = random.randint(top_low, top_high)
    outer = (r0, c0, r0 + outer_h - 1, c1)
    if depth == 1:
        return [outer]
    inner = (r0 + 2, c0 + 2, r0 + outer_h - 3, c1 - 2)
    if inner[2] - inner[0] < 2 or inner[3] - inner[1] < 2:
        return None
    return [outer, inner]


def _rect_region_masks(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    if len(rects) == 1:
        return [_strict_interior_mask(shape, rects[0])]
    if len(rects) == 2:
        outer, inner = rects
        return [
            _strict_interior_mask(shape, outer) & ~_full_rect_mask(shape, inner),
            _strict_interior_mask(shape, inner),
        ]
    outer, middle, inner = rects
    return [
        _strict_interior_mask(shape, outer) & ~_full_rect_mask(shape, middle),
        _strict_interior_mask(shape, middle) & ~_full_rect_mask(shape, inner),
        _strict_interior_mask(shape, inner),
    ]


def _mask_signature(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset(map(tuple, np.argwhere(mask)))


def _semantic_check(grid: np.ndarray, expected: np.ndarray, intended_regions: list[tuple[np.ndarray, int]]) -> bool:
    actual_regions = _enclosed_regions(grid)
    if len(actual_regions) != len(intended_regions):
        return False

    expected_sigs = {_mask_signature(mask): color for mask, color in intended_regions}
    actual_sigs = {_mask_signature(mask) for mask in actual_regions}
    if set(expected_sigs) != actual_sigs:
        return False

    controls = _control_pixels(grid)
    control_cols = [c for c, _ in controls]
    control_colors = {c: color for c, color in controls}
    seen_crossings = set()
    seen_multi_ray = False
    for region in actual_regions:
        sig = _mask_signature(region)
        col_candidates = [c for c in control_cols if region[:, c].any()]
        if len(col_candidates) >= 2:
            seen_multi_ray = True
        chosen = _region_ray(region, control_cols)
        first_row = int(np.flatnonzero(region[:, chosen])[0])
        crossings = _count_contour_crossings(grid, chosen, first_row)
        seen_crossings.add(crossings)
        if _fill_color(control_colors[chosen], crossings) != expected_sigs[sig]:
            return False

    if seen_crossings != {1, 2, 3}:
        return False
    if not seen_multi_ray:
        return False
    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected) and not np.array_equal(grid, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    n_controls = random.choice([2, 3])
    control_cols = _sample_control_columns(w, n_controls)
    if control_cols is None:
        return None
    control_colors = random.sample(CONTROL_PALETTE, n_controls)

    grid = np.full((h, w), BG, dtype=int)
    for col, color in zip(control_cols, control_colors):
        grid[0, col] = color

    if n_controls == 2:
        pair_start = 0
    else:
        pair_start = random.choice([0, 1])
    left_idx = pair_start
    right_idx = pair_start + 1
    prefer_right = random.choice([False, True])
    preferred_idx = right_idx if prefer_right else left_idx
    preferred_color = control_colors[preferred_idx]

    rects = _main_stack_rects((h, w), control_cols[left_idx], control_cols[right_idx], prefer_right)
    if rects is None:
        return None
    outer, middle, inner = rects
    for rect in rects:
        _draw_rect_outline(grid, rect, CONTOUR)

    intended_regions: list[tuple[np.ndarray, int]] = []
    for depth, mask in enumerate(_rect_region_masks((h, w), list(rects)), start=1):
        intended_regions.append((mask, _fill_color(preferred_color, depth)))

    if n_controls == 3:
        spare_idx = 2 if pair_start == 0 else 0
        spare_col = control_cols[spare_idx]
        spare_color = control_colors[spare_idx]
        on_left = spare_col < outer[1]
        extra_depth = random.choice([1, 2])
        extra_rects = _side_stack_rects((h, w), outer, spare_col, extra_depth, on_left)
        if extra_rects is None:
            return None
        for rect in extra_rects:
            _draw_rect_outline(grid, rect, CONTOUR)
        for depth, mask in enumerate(_rect_region_masks((h, w), extra_rects), start=1):
            if np.any(mask & (grid != BG)):
                return None
            intended_regions.append((mask, _fill_color(spare_color, depth)))

    for mask, _ in intended_regions:
        if not mask.any():
            return None
        if np.any(grid[mask] != BG):
            return None
    for i, (mask_i, _) in enumerate(intended_regions):
        for mask_j, _ in intended_regions[i + 1 :]:
            if np.any(mask_i & mask_j):
                return None

    expected = grid.copy()
    for mask, color in intended_regions:
        expected[mask] = color

    if not _semantic_check(grid, expected, intended_regions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-ray sample")
