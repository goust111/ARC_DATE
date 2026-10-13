from __future__ import annotations

import numpy as np

from arc_utils import Color, components_mask, find_components


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _infer_bg_and_foreground(g: np.ndarray) -> tuple[int, int]:
    colors = {int(c) for c in np.unique(g)}
    assert len(colors) == 2, "expected one background color and one foreground color"

    comps_by_color = {color: components_mask(g == color, connectivity=4) for color in colors}
    bg_candidates = [color for color, comps in comps_by_color.items() if len(comps) == 1]
    fg_candidates = [color for color, comps in comps_by_color.items() if len(comps) > 1]
    assert len(bg_candidates) == 1, "expected a unique connected background color"
    assert len(fg_candidates) == 1, "expected a unique multi-object foreground color"
    return bg_candidates[0], fg_candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill unobstructed horizontal gaps between neighboring foreground objects with maroon.

    Input structure
    - The grid has one connected background color and one foreground color split into several 4-connected solid rectangles.
    - Foreground rectangles can be horizontally separated while overlapping in their row spans.
    Rule
    - For every pair of foreground rectangles with separated column spans and overlapping row spans, consider the rectangle between their bounding boxes.
    - Paint that between-box region maroon when no third foreground rectangle has a bounding box crossing both that horizontal gap and the same shared row span.
    - Keep all original pixels unchanged.
    Key insight
    - The bridges are determined by adjacency of object bounding boxes in horizontal layers, not by scanning each row independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, foreground = _infer_bg_and_foreground(g)
    bridge = Color.MAROON
    assert bridge not in set(map(int, np.unique(g))), "maroon is reserved for the added bridges"

    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[foreground],
        return_format="dict",
    )
    assert len(objects) >= 2, "expected multiple foreground rectangles"

    boxes = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert obj["area"] == expected_area, "foreground objects must be solid rectangles"
        boxes.append((r0, c0, r1, c1))

    out = g.copy()
    painted_any = False
    for i in range(len(boxes)):
        ar0, ac0, ar1, ac1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            br0, bc0, br1, bc1 = boxes[j]
            if ac1 < bc0:
                left = (ar0, ac0, ar1, ac1)
                right = (br0, bc0, br1, bc1)
            elif bc1 < ac0:
                left = (br0, bc0, br1, bc1)
                right = (ar0, ac0, ar1, ac1)
            else:
                continue

            lr0, _, lr1, lc1 = left
            rr0, rc0, rr1, _ = right
            row0 = max(lr0, rr0)
            row1 = min(lr1, rr1)
            if row0 > row1:
                continue

            gap0 = lc1 + 1
            gap1 = rc0 - 1
            assert gap0 <= gap1, "separate rectangles should leave a nonempty gap"

            blocked = False
            for k, (blocker_r0, blocker_c0, blocker_r1, blocker_c1) in enumerate(boxes):
                if k in (i, j):
                    continue
                blocks_gap = _ranges_overlap(gap0, gap1, blocker_c0, blocker_c1)
                blocks_rows = _ranges_overlap(row0, row1, blocker_r0, blocker_r1)
                if blocks_gap and blocks_rows:
                    blocked = True
                    break
            if blocked:
                continue

            assert np.all(g[row0 : row1 + 1, gap0 : gap1 + 1] == bg), "bridge region must be empty"
            out[row0 : row1 + 1, gap0 : gap1 + 1] = bridge
            painted_any = True

    assert painted_any, "expected at least one horizontal bridge"
    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
_GEN_GRID_SIZE = 10
_GEN_LAST_INDEX = _GEN_GRID_SIZE - 1
_GEN_BACKGROUND = 0
_GEN_FOREGROUND = Color.RED
_GEN_BRIDGE = Color.MAROON

# Rectangle dimension and coverage bounds.
_GEN_MIN_RECT_HEIGHT = 2
_GEN_MAX_RECT_HEIGHT = 6
_GEN_MIN_RECT_WIDTH = 1
_GEN_MAX_RECT_WIDTH = 4
_GEN_MIN_RECT_AREA = 4
_GEN_MAX_RECT_AREA = 18
_GEN_MIN_TOTAL_AREA = 24
_GEN_MAX_TOTAL_AREA = 54
_GEN_MAX_UNUSED_WIDTH = 1

# Sampling weights for rectangle count and dimensions.
_GEN_RECT_COUNTS = (3, 4, 5)
_GEN_RECT_COUNT_PROBS = (0.55, 0.30, 0.15)
_GEN_DIM_AREA_EXP = 2.0
_GEN_DIM_WIDTH_EXP = 3.0
_GEN_DIM_HEIGHT_EXP = 1.5

# Horizontal layout gap patterns.
_GEN_GAP_PATTERNS = {
    3: ((1, 5), (5, 1), (2, 4), (4, 2), (3, 3), (2, 3), (3, 2), (1, 4), (4, 1), (1, 3), (3, 1), (2, 2), (1, 2), (2, 1), (1, 1)),
    4: ((4, 1, 1), (1, 4, 1), (1, 1, 4), (3, 2, 1), (3, 1, 2), (2, 3, 1), (1, 3, 2), (2, 1, 3), (1, 2, 3), (2, 2, 2),
        (3, 1, 1), (1, 3, 1), (1, 1, 3), (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 1, 1)),
    5: ((2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2), (1, 1, 1, 1)),
}

# Bridge geometry and retry limits.
_GEN_MIN_ROW_OVERLAP = 1
_GEN_MAX_ROW_OVERLAP = 3
_GEN_MIN_BRIDGES = 2
_GEN_MAX_BRIDGES = 4
_GEN_MIN_GAP_WIDTH = 1
_GEN_MAX_GAP_WIDTH = 5
_GEN_MIN_BRIDGE_HEIGHT = 1
_GEN_MAX_BRIDGE_HEIGHT = 3
_GEN_FLIP_PROB = 0.5
_GEN_RETRY_LIMIT = 300

_Rect = tuple[int, int, int, int]
_Region = tuple[int, int, int, int]

_GEN_DIM_OPTIONS = tuple(
    (h, w)
    for h in range(_GEN_MIN_RECT_HEIGHT, _GEN_MAX_RECT_HEIGHT + 1)
    for w in range(_GEN_MIN_RECT_WIDTH, _GEN_MAX_RECT_WIDTH + 1)
    if _GEN_MIN_RECT_AREA <= h * w <= _GEN_MAX_RECT_AREA
)


def _gen_shuffled(seq):
    order = np.random.permutation(len(seq))
    return [seq[int(i)] for i in order]


def _gen_weighted_dims() -> list[tuple[int, int]]:
    remaining = list(_GEN_DIM_OPTIONS)
    ordered = []
    while remaining:
        weights = np.array(
            [((h * w) ** _GEN_DIM_AREA_EXP) * (w**_GEN_DIM_WIDTH_EXP) / (h**_GEN_DIM_HEIGHT_EXP) for h, w in remaining],
            dtype=float,
        )
        weights /= weights.sum()
        idx = int(np.random.choice(len(remaining), p=weights))
        ordered.append(remaining.pop(idx))
    return ordered


def _gen_choose_gaps(n: int) -> list[int]:
    patterns = _GEN_GAP_PATTERNS[n]
    return list(patterns[int(np.random.randint(0, len(patterns)))])


def _gen_choose_dimensions(n: int, max_width_sum: int) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []

    def rec(i: int, width_sum: int, area_sum: int) -> list[tuple[int, int]] | None:
        if i == n:
            if (
                _GEN_MIN_TOTAL_AREA <= area_sum <= _GEN_MAX_TOTAL_AREA
                and width_sum >= max_width_sum - _GEN_MAX_UNUSED_WIDTH
            ):
                return list(chosen)
            return None

        remaining = n - i - 1
        for h, w in _gen_weighted_dims():
            new_width_sum = width_sum + w
            new_area_sum = area_sum + h * w
            if new_width_sum + _GEN_MIN_RECT_WIDTH * remaining > max_width_sum:
                continue
            if (
                new_area_sum > _GEN_MAX_TOTAL_AREA
                or new_area_sum + _GEN_MAX_RECT_AREA * remaining < _GEN_MIN_TOTAL_AREA
            ):
                continue
            chosen.append((h, w))
            result = rec(i + 1, new_width_sum, new_area_sum)
            if result is not None:
                return result
            chosen.pop()
        return None

    return rec(0, 0, 0)


def _gen_column_intervals(widths: list[int], gaps: list[int]) -> list[tuple[int, int]] | None:
    used = sum(widths) + sum(gaps)
    if used > _GEN_GRID_SIZE:
        return None
    left_margin = int(np.random.randint(0, _GEN_GRID_SIZE + 1 - used))
    intervals = []
    c = left_margin
    for i, w in enumerate(widths):
        intervals.append((c, c + w - 1))
        c += w
        if i < len(gaps):
            c += gaps[i]
    return intervals


def _gen_choose_overlaps(heights: list[int]) -> list[int] | None:
    overlaps = []
    for i in range(len(heights) - 1):
        lo = max(_GEN_MIN_ROW_OVERLAP, heights[i] + heights[i + 1] - _GEN_GRID_SIZE)
        hi = min(_GEN_MAX_ROW_OVERLAP, heights[i], heights[i + 1])
        if lo > hi:
            return None
        overlaps.append(int(np.random.randint(lo, hi + 1)))
    return overlaps


def _gen_row_intervals(heights: list[int], overlaps: list[int]) -> list[tuple[int, int]] | None:
    starts = _gen_shuffled(list(range(0, _GEN_GRID_SIZE + 1 - heights[0])))
    intervals: list[tuple[int, int]] = []

    def rec(i: int) -> list[tuple[int, int]] | None:
        if i == len(heights):
            return list(intervals)

        pr0, pr1 = intervals[-1]
        h = heights[i]
        overlap = overlaps[i - 1]
        candidates = []
        for r0 in (pr1 - overlap + 1, pr0 + overlap - h):
            if 0 <= r0 <= _GEN_GRID_SIZE - h and r0 not in candidates:
                candidates.append(r0)

        for r0 in _gen_shuffled(candidates):
            intervals.append((r0, r0 + h - 1))
            result = rec(i + 1)
            if result is not None:
                return result
            intervals.pop()
        return None

    for start in starts:
        intervals[:] = [(start, start + heights[0] - 1)]
        result = rec(1)
        if result is not None:
            return result
    return None


def _gen_transform_rects(rects: list[_Rect]) -> list[_Rect]:
    transformed = list(rects)
    if np.random.random() < _GEN_FLIP_PROB:
        transformed = [(_GEN_LAST_INDEX - r1, c0, _GEN_LAST_INDEX - r0, c1) for r0, c0, r1, c1 in transformed]
    if np.random.random() < _GEN_FLIP_PROB:
        transformed = [(r0, _GEN_LAST_INDEX - c1, r1, _GEN_LAST_INDEX - c0) for r0, c0, r1, c1 in transformed]
    return transformed


def _gen_order_horizontal_pair(a: _Rect, b: _Rect) -> tuple[_Rect, _Rect] | None:
    if a[3] < b[1]:
        return a, b
    if b[3] < a[1]:
        return b, a
    return None


def _gen_gap_region(left: _Rect, right: _Rect) -> _Region | None:
    lr0, _, lr1, lc1 = left
    rr0, rc0, rr1, _ = right
    row0 = max(lr0, rr0)
    row1 = min(lr1, rr1)
    if row0 > row1:
        return None

    gap0 = lc1 + 1
    gap1 = rc0 - 1
    if gap0 > gap1:
        return None
    return row0, gap0, row1, gap1


def _gen_region_is_blocked(rects: list[_Rect], region: _Region, skip: tuple[int, int]) -> bool:
    row0, gap0, row1, gap1 = region
    for k, (kr0, kc0, kr1, kc1) in enumerate(rects):
        if k in skip:
            continue
        if _ranges_overlap(gap0, gap1, kc0, kc1) and _ranges_overlap(row0, row1, kr0, kr1):
            return True
    return False


def _gen_bridge_regions(rects: list[_Rect]) -> list[_Region]:
    regions = []
    for i, a in enumerate(rects):
        for j in range(i + 1, len(rects)):
            ordered = _gen_order_horizontal_pair(a, rects[j])
            if ordered is None:
                continue

            region = _gen_gap_region(*ordered)
            if region is None:
                continue

            if not _gen_region_is_blocked(rects, region, (i, j)):
                regions.append(region)
    return regions


def _gen_design_rects(n: int, gaps: list[int]) -> list[_Rect] | None:
    """Sample rectangle boxes from dimensions, columns, row overlaps, and mirror flips."""
    dims = _gen_choose_dimensions(n, _GEN_GRID_SIZE - sum(gaps))
    if dims is None:
        return None

    heights = [h for h, _ in dims]
    widths = [w for _, w in dims]
    col_intervals = _gen_column_intervals(widths, gaps)
    overlaps = _gen_choose_overlaps(heights)
    if col_intervals is None or overlaps is None:
        return None

    row_intervals = _gen_row_intervals(heights, overlaps)
    if row_intervals is None:
        return None

    return _gen_transform_rects(
        [(r0, c0, r1, c1) for (r0, r1), (c0, c1) in zip(row_intervals, col_intervals)]
    )


def _gen_render_rects(rects: list[_Rect]) -> np.ndarray:
    inp = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    for r0, c0, r1, c1 in rects:
        inp[r0 : r1 + 1, c0 : c1 + 1] = _GEN_FOREGROUND
    return inp


def _gen_component_counts_match(inp: np.ndarray, n: int) -> bool:
    if len(components_mask(inp == _GEN_BACKGROUND, connectivity=4)) != 1:
        return False
    return len(components_mask(inp == _GEN_FOREGROUND, connectivity=4)) == n


def _gen_regions_are_valid(inp: np.ndarray, regions: list[_Region]) -> bool:
    if not _GEN_MIN_BRIDGES <= len(regions) <= _GEN_MAX_BRIDGES:
        return False
    for r0, c0, r1, c1 in regions:
        gap_width = c1 - c0 + 1
        bridge_height = r1 - r0 + 1
        if not (_GEN_MIN_GAP_WIDTH <= gap_width <= _GEN_MAX_GAP_WIDTH):
            return False
        if not (_GEN_MIN_BRIDGE_HEIGHT <= bridge_height <= _GEN_MAX_BRIDGE_HEIGHT):
            return False
        if not np.all(inp[r0 : r1 + 1, c0 : c1 + 1] == _GEN_BACKGROUND):
            return False
    return True


def _gen_apply_bridges(inp: np.ndarray, regions: list[_Region]) -> np.ndarray:
    out = inp.copy()
    for r0, c0, r1, c1 in regions:
        out[r0 : r1 + 1, c0 : c1 + 1] = _GEN_BRIDGE
    return out


def _gen_solver_matches(inp: np.ndarray, regions: list[_Region]) -> bool:
    expected = _gen_apply_bridges(inp, regions)
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(_GEN_RECT_COUNTS, p=_GEN_RECT_COUNT_PROBS))
    gaps = _gen_choose_gaps(n)
    rects = _gen_design_rects(n, gaps)
    if rects is None:
        return None

    inp = _gen_render_rects(rects)
    if not _gen_component_counts_match(inp, n):
        return None

    regions = _gen_bridge_regions(rects)
    if not _gen_regions_are_valid(inp, regions):
        return None
    if not _gen_solver_matches(inp, regions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ef135b50 input")
