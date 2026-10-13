from __future__ import annotations

import random

import numpy as np


DIVIDER = 0
REGION_BG = 8
BAR_COLOR = 5
RANK_COLORS = (2, 3, 4, 7, 6, 9)

N_CHOICES = (3, 4, 5, 6)
N_WEIGHTS = (1, 2, 3, 4)
M_CHOICES = (2, 3, 4)
M_WEIGHTS = (3, 3, 2)
MAX_GENERATION_ATTEMPTS = 500


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _spans_from_full_lines(full_line_mask: np.ndarray) -> list[tuple[int, int]]:
    indices = [int(idx) for idx in np.flatnonzero(full_line_mask)]
    spans: list[tuple[int, int]] = []
    start = 0
    for run_start, run_end in _group_runs(indices):
        if start < run_start:
            spans.append((start, run_start))
        start = run_end + 1
    if start < len(full_line_mask):
        spans.append((start, len(full_line_mask)))
    return spans


def _partition_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _spans_from_full_lines(np.all(grid == DIVIDER, axis=1))
    col_spans = _spans_from_full_lines(np.all(grid == DIVIDER, axis=0))
    assert row_spans and col_spans, "expected at least one partition row and one partition column"
    return row_spans, col_spans


def _extract_bar_height(region: np.ndarray) -> int:
    assert region.size > 0
    allowed = {DIVIDER, REGION_BG, BAR_COLOR}
    assert set(map(int, np.unique(region))) <= allowed
    assert int((region == DIVIDER).sum()) == 0, "divider cells must not appear inside a region"

    coords = np.argwhere(region == BAR_COLOR)
    assert len(coords) > 0, "expected one vertical bar per region"
    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, "bar must be one cell wide"
    rows = np.sort(coords[:, 0])
    assert int(rows[-1] - rows[0] + 1) == len(rows), "bar must be vertically contiguous"

    bar_col = int(cols[0])
    assert np.all(region[rows[0] : rows[-1] + 1, bar_col] == BAR_COLOR)
    return int(len(rows))


def _rank_output(height_rows: list[list[int]]) -> np.ndarray:
    m = len(height_rows)
    n = len(height_rows[0])
    assert n <= len(RANK_COLORS), "rank palette only supports up to 6 columns"

    out = np.zeros((m, n), dtype=int)
    for r, heights in enumerate(height_rows):
        assert len(heights) == n
        assert len(set(heights)) == n, "heights must be unique within each partition row"
        order = sorted(range(n), key=lambda c: heights[c], reverse=True)
        for rank, c in enumerate(order, start=1):
            out[r, c] = RANK_COLORS[rank - 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each partition to the color of its bar's within-row height rank.

    Input structure:
    - Full black rows and full black columns partition the grid into an M x N array of rectangles.
    - Each rectangle has teal background and contains exactly one grey one-cell-wide vertical bar.
    - Bar heights are distinct within each partition row, though heights may repeat across different rows.

    Rule:
    - Detect the partition layout from the full black divider lines.
    - For each partition row, measure the height of the grey bar inside every region and rank those heights from tallest to shortest.
    - Output an M x N grid where each cell is recolored by that region's rank: 1st->red, 2nd->green, 3rd->yellow, 4th->orange, 5th->pink, 6th->maroon.

    Key insight:
    - The black partition lines define both the compressed output layout and the comparison groups, so rank is local to each partition row rather than global across the whole image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans = _partition_spans(g)
    height_rows: list[list[int]] = []
    for r0, r1 in row_spans:
        row_heights: list[int] = []
        for c0, c1 in col_spans:
            row_heights.append(_extract_bar_height(g[r0:r1, c0:c1]))
        height_rows.append(row_heights)
    return _rank_output(height_rows)


def _weighted_choice(values: tuple[int, ...], weights: tuple[int, ...]) -> int:
    return random.choices(values, weights=weights, k=1)[0]


def _choose_layout() -> tuple[list[int], list[int]]:
    for _ in range(200):
        n = _weighted_choice(N_CHOICES, N_WEIGHTS)
        m = _weighted_choice(M_CHOICES, M_WEIGHTS)

        row_min = max(4, n)
        row_max = min(8, (30 - (m - 1)) // m)
        col_min = 3
        col_max = min(6, (30 - (n - 1)) // n)
        if row_max < row_min or col_max < col_min:
            continue
        if row_max == n:
            continue

        row_heights = [random.randint(row_min, row_max) for _ in range(m)]
        if max(row_heights) == n:
            continue
        col_widths = [random.randint(col_min, col_max) for _ in range(n)]
        total_h = sum(row_heights) + (m - 1)
        total_w = sum(col_widths) + (n - 1)
        if total_h <= 30 and total_w <= 30:
            return row_heights, col_widths
    raise RuntimeError("failed to choose a valid partition layout")


def _build_row_heights(max_height: int, n: int, forced_height: int | None = None, forced_rank: int | None = None) -> list[int] | None:
    if forced_height is None:
        return random.sample(list(range(1, max_height + 1)), n)

    assert forced_rank is not None
    greater_needed = forced_rank - 1
    lower_needed = n - forced_rank
    greater_pool = list(range(forced_height + 1, max_height + 1))
    lower_pool = list(range(1, forced_height))
    if len(greater_pool) < greater_needed or len(lower_pool) < lower_needed:
        return None

    heights = [forced_height]
    heights.extend(random.sample(greater_pool, greater_needed))
    heights.extend(random.sample(lower_pool, lower_needed))
    random.shuffle(heights)
    return heights


def _has_shared_height_with_different_rank(height_rows: list[list[int]]) -> bool:
    for i in range(len(height_rows)):
        for j in range(i + 1, len(height_rows)):
            shared = set(height_rows[i]) & set(height_rows[j])
            for height in shared:
                rank_i = 1 + sum(other > height for other in height_rows[i])
                rank_j = 1 + sum(other > height for other in height_rows[j])
                if rank_i != rank_j:
                    return True
    return False


def _sample_height_rows(region_heights: list[int], n: int) -> list[list[int]]:
    m = len(region_heights)
    out: list[list[int] | None] = [None] * m

    if m >= 2:
        a, b = random.sample(range(m), 2)
        configs: list[tuple[int, int, int, int]] = []
        for rank_a in range(1, n + 1):
            for rank_b in range(1, n + 1):
                if rank_a == rank_b:
                    continue
                low = max(n - rank_a, n - rank_b) + 1
                high = min(region_heights[a] - (rank_a - 1), region_heights[b] - (rank_b - 1))
                if low <= high:
                    configs.append((rank_a, rank_b, low, high))
        if configs:
            rank_a, rank_b, low, high = random.choice(configs)
            shared_height = random.randint(low, high)
            out[a] = _build_row_heights(region_heights[a], n, forced_height=shared_height, forced_rank=rank_a)
            out[b] = _build_row_heights(region_heights[b], n, forced_height=shared_height, forced_rank=rank_b)

    for idx in range(m):
        if out[idx] is None:
            out[idx] = _build_row_heights(region_heights[idx], n)

    height_rows = [row for row in out if row is not None]
    assert len(height_rows) == m
    if not _has_shared_height_with_different_rank(height_rows):
        raise AssertionError("generator failed to create a cross-row repeated height counterexample")
    return height_rows


def _build_grid(row_heights: list[int], col_widths: list[int], height_rows: list[list[int]]) -> np.ndarray:
    m = len(row_heights)
    n = len(col_widths)
    h = sum(row_heights) + (m - 1)
    w = sum(col_widths) + (n - 1)
    grid = np.full((h, w), REGION_BG, dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, region_h in enumerate(row_heights):
        row_spans.append((cursor, cursor + region_h))
        cursor += region_h
        if idx < m - 1:
            grid[cursor, :] = DIVIDER
            cursor += 1

    col_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, region_w in enumerate(col_widths):
        col_spans.append((cursor, cursor + region_w))
        cursor += region_w
        if idx < n - 1:
            grid[:, cursor] = DIVIDER
            cursor += 1

    anchor_modes = ("bottom", "top", "floating")
    anchor_weights = (3, 2, 3)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            height = height_rows[r][c]
            region_h = r1 - r0
            slack = region_h - height
            bar_c = random.randint(c0, c1 - 1)
            mode = _weighted_choice(anchor_modes, anchor_weights)
            if slack == 0:
                top = r0
            elif mode == "bottom":
                top = r1 - height
            elif mode == "top":
                top = r0
            else:
                top = random.randint(r0, r1 - height)
            grid[top : top + height, bar_c] = BAR_COLOR

    return grid


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    row_spans, col_spans = _partition_spans(grid)
    if expected.shape != (len(row_spans), len(col_spans)):
        return False

    parsed_heights: list[list[int]] = []
    for r0, r1 in row_spans:
        row_heights: list[int] = []
        for c0, c1 in col_spans:
            region = grid[r0:r1, c0:c1]
            if not np.all(np.isin(region, [REGION_BG, BAR_COLOR])):
                return False
            row_heights.append(_extract_bar_height(region))
        if len(set(row_heights)) != len(row_heights):
            return False
        parsed_heights.append(row_heights)

    if not _has_shared_height_with_different_rank(parsed_heights):
        return False

    parsed_expected = _rank_output(parsed_heights)
    return np.array_equal(parsed_expected, expected) and np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            row_heights, col_widths = _choose_layout()
            n = len(col_widths)
            height_rows = _sample_height_rows(row_heights, n)
            expected = _rank_output(height_rows)
            grid = _build_grid(row_heights, col_widths, height_rows)
            if _validate_generated(grid, expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input grid")
