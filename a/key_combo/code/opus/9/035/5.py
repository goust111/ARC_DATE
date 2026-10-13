from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
DIVIDER = 5
FG = 8

OUTPUT_COLORS = (BLUE, RED, GREEN, YELLOW)
COUNT_VALUES = tuple(range(1, 17))

REGION_SIZE_MIN = 4
REGION_SIZE_MAX = 6
GENERATION_RETRIES = 500

_SEEN_OUTPUTS: set[bytes] = set()


def _full_line_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _intervals_from_runs(size: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for start, end in runs:
        if start > cursor:
            intervals.append((cursor, start))
        cursor = end + 1
    if cursor < size:
        intervals.append((cursor, size))
    return intervals


def _partition_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_runs = _full_line_runs(np.all(grid == DIVIDER, axis=1))
    col_runs = _full_line_runs(np.all(grid == DIVIDER, axis=0))
    row_intervals = _intervals_from_runs(grid.shape[0], row_runs)
    col_intervals = _intervals_from_runs(grid.shape[1], col_runs)
    assert len(row_runs) == 3 and len(col_runs) == 3, "expected three horizontal and three vertical divider lines"
    assert len(row_intervals) == 4 and len(col_intervals) == 4, "expected a 4x4 partition"
    assert all(r1 > r0 for r0, r1 in row_intervals)
    assert all(c1 > c0 for c0, c1 in col_intervals)
    return row_intervals, col_intervals


def _rank_colors(counts: np.ndarray) -> np.ndarray:
    flat = counts.reshape(-1)
    unique = {int(v) for v in flat.tolist()}
    assert len(unique) == flat.size, "region counts must be strictly distinct"

    order = sorted(range(flat.size), key=lambda idx: int(flat[idx]), reverse=True)
    out = np.zeros(flat.size, dtype=int)
    for rank, idx in enumerate(order):
        out[idx] = OUTPUT_COLORS[rank // 4]
    return out.reshape(counts.shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each same-color partition to one pixel colored by its cross-region count rank.

    Input structure:
    - Full gray rows and full gray columns divide the grid into a 4x4 array of rectangular regions.
    - Divider cells are gray, every region background is black, and each region contains scattered teal pixels.
    - All foreground pixels use the same teal color, so color alone does not distinguish the regions.

    Rule:
    - Locate the full gray divider rows and columns to recover the 4x4 region layout.
    - Count the teal pixels inside each region.
    - Rank all 16 regions globally from most teal pixels to fewest.
    - Output a 4x4 grid whose cell colors are blue for ranks 1-4, red for ranks 5-8, green for ranks 9-12, and yellow for ranks 13-16.

    Key insight:
    - The only informative signal is how many teal pixels each region contains relative to the others; collapsing by region color would lose all information because every region uses the same foreground color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    row_intervals, col_intervals = _partition_grid(g)
    counts = np.zeros((4, 4), dtype=int)
    for out_r, (r0, r1) in enumerate(row_intervals):
        for out_c, (c0, c1) in enumerate(col_intervals):
            region = g[r0:r1, c0:c1]
            counts[out_r, out_c] = int(np.count_nonzero(region == FG))

    return _rank_colors(counts)


def _random_sizes() -> list[int]:
    while True:
        sizes = [random.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(4)]
        if sum(sizes) + 3 <= 30:
            return sizes


def _region_weight_grid(height: int, width: int, mode: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    if mode == 0:
        weights = np.ones((height, width), dtype=float)
    elif mode == 1:
        center_r = random.randint(0, height - 1)
        center_c = random.randint(0, width - 1)
        dist = np.abs(rr - center_r) + np.abs(cc - center_c)
        weights = 1.0 / (dist + 1.0)
    elif mode == 2:
        favor_rows = set(random.sample(range(height), k=random.randint(1, min(2, height))))
        favor_cols = set(random.sample(range(width), k=random.randint(1, min(2, width))))
        weights = np.ones((height, width), dtype=float)
        weights += np.isin(rr, list(favor_rows)).astype(float) * 1.7
        weights += np.isin(cc, list(favor_cols)).astype(float) * 1.3
    else:
        parity = (rr + cc + random.randint(0, 1)) % 2
        border = (rr == 0) | (rr == height - 1) | (cc == 0) | (cc == width - 1)
        weights = 0.8 + parity.astype(float) * 1.4 + border.astype(float) * 0.8
    weights += np.random.random((height, width)) * 0.05
    return weights


def _sample_region_cells(height: int, width: int, count: int) -> np.ndarray:
    total = height * width
    assert 1 <= count <= total
    mode = random.randrange(4)
    weights = _region_weight_grid(height, width, mode).reshape(-1)
    weights /= weights.sum()
    chosen = np.random.choice(total, size=count, replace=False, p=weights)
    mask = np.zeros(total, dtype=bool)
    mask[chosen] = True
    return mask.reshape(height, width)


def _build_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    counts: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_sizes) + 3
    width = sum(col_sizes) + 3
    grid = np.full((height, width), BG, dtype=int)

    row_intervals: list[tuple[int, int]] = []
    cursor = 0
    for idx, size in enumerate(row_sizes):
        row_intervals.append((cursor, cursor + size))
        cursor += size
        if idx < 3:
            grid[cursor, :] = DIVIDER
            cursor += 1

    col_intervals: list[tuple[int, int]] = []
    cursor = 0
    for idx, size in enumerate(col_sizes):
        col_intervals.append((cursor, cursor + size))
        cursor += size
        if idx < 3:
            grid[:, cursor] = DIVIDER
            cursor += 1

    for r_idx, (r0, r1) in enumerate(row_intervals):
        for c_idx, (c0, c1) in enumerate(col_intervals):
            region_mask = _sample_region_cells(r1 - r0, c1 - c0, int(counts[r_idx, c_idx]))
            region = grid[r0:r1, c0:c1]
            region[region_mask] = FG

    return grid, row_intervals, col_intervals


def _region_counts_from_intervals(
    grid: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> np.ndarray:
    counts = np.zeros((4, 4), dtype=int)
    for r_idx, (r0, r1) in enumerate(row_intervals):
        for c_idx, (c0, c1) in enumerate(col_intervals):
            region = grid[r0:r1, c0:c1]
            assert set(np.unique(region).tolist()) <= {BG, FG}
            counts[r_idx, c_idx] = int(np.count_nonzero(region == FG))
    return counts


def _non_degenerate_areas(row_sizes: list[int], col_sizes: list[int], counts: np.ndarray) -> bool:
    areas = np.array([[rh * cw for cw in col_sizes] for rh in row_sizes], dtype=int)
    largest_count = tuple(map(int, np.argwhere(counts == counts.max())[0]))
    smallest_count = tuple(map(int, np.argwhere(counts == counts.min())[0]))
    area_max = tuple(map(int, np.argwhere(areas == areas.max())[0]))
    area_min = tuple(map(int, np.argwhere(areas == areas.min())[0]))
    if largest_count == area_max or smallest_count == area_min:
        return False

    found_cross_pair = False
    flat_counts = counts.reshape(-1)
    flat_areas = areas.reshape(-1)
    for i in range(flat_counts.size):
        for j in range(flat_counts.size):
            if flat_counts[i] > flat_counts[j] and flat_areas[i] < flat_areas[j]:
                found_cross_pair = True
                break
        if found_cross_pair:
            break
    return found_cross_pair and len(set(flat_areas.tolist())) >= 2


def _validate_example(
    grid: np.ndarray,
    counts: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> np.ndarray | None:
    parsed_rows, parsed_cols = _partition_grid(grid)
    assert parsed_rows == row_intervals and parsed_cols == col_intervals

    observed_counts = _region_counts_from_intervals(grid, row_intervals, col_intervals)
    if not np.array_equal(observed_counts, counts):
        return None
    if sorted(observed_counts.reshape(-1).tolist()) != list(COUNT_VALUES):
        return None

    expected = _rank_colors(counts)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    unique_colors, color_counts = np.unique(actual, return_counts=True)
    if unique_colors.tolist() != list(OUTPUT_COLORS):
        return None
    if any(int(v) != 4 for v in color_counts.tolist()):
        return None

    return actual


def _gen_attempt() -> np.ndarray | None:
    row_sizes = _random_sizes()
    col_sizes = _random_sizes()
    counts = np.array(random.sample(COUNT_VALUES, k=16), dtype=int).reshape(4, 4)

    if not _non_degenerate_areas(row_sizes, col_sizes, counts):
        return None

    grid, row_intervals, col_intervals = _build_grid(row_sizes, col_sizes, counts)
    output = _validate_example(grid, counts, row_intervals, col_intervals)
    if output is None:
        return None

    signature = output.tobytes()
    if signature in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(signature)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRIES} attempts")
