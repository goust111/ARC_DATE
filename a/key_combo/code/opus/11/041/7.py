import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 7)
GEN_ATTEMPTS = 600


def _count_false_runs(mask: np.ndarray) -> int:
    count = 0
    inside = False
    for value in mask.tolist():
        if not value and not inside:
            count += 1
            inside = True
        elif value:
            inside = False
    return count


def _band_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if not value and start is None:
            start = idx
        elif value and start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


def _partition_info(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    full_rows = np.all(grid == GREY, axis=1)
    full_cols = np.all(grid == GREY, axis=0)
    row_bands = _band_intervals(full_rows)
    col_bands = _band_intervals(full_cols)
    return full_rows, full_cols, row_bands, col_bands


def _color_totals(grid: np.ndarray) -> Counter:
    values = grid[(grid != BLACK) & (grid != GREY)]
    return Counter(map(int, values.flat))


def _sorted_colors_by_frequency(grid: np.ndarray) -> list[int]:
    totals = _color_totals(grid)
    assert totals, "expected at least one non-black non-grey color"
    counts = list(totals.values())
    assert len(counts) == len(set(counts)), "color frequencies must be unique"
    return [color for color, _count in sorted(totals.items(), key=lambda item: item[1], reverse=True)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count partition bands to size a summary canvas, then fill its rows by descending color frequency.

    Input structure:
    - Black background contains complete grey rows and complete grey columns that partition the grid into rectangles.
    - Every rectangle contains scattered pixels of exactly one non-black, non-grey color.
    - The visible colors come from a small palette, and the number of distinct non-grey colors equals the number of row partitions.

    Rule:
    - Find all complete grey rows and complete grey columns, then count the contiguous non-grey row bands and column bands.
    - Create an output grid whose height is the row-band count and whose width is the column-band count.
    - Count all non-black, non-grey pixels by color, sort the colors from most frequent to least frequent, and fill output row i entirely with the i-th color in that order.

    Key insight:
    - The grey separator geometry determines only the output size, while the scattered colored pixels determine only the row ordering through global frequency counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    full_rows, full_cols, row_bands, col_bands = _partition_info(g)
    assert row_bands and col_bands, "expected separator-defined row and column partitions"

    grey_mask = g == GREY
    assert np.array_equal(grey_mask, full_rows[:, None] | full_cols[None, :]), "grey must appear only on complete separator rows/cols"

    colors = _sorted_colors_by_frequency(g)
    assert len(colors) == len(row_bands), "distinct color count must equal row partition count"

    out = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for idx, color in enumerate(colors):
        out[idx, :] = color
    return out


def _conflicting_order(metric_a: dict[int, int], metric_b: dict[int, int], colors: list[int]) -> bool:
    for color_a in colors:
        for color_b in colors:
            if color_a == color_b:
                continue
            if metric_a[color_a] > metric_a[color_b] and metric_b[color_a] < metric_b[color_b]:
                return True
    return False


def _sample_band_sizes(parts: int) -> list[int]:
    return [random.randint(2, 5) for _ in range(parts)]


def _intervals_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    intervals: list[tuple[int, int]] = []
    separators: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        intervals.append((cursor, cursor + size))
        cursor += size
        if idx != len(sizes) - 1:
            separators.append(cursor)
            cursor += 1
    return intervals, separators


def _choose_region_colors(rows: int, cols: int, colors: list[int]) -> np.ndarray:
    cells = [color for color in colors]
    cells.extend(random.choice(colors) for _ in range(rows * cols - len(colors)))
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(rows, cols)


def _density_bounds(area: int) -> tuple[int, int]:
    low = max(1, int(np.ceil(area * 0.2)))
    high = min(area - 1, int(np.floor(area * 0.85)))
    if high < low:
        high = low
    return low, high


def _paint_region(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int, count: int) -> None:
    cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    chosen = random.sample(cells, count)
    for r, c in chosen:
        grid[r, c] = color


def _region_stats(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    region_colors: np.ndarray,
    region_counts: np.ndarray,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    totals = Counter()
    region_occurrences = Counter()
    area_sums = Counter()
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            color = int(region_colors[i, j])
            area = (r1 - r0) * (c1 - c0)
            totals[color] += int(region_counts[i, j])
            region_occurrences[color] += 1
            area_sums[color] += area
    return dict(totals), dict(region_occurrences), dict(area_sums)


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    full_rows, full_cols, row_bands, col_bands = _partition_info(grid)
    assert 3 <= len(row_bands) <= 5
    assert 3 <= len(col_bands) <= 5
    assert np.any(full_rows)
    assert np.any(full_cols)
    assert int(np.count_nonzero(full_rows)) == len(row_bands) - 1
    assert int(np.count_nonzero(full_cols)) == len(col_bands) - 1
    assert np.array_equal(grid == GREY, full_rows[:, None] | full_cols[None, :])

    colors = sorted(set(map(int, grid[(grid != BLACK) & (grid != GREY)].flat)))
    assert len(colors) == len(row_bands)
    assert set(colors).issubset(set(PALETTE))

    totals = _color_totals(grid)
    assert len(totals) == len(row_bands)
    assert len(set(totals.values())) == len(totals)

    region_occurrences = Counter()
    area_sums = Counter()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            region = grid[r0:r1, c0:c1]
            region_colors = set(map(int, region[(region != BLACK) & (region != GREY)].flat))
            assert len(region_colors) == 1, "each rectangle must carry exactly one color"
            color = next(iter(region_colors))
            region_occurrences[color] += 1
            area_sums[color] += (r1 - r0) * (c1 - c0)
            assert np.any(region == BLACK), "every region should keep visible black background"

    assert _conflicting_order(region_occurrences, totals, colors), "pixel ranking must not collapse to region-count ranking"
    assert _conflicting_order(area_sums, totals, colors), "pixel ranking must not collapse to occupied-area ranking"

    out = main(grid)
    assert out.shape == (len(row_bands), len(col_bands))
    ordered = _sorted_colors_by_frequency(grid)
    for idx, color in enumerate(ordered):
        assert np.all(out[idx] == color)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    colors = random.sample(PALETTE, rows)

    row_sizes = _sample_band_sizes(rows)
    col_sizes = _sample_band_sizes(cols)
    row_bands, row_separators = _intervals_from_sizes(row_sizes)
    col_bands, col_separators = _intervals_from_sizes(col_sizes)

    height = row_bands[-1][1]
    width = col_bands[-1][1]
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for r in row_separators:
        grid[r, :] = GREY
    for c in col_separators:
        grid[:, c] = GREY

    region_colors = _choose_region_colors(rows, cols, colors)
    region_counts = np.zeros((rows, cols), dtype=int)

    low_density_seen = False
    high_density_seen = False
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            area = (r1 - r0) * (c1 - c0)
            low, high = _density_bounds(area)
            count = random.randint(low, high)
            region_counts[i, j] = count
            density = count / area
            low_density_seen = low_density_seen or density <= 0.4
            high_density_seen = high_density_seen or density >= 0.7

    totals, region_occurrences, area_sums = _region_stats(row_bands, col_bands, region_colors, region_counts)
    if len(totals) != rows:
        return None
    if len(set(totals.values())) != rows:
        return None
    if not low_density_seen or not high_density_seen:
        return None
    if max(region_occurrences.values()) == 1:
        return None
    if not _conflicting_order(region_occurrences, totals, colors):
        return None
    if not _conflicting_order(area_sums, totals, colors):
        return None

    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            _paint_region(grid, r0, r1, c0, c1, int(region_colors[i, j]), int(region_counts[i, j]))

    try:
        _validate_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
