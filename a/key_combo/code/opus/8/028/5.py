from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

GEN_RETRY_LIMIT = 400


def _count_bands(is_separator: np.ndarray) -> int:
    bands = 0
    in_band = False
    for flag in map(bool, is_separator):
        if flag:
            in_band = False
        elif not in_band:
            bands += 1
            in_band = True
    return bands


def _line_runs(is_separator: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(map(bool, is_separator)):
        if flag:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, len(is_separator) - 1))
    return runs


def _build_pyramid(height: int, width: int) -> np.ndarray:
    out = np.full((int(height), int(width)), BLACK, dtype=int)
    center = int(width) // 2
    for r in range(int(height)):
        for dc in range(-r, r + 1):
            c = center + dc
            out[r, c] = RED if abs(dc) % 2 == 0 else YELLOW
    return out


def _red_segment(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == RED)
    assert coords.size > 0, "expected one red line segment"
    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, "red object must be vertical"
    col = int(cols[0])
    rows = np.sort(coords[:, 0])
    expected_rows = np.arange(int(rows[0]), int(rows[-1]) + 1)
    assert np.array_equal(rows, expected_rows), "red line must be contiguous"
    return int(rows[0]), col, int(rows[-1]), col


def main(grid: np.ndarray) -> np.ndarray:
    """Count separator-defined row/column regions and use them as the size of an alternating pyramid.

    Input structure:
    - The grid has black background, grey full-row/full-column separator lines, and one red vertical line segment placed inside a non-grey partition.
    - Grey pixels are exactly the union of complete horizontal and vertical lines.
    - All non-grey, non-red cells are background.

    Rule:
    - Count the contiguous non-grey row bands to get the output height R.
    - Count the contiguous non-grey column bands to get the output width C.
    - Return an R x C black grid containing a centered pyramid of height R, where row i spans 2i+1 cells around column floor(C/2).
    - Color pyramid cells red when their horizontal distance from the center is even, otherwise yellow.

    Key insight:
    - The separators do not survive directly; they encode the output geometry, while the lone red line only tells you to perform the pyramid expansion rather than returning a blank counted grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED, GREY}, f"unexpected colors: {colors}"

    full_rows = np.all(g == GREY, axis=1)
    full_cols = np.all(g == GREY, axis=0)
    grey_mask = full_rows[:, None] | full_cols[None, :]
    assert np.array_equal(g == GREY, grey_mask), "grey pixels must be complete separator lines"

    r0, c0, r1, _c1 = _red_segment(g)
    assert not full_cols[c0], "red line cannot sit on a separator column"
    assert not np.any(full_rows[r0 : r1 + 1]), "red line cannot cross a separator row"
    assert np.all(g[(g != GREY) & (g != RED)] == BLACK), "non-separator, non-line cells must be background"

    out_h = _count_bands(full_rows)
    out_w = _count_bands(full_cols)
    assert 1 <= out_h <= 30 and 1 <= out_w <= 30
    assert out_w >= 2 * out_h - 1, "counted width must fit the pyramid"
    return _build_pyramid(out_h, out_w)


def _random_composition(total: int, parts: int, minimum: int) -> list[int]:
    assert total >= parts * minimum
    remain = total - parts * minimum
    cuts = sorted(random.sample(range(remain + parts - 1), parts - 1)) if parts > 1 else []
    extras: list[int] = []
    prev = -1
    for cut in cuts + [remain + parts - 1]:
        extras.append(cut - prev - 1)
        prev = cut
    return [minimum + extra for extra in extras]


def _bands_to_separators(bands: list[int]) -> tuple[int, list[int]]:
    size = sum(bands) + len(bands) - 1
    separators: list[int] = []
    pos = 0
    for idx, band in enumerate(bands):
        pos += int(band)
        if idx < len(bands) - 1:
            separators.append(pos)
            pos += 1
    return size, separators


def _sample_dimensions(region_count: int, min_band: int, max_size: int) -> list[int]:
    total_bands = random.randint(region_count * min_band, max_size - (region_count - 1))
    return _random_composition(total_bands, region_count, min_band)


def _partition_runs(size: int, separators: list[int]) -> list[tuple[int, int]]:
    flags = np.zeros(size, dtype=bool)
    flags[separators] = True
    return _line_runs(flags)


def _expected_output(row_regions: int, col_regions: int) -> np.ndarray:
    return _build_pyramid(row_regions, col_regions)


def _validate_generated(grid: np.ndarray, row_regions: int, col_regions: int, line_len: int) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    full_rows = np.all(g == GREY, axis=1)
    full_cols = np.all(g == GREY, axis=0)
    assert _count_bands(full_rows) == row_regions
    assert _count_bands(full_cols) == col_regions
    assert np.array_equal(g == GREY, full_rows[:, None] | full_cols[None, :])

    r0, c0, r1, _ = _red_segment(g)
    assert r1 - r0 + 1 == line_len
    assert line_len != row_regions
    assert not np.any(full_rows[r0 : r1 + 1])
    assert not full_cols[c0]

    expected = _expected_output(row_regions, col_regions)
    actual = main(g)
    assert np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    row_regions = random.randint(3, 5)
    min_cols = max(5, 2 * row_regions - 1)
    col_regions = random.randint(min_cols, 9)

    row_bands = _sample_dimensions(row_regions, min_band=2, max_size=22)
    col_bands = _sample_dimensions(col_regions, min_band=1, max_size=28)
    height, row_separators = _bands_to_separators(row_bands)
    width, col_separators = _bands_to_separators(col_bands)
    if max(height, width) > 30:
        return None

    line_choices = [length for length in range(2, 6) if length != row_regions and length <= max(row_bands)]
    if not line_choices:
        return None
    line_len = random.choice(line_choices)

    row_runs = _partition_runs(height, row_separators)
    valid_row_regions = [idx for idx, (a, b) in enumerate(row_runs) if b - a + 1 >= line_len]
    if not valid_row_regions:
        return None

    col_runs = _partition_runs(width, col_separators)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_separators, :] = GREY
    grid[:, col_separators] = GREY

    row_region_idx = random.choice(valid_row_regions)
    col_region_idx = random.randrange(len(col_runs))
    rr0, rr1 = row_runs[row_region_idx]
    cc0, cc1 = col_runs[col_region_idx]
    top = random.randint(rr0, rr1 - line_len + 1)
    col = random.randint(cc0, cc1)
    grid[top : top + line_len, col] = RED

    try:
        _validate_generated(grid, row_regions, col_regions, line_len)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
