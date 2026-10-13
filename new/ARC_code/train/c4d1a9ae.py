from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import math
import random


def _foreground_bands(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    bands: list[tuple[int, int, int]] = []
    for color in colors_present(g, exclude={bg}):
        cols = np.where(g == color)[1]
        c0, c1 = int(cols.min()), int(cols.max())
        region_colors = {int(v) for v in np.unique(g[:, c0 : c1 + 1])}
        assert region_colors <= {int(bg), int(color)}, "each band must contain only its own color and bg"
        assert np.any(g[:, c0 : c1 + 1] == bg), "each band must have bg cells to fill"
        bands.append((c0, c1, int(color)))

    assert len(bands) >= 2, "expected multiple foreground bands"
    bands.sort(key=lambda item: item[0])
    for left, right in zip(bands, bands[1:]):
        assert left[1] < right[0], "foreground bands must be non-overlapping"
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each vertical foreground band's background cells with the color of the next band to its right.

    Input structure
    - The dominant color is the background.
    - Each other color occupies a separate vertical band defined by that color's
      minimum and maximum columns.
    - Within a band, all cells are either that band's color or background.

    Rule
    - Order the bands from left to right.
    - In each band, replace background pixels with the color of the next band in
      that order, wrapping from the rightmost band back to the leftmost.
    - Preserve all existing foreground pixels and all background pixels outside
      the bands.

    Key insight
    - The column bands form a cyclic palette legend: a band's gaps are completed
      by borrowing the color from its right-hand neighbor, not by changing the
      band's existing pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    bands = _foreground_bands(g, bg)

    out = g.copy()
    for index, (c0, c1, _color) in enumerate(bands):
        fill_color = bands[(index + 1) % len(bands)][2]
        band = out[:, c0 : c1 + 1]
        band[band == bg] = fill_color

    return out

# --------------- generator (att2) ---------------

# Canvas bounds.
GRID_ROWS_RANGE = (8, 12)
GRID_COLS_RANGE = (7, 10)
# Band structure and foreground occupancy.
BAND_COUNT = 3
BAND_WIDTH_RANGE = (1, 3)
BAND_DENSITY_RANGE = (0.30, 0.75)
# Palette and retry budget.
PALETTE_SIZE = 4
MAX_GENERATION_ATTEMPTS = 300


def _gen_band_widths(total_cols: int, band_count: int) -> list[int] | None:
    band_cols = total_cols - (band_count - 1)
    options: list[tuple[int, ...]] = []
    min_width, max_width = BAND_WIDTH_RANGE

    def rec(prefix: list[int], remaining: int) -> None:
        if len(prefix) == band_count:
            if remaining == 0:
                options.append(tuple(prefix))
            return
        slots_left = band_count - len(prefix) - 1
        for width in range(min_width, max_width + 1):
            remainder = remaining - width
            if min_width * slots_left <= remainder <= max_width * slots_left:
                rec([*prefix, width], remainder)

    rec([], band_cols)
    if not options:
        return None
    return list(random.choice(options))


def _gen_fill_band(grid: np.ndarray, c0: int, width: int, color: int) -> None:
    rows = grid.shape[0]
    area = rows * width
    min_density, max_density = BAND_DENSITY_RANGE
    min_fg = max(width, math.ceil(min_density * area))
    max_fg = max(min_fg, math.floor(max_density * area))
    fg_count = random.randint(min_fg, max_fg)

    required = {(random.randrange(rows), c0 + dc) for dc in range(width)}
    cells = [(r, c0 + dc) for r in range(rows) for dc in range(width)]
    remaining = [cell for cell in cells if cell not in required]
    chosen = set(required)
    chosen.update(random.sample(remaining, fg_count - len(required)))
    for r, c in chosen:
        grid[r, c] = color


def _gen_expected_output(
    grid: np.ndarray, bands: list[tuple[int, int, int]], bg: int
) -> np.ndarray:
    out = grid.copy()
    for i, (c0, c1, _color) in enumerate(bands):
        fill_color = bands[(i + 1) % len(bands)][2]
        band = out[:, c0 : c1 + 1]
        band[band == bg] = fill_color
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    band_count = BAND_COUNT
    widths = _gen_band_widths(cols, band_count)
    if widths is None:
        return None

    colors = random.sample(range(10), PALETTE_SIZE)
    bg, *fg_colors = colors
    grid = np.full((rows, cols), bg, dtype=int)

    bands: list[tuple[int, int, int]] = []
    col = 0
    for i, (width, color) in enumerate(zip(widths, fg_colors)):
        _gen_fill_band(grid, col, width, color)
        bands.append((col, col + width - 1, color))
        col += width
        if i != band_count - 1:
            col += 1

    if col != cols:
        return None

    expected = _gen_expected_output(grid, bands, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
