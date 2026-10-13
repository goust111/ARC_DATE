from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, size]
    bands = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    assert all(start < stop for start, stop in bands), "separator lines must split nonempty regions"
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the separator-delimited board into a summary grid marking cells with two marker pixels.

    Input structure
    - The dominant color is the empty background.
    - One non-background color forms all full separator rows and all full
      separator columns.
    - The remaining non-background color is the marker color; every separated
      region contains either one or two marker pixels.

    Rule
    - Find the separator rows and columns, and use them to partition the input
      into rectangular regions.
    - For each region, output blue if it contains two marker pixels; otherwise
      output the background color.

    Key insight
    - The large grid is a 3x3 table of tiny marker counts, and the output records
      only which table cells have the doubled marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    separator_candidates = []
    for color in foreground_colors:
        sep_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if sep_rows and sep_cols:
            separator_candidates.append((color, sep_rows, sep_cols))
    assert len(separator_candidates) == 1, "expected one separator color with full rows and columns"

    separator, sep_rows, sep_cols = separator_candidates[0]
    marker_colors = [color for color in foreground_colors if color != separator]
    assert len(marker_colors) == 1, "expected one marker color besides the separators"
    marker = marker_colors[0]

    row_bands = _bands(height, sep_rows)
    col_bands = _bands(width, sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3, "expected a 3x3 separated table"
    out = np.full((len(row_bands), len(col_bands)), bg, dtype=int)

    allowed = {bg, marker}
    for out_r, (r0, r1) in enumerate(row_bands):
        for out_c, (c0, c1) in enumerate(col_bands):
            region = g[r0:r1, c0:c1]
            assert set(map(int, region.reshape(-1))) <= allowed, "regions may contain only background and markers"
            marker_count = int(np.sum(region == marker))
            assert marker_count in (1, 2), "each region must contain one or two markers"
            if marker_count == 2:
                out[out_r, out_c] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------
# Canvas/layout.
GRID_SHAPE, OUT_SHAPE = (11, 11), (3, 3)
REGION_SIZE, REGION_STARTS, SEPARATOR_LINES = 3, [0, 4, 8], [3, 7]
REGION_COUNT, CELL_COUNT = OUT_SHAPE[0] * OUT_SHAPE[1], REGION_SIZE * REGION_SIZE
# Colors.
BG_COLOR, SEPARATOR_COLOR, MARKER_COLOR, FLAG_COLOR = Color.BLACK, Color.TEAL, Color.PINK, Color.BLUE
# Sampling/retry.
TWO_MARKER_RANGE, MAX_ATTEMPTS = (2, 4), 300

def _gen_attempt() -> np.ndarray | None:
    grid = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    grid[SEPARATOR_LINES, :] = SEPARATOR_COLOR
    grid[:, SEPARATOR_LINES] = SEPARATOR_COLOR

    out = np.full(OUT_SHAPE, BG_COLOR, dtype=int)
    two_marker_regions = set(random.sample(range(REGION_COUNT), random.randint(*TWO_MARKER_RANGE)))

    for region_i in range(REGION_COUNT):
        rr, cc = divmod(region_i, REGION_SIZE)
        marker_count = 2 if region_i in two_marker_regions else 1
        if marker_count == 2:
            out[rr, cc] = FLAG_COLOR

        r0, c0 = REGION_STARTS[rr], REGION_STARTS[cc]
        for pos in random.sample(range(CELL_COUNT), marker_count):
            dr, dc = divmod(pos, REGION_SIZE)
            grid[r0 + dr, c0 + dc] = MARKER_COLOR

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
