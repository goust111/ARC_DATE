import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Paint a yellow 3x3 ring around the crossing of the full foreground row and column.

    Input structure
    - The grid has a dominant background and two one-cell-wide foreground bars.
    - One bar fills a complete row and the other fills a complete column.
    - Their crossing pixel lies away from the border.

    Rule
    - Locate the unique row and unique column that are entirely non-background.
    - Paint the 3x3 neighborhood centered on their crossing yellow.
    - Restore the original crossing pixel, leaving only the eight surrounding cells changed.

    Key insight
    - The row/column crossing is the center of a local marker that replaces both nearby bar pixels and nearby background pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    foreground = g != bg

    crossing_rows = np.flatnonzero(foreground.all(axis=1))
    crossing_cols = np.flatnonzero(foreground.all(axis=0))
    assert len(crossing_rows) == 1
    assert len(crossing_cols) == 1

    r = int(crossing_rows[0])
    c = int(crossing_cols[0])
    height, width = g.shape
    assert 0 < r < height - 1
    assert 0 < c < width - 1

    out = g.copy()
    out[r - 1 : r + 2, c - 1 : c + 2] = Color.YELLOW
    out[r, c] = g[r, c]
    return out

# --------------- generator (o5x) ---------------
# Canvas / crossing bounds.
SIZE_RANGE = (4, 12)
INTERIOR_MARGIN = 1
HALO_RADIUS = 1
# Palette / sampling controls.
FOREGROUND_COLORS = tuple(c for c in range(1, 10) if c != Color.YELLOW)
ROW_FIRST_PROBABILITY = 0.5
MAX_GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*SIZE_RANGE)
    row_color, col_color = random.sample(FOREGROUND_COLORS, 2)
    row = random.randint(INTERIOR_MARGIN, n - 1 - INTERIOR_MARGIN)
    col = random.randint(INTERIOR_MARGIN, n - 1 - INTERIOR_MARGIN)
    grid = np.zeros((n, n), dtype=int)
    if random.random() < ROW_FIRST_PROBABILITY:
        grid[row, :] = row_color
        grid[:, col] = col_color
        crossing_color = col_color
    else:
        grid[:, col] = col_color
        grid[row, :] = row_color
        crossing_color = row_color

    r = HALO_RADIUS
    expected = grid.copy()
    expected[row - r : row + r + 1, col - r : col + r + 1] = Color.YELLOW
    expected[row, col] = crossing_color
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
