from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


def _single_foreground_color(g: np.ndarray) -> int:
    colors = np.unique(g)
    foreground = colors[colors != Color.BLACK]
    assert foreground.size == 1, "expected exactly one non-black foreground color"
    return int(foreground[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 pattern by replacing each colored pixel with a tile shaped like the pattern's black pixels.

    Input structure:
    - The input is a 3x3 grid with black background cells.
    - All non-black pixels share one foreground color.

    Rule:
    - Treat the black-cell mask of the whole input as a 3x3 tile template.
    - For each foreground input pixel, place that tile in the corresponding
      3x3 output block using the foreground color.
    - For each black input pixel, leave the corresponding output block black.

    Key insight:
    - The input is both the coarse placement mask and, by complement, the local
      tile stamped at every foreground position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3), "expected a 3x3 input grid"

    color = _single_foreground_color(g)
    source_mask = g == color
    tile_mask = g == Color.BLACK
    assert np.any(source_mask), "expected foreground pixels"
    assert np.any(tile_mask), "expected black template pixels"

    out = np.kron(source_mask.astype(int), tile_mask.astype(int)) * color
    assert out.shape == (9, 9), "expanded output must be 9x9"
    return out.astype(int)

# --------------- generator (o5x) ---------------

# Fixed grid geometry, foreground sampling bounds, and retry budget.
GRID_SHAPE, CELL_COUNT = (3, 3), 9
COLOR_MIN, COLOR_MAX = 1, 9
FOREGROUND_MIN, FOREGROUND_MAX = 3, 5
MAX_GENERATION_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    color = random.randint(COLOR_MIN, COLOR_MAX)
    foreground_count = random.randint(FOREGROUND_MIN, FOREGROUND_MAX)

    cells = random.sample(range(CELL_COUNT), foreground_count)
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[np.unravel_index(cells, GRID_SHAPE)] = color

    expected = np.kron((grid != 0).astype(int), (grid == 0).astype(int)) * color
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 0692e18c")
