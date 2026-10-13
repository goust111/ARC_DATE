from __future__ import annotations

import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the upper two-pixel segment and the lower two-pixel segment around the fixed center pixel.

    Input structure:
    - The grid is a 5x1 vertical column of individual colored pixels.
    - The middle pixel is a separator between two same-size two-pixel segments.

    Rule:
    - Keep the middle pixel in place.
    - Reverse the order of the two pixels above the middle.
    - Reverse the order of the two pixels below the middle.

    Key insight:
    - The transformation is local vertical order reversal on each side of the center, not a color-specific rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (5, 1), "expected a 5x1 vertical column"

    out = g.copy()
    out[0, 0], out[1, 0] = g[1, 0], g[0, 0]
    out[3, 0], out[4, 0] = g[4, 0], g[3, 0]
    return out

# --------------- generator (att2) ---------------

# Fixed 5x1 column and color vocabulary.
GRID_SHAPE = (5, 1)
COLOR_DOMAIN = range(10)
COLORS_PER_GRID = GRID_SHAPE[0]

# Oracle/retry controls.
OUTPUT_ROW_ORDER = (1, 0, 2, 4, 3)
MAX_ATTEMPTS = 300


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = np.array(random.sample(COLOR_DOMAIN, COLORS_PER_GRID), dtype=int).reshape(GRID_SHAPE)
        if np.array_equal(main(inp), inp[OUTPUT_ROW_ORDER, :]):
            return inp
    raise RuntimeError("failed to generate valid input")
