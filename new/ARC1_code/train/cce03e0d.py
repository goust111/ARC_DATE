from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each red pixel with a copy of the whole input pattern in the corresponding enlarged block.

    Input structure
    A small grid uses black as empty space, with red pixels acting as control cells inside a colored motif.

    Rule
    Create an output whose block grid has one block for each input pixel, where each block is the same shape as the input. For every red input pixel, fill its corresponding output block with an exact copy of the input; every non-red pixel contributes an all-black block.

    Key insight
    The red pixels mark the macro-locations where the entire input pattern is recursively stamped, not just where red itself should be enlarged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0
    assert np.any(g == Color.BLACK)
    assert np.any(g == Color.RED)

    out = np.full((h * h, w * w), Color.BLACK, dtype=int)
    for r, c in np.argwhere(g == Color.RED):
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = g
    return out

# --------------- generator (o5x) ---------------

# Input shape, color-count constraints, and retry budget.
GRID_SIZE = 3
GRID_SHAPE = (GRID_SIZE, GRID_SIZE)
CELL_COUNT = GRID_SIZE * GRID_SIZE
BLUE_COUNT = 3
RED_COUNT_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 300


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h * h, w * w), Color.BLACK, dtype=int)
    for r, c in np.argwhere(grid == Color.RED):
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = grid
    return out


def _gen_attempt() -> np.ndarray | None:
    grid = np.full(GRID_SHAPE, Color.BLACK, dtype=int)
    red_count = random.randint(*RED_COUNT_RANGE)
    chosen = random.sample(range(CELL_COUNT), BLUE_COUNT + red_count)
    for idx in chosen[:BLUE_COUNT]:
        grid[idx // GRID_SIZE, idx % GRID_SIZE] = Color.BLUE
    for idx in chosen[BLUE_COUNT:]:
        grid[idx // GRID_SIZE, idx % GRID_SIZE] = Color.RED
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, _gen_expected_output(inp)):
            return inp
    raise RuntimeError("failed to generate valid input for cce03e0d")
