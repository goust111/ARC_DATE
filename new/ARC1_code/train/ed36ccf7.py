import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the entire square grid 90 degrees counterclockwise.

    Input structure
    The input is a square 2D grid whose cells form a small black-and-one-color
    pattern.

    Rule
    Preserve every cell color and move each cell to its position under a 90-degree
    counterclockwise rotation of the whole grid.

    Key insight
    The foreground shape is not rewritten internally; the complete canvas, including
    black cells, is rotated as one pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w

    colors = set(map(int, np.unique(g)))
    assert 0 in colors
    assert len(colors) == 2

    return np.rot90(g, 1)

# --------------- generator (o5x) ---------------
GRID_SHAPE = (3, 3)  # Canvas size.
CELL_COUNT = GRID_SHAPE[0] * GRID_SHAPE[1]
FOREGROUND_COLOR_RANGE = (1, 9)  # Non-black color bounds.
FOREGROUND_COUNT_RANGE = (3, 7)  # Filled-cell count bounds.
ROT90_K = 1  # Expected transform.
MAX_ATTEMPTS = 300  # Retry limit.

def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    color = random.randint(*FOREGROUND_COLOR_RANGE)
    k = random.randint(*FOREGROUND_COUNT_RANGE)
    for idx in random.sample(range(CELL_COUNT), k):
        grid[idx // GRID_SHAPE[1], idx % GRID_SHAPE[1]] = color
    expected = np.rot90(grid, ROT90_K)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ed36ccf7 input")
