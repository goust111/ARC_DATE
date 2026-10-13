import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the whole square input tile once per input row and column.

    Input structure
    - A nonempty square 2D color grid.
    - The entire grid is the motif to be repeated; individual pixels are not enlarged.

    Rule
    - Let N be the side length of the input grid.
    - Tile the full input grid N times vertically and N times horizontally.

    Key insight
    The output size grows from N x N to N^2 x N^2 because the unit of repetition is the complete input tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"
    assert height == width, "expected a square tile"

    return np.tile(g, (height, width))

# --------------- generator (o5x) ---------------

# Tile/color sampling ranges and retry limit; randint upper bounds are exclusive.
TILE_SIZE_RANGE = (2, 6)
NONBLACK_COLOR_RANGE = (1, 10)
MAX_ATTEMPTS = 300


def _sample_tile() -> np.ndarray:
    """Sample a valid tile with one non-black color and at least one black cell."""
    n = int(np.random.randint(*TILE_SIZE_RANGE))
    color = int(np.random.randint(*NONBLACK_COLOR_RANGE))
    total = n * n
    colored_count = int(np.random.randint(1, total))

    flat = np.zeros(total, dtype=int)
    flat[np.random.choice(total, size=colored_count, replace=False)] = color
    return flat.reshape(n, n)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _sample_tile()
        n = grid.shape[0]
        if np.array_equal(main(grid), np.tile(grid, (n, n))):
            return grid
    raise RuntimeError("failed to generate valid ccd554ac input")
