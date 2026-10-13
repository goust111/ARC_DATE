import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 color grid to 5x5 by copying its border pixels outward one cell.

    Input structure
    - The input is a 3x3 grid of ARC colors.

    Rule
    - Place the input at the center of a 5x5 output.
    - Copy the top and bottom input rows into the adjacent outer rows, and the
      left and right input columns into the adjacent outer columns.
    - The corner pixels are copied outward in both directions, forming 2x2 corner blocks.

    Key insight
    - This is a one-cell edge padding of the entire grid, not a per-color or
      object-specific transformation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected the 3x3 source grid"

    return np.pad(g, ((1, 1), (1, 1)), mode="edge")

# --------------- generator (o5x) ---------------

# Grid, palette, padding, and retry constraints.
GRID_SHAPE = (3, 3)
GRID_CELL_COUNT = GRID_SHAPE[0] * GRID_SHAPE[1]
ARC_COLORS = range(1, 10)
DISTINCT_COLOR_RANGE = (5, 9)
EDGE_PAD_WIDTH = 1
RETRY_LIMIT = 300


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        distinct_count = random.randint(*DISTINCT_COLOR_RANGE)
        palette = random.sample(ARC_COLORS, distinct_count)
        cells = palette[:] + [random.choice(palette) for _ in range(GRID_CELL_COUNT - distinct_count)]
        random.shuffle(cells)

        inp = np.array(cells, dtype=int).reshape(GRID_SHAPE)
        if np.array_equal(main(inp), np.pad(inp, EDGE_PAD_WIDTH, mode="edge")):
            return inp
    raise RuntimeError("failed to generate valid e633a9e5 input")
