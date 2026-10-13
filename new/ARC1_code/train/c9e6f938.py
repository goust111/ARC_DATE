import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the grid to the right with a left-right mirror copy of the whole input.

    Input structure
    - The input is a non-empty 2D grid that forms the complete left half of the output.

    Rule
    - Keep the original grid unchanged as the left half.
    - Append the same rows in reverse column order as the right half.

    Key insight
    - The transformation is global horizontal reflection completion over the whole grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "expected a non-empty grid"

    return np.concatenate((g, np.fliplr(g)), axis=1)

# --------------- generator (o5x) ---------------

# Canvas/object sampling bounds.
ROW_RANGE = (2, 5)
COL_RANGE = (2, 5)
MIN_MARKED_CELLS = 1
MAX_MARKED_CELLS = 4
MARK_COLOR = 7
# Generation retry budget.
MAX_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray:
    rows = random.randint(*ROW_RANGE)
    cols = random.randint(*COL_RANGE)
    max_cells = min(MAX_MARKED_CELLS, (rows * cols + 1) // 2)
    count = random.randint(MIN_MARKED_CELLS, max_cells)

    grid = np.zeros((rows, cols), dtype=int)

    asymmetric_cells = [(r, c) for r in range(rows) for c in range(cols) if c != cols - 1 - c]
    ar, ac = random.choice(asymmetric_cells)
    grid[ar, ac] = MARK_COLOR
    forbidden = (ar, cols - 1 - ac)

    choices = [(r, c) for r in range(rows) for c in range(cols)
               if (r, c) != (ar, ac) and (r, c) != forbidden]
    for r, c in random.sample(choices, count - 1):
        grid[r, c] = MARK_COLOR

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        out = np.concatenate((inp, np.fliplr(inp)), axis=1)
        if not np.array_equal(inp, np.fliplr(inp)) and np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate valid input")
