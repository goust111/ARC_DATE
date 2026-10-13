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

# Grid geometry.
_GRID_SHAPE = (3, 3)
_GRID_CELLS = _GRID_SHAPE[0] * _GRID_SHAPE[1]
# Color sampling.
_GEN_PALETTE = (1, 6, 8)
_COLOR_COUNT_RANGE = (2, 3)
# Validation retry budget.
_MAX_ATTEMPTS = 300


def _sample_grid() -> np.ndarray:
    color_count = random.randint(*_COLOR_COUNT_RANGE)
    colors = random.sample(_GEN_PALETTE, color_count)
    cells = list(colors) + [random.choice(colors) for _ in range(_GRID_CELLS - color_count)]
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(_GRID_SHAPE)


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _sample_grid()
        expected = np.concatenate((grid, np.fliplr(grid)), axis=1)
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid 6d0aefbc input")
