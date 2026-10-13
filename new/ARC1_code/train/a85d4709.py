import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each row with the color selected by that row's grey marker column.

    Input structure:
    - The grid is a 3x3 black background with grey marker pixels.
    - Each row contains exactly one grey marker and two black pixels.

    Rule:
    - Read the grey marker column in each row.
    - Replace the whole row by the color assigned to that column: left -> red, middle -> yellow, right -> green.

    Key insight:
    - The grey pixels are row-wise control pixels; their horizontal position, not their connected shape, determines the output row color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3), "expected a 3x3 grid"
    assert set(map(int, np.unique(g))) <= {0, 5}, "expected only black background and grey markers"
    assert np.all(np.sum(g == 5, axis=1) == 1), "expected exactly one grey marker per row"

    column_colors = np.array([2, 4, 3], dtype=int)
    marker_cols = np.argmax(g == 5, axis=1)
    return np.repeat(column_colors[marker_cols, None], 3, axis=1)

# --------------- generator (o5x) ---------------

# CFG: fixed grid/color contract and oracle retry budget.
GRID_SHAPE = (3, 3)
BACKGROUND_COLOR, MARKER_COLOR = 0, 5
COLUMN_COLORS = np.array([2, 4, 3], dtype=int)
MAX_ATTEMPTS = 300

def _gen_attempt():
    grid = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    for r in range(GRID_SHAPE[0]):
        grid[r, random.randrange(GRID_SHAPE[1])] = MARKER_COLOR
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        marker_cols = np.argmax(inp == MARKER_COLOR, axis=1)
        expected = np.repeat(COLUMN_COLORS[marker_cols, None], GRID_SHAPE[1], axis=1)
        try:
            out = main(inp)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
