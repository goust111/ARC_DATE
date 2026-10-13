import numpy as np

from arc_utils import Color
import itertools
import random


TALLY_CELLS = ((0, 0), (0, 1), (0, 2), (1, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace scattered blue pixels with an upper-left red tally of the same size.

    Input structure
    - The grid is a 3x3 black-background mask.
    - Blue pixels are the only foreground pixels, with a count from one to four.
    Rule
    - Ignore the source positions and count the blue pixels.
    - Clear the grid and paint that many red pixels in the canonical tally order:
      top row from left to right, then the center cell.
    Key insight
    - The blue mask is summarized by pixel count rather than moved as a shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected a 3x3 grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE}, "expected only black bg and blue foreground"

    count = int(np.count_nonzero(g == Color.BLUE))
    assert 1 <= count <= len(TALLY_CELLS), "expected one to four blue pixels"

    out = np.full_like(g, Color.BLACK)
    for r, c in TALLY_CELLS[:count]:
        out[r, c] = Color.RED
    return out

# --------------- generator (o5x) ---------------
# Canvas/color model.
_GEN_GRID_SHAPE = (3, 3)
_GEN_BG_COLOR, _GEN_INPUT_COLOR, _GEN_OUTPUT_COLOR = Color.BLACK, Color.BLUE, Color.RED
# Count/retry controls.
_GEN_BLUE_COUNT_RANGE, _GEN_MAX_ATTEMPTS = (1, 4), 300
# Tally placement.
_GEN_TALLY_CELLS = ((0, 0), (0, 1), (0, 2), (1, 1))
_GEN_ALL_CELLS = tuple((r, c) for r in range(_GEN_GRID_SHAPE[0]) for c in range(_GEN_GRID_SHAPE[1]))

def _choose_blue_cells(blue_count):
    """Return a non-tally-prefix combination for the requested count."""
    forbidden = frozenset(_GEN_TALLY_CELLS[:blue_count])
    return random.choice([
        cells
        for cells in itertools.combinations(_GEN_ALL_CELLS, blue_count)
        if frozenset(cells) != forbidden
    ])

def _paint_cells(cells, color):
    """Paint selected cells on a fresh black 3x3 grid."""
    grid = np.full(_GEN_GRID_SHAPE, _GEN_BG_COLOR, dtype=int)
    for r, c in cells:
        grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    blue_count = random.randint(*_GEN_BLUE_COUNT_RANGE)
    grid = _paint_cells(_choose_blue_cells(blue_count), _GEN_INPUT_COLOR)
    expected = _paint_cells(_GEN_TALLY_CELLS[:blue_count], _GEN_OUTPUT_COLOR)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 794b24be input")
