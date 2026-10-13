import numpy as np

from arc_utils import Color, colors_present
import random


FOREGROUND_TO_FILL = {
    Color.TEAL: Color.RED,
    Color.GREEN: Color.BLUE,
    Color.GREY: Color.YELLOW,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert the single-color 3x3 mask, turning marked cells black and unmarked cells into the marker color's paired color.

    Input structure:
    - The grid is 3x3.
    - Black cells form the background; every non-black cell has the same foreground color.
    - The foreground color is one of the trained paired colors: teal, green, or grey.

    Rule:
    - Choose the paired fill color for the foreground: teal -> red, green -> blue, grey -> yellow.
    - Paint every black input cell with that paired color.
    - Paint every foreground input cell black.

    Key insight:
    - Geometry only selects which cells swap roles; the output color is determined by the foreground color's visible pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3), "expected a 3x3 grid"
    assert np.any(g == Color.BLACK), "expected at least one black background cell"

    foreground_colors = colors_present(g, exclude={Color.BLACK})
    assert len(foreground_colors) == 1, "expected exactly one non-black foreground color"

    foreground = foreground_colors[0]
    assert foreground in FOREGROUND_TO_FILL, "foreground color must have a trained paired fill color"

    out = np.full(g.shape, FOREGROUND_TO_FILL[foreground], dtype=int)
    out[g == foreground] = Color.BLACK
    return out

# --------------- generator (o5x) ---------------

# Canvas, population, and retry bounds.
GRID_SHAPE = (3, 3)
CELL_COUNT = GRID_SHAPE[0] * GRID_SHAPE[1]
FOREGROUND_COUNT_RANGE = (1, CELL_COUNT - 1)
MAX_GENERATION_ATTEMPTS = 300

# Color mapping.
FOREGROUND_COLORS = [Color.TEAL, Color.GREEN, Color.GREY]
FILL_BY_FOREGROUND = FOREGROUND_TO_FILL


def _gen_attempt() -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    foreground = random.choice(FOREGROUND_COLORS)
    k = random.randint(*FOREGROUND_COUNT_RANGE)
    for pos in random.sample(range(CELL_COUNT), k):
        grid[pos // GRID_SHAPE[1], pos % GRID_SHAPE[1]] = foreground
    return grid


def _expected_output(inp: np.ndarray, foreground: int) -> np.ndarray:
    out = np.full(inp.shape, FILL_BY_FOREGROUND[foreground], dtype=int)
    out[inp == foreground] = Color.BLACK
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()

        foregrounds = [color for color in np.unique(inp).tolist() if color != Color.BLACK]
        if len(foregrounds) != 1:
            continue

        foreground = foregrounds[0]
        try:
            if np.array_equal(main(inp), _expected_output(inp, foreground)):
                return inp
        except Exception:
            pass

    raise RuntimeError("failed to generate a valid 6ea4a07e input")
