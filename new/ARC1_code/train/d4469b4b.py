from __future__ import annotations

import numpy as np

from arc_utils import Color


_COLOR_TO_SYMBOL = {
    Color.RED: np.array(
        [
            [Color.GREY, Color.GREY, Color.GREY],
            [Color.BLACK, Color.GREY, Color.BLACK],
            [Color.BLACK, Color.GREY, Color.BLACK],
        ],
        dtype=int,
    ),
    Color.BLUE: np.array(
        [
            [Color.BLACK, Color.GREY, Color.BLACK],
            [Color.GREY, Color.GREY, Color.GREY],
            [Color.BLACK, Color.GREY, Color.BLACK],
        ],
        dtype=int,
    ),
    Color.GREEN: np.array(
        [
            [Color.BLACK, Color.BLACK, Color.GREY],
            [Color.BLACK, Color.BLACK, Color.GREY],
            [Color.GREY, Color.GREY, Color.GREY],
        ],
        dtype=int,
    ),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the noisy 5x5 single-color mask with the fixed 3x3 grey symbol encoded by its foreground color.

    Input structure:
    - The input is a 5x5 grid with black background.
    - All foreground pixels share one visible color: red, blue, or green.
    - The exact foreground pixel positions vary within each color class.

    Rule:
    - Red emits a grey top-T symbol.
    - Blue emits a grey plus symbol.
    - Green emits a grey lower-right L symbol.
    - The output background is black and the symbol pixels are grey.

    Key insight:
    - The foreground shape is distractor noise; the single foreground color is the code for the 3x3 output symbol.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (5, 5), "expected a 5x5 input grid"

    foreground_colors = [int(color) for color in np.unique(g) if int(color) != Color.BLACK]
    assert len(foreground_colors) == 1, "expected exactly one foreground color"

    color = foreground_colors[0]
    assert color in _COLOR_TO_SYMBOL, "unexpected foreground color"
    return _COLOR_TO_SYMBOL[color].copy()

# --------------- generator (o5x) ---------------

# Canvas, colors, sampling bounds, and retry budget.
INPUT_SHAPE = (5, 5)
OUTPUT_SHAPE = (3, 3)
BACKGROUND_COLOR = Color.BLACK
SYMBOL_COLOR = Color.GREY
FOREGROUND_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
FOREGROUND_COUNT_MIN = 8
FOREGROUND_COUNT_MAX = 14
MAX_GENERATION_ATTEMPTS = 300

_SYMBOL_COORDS_BY_COLOR = {
    Color.BLUE: ((1, 0), (1, 1), (1, 2), (0, 1), (2, 1)),
    Color.RED: ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    Color.GREEN: ((2, 0), (2, 1), (2, 2), (0, 2), (1, 2)),
}


def _gen_symbol_for_color(color: int) -> np.ndarray:
    symbol = np.full(OUTPUT_SHAPE, BACKGROUND_COLOR, dtype=int)
    rows, cols = zip(*_SYMBOL_COORDS_BY_COLOR[color])
    symbol[rows, cols] = SYMBOL_COLOR
    return symbol


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    color = int(rng.choice(FOREGROUND_COLORS))
    count = int(rng.integers(FOREGROUND_COUNT_MIN, FOREGROUND_COUNT_MAX + 1))
    grid = np.full(INPUT_SHAPE, BACKGROUND_COLOR, dtype=int)
    chosen = rng.choice(np.prod(INPUT_SHAPE), size=count, replace=False)
    grid.flat[chosen] = color
    out = _gen_symbol_for_color(color)
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d4469b4b input")
