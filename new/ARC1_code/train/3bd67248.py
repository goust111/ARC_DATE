from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a left-edge vertical bar into a right-triangle outline.

    Input structure:
    - The grid is square with a uniform background.
    - A single non-background color forms a full-height vertical bar in column 0.
    - All cells to the right of the bar are background.

    Rule:
    - Preserve the input bar.
    - Paint a red diagonal from the top-right corner down-left to the cell just
      above the bottom-left bar endpoint.
    - Paint a yellow base on the bottom row from column 1 to the right edge.

    Key insight:
    - The given bar is the left side of an implied right triangle; the output
      draws the other two sides in fixed palette colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    assert h == w and h >= 2, "expected a square grid of side at least 2"

    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground bar color"
    bar_color = foreground_colors[0]
    assert np.all(g[:, 0] == bar_color), "foreground must be a full-height left bar"
    assert np.all(g[:, 1:] == bg), "all non-bar cells must be background"

    out = g.copy()
    rows = np.arange(h - 1)
    cols = (w - 1) - rows
    out[rows, cols] = Color.RED
    out[h - 1, 1:w] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and retry configuration for this direct generator.
GRID_SIZE_MIN, GRID_SIZE_STOP = 3, 16
BAR_COLOR_CHOICES = [1, 3, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    n = inp.shape[0]
    rows = np.arange(n - 1)
    cols = (n - 1) - rows
    out[rows, cols] = Color.RED
    out[n - 1, 1:n] = Color.YELLOW
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_STOP))
    bar_color = int(np.random.choice(BAR_COLOR_CHOICES))

    inp = np.zeros((n, n), dtype=int)
    inp[:, 0] = bar_color

    expected = _gen_expected_output(inp)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input for 3bd67248")
