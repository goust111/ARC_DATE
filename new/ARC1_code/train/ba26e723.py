from __future__ import annotations

import numpy as np

from arc_utils import Color
import random


def _alternating_row(start: int, width: int) -> np.ndarray:
    assert start in (Color.BLACK, Color.YELLOW), "alternating row must start with black or yellow"
    other = Color.YELLOW if start == Color.BLACK else Color.BLACK
    return np.where(np.arange(width) % 2 == 0, start, other)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the yellow pixels in every third column of the three-row alternating lattice to pink.

    Input structure
    - A 3-row grid containing only black bg and yellow foreground.
    - The middle row is solid yellow.
    - The top row alternates black/yellow, and the bottom row is its color complement.

    Rule
    - Keep the grid shape and all black pixels unchanged.
    - In columns 0, 3, 6, ..., recolor any yellow pixel to pink.

    Key insight
    - The pink marks are not new objects; they are a regular column-period highlight applied only where the yellow lattice already has pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == 3, "expected exactly three rows"
    assert width > 0, "expected a nonempty lattice"
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.YELLOW}, "expected only black and yellow"
    assert np.all(g[1, :] == Color.YELLOW), "middle row must be solid yellow"

    expected_top = _alternating_row(int(g[0, 0]), width)
    expected_bottom = np.where(expected_top == Color.YELLOW, Color.BLACK, Color.YELLOW)
    assert np.array_equal(g[0, :], expected_top), "top row must alternate"
    assert np.array_equal(g[2, :], expected_bottom), "bottom row must complement the top row"

    out = g.copy()
    recolor = np.zeros(g.shape, dtype=bool)
    recolor[:, ::3] = g[:, ::3] == Color.YELLOW
    out[recolor] = Color.PINK
    return out

# --------------- generator (o5x) ---------------
# Sampling bounds and validation knobs.
WIDTH_BOUNDS = (10, 17)
START_COLORS = (Color.BLACK, Color.YELLOW)
HIGHLIGHT_STRIDE = 3
MAX_GENERATION_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_BOUNDS)
    start = random.choice(START_COLORS)

    top = _alternating_row(start, width)
    middle = np.full(width, Color.YELLOW, dtype=int)
    bottom = np.where(top == Color.YELLOW, Color.BLACK, Color.YELLOW)
    grid = np.vstack([top, middle, bottom]).astype(int)

    out = grid.copy()
    recolor = np.zeros(grid.shape, dtype=bool)
    recolor[:, ::HIGHLIGHT_STRIDE] = grid[:, ::HIGHLIGHT_STRIDE] == Color.YELLOW
    out[recolor] = Color.PINK

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ba26e723 input")
