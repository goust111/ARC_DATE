from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each vertical red bar by recoloring its lower half teal.

    Input structure:
    - The background is black.
    - The only foreground color is red, arranged as one or more contiguous vertical bars.
    - Empty columns contain only background pixels.

    Rule:
    - For each red vertical bar, leave the upper ceil(height/2) cells red.
    - Recolor the lower floor(height/2) cells of that same bar to teal.
    - Preserve all background cells.

    Key insight:
    - Each occupied column is an independent height marker; the transformation is a
      per-bar bottom-half recolor, not a global shape or path operation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == Color.TEAL), "teal must be absent from the input"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    foreground = colors_present(g, exclude={bg})
    assert foreground == [Color.RED], "expected red as the only foreground color"

    out = g.copy()
    bar_count = 0
    for col in range(g.shape[1]):
        rows = np.flatnonzero(g[:, col] == Color.RED)
        if rows.size == 0:
            assert np.all(g[:, col] == bg), "empty columns must contain only background"
            continue

        bar_count += 1
        assert np.all(np.diff(rows) == 1), "red cells in each occupied column must be contiguous"
        assert rows.size >= 2, "each vertical bar must have a lower half to recolor"
        lower_count = rows.size // 2
        out[rows[-lower_count:], col] = Color.TEAL

    assert bar_count > 0, "expected at least one vertical bar"
    return out

# --------------- generator (o5x) ---------------

# Canvas and bar geometry.
_GEN_ROW_BOUNDS = (7, 11)
_GEN_GRID_WIDTH = 9
_GEN_BAR_COLUMNS = (1, 3, 5, 7)
_GEN_MIN_BAR_HEIGHT = 2

# Retry budget.
_GEN_MAX_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GEN_ROW_BOUNDS)
    inp = np.full((rows, _GEN_GRID_WIDTH), Color.BLACK, dtype=int)
    expected = inp.copy()

    heights = random.sample(range(_GEN_MIN_BAR_HEIGHT, rows), k=len(_GEN_BAR_COLUMNS))
    for col, height in zip(_GEN_BAR_COLUMNS, heights):
        start = rows - height
        inp[start:rows, col] = Color.RED
        expected[start:rows, col] = Color.RED
        expected[rows - (height // 2) : rows, col] = Color.TEAL

    if not np.all(inp[0, :] == Color.BLACK):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate a valid ce9e57f2 input")
