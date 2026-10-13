from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each background cell at the intersections of the marked rows and marked columns with red.

    Input structure:
    - The grid contains a background color and one marker color.
    - All marker pixels are headers: some are in the top row and the rest are in
      the rightmost column.

    Rule:
    - The top-row markers select columns, and the right-edge markers select rows.
    - Preserve the input markers and paint every selected row/column intersection red.

    Key insight:
    - The edge markers form row and column headers for a sparse table; the new red
      pixels are exactly the Cartesian product of those header positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H >= 2 and W >= 2, "expected room for top and right headers"
    assert not np.any(g == Color.RED), "red must be absent from the input"

    bg = get_bg(g)
    marker_colors = colors_present(g, exclude={bg})
    assert len(marker_colors) == 1, "expected exactly one marker color"
    marker = marker_colors[0]

    marker_coords = np.argwhere(g == marker)
    assert marker_coords.size > 0, "expected header markers"
    assert all(r == 0 or c == W - 1 for r, c in marker_coords), (
        "markers must lie only on the top row or right edge"
    )

    marker_cols = [int(c) for c in np.flatnonzero(g[0, :] == marker)]
    marker_rows = [int(r) for r in np.flatnonzero(g[:, W - 1] == marker)]
    assert marker_cols and marker_rows, "expected both row and column headers"
    assert all(c < W - 1 for c in marker_cols), "top-right marker is not part of this task"
    assert all(r > 0 for r in marker_rows), "top-right marker is not part of this task"

    out = g.copy()
    for r in marker_rows:
        for c in marker_cols:
            assert out[r, c] == bg, "header intersection must be background"
            out[r, c] = Color.RED
    return out

# --------------- generator (o5x) ---------------
# Geometry and sampling controls.
GRID_SHAPE = (10, 10)
TOP_ROW = 0
RIGHT_COL = GRID_SHAPE[1] - 1
COL_HEADER_CHOICES = range(TOP_ROW, RIGHT_COL)
ROW_HEADER_CHOICES = range(TOP_ROW + 1, GRID_SHAPE[0])
HEADER_COUNT_BOUNDS = (2, 5)
GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    g = np.full(GRID_SHAPE, Color.BLACK, dtype=int)
    col_count = random.randint(*HEADER_COUNT_BOUNDS)
    row_count = random.randint(*HEADER_COUNT_BOUNDS)
    col_headers = random.sample(COL_HEADER_CHOICES, col_count)
    row_headers = random.sample(ROW_HEADER_CHOICES, row_count)

    g[TOP_ROW, col_headers] = Color.GRAY
    g[row_headers, RIGHT_COL] = Color.GRAY

    expected = g.copy()
    expected[np.ix_(row_headers, col_headers)] = Color.RED

    if np.array_equal(main(g), expected):
        return g
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid 2281f1f4 input")
