from __future__ import annotations

import numpy as np

from arc_utils import colors_present
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the red cell-center columns by how many filled centers they contain, from fullest to emptiest.

    Input structure:
    - The grid is a square macro-grid of red 3x3 cells separated by one black vertical gutter column.
    - Every macro-cell can differ only at its center pixel, which is either blank or filled.

    Rule:
    - Treat the center pixels as columns of a binary matrix.
    - Count filled centers in each macro-column.
    - Move whole center columns into descending count order, preserving each column's vertical pattern and leaving the frame/gutters unchanged.

    Key insight:
    - The visible swaps are a column sort of the hidden center matrix; no center pixel is created or deleted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 7, "expected at least a 2x2 macro-grid"
    assert h % 2 == 1 and (w + 1) % 4 == 0, "expected 2r+1 by 4c-1 macro-grid geometry"

    n_rows = (h - 1) // 2
    n_cols = (w + 1) // 4
    assert n_rows == n_cols and n_cols >= 2, "expected a square macro-grid"

    center_rows = np.arange(1, h, 2)
    center_cols = np.arange(1, w, 4)
    sep_cols = np.arange(3, w, 4)

    blank = int(g[0, sep_cols[0]])
    filled = int(g[0, 0])
    assert filled != blank, "expected distinct frame and blank colors"
    assert set(colors_present(g)) == {blank, filled}, "expected only frame and blank colors"
    assert np.all(g[:, sep_cols] == blank), "expected blank vertical gutters"

    variable = np.zeros(g.shape, dtype=bool)
    variable[np.ix_(center_rows, center_cols)] = True
    frame = np.ones(g.shape, dtype=bool)
    frame[:, sep_cols] = False
    frame &= ~variable
    assert np.all(g[frame] == filled), "expected fixed filled cell frames"

    centers = g[np.ix_(center_rows, center_cols)]
    assert np.all((centers == blank) | (centers == filled)), "expected binary cell centers"

    filled_mask = centers == filled
    counts = filled_mask.sum(axis=0)
    assert len(set(map(int, counts))) == n_cols, "expected unique filled-center counts"
    order = np.argsort(-counts)

    out = g.copy()
    out[np.ix_(center_rows, center_cols)] = centers[:, order]
    return out

# --------------- generator (o5x) ---------------
# Colors and macro-grid geometry.
FILLED_COLOR, BLANK_COLOR = 2, 0
MIN_N, MAX_N = 3, 5
ROW_STRIDE, COL_STRIDE = 2, 4
GRID_EXTRA_ROWS, GRID_EXTRA_COLS = 1, -1
CENTER_OFFSET, GUTTER_OFFSET = 1, 3
# Solver-oracle retry budget.
MAX_GENERATION_ATTEMPTS = 300

def _gen_grid_from_centers(center_matrix: np.ndarray) -> np.ndarray:
    n = int(center_matrix.shape[0])
    h, w = ROW_STRIDE * n + GRID_EXTRA_ROWS, COL_STRIDE * n + GRID_EXTRA_COLS
    grid = np.full((h, w), FILLED_COLOR, dtype=int)
    sep_cols = np.arange(GUTTER_OFFSET, w, COL_STRIDE)
    center_rows = np.arange(CENTER_OFFSET, h, ROW_STRIDE)
    center_cols = np.arange(CENTER_OFFSET, w, COL_STRIDE)
    grid[:, sep_cols] = BLANK_COLOR
    grid[np.ix_(center_rows, center_cols)] = np.where(center_matrix, FILLED_COLOR, BLANK_COLOR)
    return grid


def _gen_sorted_centers(n: int) -> np.ndarray:
    centers = np.zeros((n, n), dtype=bool)
    for c in range(n):
        count = n - 1 - c
        if count:
            rows = np.random.choice(n, size=count, replace=False)
            centers[rows, c] = True
    return centers


def _gen_non_identity_permutation(n: int) -> np.ndarray:
    perm = np.random.permutation(n)
    if np.array_equal(perm, np.arange(n)):
        i, j = random.sample(range(n), 2)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(MIN_N, MAX_N)
    sorted_centers = _gen_sorted_centers(n)
    perm = _gen_non_identity_permutation(n)
    input_centers = sorted_centers[:, perm]
    inp = _gen_grid_from_centers(input_centers)
    expected = _gen_grid_from_centers(sorted_centers)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 42a15761 input")
