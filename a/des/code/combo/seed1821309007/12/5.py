import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MATRIX_COLORS = (3, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 300


def _nonempty_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.any(grid[r] != BG)]


def _find_matrix_height(grid: np.ndarray) -> int:
    top_h = 0
    for r in range(grid.shape[0]):
        if np.any(grid[r] == BG):
            break
        top_h += 1
    assert top_h > 0, "expected a non-empty top matrix"
    assert top_h < grid.shape[0], "expected marker rows below the matrix"
    return top_h


def _find_matrix_width(grid: np.ndarray, top_h: int) -> int:
    cols = np.flatnonzero(np.any(grid[:top_h] != BG, axis=0))
    assert cols.size > 0, "expected non-empty matrix columns"
    left = int(cols[0])
    right = int(cols[-1])
    assert left == 0, "matrix must be left-aligned"
    width = right + 1
    assert np.all(grid[:top_h, :width] != BG), "matrix must be a solid rectangle"
    assert np.all(grid[:top_h, width:] == BG), "matrix rows may only contain trailing background"
    return width


def _marker_columns(row: np.ndarray, color: int) -> list[int]:
    vals = set(int(v) for v in np.unique(row))
    assert vals <= {BG, color}, "marker row must contain only background and its marker color"
    cols = np.flatnonzero(row == color)
    assert cols.size > 0, "marker row must contain at least one marker"
    return [int(c) for c in cols]


def main(grid: np.ndarray) -> np.ndarray:
    """Extract a submatrix using bottom row/column selectors."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    top_h = _find_matrix_height(g)
    top_w = _find_matrix_width(g, top_h)
    matrix = g[:top_h, :top_w]

    marker_rows = [r for r in _nonempty_rows(g[top_h:]) if r is not None]
    marker_rows = [top_h + r for r in marker_rows]
    assert len(marker_rows) == 2, "expected exactly two non-empty marker rows below the matrix"
    assert np.all(g[top_h:marker_rows[0]] == BG), "only background may separate matrix and markers"

    row_a, row_b = marker_rows
    cols_a = set(int(v) for v in np.unique(g[row_a])) - {BG}
    cols_b = set(int(v) for v in np.unique(g[row_b])) - {BG}
    assert len(cols_a) == 1 and len(cols_b) == 1 and cols_a != cols_b, "marker rows must be single-color and distinct"

    color_a = int(next(iter(cols_a)))
    color_b = int(next(iter(cols_b)))
    assert {color_a, color_b} == {RED, BLUE}, "expected one red row and one blue row"

    red_row = row_a if color_a == RED else row_b
    blue_row = row_a if color_a == BLUE else row_b
    red_cols = _marker_columns(g[red_row], RED)
    blue_cols = _marker_columns(g[blue_row], BLUE)

    assert max(red_cols) < top_w, "red markers must point to matrix columns"
    assert max(blue_cols) < top_h, "blue markers must point to matrix rows"

    return matrix[np.array(blue_cols), :][:, np.array(red_cols)]


def _gen_attempt(rng: random.Random) -> np.ndarray:
    top_h = rng.randint(2, 8)
    top_w = rng.randint(max(3, top_h), 10)
    canvas_w = top_w + rng.randint(0, 3)

    color_count = rng.randint(2, min(6, len(MATRIX_COLORS)))
    palette = rng.sample(MATRIX_COLORS, color_count)
    matrix = np.array(
        [[palette[rng.randrange(color_count)] for _ in range(top_w)] for _ in range(top_h)],
        dtype=int,
    )

    red_count = rng.randint(1, min(5, top_w))
    blue_count = rng.randint(1, min(4, top_h))
    red_cols = sorted(rng.sample(range(top_w), red_count))
    blue_cols = sorted(rng.sample(range(top_h), blue_count))

    gap_after_matrix = rng.randint(1, 3)
    gap_between_markers = rng.randint(0, 2)
    tail_gap = rng.randint(0, 2)
    red_first = bool(rng.randrange(2))

    total_h = top_h + gap_after_matrix + 2 + gap_between_markers + tail_gap
    grid = np.zeros((total_h, canvas_w), dtype=int)
    grid[:top_h, :top_w] = matrix

    first_marker_r = top_h + gap_after_matrix
    second_marker_r = first_marker_r + 1 + gap_between_markers
    red_r, blue_r = (first_marker_r, second_marker_r) if red_first else (second_marker_r, first_marker_r)
    grid[red_r, red_cols] = RED
    grid[blue_r, blue_cols] = BLUE

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.size == 0:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
