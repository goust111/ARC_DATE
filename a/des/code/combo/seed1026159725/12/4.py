from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(10))
GRID_HEIGHT_RANGE = (6, 14)
GRID_WIDTH_RANGE = (6, 14)
MAX_LABELS_PER_AXIS = 5
GENERATION_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _top_labels(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    return [(c, int(grid[0, c])) for c in range(1, grid.shape[1]) if int(grid[0, c]) != bg]


def _left_labels(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    return [(r, int(grid[r, 0])) for r in range(1, grid.shape[0]) if int(grid[r, 0]) != bg]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a sparse row/column color code by coloring only matching label intersections.

    The first row stores column labels and the first column stores row labels.
    An interior cell becomes colored exactly when its row-label color equals its
    column-label color. All non-matching intersections remain background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = _background_color(g)
    out = g.copy()
    out[1:, 1:] = bg

    top = _top_labels(g, bg)
    left = _left_labels(g, bg)
    assert top and left

    for r, row_color in left:
        for c, col_color in top:
            if row_color == col_color:
                out[r, c] = row_color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    bg = rng.choice(PALETTE)

    max_row_labels = min(MAX_LABELS_PER_AXIS, width - 1)
    max_col_labels = min(MAX_LABELS_PER_AXIS, height - 1)
    if max_row_labels < 2 or max_col_labels < 2:
        return None

    row_count = rng.randint(2, max_row_labels)
    col_count = rng.randint(2, max_col_labels)
    shared_count = rng.randint(1, min(row_count, col_count) - 1)

    color_pool = [color for color in PALETTE if color != bg]
    rng.shuffle(color_pool)

    shared = color_pool[:shared_count]
    cursor = shared_count

    row_only_count = row_count - shared_count
    col_only_count = col_count - shared_count
    row_only = color_pool[cursor : cursor + row_only_count]
    cursor += row_only_count
    col_only = color_pool[cursor : cursor + col_only_count]

    row_colors = shared + row_only
    col_colors = shared + col_only
    rng.shuffle(row_colors)
    rng.shuffle(col_colors)

    row_positions = sorted(rng.sample(range(1, width), row_count))
    col_positions = sorted(rng.sample(range(1, height), col_count))

    grid = np.full((height, width), bg, dtype=int)
    for c, color in zip(row_positions, row_colors):
        grid[0, c] = color
    for r, color in zip(col_positions, col_colors):
        grid[r, 0] = color

    top = _top_labels(grid, bg)
    left = _left_labels(grid, bg)
    assert len(top) == row_count
    assert len(left) == col_count
    assert len({color for _, color in top}) == row_count
    assert len({color for _, color in left}) == col_count

    out = main(grid)
    interior = out[1:, 1:]
    filled = int(np.count_nonzero(interior != bg))
    assert filled == shared_count
    assert filled >= 1
    assert filled < row_count * col_count
    assert np.array_equal(out[0, :], grid[0, :])
    assert np.array_equal(out[:, 0], grid[:, 0])

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
