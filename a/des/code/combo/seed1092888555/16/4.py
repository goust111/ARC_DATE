from __future__ import annotations

import random

import numpy as np


def _full_rows_and_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r] == g[r, 0]))]
    full_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == g[0, c]))]
    return full_rows, full_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only pixels that belong to complete rows or complete columns.

    Complete rows/columns are those whose cells all have the same color.
    The output is black elsewhere. Complete-row color takes priority over
    complete-column color at intersections, though valid inputs force the
    two colors to agree at those cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_rows, full_cols = _full_rows_and_cols(g)
    out = np.zeros_like(g)

    for c in full_cols:
        out[:, c] = g[0, c]
    for r in full_rows:
        out[r, :] = g[r, 0]

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (4, 10)
GRID_WIDTH_RANGE = (4, 10)
COLOR_POOL = tuple(range(1, 10))
FILLER_COLOR_COUNT_RANGE = (1, 4)
GENERATION_RETRIES = 300


def _expected_output(grid: np.ndarray) -> np.ndarray:
    full_rows, full_cols = _full_rows_and_cols(grid)
    out = np.zeros_like(grid)
    for c in full_cols:
        out[:, c] = int(grid[0, c])
    for r in full_rows:
        out[r, :] = int(grid[r, 0])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    line_color = random.choice(COLOR_POOL)
    filler_palette = [color for color in COLOR_POOL if color != line_color]
    filler_count = random.randint(*FILLER_COLOR_COUNT_RANGE)
    fillers = random.sample(filler_palette, filler_count)

    full_row_count = random.randint(1, height - 1)
    full_col_count = random.randint(1, width - 1)
    full_rows = set(random.sample(range(height), full_row_count))
    full_cols = set(random.sample(range(width), full_col_count))

    # Start from non-line colors so non-complete rows/cols stay mixed once
    # full rows and full columns are stamped in.
    grid = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            grid[r, c] = random.choice(fillers)

    for r in full_rows:
        grid[r, :] = line_color
    for c in full_cols:
        grid[:, c] = line_color

    detected_rows, detected_cols = _full_rows_and_cols(grid)
    if set(detected_rows) != full_rows or set(detected_cols) != full_cols:
        return None

    if len(full_rows) == height or len(full_cols) == width:
        return None

    # Ensure both row-only and col-only branches are present in this sample.
    if not any(c not in full_cols for c in range(width)):
        return None
    if not any(r not in full_rows for r in range(height)):
        return None

    expected = _expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
