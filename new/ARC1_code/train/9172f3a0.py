from __future__ import annotations

import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each input pixel into a solid 3x3 block of the same color.

    Input structure
    - The input is a 3x3 color grid.

    Rule
    - Replace every pixel at input position (r, c) with a 3x3 block in output
      rows 3*r..3*r+2 and columns 3*c..3*c+2.
    - Preserve the original pixel color, including the background color.

    Key insight
    - The output is a fixed nearest-neighbor scale-up of the entire 3x3 grid,
      not a foreground-only object transformation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (3, 3), "expected the 3x3 source grid"

    return np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)

# --------------- generator (o5x) ---------------

# Grid geometry and output scaling.
GRID_SHAPE = (3, 3)
GRID_CELLS = GRID_SHAPE[0] * GRID_SHAPE[1]
UPSCALE_FACTOR = 3
# Palette, occupancy, and per-color count bounds.
COLOR_RANGE = range(1, 10)
NON_BLACK_COLOR_BOUNDS = (2, 3)
FILLED_CELL_BOUNDS = (4, 5)
COLOR_COUNT_BOUNDS = (1, 3)
# Retry budget.
GENERATION_RETRIES = 300

def _gen_color_counts(n_colors: int, n_filled: int) -> list[int]:
    min_count, max_count = COLOR_COUNT_BOUNDS
    counts = [min_count] * n_colors
    remaining = n_filled - n_colors * min_count
    while remaining:
        eligible = [i for i, count in enumerate(counts) if count < max_count]
        i = random.choice(eligible)
        counts[i] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts

def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    n_colors = random.randint(*NON_BLACK_COLOR_BOUNDS)
    n_filled = random.randint(*FILLED_CELL_BOUNDS)
    colors = random.sample(COLOR_RANGE, n_colors)
    counts = _gen_color_counts(n_colors, n_filled)
    values = [color for color, count in zip(colors, counts) for _ in range(count)]
    random.shuffle(values)

    cells = random.sample(range(GRID_CELLS), n_filled)
    for idx, value in zip(cells, values):
        grid[idx // GRID_SHAPE[1], idx % GRID_SHAPE[1]] = value

    expected = np.repeat(np.repeat(grid, UPSCALE_FACTOR, axis=0), UPSCALE_FACTOR, axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
