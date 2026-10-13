import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the foreground pixels of the input across a doubled canvas along the down-right diagonal.

    Input structure:
    - The input is a nonempty 2D grid with a unique dominant background color.
    - The non-background pixels form the sparse motif to be repeated.

    Rule:
    - Create an output canvas with twice the input height and twice the input width, filled with background.
    - For each foreground pixel at input position (r, c), paint that color at every output position (r+k, c+k) while it remains in bounds.
    - If two painted diagonal traces meet, they must agree in color.

    Key insight:
    - The input is stamped with transparent background at every one-cell down-right offset; it is not enlarged by scaling individual cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    bg = get_bg(g)
    values, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    assert np.count_nonzero(counts == max_count) == 1, "expected a unique dominant background"
    assert int(counts[values == bg][0]) == max_count, "expected inferred background to be dominant"

    out_height, out_width = 2 * height, 2 * width
    out = np.full((out_height, out_width), bg, dtype=int)
    coords = np.argwhere(g != bg)
    assert len(coords) > 0, "expected at least one foreground pixel"

    for r, c in coords:
        color = int(g[r, c])
        max_steps = min(out_height - int(r), out_width - int(c))
        for step in range(max_steps):
            rr = int(r) + step
            cc = int(c) + step
            target = int(out[rr, cc])
            assert target == bg or target == color, "expected diagonal stamps not to conflict"
            out[rr, cc] = color

    return out

# --------------- generator (o5x) ---------------

# Generator controls: canvas/colors, foreground sparsity, and retry budget.
MIN_GRID_SIZE, MAX_GRID_SIZE = 2, 5
MIN_FOREGROUND_PIXELS, MAX_FOREGROUND_PIXELS = 1, 5
BACKGROUND_COLOR, MIN_BACKGROUND_CELLS, MAX_GENERATION_ATTEMPTS = 0, 2, 300
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _gen_attempt():
    height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)

    max_fg = min(MAX_FOREGROUND_PIXELS, height + width - 1, height * width - MIN_BACKGROUND_CELLS)
    n_fg = random.randint(MIN_FOREGROUND_PIXELS, max_fg)
    colors = random.sample(FOREGROUND_COLORS, n_fg)

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    diagonals = random.sample(range(1 - width, height), n_fg)
    for color, diagonal in zip(colors, diagonals):
        cells = [(r, r - diagonal) for r in range(height) if 0 <= r - diagonal < width]
        r, c = random.choice(cells)
        grid[r, c] = color

    out = main(grid)
    if out.shape != (2 * height, 2 * width):
        return None
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input")
