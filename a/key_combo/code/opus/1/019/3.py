import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Draw paired guide lines on a checkerboard and invert colors with XOR parity.

    Input structure: A two-color checkerboard has yellow markers on opposite
    borders. Matching left-right markers define horizontal guide rows; matching
    top-bottom markers define vertical guide columns.
    Rule: Interior checkerboard cells crossed by exactly one guide line swap the
    two checker colors, cells crossed by zero or two keep their color, then every
    guided row or column cell is painted yellow. Border cells remain unchanged.
    Key insight: Row and column inversions combine by parity, so intersections
    receive two inversions that cancel before the visible yellow line is drawn.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    hrows = {r for r in range(1, h - 1) if grid[r, 0] == 4 and grid[r, w - 1] == 4}
    vcols = {c for c in range(1, w - 1) if grid[0, c] == 4 and grid[h - 1, c] == 4}

    colors = [int(x) for x in np.unique(grid) if x not in (0, 4)]
    if len(colors) < 2:
        return out
    c1, c2 = colors[:2]
    swap = {c1: c2, c2: c1}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            k = (r in hrows) + (c in vcols)
            if k % 2 == 1 and out[r, c] in swap:
                out[r, c] = swap[int(out[r, c])]
    for r in hrows:
        out[r, 1:w - 1] = 4
    for c in vcols:
        out[1:h - 1, c] = 4
    return out


def generate_input() -> np.ndarray:
    h = random.randint(10, 14)
    w = random.randint(10, 14)
    colors = random.sample([1, 2, 3, 6, 7, 8, 9], 2)
    grid = np.fromfunction(lambda r, c: np.where((r + c) % 2 == 0, colors[0], colors[1]), (h, w), dtype=int)
    hrows = random.sample(range(2, h - 2), random.randint(1, min(3, h - 4)))
    vcols = random.sample(range(2, w - 2), random.randint(1, min(3, w - 4)))
    for r in hrows:
        grid[r, 0] = 4
        grid[r, w - 1] = 4
    for c in vcols:
        grid[0, c] = 4
        grid[h - 1, c] = 4
    return grid.astype(int)

