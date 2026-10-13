import random
from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    sep = w // 2
    left = grid[:, :sep] == 1
    right = np.fliplr(grid[:, sep + 1 :]) == 3
    mask = np.logical_xor(left, right)
    run = 0
    while run < h and grid[run, sep] == 5:
        run += 1
    run = max(run, 1)
    out = np.zeros((h, sep), dtype=int)
    colors = [1, 3]
    idx = 0
    for r, c in map(tuple, np.argwhere(mask)):
        out[r, c] = colors[(idx // run) % 2]
        idx += 1
    return out


def _grow(seed, n, steps):
    cells = {seed}
    for _ in range(steps):
        r, c = random.choice(tuple(cells))
        opts = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        random.shuffle(opts)
        for nr, nc in opts:
            if 1 <= nr < n - 1 and 1 <= nc < n - 1:
                cells.add((nr, nc))
                break
    return cells


def generate_input() -> np.ndarray:
    h, half = 14, 12
    grid = np.zeros((h, half * 2 + 1), dtype=int)
    run = random.randint(1, 4)
    grid[:run, half] = 5
    base = _grow((h // 2, half // 2), min(h, half), random.randint(18, 26))
    left = set(base)
    right = set(base)
    for cell in random.sample(tuple(base), random.randint(4, 8)):
        right.discard(cell)
    for _ in range(random.randint(5, 9)):
        right.add((random.randint(1, h - 2), random.randint(1, half - 2)))
    for r, c in left:
        grid[r, c] = 1
    for r, c in right:
        grid[r, half + 1 + (half - 1 - c)] = 3
    return grid

