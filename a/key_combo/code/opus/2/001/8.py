import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    band = np.concatenate([grid, np.fliplr(grid)], axis=1)
    tiled = np.concatenate([np.flipud(band), band, np.flipud(band)], axis=0)
    h, w = tiled.shape
    out = np.zeros_like(tiled)
    for c in range(w):
        count = int(np.count_nonzero(tiled[:, c]))
        if count == 0:
            continue
        color = 1 if count == 1 else 2 if count == 2 else 3 if count == 3 else 4
        out[h - count : h, c] = color
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.choice([3, 4])
        w = random.choice([4, 5])
        grid = np.zeros((h, w), dtype=int)
        cells = random.sample([(r, c) for r in range(h) for c in range(w)], random.randint(3, 6))
        for r, c in cells:
            grid[r, c] = 1
        if any(np.all(grid[:, c] == 1) for c in range(w)):
            continue
        band = np.concatenate([grid, np.fliplr(grid)], axis=1)
        heights = [int(np.count_nonzero(np.concatenate([np.flipud(band), band, np.flipud(band)], axis=0)[:, c])) for c in range(band.shape[1])]
        if len(set(heights)) >= 3:
            return grid
