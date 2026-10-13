import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    sources = [tuple(p) for p in np.argwhere(grid == 5)]
    bars = sorted([(grid.shape[0] - r, r, c) for r, c in sources], reverse=True)
    colors = [1, 4, 2]
    for i, (_, r, c) in enumerate(bars):
        out[r:, c] = colors[i]
    return out


def generate_input() -> np.ndarray:
    h, w = random.choice([(12, 15), (14, 16), (15, 18)])
    grid = np.zeros((h, w), dtype=int)
    rows = random.sample(range(1, h - 2), 3)
    while min(abs(a - b) for i, a in enumerate(rows) for b in rows[i + 1 :]) < 2:
        rows = random.sample(range(1, h - 2), 3)
    cols = random.sample(range(1, w - 1), 3)
    for r, c in zip(rows, cols):
        grid[r, c] = 5
    return grid
