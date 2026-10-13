import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    sep_rows = [r for r in range(grid.shape[0]) if np.all(np.isin(grid[r], [1, 4])) and len(set(grid[r])) == 1]
    yellow = [r for r in sep_rows if grid[r, 0] == 4]
    n = len(sep_rows) - 1
    a, b = min(yellow), max(yellow)
    region = grid[a + 1:b]
    pts = np.argwhere((region != 0) & (region != 1) & (region != 4))
    r0, r1 = pts[:, 0].min(), pts[:, 0].max()
    c0, c1 = pts[:, 1].min(), pts[:, 1].max()
    motif = region[r0:r1 + 1, c0:c1 + 1]
    mh, mw = motif.shape
    out = np.zeros((n * mh, n * mw), dtype=int)
    for i in range(n):
        for j in range(n):
            tile = motif.copy()
            if j % 2:
                tile = np.fliplr(tile)
            if i % 2:
                tile = np.flipud(tile)
            out[i * mh:(i + 1) * mh, j * mw:(j + 1) * mw] = tile
    return out


def generate_input() -> np.ndarray:
    n = random.randint(2, 4)
    ph = 3
    period = ph + 1
    h = n * period + 1
    w = random.randint(8, 12)
    grid = np.zeros((h, w), dtype=int)
    separators = [i * period for i in range(n + 1)]
    marked = random.randint(0, n - 1)
    for idx, row in enumerate(separators):
        grid[row, :] = 4 if idx in (marked, marked + 1) else 1
    motif = np.zeros((ph, random.randint(3, 5)), dtype=int)
    pts = [(0, 0), (0, 2), (1, 1), (2, 1), (2, motif.shape[1] - 1)]
    for r, c in pts:
        motif[r, c] = random.choice([2, 3])
    off = random.randint(1, w - motif.shape[1] - 1)
    for i in range(n):
        top = separators[i] + 1
        grid[top:top + ph, off:off + motif.shape[1]] = motif
    return grid
