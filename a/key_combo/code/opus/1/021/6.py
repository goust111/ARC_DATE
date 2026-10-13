import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == 1)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 1)]
    rows = sorted(rows)
    cols = sorted(cols)
    ctrl = tuple(np.argwhere(grid == 6)[0])
    rmid = sum(rows) / 2
    cmid = sum(cols) / 2
    sr = rows[0] if ctrl[0] < rmid else rows[1]
    sc = cols[0] if ctrl[1] < cmid else cols[1]
    for r in range(sr - 1, sr + 2):
        for c in range(sc - 1, sc + 2):
            if (r, c) != (sr, sc):
                out[r, c] = 4
    out[sr, sc] = 1
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([15, 17, 19])
    grid = np.zeros((h, w), dtype=int)
    rows = sorted(random.sample(range(3, h - 3), 2))
    cols = sorted(random.sample(range(3, w - 3), 2))
    while rows[1] - rows[0] < 4:
        rows = sorted(random.sample(range(3, h - 3), 2))
    while cols[1] - cols[0] < 4:
        cols = sorted(random.sample(range(3, w - 3), 2))
    grid[rows, :] = 1
    grid[:, cols] = 1
    rmid = sum(rows) / 2
    cmid = sum(cols) / 2
    quad = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    rr = range(1, int(rmid) - 1) if quad[0] == 0 else range(int(rmid) + 2, h - 1)
    cc = range(1, int(cmid) - 1) if quad[1] == 0 else range(int(cmid) + 2, w - 1)
    choices = [(r, c) for r in rr for c in cc if r not in rows and c not in cols]
    grid[random.choice(choices)] = 6
    return grid
