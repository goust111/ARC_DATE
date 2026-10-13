import random

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    h, w = grid.shape
    for r, c in sorted(map(tuple, np.argwhere(grid == 2))):
        for dr, dc in DIRS:
            cells = []
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 5:
                cells.append((nr, nc))
                nr += dr
                nc += dc
            if not cells:
                continue
            for cr, cc in cells:
                if out[cr, cc] == 0:
                    out[cr, cc] = 3
            lr, lc = cells[-1]
            out[lr, lc] = 7
        out[r, c] = 3
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    rows = sorted(random.sample(range(4, 15), 2))
    cols = sorted(random.sample(range(4, 15), 2))
    for r in rows:
        grid[r, :] = 5
    for c in cols:
        grid[:, c] = 5
    regions = [
        (1, rows[0] - 1, 1, cols[0] - 1),
        (rows[0] + 1, rows[1] - 1, cols[0] + 1, cols[1] - 1),
        (rows[1] + 1, 16, cols[1] + 1, 16),
    ]
    for r0, r1, c0, c1 in regions[: random.randint(2, 3)]:
        if r0 <= r1 and c0 <= c1:
            grid[random.randint(r0, r1), random.randint(c0, c1)] = 2
    for _ in range(random.randint(4, 8)):
        r, c = random.randrange(18), random.randrange(18)
        if grid[r, c] == 0:
            grid[r, c] = 1
    return grid

