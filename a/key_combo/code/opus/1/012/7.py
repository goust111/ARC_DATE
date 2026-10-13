import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    horizontal = []
    for color in sorted(set(np.unique(grid)) - {0}):
        pts = [tuple(p) for p in np.argwhere(grid == color)]
        if len(pts) != 2:
            continue
        (r1, c1), (r2, c2) = pts
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                out[r1, c] = color
            horizontal.append((r1, min(c1, c2), max(c1, c2), int(color)))
        elif c1 == c2:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                out[r, c1] = color
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if out[r, c] != 0:
                continue
            candidates = [(hr, col) for hr, c0, c1, col in horizontal if hr < r and c0 <= c <= c1]
            if candidates:
                out[r, c] = max(candidates)[1]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((14, 14), dtype=int)
    while True:
        rows = sorted(random.sample(range(2, 12), 2))
        if rows[1] - rows[0] >= 3:
            break
    colors = random.sample([2, 3, 4], 3)
    for row, color in zip(rows, colors[:2]):
        c0 = random.randint(1, 3)
        c1 = random.randint(9, 12)
        grid[row, c0] = color
        grid[row, c1] = color
    if random.random() < 0.8:
        col = random.choice([1, 12])
        r0 = random.randint(rows[0] + 1, rows[1] - 1)
        r1 = random.randint(rows[1] + 1, 12)
        grid[r0, col] = colors[2]
        grid[r1, col] = colors[2]
    return grid
