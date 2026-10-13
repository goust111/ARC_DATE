import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    rows = []
    for r in range(grid.shape[0]):
        blues = np.where(grid[r] == 1)[0]
        cyans = np.where(grid[r] == 8)[0]
        if len(blues) == 1 and len(cyans) == 1 and blues[0] < cyans[0]:
            c1, c2 = int(blues[0]), int(cyans[0])
            if c2 - c1 == 5:
                rows.append(grid[r, c1 + 1:c2].copy())
    if not rows:
        return np.zeros((1, 1), dtype=int)
    mat = np.vstack(rows)
    k = (mat.shape[0] // 2) * 2
    mat = mat[:k]
    blocks = []
    for br in range(0, mat.shape[0], 2):
        for bc in range(0, 4, 2):
            blocks.append(mat[br:br + 2, bc:bc + 2])
    K = len(blocks)
    n = 2 * K - 1
    center = K - 1
    out = np.zeros((n, n), dtype=int)
    for m in range(K):
        block = blocks[K - 1 - m]
        dist = K - 1 - m
        for r in range(n):
            for c in range(n):
                if max(abs(r - center), abs(c - center)) == dist:
                    rr = 0 if r < center else 1
                    cc = 0 if c < center else 1
                    out[r, c] = block[rr, cc]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(10, 15)
    w = random.randint(12, 18)
    grid = np.zeros((h, w), dtype=int)
    nrows = random.choice([2, 4])
    start = random.randint(1, h - nrows - 1)
    step = random.choice([1, 2])
    if start + (nrows - 1) * step >= h:
        step = 1
        start = random.randint(1, h - nrows - 1)
    left = random.randint(0, w - 6)
    colors = [2, 3, 4, 5, 7, 9]
    for i in range(nrows):
        r = start + i * step
        grid[r, left] = 1
        grid[r, left + 5] = 8
        grid[r, left + 1:left + 5] = [random.choice(colors) for _ in range(4)]
    return grid
