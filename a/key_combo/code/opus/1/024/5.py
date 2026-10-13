import random
import numpy as np


def _extend(panel, color):
    h, w = panel.shape
    out = np.zeros_like(panel)
    for r in range(h):
        cols = np.where(panel[r] == color)[0]
        if cols.size == 0:
            continue
        left, right = int(cols.min()), int(cols.max())
        length = right - left + 1
        start = left % length
        for c in range(start, w, length):
            out[r, c:min(w, c + length)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    sep = int(np.where(np.all(grid == 6, axis=1))[0][0])
    top = _extend(grid[:sep], 1)
    bottom = _extend(grid[sep + 1:], 2)
    h, w = top.shape
    out = np.zeros((h, w), dtype=int)
    out[(top == 1) & (bottom == 0)] = 1
    out[(top == 0) & (bottom == 2)] = 2
    out[(top == 1) & (bottom == 2)] = 3
    return out


def generate_input() -> np.ndarray:
    h, w = 15, 10
    half = 7
    g = np.zeros((h, w), dtype=int)
    g[half, :] = 6
    rows_top = random.sample(range(half), random.randint(3, 5))
    rows_bottom = set(random.sample(rows_top, random.randint(2, min(4, len(rows_top)))))
    while len(rows_bottom) < 3:
        rows_bottom.add(random.randrange(half))
    for r in rows_top:
        length = random.randint(2, 3)
        c = random.randrange(0, w - length + 1)
        g[r, c:c + length] = 1
    for r in rows_bottom:
        length = random.randint(2, 4)
        c = random.randrange(0, w - length + 1)
        g[half + 1 + r, c:c + length] = 2
    return g
