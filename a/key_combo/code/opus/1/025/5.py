import random
import numpy as np


def _sgn(x):
    return (x > 0) - (x < 0)


def _path(a, b):
    r, c = a
    tr, tc = b
    out = []
    while max(abs(tr - r), abs(tc - c)) > 1:
        dr, dc = tr - r, tc - c
        nr, nc = r + _sgn(dr), c + _sgn(dc)
        if abs(dr) and abs(dc):
            pass
        elif abs(dr):
            nr, nc = r + _sgn(dr), c
        else:
            nr, nc = r, c + _sgn(dc)
        r, c = nr, nc
        out.append((r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = grid.copy()
    reds = np.argwhere(grid == 2)
    yellows = np.argwhere(grid == 4)
    if len(reds) != 1 or len(yellows) != 1:
        return g
    start = tuple(map(int, reds[0]))
    end = tuple(map(int, yellows[0]))
    for r, c in _path(start, end):
        if grid[r, c] == 1:
            g[r, c] = 7
        elif grid[r, c] == 3:
            g[r, c] = 5
    g[start] = 2
    g[end] = 4
    return g


def generate_input() -> np.ndarray:
    n = random.randint(11, 15)
    g = np.zeros((n, n), dtype=int)
    offset = random.randint(-2, 2)
    wiggles = [random.randint(-1, 1) for _ in range(n)]
    for r in range(n):
        boundary = r + offset + wiggles[r]
        for c in range(n):
            g[r, c] = 1 if c <= boundary else 3
    sr = random.randint(0, max(1, n // 4))
    sc = random.randint(0, max(1, n // 4))
    er = random.randint(3 * n // 4, n - 1)
    ec = random.randint(3 * n // 4, n - 1)
    g[sr, sc] = 2
    g[er, ec] = 4
    return g
