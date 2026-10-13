import random

import numpy as np


def _find_block(grid):
    h, w = grid.shape
    colors = [x for x in np.unique(grid) if x != 0]
    for color in colors:
        for r0 in range(h - 8):
            for c0 in range(w - 8):
                sub = grid[r0 : r0 + 9, c0 : c0 + 9]
                if np.any((sub != 0) & (sub != color)):
                    continue
                m = np.zeros((3, 3), dtype=int)
                ok = True
                for i in range(3):
                    for j in range(3):
                        tile = sub[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3]
                        if np.all(tile == color):
                            m[i, j] = 1
                        elif not np.all(tile == 0):
                            ok = False
                corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
                if ok and 3 <= m.sum() <= 5 and sum(m[x] for x in corners) == 1:
                    return r0, c0, int(color), m
    return 0, 0, 1, np.zeros((3, 3), dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    r0, c0, color, m = _find_block(grid)
    dirs = {(0, 0): (-1, -1), (0, 2): (-1, 1), (2, 0): (1, -1), (2, 2): (1, 1)}
    corner = next(pos for pos, d in dirs.items() if m[pos] == 1)
    dr, dc = dirs[corner]
    k = np.kron(m, m)
    out[r0 : r0 + 9, c0 : c0 + 9] = 0
    h, w = out.shape
    nr0, nc0 = r0 + dr, c0 + dc
    for r in range(9):
        for c in range(9):
            if k[r, c]:
                rr, cc = nr0 + r, nc0 + c
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    corner = random.choice([(0, 0), (0, 2), (2, 0), (2, 2)])
    m = np.zeros((3, 3), dtype=int)
    m[corner] = 1
    choices = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    for pos in random.sample(choices, random.randint(2, 4)):
        m[pos] = 1
    r0 = random.randint(1 if corner[0] == 0 else 0, 5 if corner[0] == 2 else 6)
    c0 = random.randint(1 if corner[1] == 0 else 0, 5 if corner[1] == 2 else 6)
    for i in range(3):
        for j in range(3):
            if m[i, j]:
                grid[r0 + 3 * i : r0 + 3 * i + 3, c0 + 3 * j : c0 + 3 * j + 3] = color
    return grid

