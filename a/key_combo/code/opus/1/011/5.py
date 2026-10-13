import random
from collections import deque

import numpy as np


def _red_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 2 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == 2:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _ray(out, r, c, dr, dc, color):
    r += dr
    c += dc
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == 0:
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    micros = []
    for comp in _red_components(grid):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(comp) == 1:
            out[comp[0]] = 0
        elif len(comp) == 2:
            micros.append((r0, r1, c0, c1))
        else:
            markers = [(r, c, int(grid[r, c])) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                       if grid[r, c] in (1, 3)]
            if not markers:
                continue
            color = markers[0][2]
            for p in comp:
                out[p] = color
            out[markers[0][0], markers[0][1]] = color
            if color == 1:
                cr = (r0 + r1) // 2
                _ray(out, cr, c0, 0, -1, 1)
                _ray(out, cr, c1, 0, 1, 1)
            else:
                cc = (c0 + c1) // 2
                _ray(out, r0, cc, -1, 0, 3)
                _ray(out, r1, cc, 1, 0, 3)
    for r0, r1, c0, c1 in micros:
        a, b = max(0, r0 - 1), min(out.shape[0] - 1, r1 + 1)
        x, y = max(0, c0 - 1), min(out.shape[1] - 1, c1 + 1)
        for c in range(x, y + 1):
            if out[a, c] == 0:
                out[a, c] = 3
            if out[b, c] == 0:
                out[b, c] = 3
        for r in range(a, b + 1):
            if out[r, x] == 0:
                out[r, x] = 3
            if out[r, y] == 0:
                out[r, y] = 3
    return out


def _place_large(grid, r, c, marker):
    border = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    for dr, dc in border:
        grid[r + dr, c + dc] = 2
    grid[r + 1, c + 1] = marker


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    _place_large(grid, random.randint(2, 3), random.randint(2, 3), 1)
    _place_large(grid, random.randint(9, 10), random.randint(9, 10), 3)
    if random.random() < 0.5:
        _place_large(grid, 9, 2, random.choice([1, 3]))
    grid[2, 11:13] = 2
    grid[12:14, 4] = 2
    if random.random() < 0.6:
        grid[6, 12:14] = 2
    for _ in range(random.randint(1, 3)):
        for _attempt in range(50):
            r, c = random.randrange(16), random.randrange(16)
            if grid[r, c] == 0 and all(not (0 <= r + dr < 16 and 0 <= c + dc < 16 and grid[r + dr, c + dc] == 2)
                                      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                grid[r, c] = 2
                break
    return grid
