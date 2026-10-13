import random
from collections import deque

import numpy as np


BG = 0
WALL = 5


def _regions(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == WALL or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] != WALL and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cur)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    for reg in _regions(grid):
        seeds = [(r, c) for r, c in reg if grid[r, c] not in (BG, WALL)]
        n = len(seeds)
        regset = set(reg)
        for r, c in seeds:
            color = int(grid[r, c])
            for k in range(1, n + 1):
                for rr, cc in ((r - k, c), (r + k, c)):
                    if (rr, cc) in regset:
                        out[rr, cc] = color
            for k in range(1, n + 1):
                for rr, cc in ((r, c - k), (r, c + k)):
                    if (rr, cc) in regset:
                        out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL
    for r in range(4, h - 4, 5):
        grid[r, 1:-1] = WALL
    for c in range(4, w - 4, 5):
        grid[1:-1, c] = WALL
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for r0 in range(1, h - 1, 5):
        for c0 in range(1, w - 1, 5):
            if r0 + 3 >= h - 1 or c0 + 3 >= w - 1:
                continue
            for _ in range(random.randint(1, 3)):
                rr = random.randint(r0, min(r0 + 3, h - 2))
                cc = random.randint(c0, min(c0 + 3, w - 2))
                if grid[rr, cc] == BG:
                    grid[rr, cc] = random.choice(colors)
    return grid
