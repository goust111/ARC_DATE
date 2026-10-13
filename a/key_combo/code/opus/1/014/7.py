import random
from collections import deque
import numpy as np


def _component_sizes(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes = {}
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != 1:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == 1:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            for p in comp:
                sizes[p] = len(comp)
    return sizes


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h * 2, w * 2), 5, dtype=int)
    sizes = _component_sizes(grid)
    group_diags = {r - c for (r, c), size in sizes.items() if size >= 2}
    for r in range(h):
        for c in range(w):
            rr, cc = 2 * r, 2 * c
            if grid[r, c] == 1:
                out[rr:rr + 2, cc:cc + 2] = 2 if sizes[(r, c)] >= 2 else 3
            elif grid[r, c] == 5 and (r - c) in group_diags:
                out[rr, cc] = 1
                out[rr + 1, cc + 1] = 1
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(6, 9)
    g = np.full((h, w), 5, dtype=int)
    for _ in range(random.randint(1, 2)):
        r = random.randint(1, h - 3)
        c = random.randint(1, w - 3)
        g[r, c] = 1
        g[r, c + 1] = 1
        if random.random() < 0.5:
            g[r + 1, c] = 1
    placed = 0
    while placed < random.randint(2, 4):
        r = random.randrange(h)
        c = random.randrange(w)
        if g[r, c] == 5 and not np.any(g[max(0, r - 1):r + 2, max(0, c - 1):c + 2] == 1):
            g[r, c] = 1
            placed += 1
    return g
