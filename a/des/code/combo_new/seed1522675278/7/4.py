import random
from collections import deque

import numpy as np


BG = 0
PIPE = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cur)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    blocks = []
    for color in [c for c in np.unique(grid) if c not in (BG, PIPE)]:
        for comp in _components(grid == color):
            blocks.append((color, set(comp)))
    for comp in _components(grid == PIPE):
        end_cols = []
        for color, block in blocks:
            if any(any((r + dr, c + dc) in block for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))) for r, c in comp):
                end_cols.append(color)
        if len(end_cols) < 2:
            continue
        a, b = sorted(end_cols[:2])
        path = sorted(comp)
        for i, (r, c) in enumerate(path):
            out[r, c] = a if i % 2 == 0 else b
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    cols = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    blocks = []
    for i in range(0, 4, 2):
        c1, c2 = cols[i], cols[i + 1]
        r1, c1p = random.randint(2, h - 5), random.randint(1, w // 3)
        r2, c2p = random.randint(2, h - 5), random.randint(2 * w // 3, w - 4)
        grid[r1 : r1 + 2, c1p : c1p + 2] = c1
        grid[r2 : r2 + 2, c2p : c2p + 2] = c2
        rr = min(r1 + 1, r2 + 1)
        for c in range(c1p + 2, c2p):
            grid[rr, c] = PIPE
    return grid
