import random
from collections import deque

import numpy as np


BG = 0
LOCK = 5
FILL = 7


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
    keys = []
    locks = []
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            for r, c in comp:
                mask[r - r0, c - c0] = 1
            if color == LOCK:
                locks.append((r0, c0, mask))
            else:
                keys.append((color, r0, c0, mask))
    for color, kr, kc, km in keys:
        for i, (lr, lc, lm) in enumerate(locks):
            if km.shape == lm.shape and np.all(km + lm == 1):
                out[kr : kr + km.shape[0], kc : kc + km.shape[1]] = BG
                for rr in range(km.shape[0]):
                    for cc in range(km.shape[1]):
                        out[lr + rr, lc + cc] = color if km[rr, cc] else FILL
                locks.pop(i)
                break
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 8, 9], random.randint(2, 3)):
        mask = np.array([[1, 1], [1, 0]])
        lr = random.randint(1, h - 4)
        lc = random.randint(1, w // 2 - 3)
        kr = random.randint(1, h - 4)
        kc = random.randint(w // 2 + 1, w - 4)
        for rr in range(2):
            for cc in range(2):
                if mask[rr, cc]:
                    grid[kr + rr, kc + cc] = color
                else:
                    grid[lr + rr, lc + cc] = LOCK
    return grid
