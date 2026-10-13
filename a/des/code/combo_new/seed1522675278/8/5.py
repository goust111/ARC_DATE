import random
from collections import deque

import numpy as np


BG = 0


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
    palette = [int(grid[0, c]) for c in range(grid.shape[1]) if grid[0, c] != BG]
    out[0, :] = BG
    for comp in _components(grid[1:] != BG):
        comp = [(r + 1, c) for r, c in comp]
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        out[r0 : min(out.shape[0], r0 + 20), c0 : c1 + 1] = BG
        top = r0
        for color in palette:
            if color not in block:
                continue
            layer = np.where(block == color, color, BG)
            hh = layer.shape[0]
            if top + hh > out.shape[0]:
                break
            out[top : top + hh, c0 : c1 + 1] = layer
            top += hh + 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(20, 28)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 4))
    c = 1
    for color in palette:
        grid[0, c] = color
        c += 2
    for left in range(1, w - 5, 7):
        top = random.randint(3, 5)
        bh = random.randint(2, 3)
        bw = random.randint(2, 3)
        used = random.sample(palette, random.randint(2, len(palette)))
        for r in range(top, top + bh):
            for cc in range(left, left + bw):
                grid[r, cc] = random.choice(used)
    return grid
