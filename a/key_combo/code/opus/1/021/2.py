import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            x, y = q.popleft()
            cells.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
        comps.append(cells)
    return comps


def _crop(grid, cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return grid[min(rs) : max(rs) + 1, min(cs) : max(cs) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    seed = None
    for comp in _components(grid == 3):
        crop = _crop(grid, comp)
        mask = crop == 3
        if not np.array_equal(mask, np.fliplr(mask)):
            seed = mask
            break
    out = np.zeros((seed.shape[0] * seed.shape[0], seed.shape[1] * seed.shape[1]), dtype=grid.dtype)
    h, w = seed.shape
    for r in range(h):
        for c in range(w):
            color = 3 if seed[r, c] else 1
            block = np.where(seed, color, 0)
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = block
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    sym = [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]]),
        np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]]),
    ]
    asym = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]]),
        np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),
    ]
    positions = [(1, 1), (1, 7), (6, 3), (6, 10)]
    random.shuffle(positions)
    bad = random.randrange(4)
    for i, pos in enumerate(positions):
        pat = random.choice(asym if i == bad else sym) * 3
        r, c = pos
        grid[r : r + 3, c : c + 3] = pat
    lr, lc = 14, 2
    grid[lr, lc : lc + 4] = 1
    grid[lr : lr + 4, lc] = 1
    return grid
