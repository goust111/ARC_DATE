import random
from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    sep = w // 2
    left = grid[:, :sep]
    right = np.fliplr(grid[:, sep + 1 :])
    clean = np.where((left == 1) & (right == 1), 1, 0)
    out = clean.copy()
    seen = np.zeros(clean.shape, dtype=bool)
    hh, ww = clean.shape
    for r in range(hh):
        for c in range(ww):
            if clean[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                touches |= cr in (0, hh - 1) or cc in (0, ww - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hh and 0 <= nc < ww and clean[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if touches:
                for cr, cc in cells:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < hh and 0 <= nc < ww and clean[nr, nc] == 1:
                                out[cr, cc] = 3
            else:
                for cr, cc in cells:
                    out[cr, cc] = 4
    return out


def generate_input() -> np.ndarray:
    h, half = 12, 12
    grid = np.zeros((h, half * 2 + 1), dtype=int)
    grid[:, half] = 5
    top = random.randint(2, 3)
    left = random.randint(2, 4)
    bottom = top + random.randint(5, 6)
    right = left + random.randint(5, 6)
    shape = set()
    for c in range(left, right + 1):
        shape.add((top, c))
        shape.add((bottom, c))
    for r in range(top, bottom + 1):
        shape.add((r, left))
        shape.add((r, right))
    for r, c in shape:
        grid[r, c] = 1
        grid[r, half + 1 + (half - 1 - c)] = 1
    noise_slots = [(r, c) for r in range(h) for c in range(half) if grid[r, c] == 0]
    for r, c in random.sample(noise_slots, random.randint(5, 10)):
        grid[r, c] = 2
    return grid

