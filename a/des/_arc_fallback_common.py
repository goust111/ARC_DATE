import random
from collections import deque

import numpy as np


COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PATTERNS = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        cells = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    h, w = g.shape
    for cells in _components(g != 0):
        color = int(g[cells[0]])
        dr = (color % 3) - 1
        dc = ((color // 3) % 3) - 1
        for r, c in cells:
            nr = min(h - 1, max(0, r + dr))
            nc = min(w - 1, max(0, c + dc))
            out[nr, nc] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(100):
        h = random.randint(8, 30)
        w = random.randint(8, 30)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        count = random.randint(2, 7)
        for _i in range(count):
            pat = random.choice(PATTERNS)
            ph, pw = pat.shape
            if ph + 2 >= h or pw + 2 >= w:
                continue
            color = random.choice(COLORS)
            for _attempt in range(80):
                r0 = random.randint(1, h - ph - 1)
                c0 = random.randint(1, w - pw - 1)
                area = occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1]
                if area.any():
                    continue
                grid[r0 : r0 + ph, c0 : c0 + pw][pat] = color
                occupied[r0 : r0 + ph, c0 : c0 + pw] |= pat
                break
        if np.count_nonzero(grid):
            return grid
    grid = np.zeros((10, 10), dtype=int)
    grid[4:6, 4:6] = 2
    return grid
