import random
from collections import deque

import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    center_r = next(r for r in range(h) if np.all(grid[r, :] == 7))
    center_c = next(c for c in range(w) if np.all(grid[:, c] == 7))
    out = grid.copy()
    out[center_r, :] = 0
    out[:, center_c] = 0

    source = max(_components(grid, 1), key=len)
    r0, c0, r1, c1 = _bbox(source)
    template = (grid[r0 : r1 + 1, c0 : c1 + 1] == 1).astype(int)
    markers = sorted(map(tuple, np.argwhere(grid == 3)))
    for mr, mc in markers:
        if mr < center_r and mc > center_c:
            rot = np.rot90(template, -1)
        elif mr > center_r and mc > center_c:
            rot = np.rot90(template, 2)
        else:
            rot = np.rot90(template, 1)
        top = mr - rot.shape[0] // 2
        left = mc - rot.shape[1] // 2
        for r in range(rot.shape[0]):
            for c in range(rot.shape[1]):
                if rot[r, c]:
                    out[top + r, left + c] = 7
        if out[mr, mc] == 3:
            out[mr, mc] = 0
    return out


def generate_input() -> np.ndarray:
    n = random.choice([13, 15])
    grid = np.zeros((n, n), dtype=int)
    mid = n // 2
    grid[mid, :] = 7
    grid[:, mid] = 7
    templates = [
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    ]
    t = random.choice(templates)
    grid[1 : 1 + t.shape[0], 1 : 1 + t.shape[1]][t == 1] = 1
    candidates = [(2, n - 3), (n - 3, n - 3), (n - 3, 2)]
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(1, 3)]:
        grid[r, c] = 3
    return grid
