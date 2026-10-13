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
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for color in (1, 2, 3):
        for comp in _components(grid == color):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            n4 = int(np.sum(interior == 4))
            n7 = int(np.sum(interior == 7))
            if n4 == n7:
                continue
            fill = 4 if n4 > n7 else 7
            area = out[r0 + 1 : r1, c0 + 1 : c1]
            area[area == 0] = fill
    return out


def _draw_rect(grid, r0, c0, ih, iw, color):
    r1, c1 = r0 + ih + 1, c0 + iw + 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    return r0, r1, c0, c1


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    specs = [(1, 1, 4, 4, 2), (1, 11, 5, 3, 1), (11, 3, 3, 6, 3)]
    random.shuffle(specs)
    for r0, c0, ih, iw, color in specs:
        r0 += random.choice([0, 1])
        c0 += random.choice([0, 1])
        rr0, rr1, cc0, cc1 = _draw_rect(grid, r0, c0, ih, iw, color)
        cells = [(r, c) for r in range(rr0 + 1, rr1) for c in range(cc0 + 1, cc1)]
        random.shuffle(cells)
        total = random.choice([3, 5, 7])
        yellow = random.randrange(1, total)
        if yellow * 2 == total:
            yellow += 1
        marks = [4] * yellow + [7] * (total - yellow)
        random.shuffle(marks)
        for (r, c), mark in zip(cells, marks):
            grid[r, c] = mark
    return grid
