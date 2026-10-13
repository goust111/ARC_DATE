import random
from collections import deque

import numpy as np


def _mask_components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    mask = grid == 2
    union = mask | np.fliplr(mask)
    comps = _mask_components(union)
    comps.sort(key=lambda cells: (-len(cells), min(p[0] for p in cells), min(p[1] for p in cells)))
    rank_colors = [1, 3, 4, 7]
    out = np.zeros_like(grid)
    for i, cells in enumerate(comps):
        color = rank_colors[i]
        for p in cells:
            out[p] = color
    return out


def _red_components(grid):
    return _mask_components(grid == 2)


def _random_connected(area, h, w):
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        y, x = random.choice(tuple(cells))
        dy, dx = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            cells.add((ny, nx))
    return cells


def generate_input() -> np.ndarray:
    h = random.randint(8, 12)
    w = random.choice([8, 10, 12])
    for _ in range(2000):
        grid = np.zeros((h, w), dtype=int)
        comp_count = random.randint(1, 3)
        for _j in range(comp_count):
            area = random.randint(2, 6)
            cells = _random_connected(area, h, w)
            for y, x in cells:
                grid[y, x] = 2
        if len(_red_components(grid)) not in (1, 2, 3):
            continue
        union_comps = _mask_components((grid == 2) | np.fliplr(grid == 2))
        areas = [len(c) for c in union_comps]
        if 2 <= len(areas) <= 4 and len(set(areas)) == len(areas) and not np.array_equal(grid == 2, np.fliplr(grid == 2)):
            return grid
    return generate_input()
