import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    if np.count_nonzero(grid[-1] != BG) >= np.count_nonzero(grid[:, -1] != BG):
        legend = [int(v) for v in grid[-1] if v != BG]
        scene = grid[:-1]
        offset = (0, 0)
        out = grid.copy()
    else:
        legend = [int(v) for v in grid[:, -1] if v != BG]
        scene = grid[:, :-1]
        out = grid.copy()
    comps = _components((scene == BG) | (scene == RED))
    h, w = scene.shape
    for comp in comps:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        k = sum(scene[r, c] == RED for r, c in comp)
        fill = legend[k - 1]
        for r, c in comp:
            out[r, c] = fill
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 24), dtype=int)
    legends = random.sample([3, 4, 5, 6, 7, 8, 9], 3)
    grid[-1, :3] = legends
    rooms = [(2, 2, 6, 6, 1), (2, 11, 6, 6, 2), (10, 6, 7, 8, 3)]
    for r0, c0, h, w, count in rooms:
        r1, c1 = r0 + h - 1, c0 + w - 1
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        spots = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        for r, c in random.sample(spots, count):
            grid[r, c] = RED
    return grid
