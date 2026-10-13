import random
from collections import deque
import numpy as np


def _components_by_color(grid, allowed):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] in allowed and not seen[r, c]:
                color = int(grid[r, c])
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < h and 0 <= yy < w and not seen[xx, yy] and int(grid[xx, yy]) == color:
                            seen[xx, yy] = True
                            q.append((xx, yy))
                comps.append((color, comp))
    return comps


def _step_from_arrow(grid):
    best = 1
    for r in range(grid.shape[0]):
        run = 0
        for c in range(grid.shape[1]):
            if grid[r, c] == 6:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return max(1, best - 1)


def main(grid: np.ndarray) -> np.ndarray:
    step = _step_from_arrow(grid)
    out = np.full(grid.shape, 5, dtype=int)
    allowed = [int(c) for c in np.unique(grid) if c not in (0, 6)]
    comps = _components_by_color(grid, allowed)
    comps.sort(key=lambda item: len(item[1]))
    h, w = grid.shape
    for color, comp in comps:
        for r, c in comp:
            cc = c + step
            if 0 <= cc < w:
                out[r, cc] = color
    return out


def _shape(size):
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        cells.add((r + dr, c + dc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def generate_input() -> np.ndarray:
    h, w = 14, 16
    grid = np.zeros((h, w), dtype=int)
    grid[h - 3, 0:3] = 6
    grid[h - 4, 2] = 6
    grid[h - 2, 2] = 6
    colors = random.sample([1, 2, 3, 4, 7, 9], random.randint(4, 6))
    sizes = random.sample(range(3, 11), len(colors))
    occupied = set(zip(*np.where(grid != 0)))
    for color, size in zip(colors, sizes):
        cells = _shape(size)
        for _ in range(200):
            r0 = random.randint(0, h - 5)
            c0 = random.randint(0, w - 6)
            placed = {(r0 + r, c0 + c) for r, c in cells}
            if all(0 <= r < h and 0 <= c < w and all((r + dr, c + dc) not in occupied for dr in (-1, 0, 1) for dc in (-1, 0, 1)) for r, c in placed):
                for r, c in placed:
                    grid[r, c] = color
                occupied |= placed
                break
    return grid
