import random
from collections import deque

import numpy as np


BG = 0


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
    out = np.zeros_like(grid)
    for comp in _components(grid != BG):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(grid[comp[0]])
        box = np.full((r1 - r0 + 1, c1 - c0 + 1), color, dtype=int)
        for r, c in comp:
            box[r - r0, c - c0] = color
        missing = None
        for cell, direction in [((0, 0), (1, 1)), ((0, -1), (1, -1)), ((-1, 0), (-1, 1)), ((-1, -1), (-1, -1))]:
            rr = 0 if cell[0] == 0 else box.shape[0] - 1
            cc = 0 if cell[1] == 0 else box.shape[1] - 1
            if (r0 + rr, c0 + cc) not in comp:
                missing = direction
        dist = len(comp)
        dr, dc = missing
        for r, c in comp:
            nr, nc = r + dr * dist, c + dc * dist
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = grid[r, c]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    used = np.zeros_like(grid, dtype=bool)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
    for color in colors:
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        arr = np.full((h, w), color, dtype=int)
        miss = random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
        arr[miss] = BG
        area = int(np.count_nonzero(arr))
        dir_map = {
            (0, 0): (1, 1),
            (0, w - 1): (1, -1),
            (h - 1, 0): (-1, 1),
            (h - 1, w - 1): (-1, -1),
        }
        dr, dc = dir_map[miss]
        for _ in range(300):
            r = random.randint(1 + max(0, -dr * area), 23 - h - max(0, dr * area))
            c = random.randint(1 + max(0, -dc * area), 23 - w - max(0, dc * area))
            if np.any(used[max(0, r - 1) : min(24, r + h + 1), max(0, c - 1) : min(24, c + w + 1)]):
                continue
            grid[r : r + h, c : c + w] = np.where(arr != BG, arr, grid[r : r + h, c : c + w])
            used[max(0, r - 1) : min(24, r + h + 1), max(0, c - 1) : min(24, c + w + 1)] = True
            break
    return grid
