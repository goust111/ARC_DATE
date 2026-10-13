import random
from collections import Counter

import numpy as np


BG = 0
FENCE = 1


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    fence_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == FENCE)]
    fence_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == FENCE)]
    row_bounds = [-1] + fence_rows + [grid.shape[0]]
    col_bounds = [-1] + fence_cols + [grid.shape[1]]
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r0, r1 = row_bounds[i] + 1, row_bounds[i + 1] - 1
            c0, c1 = col_bounds[j] + 1, col_bounds[j + 1] - 1
            if r0 > r1 or c0 > c1:
                continue
            inner = grid[r0 : r1 + 1, c0 : c1 + 1]
            cnt = Counter(int(x) for x in inner.flat if x not in (BG, FENCE))
            if not cnt:
                continue
            color, num = cnt.most_common(1)[0]
            out[r0 : r1 + 1, c0 : c1 + 1] = BG
            out[r1 - num + 1 : r1 + 1, c0 : c1 + 1] = color
    return out


def generate_input() -> np.ndarray:
    cells_r = random.randint(2, 3)
    cells_c = random.randint(2, 3)
    cell_h = random.randint(3, 5)
    cell_w = random.randint(3, 5)
    h = cells_r * cell_h + cells_r + 1
    w = cells_c * cell_w + cells_c + 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(0, h, cell_h + 1):
        grid[r, :] = FENCE
    for c in range(0, w, cell_w + 1):
        grid[:, c] = FENCE
    colors = [2, 3, 4, 6, 7, 8, 9]
    for i in range(cells_r):
        for j in range(cells_c):
            r0 = i * (cell_h + 1) + 1
            c0 = j * (cell_w + 1) + 1
            choice = random.sample(colors, min(random.randint(2, 3), cell_h - 1))
            counts = random.sample(range(1, cell_h + 1), len(choice))
            counts = sorted(set(counts), reverse=True)
            choice = choice[: len(counts)]
            pts = [(r, c) for r in range(r0, r0 + cell_h) for c in range(c0, c0 + cell_w)]
            random.shuffle(pts)
            idx = 0
            for color, n in zip(choice, counts):
                for _ in range(n):
                    grid[pts[idx]] = color
                    idx += 1
    return grid
