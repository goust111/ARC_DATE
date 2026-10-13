import random
from collections import deque

import numpy as np


def _intervals(lines, limit):
    pts = [-1] + sorted(lines) + [limit]
    return [(a + 1, b) for a, b in zip(pts, pts[1:]) if b - a > 1]


def main(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    row_counts = [np.count_nonzero(grid[r] == 5) for r in range(grid.shape[0])]
    col_counts = [np.count_nonzero(grid[:, c] == 5) for c in range(grid.shape[1])]
    gray_rows = [r for r, cnt in enumerate(row_counts) if cnt >= max(row_counts) * 0.7 and cnt > 0]
    gray_cols = [c for c, cnt in enumerate(col_counts) if cnt >= max(col_counts) * 0.7 and cnt > 0]
    row_cells = _intervals(gray_rows, grid.shape[0])
    col_cells = _intervals(gray_cols, grid.shape[1])

    for r_idx, (r0, r1) in enumerate(row_cells):
        for c_idx, (c0, c1) in enumerate(col_cells):
            cell = grid[r0:r1, c0:c1]
            colors = sorted(int(x) for x in np.unique(cell) if x not in (0, 5))
            if len(colors) != 2:
                continue
            a, b = colors
            block = cell.copy()
            if (r_idx + c_idx) % 2 == 1:
                block[cell == a] = b
                block[cell == b] = a
            out[r0:r1, c0:c1] = np.where(block == 5, 0, block)
    return out


def _connected_shape(size, n):
    cells = {(random.randrange(size), random.randrange(size))}
    while len(cells) < n:
        r, c = random.choice(tuple(cells))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < size and 0 <= nc < size:
                cells.add((nr, nc))
                break
    return list(cells)


def generate_input() -> np.ndarray:
    side = random.choice([3, 4])
    rows = random.choice([3, 4])
    cols = random.choice([3, 4])
    top = left = side + 1
    h = top + rows * side + rows + 1
    w = left + cols * side + cols + 1
    grid = np.zeros((h, w), dtype=int)
    grid[:side, :side] = random.choice([3, 4, 6, 7, 8, 9])

    for i in range(rows + 1):
        grid[top + i * (side + 1), left : left + cols * side + cols + 1] = 5
    for j in range(cols + 1):
        grid[top : top + rows * side + rows + 1, left + j * (side + 1)] = 5

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    for r in range(rows):
        for c in range(cols):
            r0 = top + 1 + r * (side + 1)
            c0 = left + 1 + c * (side + 1)
            coords = _connected_shape(side, random.randint(3, min(8, side * side)))
            split = random.randint(1, len(coords) - 1)
            random.shuffle(coords)
            for rr, cc in coords[:split]:
                grid[r0 + rr, c0 + cc] = colors[0]
            for rr, cc in coords[split:]:
                grid[r0 + rr, c0 + cc] = colors[1]
    return grid
