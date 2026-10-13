import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    red_counts = [(r, int(np.sum(grid[r] == 2))) for r in range(grid.shape[0])]
    mark_row = max(red_counts, key=lambda x: x[1])[0]
    bar_cols = [c for c in range(grid.shape[1]) if np.any(grid[:mark_row, c] == 8)]
    heights = []
    for c in bar_cols:
        rows = np.where(grid[:mark_row, c] == 8)[0]
        heights.append(mark_row - int(rows.min()))
    hmax = max(heights)
    for c in bar_cols:
        out[:mark_row, c][out[:mark_row, c] == 8] = 0
        out[mark_row - hmax:mark_row, c] = 8
    sprite_mask = (grid[mark_row + 1:] != 0) & (grid[mark_row + 1:] != 2) & (grid[mark_row + 1:] != 8)
    pts = np.argwhere(sprite_mask)
    if len(pts) == 0:
        return out
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 += mark_row + 1
    r1 += mark_row + 1
    tile = grid[r0:r1 + 1, c0:c1 + 1]
    out[r0:r1 + 1, c0:c1 + 1] = 0
    top = mark_row + 2
    for i in range(len(bar_cols)):
        left = i * (tile.shape[1] + 1)
        for r in range(tile.shape[0]):
            for c in range(tile.shape[1]):
                if tile[r, c] != 0:
                    rr, cc = top + r, left + c
                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                        out[rr, cc] = tile[r, c]
    return out


def generate_input() -> np.ndarray:
    h, w = 15, 18
    grid = np.zeros((h, w), dtype=int)
    mark_row = 7
    n = random.randint(2, 5)
    for i in range(n + 1):
        grid[mark_row, i * 2] = 2
    for i in range(n):
        height = random.randint(1, 5)
        col = i * 2 + 1
        grid[mark_row - height:mark_row, col] = 8
    tile = random.choice([
        np.array([[4, 0, 6], [4, 6, 6]]),
        np.array([[1, 3], [0, 3], [1, 1]]),
        np.array([[9, 9, 0], [0, 5, 5], [9, 0, 5]]),
    ])
    top = random.randint(mark_row + 2, h - tile.shape[0])
    left = random.randint(0, w - tile.shape[1])
    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            if tile[r, c]:
                grid[top + r, left + c] = tile[r, c]
    return grid
