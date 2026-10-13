import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    rights = []
    runs = []
    for r in range(grid.shape[0]):
        cols = np.where(grid[r] == 6)[0]
        if len(cols):
            runs.append((r, int(cols.min()), int(cols.max())))
            rights.append(int(cols.max()))
    target = min(rights)
    for r, left, right in runs:
        shift = right - target
        if shift:
            width = right - left + 1
            out[r, left:right + 1] = 5
            out[r, left - shift:left - shift + width] = 6
    aligned = out.copy()
    w = out.shape[1]
    for r, c in np.argwhere(aligned == 5):
        mirror = w - 1 - int(c)
        if aligned[r, mirror] != 5:
            out[r, c] = aligned[r, mirror]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(7, 10)
    w = random.choice([11, 13])
    center = w // 2
    target = random.randint(center, center + 1)
    grid = np.zeros((h, w), dtype=int)
    for r in range(h):
        run_w = random.randint(1, 4)
        right = target + random.randint(0, min(3, w - target - 2))
        left = max(0, right - run_w + 1)
        grid[r, :left] = 5
        grid[r, left:right + 1] = 6
        grid[r, right + 1:] = 9
    if len(set(np.where(row == 6)[0].max() for row in grid)) < 2:
        return generate_input()
    return grid
