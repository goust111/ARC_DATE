import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    out = grid.copy()
    controls = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != 0) & (grid != 2) & (grid != 5) & (grid != 8))]
    pivots = np.argwhere(grid == 5)
    if len(controls) < 4 or len(pivots) == 0:
        return out
    pivot = tuple(map(int, pivots[0]))
    top = min(controls, key=lambda x: x[0])
    bottom = max(controls, key=lambda x: x[0])
    left = min(controls, key=lambda x: x[1])
    right = max(controls, key=lambda x: x[1])
    r0, r1, c0, c1 = top[0], bottom[0], left[1], right[1]
    out[r0, c0:c1 + 1] = top[2]
    out[r1, c0:c1 + 1] = bottom[2]
    out[r0 + 1:r1, c0] = left[2]
    out[r0 + 1:r1, c1] = right[2]
    pr, pc = pivot
    out[pr, c0 + 1:c1] = 5
    out[r0 + 1:r1, pc] = 5
    out[pr, 1:c0] = 8
    out[pr, c1 + 1:w - 1] = 8
    out[pr, w - 1] = 8
    out[pr, 0] = 8
    for r, c, color in controls:
        out[r, c] = color
    out[pr, pc] = 5
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 17)
    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = 2
    grid[:, w - 1] = 2
    r0 = random.randint(2, 4)
    r1 = random.randint(h - 5, h - 3)
    c0 = random.randint(3, 5)
    c1 = random.randint(w - 6, w - 4)
    colors = random.sample([1, 3, 4, 6], 4)
    pts = [(r0, random.randint(c0 + 1, c1 - 1), colors[0]),
           (r1, random.randint(c0 + 1, c1 - 1), colors[1]),
           (random.randint(r0 + 1, r1 - 1), c0, colors[2]),
           (random.randint(r0 + 1, r1 - 1), c1, colors[3])]
    for r, c, color in pts:
        grid[r, c] = color
    pr = random.randint(r0 + 1, r1 - 1)
    pc = random.randint(c0 + 1, c1 - 1)
    if pr == (r0 + r1) // 2 and pc == (c0 + c1) // 2:
        pr = min(r1 - 1, pr + 1)
    grid[pr, pc] = 5
    return grid
