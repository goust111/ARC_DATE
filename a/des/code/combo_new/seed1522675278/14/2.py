import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = np.zeros_like(grid)
    rows, cols = np.where(grid == GREY)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    out[grid == GREY] = GREY
    row_proj = {}
    col_proj = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            val = int(grid[r, c])
            if val in (BG, GREY, MAROON):
                continue
            if r < r0 and c0 < c < c1:
                out[r0 + 1, c] = val
                col_proj[c] = val
            elif r > r1 and c0 < c < c1:
                out[r1 - 1, c] = val
                col_proj[c] = val
            elif c < c0 and r0 < r < r1:
                out[r, c0 + 1] = val
                row_proj[r] = val
            elif c > c1 and r0 < r < r1:
                out[r, c1 - 1] = val
                row_proj[r] = val
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if grid[r, c] != MAROON:
                out[r, c] = grid[r, c]
                continue
            a = row_proj.get(r)
            b = col_proj.get(c)
            if a is None and b is None:
                out[r, c] = MAROON
            elif a is None:
                out[r, c] = b
            else:
                out[r, c] = a
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    r0, r1, c0, c1 = 6, 15, 6, 15
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    row_choices = random.sample(range(r0 + 1, r1), random.randint(2, 4))
    col_choices = random.sample(range(c0 + 1, c1), random.randint(2, 4))
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], len(row_choices) + len(col_choices))
    for r, color in zip(row_choices, colors[: len(row_choices)]):
        if random.choice([True, False]):
            grid[r, random.randint(0, c0 - 2)] = color
        else:
            grid[r, random.randint(c1 + 2, 21)] = color
    for c, color in zip(col_choices, colors[len(row_choices) :]):
        if random.choice([True, False]):
            grid[random.randint(0, r0 - 2), c] = color
        else:
            grid[random.randint(r1 + 2, 21), c] = color
    for _ in range(random.randint(4, 7)):
        grid[random.randint(r0 + 2, r1 - 2), random.randint(c0 + 2, c1 - 2)] = MAROON
    return grid
