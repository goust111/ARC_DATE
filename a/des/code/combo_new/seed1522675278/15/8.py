import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    seeds = [(r, c, int(grid[r, c])) for r, c in np.argwhere((grid != BG) & (grid != GREY) & (grid != MAROON))]
    cover = {}
    for r, c, color in seeds:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc] == GREY or any((nr, nc) == (sr, sc) for sr, sc, _ in seeds):
                    break
                if grid[nr, nc] == BG:
                    cover.setdefault((nr, nc), set()).add(color)
                nr += dr
                nc += dc
    for (r, c), colors in cover.items():
        out[r, c] = MAROON if len(colors) > 1 else next(iter(colors))
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    for _ in range(random.randint(4, 8)):
        if random.choice([True, False]):
            r = random.randint(1, 20)
            c0 = random.randint(1, 18)
            grid[r, c0 : c0 + random.randint(1, 3)] = GREY
        else:
            c = random.randint(1, 20)
            r0 = random.randint(1, 18)
            grid[r0 : r0 + random.randint(1, 3), c] = GREY
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4))
    for color in colors:
        for _ in range(300):
            r = random.randint(1, 20)
            c = random.randint(1, 20)
            if grid[r, c] == BG:
                grid[r, c] = color
                break
    return grid
