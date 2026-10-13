import random

import numpy as np


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    row_marks = [(r, int(grid[r, 0])) for r in range(1, grid.shape[0]) if grid[r, 0] not in (BG, GREY)]
    col_marks = [(c, int(grid[0, c])) for c in range(1, grid.shape[1]) if grid[0, c] not in (BG, GREY)]
    for r, row_color in row_marks:
        for c, col_color in col_marks:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr, cc = r + dr, c + dc
                    if not (1 <= rr < grid.shape[0] and 1 <= cc < grid.shape[1]):
                        continue
                    if row_color == col_color:
                        out[rr, cc] = row_color
                    else:
                        if dr == 0 or dc == 0:
                            out[rr, cc] = row_color
                        else:
                            out[rr, cc] = col_color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = GREY
    grid[:, 0] = GREY
    palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    row_positions = list(range(2, h - 1, 4))
    col_positions = list(range(2, w - 1, 4))
    random.shuffle(row_positions)
    random.shuffle(col_positions)
    for r in row_positions[: random.randint(2, min(4, len(row_positions)))]:
        grid[r, 0] = random.choice(palette)
    for c in col_positions[: random.randint(2, min(4, len(col_positions)))]:
        grid[0, c] = random.choice(palette)
    return grid
