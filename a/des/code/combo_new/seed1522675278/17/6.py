import random

import numpy as np


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    singletons = []
    segments = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) == 1:
            singletons.append((tuple(coords[0]), color))
        else:
            segments.append((len(coords), color))
    singletons.sort()
    start = singletons[0][0]
    end = singletons[-1][0]
    row = start[0]
    x = start[1]
    for length, color in sorted(segments):
        out[row, x : x + length] = color
        x += length
    return out


def generate_input() -> np.ndarray:
    h = random.randint(10, 16)
    w = random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    row = random.randint(2, h - 3)
    part_count = random.randint(3, 5)
    lengths = random.sample(range(2, 7), part_count)
    total = sum(lengths)
    start_col = random.randint(1, 3)
    end_col = start_col + total - 1
    if end_col >= w - 1:
        end_col = w - 2
        start_col = end_col - total + 1
    grid[row, start_col] = 8
    grid[row, end_col] = 5
    colors = random.sample([1, 2, 3, 4, 6, 7, 9], part_count)
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[row, start_col] = True
    occupied[row, end_col] = True
    for length, color in zip(lengths, colors):
        for _ in range(200):
            rr = random.randint(0, h - 1)
            cc = random.randint(0, w - length)
            if occupied[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + length + 1)].any():
                continue
            grid[rr, cc : cc + length] = color
            occupied[rr, cc : cc + length] = True
            break
    return grid
