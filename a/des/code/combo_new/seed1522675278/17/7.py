import random

import numpy as np


BG = 0
GREY = 5
YELLOW = 4


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    bounds = [-1] + sep_rows + [grid.shape[0]]
    regions = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 1]
    layers = regions[:-1]
    out_r0, out_r1 = regions[-1]
    masks = []
    for r0, r1 in layers:
        masks.append((grid[r0:r1] != BG) & (grid[r0:r1] != GREY))
    inter = masks[0].copy()
    for mask in masks[1:]:
        inter &= mask
    out[out_r0:out_r1] = BG
    out[out_r0:out_r1][inter] = YELLOW
    return out


def generate_input() -> np.ndarray:
    layer_count = random.randint(3, 4)
    rh = random.randint(3, 5)
    rw = random.randint(8, 12)
    h = layer_count * rh + layer_count + rh
    w = rw
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 6, 7, 8, 9], layer_count)
    common = np.zeros((rh, rw), dtype=bool)
    for _ in range(random.randint(4, rh * rw // 2)):
        common[random.randrange(rh), random.randrange(rw)] = True
    row = 0
    for idx in range(layer_count):
        mask = common.copy()
        for _ in range(random.randint(2, 6)):
            mask[random.randrange(rh), random.randrange(rw)] = True
        grid[row : row + rh][mask] = colors[idx]
        row += rh
        grid[row, :] = GREY
        row += 1
    return grid
