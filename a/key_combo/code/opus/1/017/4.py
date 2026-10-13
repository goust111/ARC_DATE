import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    band_h = 0
    band_color = int(grid[0, 0])
    while band_h < h and band_color != 0 and np.all(grid[band_h] == band_color):
        band_h += 1
    out = np.array(grid, copy=True)
    for i in range(band_h):
        out[h - 1 - i, :] = grid[i, :]
    bottom_top = h - band_h
    markers = []
    for r in range(band_h, bottom_top):
        for c in range(w):
            val = int(grid[r, c])
            if val not in (0, band_color):
                d_top = r - (band_h - 1)
                d_bot = bottom_top - r
                markers.append((min(d_top, d_bot), r, c, val, d_top, d_bot))
    painted = np.zeros_like(grid, dtype=bool)
    for d, r, c, val, d_top, d_bot in sorted(markers):
        far_top = d_top > d_bot
        if d_top == d_bot:
            far_top = False
        rows = range(band_h - d, band_h) if far_top else range(bottom_top, bottom_top + d)
        for rr in rows:
            for cc in range(w):
                if not painted[rr, cc]:
                    out[rr, cc] = val
                    painted[rr, cc] = True
        out[r, c] = 0
    return out


def generate_input() -> np.ndarray:
    band_h = random.randint(2, 4)
    h = random.randint(14, 18)
    w = random.randint(17, 22)
    grid = np.zeros((h, w), dtype=int)
    band_color = random.choice([1, 4, 5, 8, 9])
    grid[:band_h, :] = band_color
    marker_colors = random.sample([2, 3, 6, 7], random.randint(1, min(3, band_h)))
    used_d = set()
    for col, color in zip(random.sample(range(2, w - 2), len(marker_colors)), marker_colors):
        d = random.choice([x for x in range(1, band_h + 1) if x not in used_d])
        used_d.add(d)
        if random.random() < 0.5:
            row = band_h - 1 + d
        else:
            row = h - band_h - d
        grid[row, col] = color
    return grid
