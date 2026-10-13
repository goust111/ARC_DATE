import random
from collections import Counter

import numpy as np


def _cell_starts(grid):
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == 5)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    rows = []
    prev = 0
    for s in sep_rows + [grid.shape[0]]:
        if s - prev == 3:
            rows.append(prev)
        prev = s + 1
    cols = []
    prev = 0
    for s in sep_cols + [grid.shape[1]]:
        if s - prev == 3:
            cols.append(prev)
        prev = s + 1
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    rows, cols = _cell_starts(grid)
    masks = {}
    for r0 in rows:
        for c0 in cols:
            tile = grid[r0 : r0 + 3, c0 : c0 + 3]
            mask = tuple(map(tuple, np.argwhere((tile != 0) & (tile != 5))))
            masks[(r0, c0)] = mask
    counts = Counter(masks.values())
    ordered = sorted(counts, key=lambda m: counts[m])
    rank_color = {mask: color for mask, color in zip(ordered, [2, 3, 1, 4])}
    for (r0, c0), mask in masks.items():
        color = rank_color.get(mask, 4)
        tile = grid[r0 : r0 + 3, c0 : c0 + 3]
        for rr, cc in np.argwhere((tile != 0) & (tile != 5)):
            out[r0 + rr, c0 + cc] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    for x in (3, 7, 11):
        grid[x, :] = 5
        grid[:, x] = 5
    shapes = [
        [(0, 0), (0, 1), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2)],
    ]
    counts = random.choice([[1, 3, 5, 7], [2, 3, 4, 7], [3, 5, 8]])
    chosen = random.sample(shapes, len(counts))
    entries = []
    for shape, count in zip(chosen, counts):
        entries.extend([shape] * count)
    random.shuffle(entries)
    starts = [(r, c) for r in (0, 4, 8, 12) for c in (0, 4, 8, 12)]
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for (r0, c0), shape in zip(starts, entries):
        color = random.choice(colors)
        for r, c in shape:
            grid[r0 + r, c0 + c] = color
    return grid

