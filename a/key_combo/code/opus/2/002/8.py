import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    sep = max(range(grid.shape[1]), key=lambda c: np.count_nonzero(grid[:, c] == 5))
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    both = (left != 0) & (right != 0)
    blue = np.count_nonzero((left == 1) & both)
    red = np.count_nonzero((left == 2) & both)
    color = 1 if blue > red else 2
    return np.where(both, color, 0).astype(int)


def generate_input() -> np.ndarray:
    h = random.randint(7, 11)
    w = random.randint(5, 8)
    grid = np.zeros((h, 2 * w + 1), dtype=int)
    grid[:, w] = 5
    coords = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(coords)
    both_n = random.choice([3, 5, 7])
    both = coords[:both_n]
    majority = random.choice([1, 2])
    maj_n = both_n // 2 + 1
    for i, (r, c) in enumerate(both):
        grid[r, c] = majority if i < maj_n else 3 - majority
        grid[r, w + 1 + c] = 3
    for r, c in coords[both_n : both_n + random.randint(5, 10)]:
        grid[r, c] = random.choice([1, 2])
    for r, c in coords[both_n + 10 : both_n + 10 + random.randint(4, 9)]:
        grid[r, w + 1 + c] = 3
    return grid
