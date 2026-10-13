import random
import numpy as np

BLACK = 0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    mapping = {}
    c = 0
    while c < w:
        if grid[h - 1, c] != BLACK:
            mapping[grid[h - 1, c]] = grid[h - 1, c + 1]
            c += 2
        c += 1
    out = grid.copy()
    out[h - 1, :] = BLACK
    for r in range(h - 1):
        for c in range(w):
            color = grid[r, c]
            if color != BLACK:
                rr = (h - 2) - r
                if out[rr, c] == BLACK:
                    out[rr, c] = mapping[color]
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(14, 20), random.randint(14, 24)
    grid = np.zeros((h, w), dtype=int)
    pairs = random.randint(3, 4)
    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], pairs * 2)
    mapping = list(zip(palette[::2], palette[1::2]))
    c = 0
    for a, b in mapping:
        grid[h - 1, c] = a
        grid[h - 1, c + 1] = b
        c += 3
    for _ in range(random.randint(3, 6)):
        color, _ = random.choice(mapping)
        oh, ow = random.randint(2, 4), random.randint(2, 4)
        r0 = random.randint(0, h // 2 - oh - 1)
        c0 = random.randint(0, w - ow)
        grid[r0 : r0 + oh, c0 : c0 + ow] = color
    return grid
