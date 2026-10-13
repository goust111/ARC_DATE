import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    bounds = []
    for r in range(h):
        cols = np.where(grid[r] == 1)[0]
        bounds.append((int(cols[0]), int(cols[-1]), len(cols)))
    target = min(b[1] for b in bounds)
    out = np.zeros_like(grid)
    new_bounds = []
    for r, (_, _, length) in enumerate(bounds):
        start = max(0, target - length + 1)
        end = start + length - 1
        out[r, :start] = 5
        out[r, start:end + 1] = 1
        out[r, end + 1:] = 2
        new_bounds.append((start, end))
    for r in range(h // 2):
        mate = h - 1 - r
        if new_bounds[r] != new_bounds[mate]:
            s, e = new_bounds[r]
            out[r, s:e + 1] = 4
            s, e = new_bounds[mate]
            out[mate, s:e + 1] = 4
    return out


def generate_input() -> np.ndarray:
    h = random.choice([8, 10, 12])
    w = random.randint(10, 15)
    widths = [0] * h
    pairs = list(range(h // 2))
    sym_pair = random.choice(pairs)
    asym_pair = random.choice([p for p in pairs if p != sym_pair])
    for i in pairs:
        a = random.randint(1, min(6, w - 3))
        b = a if i == sym_pair else random.randint(1, min(6, w - 3))
        if i == asym_pair:
            while b == a:
                b = random.randint(1, min(6, w - 3))
        widths[i] = a
        widths[h - 1 - i] = b
    grid = np.zeros((h, w), dtype=int)
    for r, length in enumerate(widths):
        end = random.randint(length - 1, w - 1)
        start = end - length + 1
        grid[r, :start] = 5
        grid[r, start:end + 1] = 1
        grid[r, end + 1:] = 2
    return grid
