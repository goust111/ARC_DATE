import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    tile = g[1:, 1:]
    th, tw = tile.shape
    cols = [c - 1 for c in np.where(g[0] == 9)[0]]
    rows = [r - 1 for r in np.where(g[:, 0] == 9)[0]]
    cols.sort()
    rows.sort()
    out = np.zeros((3 * th, 3 * tw), dtype=g.dtype)
    for i, rshift in enumerate(rows[:3]):
        for j, cshift in enumerate(cols[:3]):
            block = tile.copy()
            if (rshift + cshift) % 2:
                block = np.fliplr(block)
            block = np.roll(block, shift=(rshift % th, cshift % tw), axis=(0, 1))
            out[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = block
    return out


def generate_input() -> np.ndarray:
    n = random.randint(4, 8)
    g = np.zeros((n + 1, n + 1), dtype=int)
    g[0, :] = 8
    g[:, 0] = 8
    for c in random.sample(range(1, n + 1), 3):
        g[0, c] = 9
    for r in random.sample(range(1, n + 1), 3):
        g[r, 0] = 9
    colors = [0, 1, 2, 3, 4, 5, 6, 7]
    while True:
        tile = np.array([[random.choice(colors) for _ in range(n)] for _ in range(n)], dtype=int)
        if not np.array_equal(tile, np.fliplr(tile)) and len(np.unique(tile)) >= 4:
            break
    g[1:, 1:] = tile
    return g
