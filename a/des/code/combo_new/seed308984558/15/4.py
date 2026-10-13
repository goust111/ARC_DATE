import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Read cyclic row/column orders from the broken edge bands and permute the inner grid."""
    g = np.array(grid, dtype=int)
    top_break = int(np.where(g[0, 1:] == 0)[0][0] + 1)
    left_break = int(np.where(g[1:, 0] == 0)[0][0] + 1)
    cols = [c for c in list(range(top_break + 1, g.shape[1])) + list(range(1, top_break)) if c != top_break]
    rows = [r for r in list(range(left_break + 1, g.shape[0])) + list(range(1, left_break)) if r != left_break]
    return g[rows][:, cols]


def generate_input() -> np.ndarray:
    h, w = random.randint(8, 12), random.randint(8, 12)
    g = np.zeros((h + 1, w + 1), dtype=int)
    inner = np.random.randint(1, 10, size=(h, w))
    g[1:, 1:] = inner
    g[0, 1:] = np.random.randint(1, 10, size=w)
    g[1:, 0] = np.random.randint(1, 10, size=h)
    g[0, 0] = 0
    g[0, random.randint(1, w)] = 0
    g[random.randint(1, h), 0] = 0
    return g
