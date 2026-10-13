import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Push colored pixels toward row markers first, then column markers, preserving order."""
    g = np.array(grid, dtype=int)
    inner = g[1:-1, 1:-1].copy()
    for r in range(inner.shape[0]):
        vals = [int(v) for v in inner[r] if int(v) != 0]
        inner[r] = 0
        if not vals:
            continue
        if g[r + 1, 0] == 2:
            inner[r, : len(vals)] = vals
        else:
            inner[r, -len(vals) :] = vals
    tmp = inner.copy()
    for c in range(inner.shape[1]):
        vals = [int(v) for v in tmp[:, c] if int(v) != 0]
        inner[:, c] = 0
        if not vals:
            continue
        if g[0, c + 1] == 2:
            inner[: len(vals), c] = vals
        else:
            inner[-len(vals) :, c] = vals
    return inner


def generate_input() -> np.ndarray:
    n = random.randint(8, 12)
    g = np.zeros((n + 2, n + 2), dtype=int)
    for r in range(n):
        g[r + 1, 0 if random.choice([True, False]) else -1] = 2
    for c in range(n):
        g[0 if random.choice([True, False]) else -1, c + 1] = 2
    for _ in range(random.randint(n, 2 * n)):
        g[random.randint(1, n), random.randint(1, n)] = random.choice([1, 3, 4, 5, 6, 7, 8, 9])
    return g
