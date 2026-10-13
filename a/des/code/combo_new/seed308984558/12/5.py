from __future__ import annotations
import random
import numpy as np

BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror only the rows that contain an odd number of non-black pixels across the center column.

    Input structure:
    - Black background with an odd grid width.
    - Each row contains any number of colored pixels, including zero.

    Rule:
    - Count non-black pixels per row.
    - Leave zero-count and positive even-count rows unchanged.
    - For odd-count rows, mirror colored pixels across the center column into empty opposite cells, preserving any original occupied target cells.
    - Keep the center-column pixel unchanged.

    Key insight:
    - The transform is row-local and depends only on parity, not on the colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mid = g.shape[1] // 2
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] != BG)
        if len(cols) == 0 or len(cols) % 2 == 0:
            continue
        for c in cols:
            mc = 2*mid - c
            if 0 <= mc < g.shape[1] and out[r, mc] == BG:
                out[r, mc] = out[r, c]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(10,18)
    w = random.choice([11,13,15,17])
    g = np.zeros((h,w), dtype=int)
    for r in range(h):
        k = random.randint(0,6)
        cols = random.sample(range(w), k)
        for c in cols:
            g[r,c] = random.choice([1,2,3,4,6,7,8,9])
    return g
