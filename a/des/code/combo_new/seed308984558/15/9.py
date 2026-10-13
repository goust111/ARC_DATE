import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Project upper sprites downward, recoloring each projected column from the filter strip."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    band = int(np.where(np.all(g == 5, axis=1))[0][0])
    filt = g[band + 1]
    for r in range(band):
        for c in range(g.shape[1]):
            if g[r, c] not in (0, 5) and filt[c] != 0:
                rr = band + 2 + r
                if rr < g.shape[0]:
                    out[rr, c] = int(filt[c])
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(14, 20)
    band = h // 2 - 1
    g = np.zeros((h, w), dtype=int)
    g[band] = 5
    g[band + 1] = np.random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9], size=w)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4)):
        r0 = random.randint(1, band - 4)
        c0 = random.randint(1, w - 4)
        g[r0 : r0 + 2, c0 : c0 + 2] = color
    return g
