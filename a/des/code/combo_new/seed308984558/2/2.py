import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color pair with a vertical line that borrows the stripe color under each segment.

    Input structure:
    - The whole grid is filled with alternating full-width horizontal stripes in two background colors.
    - Every non-stripe color appears exactly twice as tiny fragments in the same column.
    - The paired fragments sit on different stripes.

    Rule:
    - Keep each colored fragment unchanged.
    - For every color pair, draw a vertical segment between them in their shared column.
    - Color each drawn cell with the stripe color already present at that row.

    Key insight:
    - Pairing uses the fragment color, but the connector color is determined locally by the background stripe the connector passes through.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    vals, cnts = np.unique(g, return_counts=True)
    stripe_colors = {int(v) for v, c in zip(vals, cnts, strict=False) if c >= g.shape[1]}
    for color in [v for v in vals if v not in stripe_colors]:
        pts = list(zip(*np.where(g == int(color)), strict=False))
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(pts)
        for r in range(r0 + 1, r1):
            out[r, c0] = g[r, c0]
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(15, 27), random.randint(12, 20)
    band = random.randint(2, 4)
    stripe_a, stripe_b = random.sample([5, 8], 2)
    g = np.zeros((h, w), dtype=int)
    for r in range(h):
        g[r, :] = stripe_a if (r // band) % 2 == 0 else stripe_b
    cols = random.sample(range(1, w - 1), random.randint(2, 4))
    colors = random.sample([1, 2, 3, 4, 6, 7, 9], len(cols))
    for c, color in zip(cols, colors):
        r0, r1 = sorted(random.sample(range(h), 2))
        g[r0, c] = color
        g[r1, c] = color
    return g
