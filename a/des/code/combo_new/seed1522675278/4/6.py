import random

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, rank colors by total paired area, and render concentric square rings in that order.

    Input structure:
    - Every non-black color appears in exactly two disconnected fragments.
    - Same-colored fragment pairs are intended to assemble into a rectangle.
    - Only the total area of each pair matters for the output.

    Rule:
    - Sum the two fragment areas for each color.
    - Sort colors by that total area descending.
    - Paint a square of side 2n-1 using those colors from outer ring to center.

    Key insight:
    - Rectangle reconstruction is unnecessary once the pair area is known.
    """
    g = np.asarray(grid, dtype=int)
    colors = [c for c in np.unique(g) if c != BLACK]
    order = sorted(colors, key=lambda c: -int(np.count_nonzero(g == c)))
    n = len(order)
    out = np.zeros((2 * n - 1, 2 * n - 1), dtype=int)
    for i, color in enumerate(order):
        out[i : 2 * n - 1 - i, i] = color
        out[i : 2 * n - 1 - i, 2 * n - 2 - i] = color
        out[i, i : 2 * n - i - 1] = color
        out[2 * n - 2 - i, i : 2 * n - i - 1] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    sizes = [8, 6, 4]
    for color, size, pos in zip(random.sample(COLORS, 3), sizes, [(2, 2), (10, 2), (2, 10)]):
        r0, c0 = pos
        a = size // 2
        g[r0 : r0 + 1, c0 : c0 + a] = color
        g[r0 + 2 : r0 + 3, c0 : c0 + (size - a)] = color
    return g
