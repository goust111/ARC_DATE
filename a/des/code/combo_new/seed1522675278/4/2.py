import random
from itertools import combinations

import numpy as np


BLACK = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the 3x3 corner mixing table and recolor every orthogonal path crossing by the table result.

    Input structure:
    - The top-left 3x3 block stores three unordered color-pair mappings.
    - Outside that block, horizontal and vertical monochrome paths cross on black background.
    - A crossing belongs to one row color and one column color.

    Rule:
    - Each table row encodes A, B -> C and also B, A -> C.
    - For every path crossing outside the table, infer the row color and column color from the rest of that row and column.
    - Recolor only the crossing cell with the looked-up mixed color.

    Key insight:
    - The corner block is a lookup table, not another object on the board.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mapping = {}
    for r in range(3):
        a, b, c = map(int, g[r, :3])
        mapping[frozenset((a, b))] = c
    for r in range(3, g.shape[0]):
        row_colors = [int(x) for x in g[r, 3:] if x != BLACK]
        if not row_colors:
            continue
        row_color = max(set(row_colors), key=row_colors.count)
        for c in range(3, g.shape[1]):
            col_colors = [int(x) for x in g[3:, c] if x != BLACK]
            if not col_colors:
                continue
            col_color = max(set(col_colors), key=col_colors.count)
            if g[r, c] != BLACK and row_color != col_color and frozenset((row_color, col_color)) in mapping:
                out[r, c] = mapping[frozenset((row_color, col_color))]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    cols = random.sample([1, 2, 3, 4, 6, 7], 3)
    res = random.sample([8, 9, 5], 3)
    for r, ((a, b), c) in enumerate(zip(combinations(cols, 2), res)):
        g[r, :3] = [a, b, c]
    for r, c in zip([6, 10], cols[:2]):
        g[r, 3:15] = c
    for c, color in zip([8, 12], cols[1:]):
        g[3:15, c] = color
    return g
