import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested grey layer from the left-edge color strip, outermost inward.

    Input structure:
    - A central figure consists of tightly nested grey rectangular borders or a grey
      innermost fill.
    - A short top-left horizontal strip lists one color per nested layer.

    Rule:
    - Read the strip left to right as outer-to-inner layer colors.
    - Remove the strip from the output.
    - Repaint the nested structure layer by layer from the outside in using those
      colors, preserving the original geometry.

    Key insight:
    - The strip is a one-dimensional encoding of the two-dimensional nesting order, so
      counting layers correctly is the bridge between the strip and the central figure.
    """
    colors = [int(v) for v in grid[0] if v != BLACK]
    out = grid.copy()
    out[0, :] = BLACK
    cells = np.argwhere(grid == GREY)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    for idx, color in enumerate(colors):
        rr0, cc0 = r0 + idx, c0 + idx
        rr1, cc1 = r1 - idx, c1 - idx
        if rr0 > rr1 or cc0 > cc1:
            break
        out[rr0, cc0 : cc1 + 1] = color
        out[rr1, cc0 : cc1 + 1] = color
        out[rr0 : rr1 + 1, cc0] = color
        out[rr0 : rr1 + 1, cc1] = color
    return out


def generate_input() -> np.ndarray:
    layers = random.randint(2, 5)
    colors = random.sample(COLORS, layers)
    side = random.randint(2 * layers + 1, min(2 * layers + 7, 15))
    h = side + 4
    w = max(side + 4, layers + 4)
    grid = np.zeros((h, w), dtype=int)
    grid[0, 0:layers] = colors
    r0 = 2
    c0 = (w - side) // 2
    for i in range(layers):
        rr0, cc0 = r0 + i, c0 + i
        rr1, cc1 = r0 + side - 1 - i, c0 + side - 1 - i
        grid[rr0, cc0 : cc1 + 1] = GREY
        grid[rr1, cc0 : cc1 + 1] = GREY
        grid[rr0 : rr1 + 1, cc0] = GREY
        grid[rr0 : rr1 + 1, cc1] = GREY
    return grid
