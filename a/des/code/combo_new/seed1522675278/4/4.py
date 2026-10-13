import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover full rectangle extents from the visible colored layers, reverse the layer order given by the side legend, and redraw only the rectangle field.

    Input structure:
    - Several solid colored rectangles overlap on a black field.
    - A full grey separator column splits off a color-order legend at the right.
    - The legend lists current top-to-bottom layer order from topmost to bottommost.

    Rule:
    - Infer each rectangle's full bbox from the visible pixels of its color.
    - Reverse the legend order.
    - Redraw the rectangles from new bottom to new top on the left field and drop the legend area.

    Key insight:
    - The visible fragments are enough because each color still exposes the rectangle's true min/max rows and columns.
    """
    g = np.asarray(grid, dtype=int)
    sep = int(np.where(np.all(g == GREY, axis=0))[0][0])
    order = [int(x) for x in g[:, sep + 1] if x != BLACK]
    rects = {}
    for color in order:
        pts = np.argwhere(g[:, :sep] == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rects[color] = (r0, c0, r1, c1)
    out = np.zeros_like(g[:, :sep])
    for color in reversed(order):
        r0, c0, r1, c1 = rects[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def generate_input() -> np.ndarray:
    h, w = 16, 20
    sep = 15
    g = np.zeros((h, w), dtype=int)
    g[:, sep] = GREY
    colors = random.sample(COLORS, 3)
    rects = {
        colors[0]: (random.randint(1, 3), random.randint(1, 3), random.randint(8, 10), random.randint(8, 10)),
        colors[1]: (random.randint(3, 5), random.randint(4, 6), random.randint(10, 12), random.randint(11, 13)),
        colors[2]: (random.randint(1, 2), random.randint(7, 9), random.randint(7, 9), 14),
    }
    order = colors[:]
    random.shuffle(order)
    for idx, color in enumerate(reversed(order)):
        r0, c0, r1, c1 = rects[color]
        g[r0 : r1 + 1, c0 : c1 + 1] = color
    for i, color in enumerate(order):
        g[2 + i, sep + 1] = color
    return g
