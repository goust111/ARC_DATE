import random

import numpy as np


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the colored dots in each grey-bounded channel and draw equal-length endpoint bars in that channel color.

    Input structure:
    - Parallel full-width grey lines split the grid into horizontal channels.
    - Each channel contains same-colored singleton dots and blank endpoint columns.
    - Different channels use different dot colors.

    Rule:
    - For each channel, count its colored dots.
    - Fill the left endpoint column and the right endpoint column from the channel top downward using that count, capped by the channel height.
    - Keep the original dots and the grey separators unchanged.

    Key insight:
    - The dots are grouped by the channel geometry rather than by connectivity, and that per-channel count becomes a line length on both edges.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    lines = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    bounds = [-1] + lines + [g.shape[0]]
    for i in range(len(bounds) - 1):
        r0, r1 = bounds[i] + 1, bounds[i + 1] - 1
        if r0 > r1:
            continue
        band = g[r0 : r1 + 1]
        colors = [v for v in np.unique(band) if v != BG]
        if not colors:
            continue
        color = int(colors[0])
        count = int(np.count_nonzero(band == color))
        length = min(count, r1 - r0 + 1)
        out[r0 : r0 + length, 0] = color
        out[r0 : r0 + length, -1] = color
    return out


def generate_input() -> np.ndarray:
    channels = random.randint(2, 4)
    ch = random.randint(3, 4)
    h = channels * ch + channels - 1
    w = random.randint(12, 20)
    if h > 30:
        return generate_input()
    g = np.zeros((h, w), dtype=int)
    row = 0
    for idx, color in enumerate(random.sample([1, 2, 3, 4, 6, 7], channels)):
        if idx:
            g[row] = GREY
            row += 1
        for _ in range(random.randint(1, ch)):
            rr = random.randint(row, row + ch - 1)
            cc = random.randint(2, w - 3)
            g[rr, cc] = color
        row += ch
    return g
