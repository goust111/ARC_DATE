import random

import numpy as np


BG = 8
GREY = 5


def _line(out, a, b, color):
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            out[r1, c] = color
    elif c1 == c2:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            out[r, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color pixel pair with one grey-guided orthogonal bend.

    Input structure:
    - Azure background with foreground colors that each occur exactly twice.
    - Grey singleton navigation pixels sit near the intended bends.
    - Pairs are separated so their L-shaped connectors do not need to cross.

    Rule:
    - Remove all grey navigation pixels.
    - For each same-color pair, draw a horizontal/vertical line if aligned.
    - Otherwise choose the L-corner closer to a grey navigation pixel, preferring the longer-axis first route, and draw the two orthogonal legs in the pair color.

    Key insight:
    - Grey pixels are route controls: they choose the bend for the color-paired connector.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == GREY] = BG
    greys = [tuple(map(int, p)) for p in np.argwhere(g == GREY)]
    for color in [int(x) for x in np.unique(g) if int(x) not in (BG, GREY)]:
        pts = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        if len(pts) != 2:
            continue
        a, b = pts
        if a[0] == b[0] or a[1] == b[1]:
            _line(out, a, b, color)
            continue
        candidates = [(a[0], b[1]), (b[0], a[1])]
        if abs(a[0] - b[0]) > abs(a[1] - b[1]):
            default = (b[0], a[1])
        else:
            default = (a[0], b[1])
        if greys:
            bend = min(candidates, key=lambda p: min(abs(p[0] - gr) + abs(p[1] - gc) for gr, gc in greys))
        else:
            bend = default
        _line(out, a, bend, color)
        _line(out, bend, b, color)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(12, 24)
    w = rng.randint(12, 24)
    g = np.full((h, w), BG, dtype=int)
    colors = rng.sample([1, 2, 3, 4, 6, 7, 9], rng.randint(2, 5))
    used = set()
    for color in colors:
        for _try in range(100):
            a = (rng.randint(1, h - 2), rng.randint(1, w - 2))
            b = (rng.randint(1, h - 2), rng.randint(1, w - 2))
            if a == b or a in used or b in used:
                continue
            bend = rng.choice([(a[0], b[1]), (b[0], a[1])])
            if bend in used:
                continue
            used.update([a, b, bend])
            g[a] = color
            g[b] = color
            g[bend] = GREY
            break
    return g
