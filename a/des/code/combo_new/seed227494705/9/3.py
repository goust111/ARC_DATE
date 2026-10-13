import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MARKERS = {1, 2, 3, 4}


def _neighbors4(r, c):
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the orange signal through black maze channels, turning by marker position.

    Input structure:
    - Grey walls surround one-pixel black maze channels.
    - One orange source pixel lies in a channel entrance.
    - Colored wall markers near intersections indicate the side to continue toward.

    Rule:
    - Start at the orange source and paint traversed black channel cells orange.
    - Continue straight through corridors; at a branch, choose the direction indicated by the nearest colored wall marker's relative position.
    - Stop at a dead end or grid edge, preserving marker pixels.

    Key insight:
    - Marker color only makes the control visible; marker position around the junction encodes the turn.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    starts = np.argwhere(g == ORANGE)
    if starts.size == 0:
        return out
    r, c = map(int, starts[0])
    prev = None
    for _ in range(g.size):
        out[r, c] = ORANGE
        options = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and (nr, nc) != prev and int(g[nr, nc]) in (BG, ORANGE):
                options.append((nr, nc))
        if not options:
            break
        if len(options) == 1:
            nxt = options[0]
        else:
            markers = []
            for rr in range(max(0, r - 2), min(g.shape[0], r + 3)):
                for cc in range(max(0, c - 2), min(g.shape[1], c + 3)):
                    if int(g[rr, cc]) in MARKERS:
                        markers.append((rr, cc))
            if markers:
                mr, mc = min(markers, key=lambda p: abs(p[0] - r) + abs(p[1] - c))
                desired = (int(np.sign(mr - r)), int(np.sign(mc - c)))
                if abs(desired[0]) >= abs(desired[1]):
                    step = (desired[0], 0)
                else:
                    step = (0, desired[1])
                candidate = (r + step[0], c + step[1])
                nxt = candidate if candidate in options else options[0]
            else:
                nxt = options[0]
        prev = (r, c)
        r, c = nxt
        if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1):
            out[r, c] = ORANGE
            break
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    g = np.full((h, w), GREY, dtype=int)
    r = rng.randint(2, h - 3)
    c = 1
    g[r, c] = ORANGE
    direction = (0, 1)
    for _ in range(rng.randint(4, 8)):
        length = rng.randint(3, 6)
        for _step in range(length):
            r += direction[0]
            c += direction[1]
            if not (1 <= r < h - 1 and 1 <= c < w - 1):
                return g
            g[r, c] = BG
        new_dir = rng.choice([(1, 0), (-1, 0), (0, 1)])
        if new_dir == (-direction[0], -direction[1]):
            new_dir = direction
        mr = min(max(r + new_dir[0] * 2, 0), h - 1)
        mc = min(max(c + new_dir[1] * 2, 0), w - 1)
        if g[mr, mc] == GREY:
            g[mr, mc] = rng.choice(list(MARKERS))
        direction = new_dir
    return g
