import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count 1x1 holes in each quadrant object, shrink the object by 1/(holes+1) using top-left subsampling, and place a grey shadow near the center lines.

    Input structure:
    - A grey vertical and horizontal center line split an even square grid into four quadrants.
    - Each quadrant contains one colored object that does not touch the center lines.
    - Some objects contain isolated one-cell black holes.

    Rule:
    - Let the shrink factor be holes+1.
    - Subsample the object's bbox by taking the top-left cell of each factor-by-factor block.
    - Draw the resulting grey shadow in the same quadrant with its inner corner adjacent to the center lines.

    Key insight:
    - Hole count is a scale instruction, and the shadow always projects toward the center cross.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    n = g.shape[0]
    mr = mc = n // 2
    quads = [((0, mr - 1, 0, mc - 1), (mr - 1, mc - 1)), ((0, mr - 1, mc + 1, n - 1), (mr - 1, mc + 1)), ((mr + 1, n - 1, 0, mc - 1), (mr + 1, mc - 1)), ((mr + 1, n - 1, mc + 1, n - 1), (mr + 1, mc + 1))]
    for (r0, r1, c0, c1), anchor in quads:
        pts = np.argwhere((g[r0 : r1 + 1, c0 : c1 + 1] != BLACK) & (g[r0 : r1 + 1, c0 : c1 + 1] != GREY))
        if not len(pts):
            continue
        pts = [(r + r0, c + c0) for r, c in pts]
        br0, br1 = min(r for r, _ in pts), max(r for r, _ in pts)
        bc0, bc1 = min(c for _, c in pts), max(c for _, c in pts)
        holes = 0
        for r in range(br0 + 1, br1):
            for c in range(bc0 + 1, bc1):
                if g[r, c] == BLACK and all(g[r + dr, c + dc] not in (BLACK, GREY) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    holes += 1
        factor = holes + 1
        sh = np.zeros(((br1 - br0) // factor + 1, (bc1 - bc0) // factor + 1), dtype=bool)
        for rr in range(sh.shape[0]):
            for cc in range(sh.shape[1]):
                if g[br0 + rr * factor, bc0 + cc * factor] not in (BLACK, GREY):
                    sh[rr, cc] = True
        ar, ac = anchor
        sr = ar - sh.shape[0] + 1 if ar < mr else ar
        sc = ac - sh.shape[1] + 1 if ac < mc else ac
        for rr in range(sh.shape[0]):
            for cc in range(sh.shape[1]):
                if sh[rr, cc]:
                    out[sr + rr, sc + cc] = GREY
    return out


def generate_input() -> np.ndarray:
    n = 18
    g = np.zeros((n, n), dtype=int)
    g[n // 2, :] = GREY
    g[:, n // 2] = GREY
    quads = [(1, 1), (1, 11), (11, 1), (11, 11)]
    for color, (r0, c0) in zip(random.sample(COLORS, 4), quads):
        g[r0 : r0 + 3, c0 : c0 + 3] = color
        if random.random() < 0.7:
            g[r0 + 1, c0 + 1] = BLACK
    return g
