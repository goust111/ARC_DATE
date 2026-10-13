import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed grey-bounded region inward by as many boundary layers as the seed's distance to the nearest wall.

    Input structure:
    - Grey orthogonal boundaries enclose one seed-colored region each on black background.
    - Every region contains exactly one colored seed.
    - Boundaries themselves stay fixed.

    Rule:
    - For each region, compute every interior cell's distance from the grey boundary.
    - Let D be the seed cell's nearest-boundary distance.
    - Recolor all interior cells whose boundary distance is at most D with the seed color.

    Key insight:
    - The seed contributes both the color and the fill depth.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if g[r, c] == GREY or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            touch = False
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touch = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != GREY and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            seeds = [(rr, cc) for rr, cc in comp if g[rr, cc] not in (BLACK, GREY)]
            if touch or not seeds:
                continue
            seed = seeds[0]
            color = int(g[seed])
            dist = {}
            dq = deque()
            for rr, cc in comp:
                if any(0 <= rr + dr < h and 0 <= cc + dc < w and g[rr + dr, cc + dc] == GREY for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    dist[(rr, cc)] = 1
                    dq.append((rr, cc))
            while dq:
                rr, cc = dq.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (nr, nc) in comp and (nr, nc) not in dist:
                        dist[(nr, nc)] = dist[(rr, cc)] + 1
                        dq.append((nr, nc))
            D = dist[seed]
            for cell, d in dist.items():
                if d <= D:
                    out[cell] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    for color, box, seed in zip(random.sample(COLORS, 2), [(2, 2, 9, 9), (11, 10, 18, 17)], [(5, 5), (14, 13)]):
        r0, c0, r1, c1 = box
        g[r0, c0 : c1 + 1] = GREY
        g[r1, c0 : c1 + 1] = GREY
        g[r0 : r1 + 1, c0] = GREY
        g[r0 : r1 + 1, c1] = GREY
        g[seed] = color
    return g
