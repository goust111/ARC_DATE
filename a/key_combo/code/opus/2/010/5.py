from collections import defaultdict, deque
import random

import numpy as np


MARK_COLORS = {1, 3, 4, 7}
DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}


def _cross_centers(g):
    h, w = g.shape
    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r, c] == 5 and all(g[r + dr, c + dc] == 5 for dr, dc in DIRS.values()):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    centers = _cross_centers(g)
    claims = defaultdict(list)
    marker_cells = set()
    for sid, (cr, cc) in enumerate(centers):
        for dr, dc in DIRS.values():
            mr, mc = cr + 2 * dr, cc + 2 * dc
            if 0 <= mr < h and 0 <= mc < w and int(g[mr, mc]) in MARK_COLORS:
                col = int(g[mr, mc])
                marker_cells.add((mr, mc))
                rr, cc2 = mr, mc
                while 0 <= rr < h and 0 <= cc2 < w:
                    if g[rr, cc2] == 5:
                        break
                    claims[(rr, cc2)].append((sid, col, (mr, mc)))
                    rr += 2 * dr
                    cc2 += 2 * dc
    out = g.copy()
    for (r, c), vals in claims.items():
        if (r, c) in marker_cells or g[r, c] == 5:
            continue
        sids = {v[0] for v in vals}
        if len(sids) >= 2:
            out[r, c] = 2
        elif len(vals) >= 2:
            out[r, c] = 5
        else:
            out[r, c] = vals[0][1]
    return out


def _draw_cross(g, r, c):
    g[r, c] = 5
    for dr, dc in DIRS.values():
        g[r + dr, c + dc] = 5


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    centers = random.choice([[(5, 5), (13, 13), (5, 14)], [(6, 6), (14, 6), (10, 14)]])
    for r, c in centers:
        _draw_cross(g, r, c)
    colors = [1, 3, 4, 7]
    for r, c in centers:
        arms = random.sample(list(DIRS.values()), random.randint(2, 4))
        for dr, dc in arms:
            mr, mc = r + 2 * dr, c + 2 * dc
            if 0 <= mr < 20 and 0 <= mc < 20 and g[mr, mc] == 0:
                g[mr, mc] = random.choice(colors)
    return g
