import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEEDS = [1, 2, 3, 4, 6, 7, 8]
YELLOW = 4


def _lines(g):
    rows = [r for r in range(g.shape[0]) if np.count_nonzero(g[r] == GREY) > g.shape[1] // 2]
    cols = [c for c in range(g.shape[1]) if np.count_nonzero(g[:, c] == GREY) > g.shape[0] // 2]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate seed colors across partitioned skeleton regions through boundary-contact evidence, then mark mixed-color crossings yellow.

    Input structure:
    - Grey partition lines divide the board into rectangular regions.
    - Each region contains a maroon one-cell skeleton; some regions also contain one colored seed on that skeleton.
    - Neighboring skeletons may approach the shared border from opposite sides.

    Rule:
    - Seeded regions color their whole skeleton with the seed color.
    - Unseeded regions inherit the color of the neighboring region with the strongest opposite-side border contact count.
    - Any cell whose colored neighbors disagree after propagation is recolored yellow.

    Key insight:
    - Color moves by regional adjacency evidence, not by global flood fill across the whole board.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rows, cols = _lines(g)
    region_color = {}
    contacts = {}
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            region = g[r0 + 1 : r1, c0 + 1 : c1]
            seeds = [int(x) for x in region[(region != BLACK) & (region != MAROON)]]
            region_color[(i, j)] = seeds[0] if seeds else None
            contacts[(i, j)] = {
                "top": np.count_nonzero(region[0, :] == MAROON),
                "bottom": np.count_nonzero(region[-1, :] == MAROON),
                "left": np.count_nonzero(region[:, 0] == MAROON),
                "right": np.count_nonzero(region[:, -1] == MAROON),
            }
    changed = True
    while changed:
        changed = False
        for i in range(len(rows) - 1):
            for j in range(len(cols) - 1):
                if region_color[(i, j)] is not None:
                    continue
                opts = []
                if i > 0 and region_color[(i - 1, j)] is not None:
                    opts.append((contacts[(i, j)]["top"] + contacts[(i - 1, j)]["bottom"], region_color[(i - 1, j)]))
                if i + 1 < len(rows) - 1 and region_color[(i + 1, j)] is not None:
                    opts.append((contacts[(i, j)]["bottom"] + contacts[(i + 1, j)]["top"], region_color[(i + 1, j)]))
                if j > 0 and region_color[(i, j - 1)] is not None:
                    opts.append((contacts[(i, j)]["left"] + contacts[(i, j - 1)]["right"], region_color[(i, j - 1)]))
                if j + 1 < len(cols) - 1 and region_color[(i, j + 1)] is not None:
                    opts.append((contacts[(i, j)]["right"] + contacts[(i, j + 1)]["left"], region_color[(i, j + 1)]))
                if opts:
                    best = max(opts)
                    if best[0] > 0:
                        region_color[(i, j)] = best[1]
                        changed = True
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            color = region_color[(i, j)] or 1
            mask = g[rows[i] + 1 : rows[i + 1], cols[j] + 1 : cols[j + 1]] == MAROON
            out[rows[i] + 1 : rows[i + 1], cols[j] + 1 : cols[j + 1]][mask] = color
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if out[r, c] in (BLACK, GREY):
                continue
            neigh = {int(out[r + dr, c + dc]) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= r + dr < out.shape[0] and 0 <= c + dc < out.shape[1] and out[r + dr, c + dc] not in (BLACK, GREY)}
            if len(neigh) > 1:
                out[r, c] = YELLOW
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    g[0, :] = g[5, :] = g[10, :] = g[-1, :] = GREY
    g[:, 0] = g[:, 5] = g[:, 10] = g[:, -1] = GREY
    cols = random.sample(SEEDS, 4)
    regs = [((0, 0), cols[0]), ((0, 1), None), ((0, 2), cols[1]), ((1, 0), None), ((1, 1), None), ((1, 2), None), ((2, 0), cols[2]), ((2, 1), None), ((2, 2), cols[3])]
    for (i, j), color in regs:
        r0, r1 = [0, 5, 10, 15][i], [5, 10, 15][i]
        c0, c1 = [0, 5, 10, 15][j], [5, 10, 15][j]
        rr = random.randint(r0 + 1, r1 - 1)
        g[rr, c0 + 1 : c1] = MAROON
        if color:
            g[rr, c0 + 1] = color
    return g
