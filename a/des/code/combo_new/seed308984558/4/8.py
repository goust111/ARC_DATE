import random

import numpy as np


BG = 0


def _neighbors(path, cell):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c + dc) in path]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the colored path from the left edge and choose each T-junction branch by comparing the current path color with the gate color.

    Input structure:
    - A multicolor 4-connected path enters from the left edge and contains T-junctions.
    - One branch at each T-junction has an adjacent gate segment with its own color.
    - Background cells not on the chosen route should disappear.

    Rule:
    - Start at the left-edge path pixel and walk along the path.
    - At a degree-3 junction, identify which candidate branch is the gated branch by the adjacent gate pixel(s).
    - If the current path color matches the gate color, take the gated branch; otherwise take the ungated branch.
    - Keep only the visited path cells and all gate pixels.

    Key insight:
    - The branch choice is path-dependent because it uses the color of the last path cell reached at each junction.
    """
    g = np.array(grid, dtype=int)
    path_colors = {1, 2, 3}
    path = {(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if int(g[r, c]) in path_colors}
    start = next((r, 0) for r in range(g.shape[0]) if (r, 0) in path)
    gates = {(r, c): int(g[r, c]) for r in range(g.shape[0]) for c in range(g.shape[1]) if int(g[r, c]) not in path_colors and int(g[r, c]) != BG}
    visited = {start}
    prev = None
    cur = start
    while True:
        neigh = [n for n in _neighbors(path, cur) if n != prev]
        if not neigh:
            break
        if len(neigh) == 1:
            prev, cur = cur, neigh[0]
            visited.add(cur)
            continue
        gated = None
        gate_color = None
        for n in neigh:
            nr, nc = n
            around = [(nr + dr, nc + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (nr + dr, nc + dc) in gates]
            if around:
                gated = n
                gate_color = gates[around[0]]
        nxt = gated if gate_color == int(g[cur]) else next(n for n in neigh if n != gated)
        prev, cur = cur, nxt
        visited.add(cur)
    out = np.zeros_like(g)
    for r, c in visited:
        out[r, c] = g[r, c]
    for (r, c), color in gates.items():
        out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = 13, 17
    g = np.zeros((h, w), dtype=int)
    path = [(6, c) for c in range(0, 6)] + [(5, 6), (6, 6), (7, 6)] + [(5, c) for c in range(7, 12)] + [(4, 12), (5, 12), (6, 12)] + [(6, c) for c in range(13, 17)]
    for r, c in path:
        g[r, c] = random.choice([1, 2, 3])
    # first gate guards upper branch
    gate1 = random.choice([1, 2, 3])
    g[4, 6] = gate1
    # second gate guards lower branch
    gate2 = random.choice([1, 2, 3])
    g[6, 11] = gate2
    return g
