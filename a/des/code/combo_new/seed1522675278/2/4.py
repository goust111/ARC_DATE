import random
from collections import Counter

import numpy as np


BLACK = 0
WALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _lines(g):
    rows = [r for r in range(g.shape[0]) if np.count_nonzero(g[r] != BLACK) > g.shape[1] // 2]
    cols = [c for c in range(g.shape[1]) if np.count_nonzero(g[:, c] != BLACK) > g.shape[0] // 2]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge adjacent rooms across monochrome shared walls and fill each merged region by the majority color on its outer perimeter.

    Input structure:
    - Rectangular rooms are partitioned by one-cell colored wall segments on black background.
    - Shared walls between rooms may have different colors from the outer boundary.
    - Room interiors are empty black cells.

    Rule:
    - Rooms between the same pair of grid lines merge when their shared wall segment has a single uniform color.
    - Merge transitively across the room lattice.
    - Fill each merged region using the most frequent color on that region's outer walls.

    Key insight:
    - Shared-wall color determines connectivity, while the exterior wall-color histogram determines the fill.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rows, cols = _lines(g)
    rooms = [(i, j) for i in range(len(rows) - 1) for j in range(len(cols) - 1)]
    parent = {x: x for x in rooms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(rows) - 1):
        for j in range(len(cols) - 2):
            wall = {int(g[r, cols[j + 1]]) for r in range(rows[i] + 1, rows[i + 1])}
            if len(wall) == 1 and BLACK not in wall:
                union((i, j), (i, j + 1))
    for i in range(len(rows) - 2):
        for j in range(len(cols) - 1):
            wall = {int(g[rows[i + 1], c]) for c in range(cols[j] + 1, cols[j + 1])}
            if len(wall) == 1 and BLACK not in wall:
                union((i, j), (i + 1, j))
    groups = {}
    for room in rooms:
        groups.setdefault(find(room), []).append(room)
    for group in groups.values():
        colors = []
        members = set(group)
        for i, j in group:
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            if (i - 1, j) not in members:
                colors += [int(g[r0, c]) for c in range(c0 + 1, c1)]
            if (i + 1, j) not in members:
                colors += [int(g[r1, c]) for c in range(c0 + 1, c1)]
            if (i, j - 1) not in members:
                colors += [int(g[r, c0]) for r in range(r0 + 1, r1)]
            if (i, j + 1) not in members:
                colors += [int(g[r, c1]) for r in range(r0 + 1, r1)]
        fill = Counter(colors).most_common(1)[0][0]
        for i, j in group:
            out[rows[i] + 1 : rows[i + 1], cols[j] + 1 : cols[j + 1]] = fill
    out[g != BLACK] = g[g != BLACK]
    return out


def generate_input() -> np.ndarray:
    nr, nc = 2, 3
    room_h, room_w = 3, 4
    h = nr * room_h + nr + 1
    w = nc * room_w + nc + 1
    g = np.zeros((h, w), dtype=int)
    rows = [i * (room_h + 1) for i in range(nr + 1)]
    cols = [i * (room_w + 1) for i in range(nc + 1)]
    for i in range(nr + 1):
        for j in range(nc):
            g[rows[i], cols[j] : cols[j + 1] + 1] = random.choice(WALL_COLORS)
    for j in range(nc + 1):
        for i in range(nr):
            g[rows[i] : rows[i + 1] + 1, cols[j]] = random.choice(WALL_COLORS)
    return g
