import random
from collections import Counter

import numpy as np


BLACK = 0
WALL = 9
COLORS = [1, 2, 3, 4, 6, 7, 8]
SHAPES = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0)},
]


def _canon(shape):
    pts = list(shape)
    outs = []
    for flip in (False, True):
        cur = [(-r, c) if flip else (r, c) for r, c in pts]
        for _ in range(4):
            r0 = min(r for r, _ in cur)
            c0 = min(c for _, c in cur)
            norm = frozenset((r - r0, c - c0) for r, c in cur)
            outs.append(norm)
            cur = [(-c, r) for r, c in cur]
    return min(outs, key=lambda x: sorted(x))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group rooms by shape equivalence under rotation/flip, open single-cell doors between adjacent equal-shape rooms, and fill each group by its majority marker color.

    Input structure:
    - Maroon walls partition the grid into rectangular rooms on black background.
    - Each room contains one small monochrome marker shape.
    - Matching is based on geometry up to rotation and reflection.

    Rule:
    - Canonicalize each room's shape under the dihedral symmetries.
    - Adjacent rooms with the same canonical shape get a one-cell opening in their shared wall center.
    - Fill every room in a shape group with that group's majority marker color, leaving the marker itself unchanged.

    Key insight:
    - Geometry defines connectivity groups, while color only decides the shared fill color after grouping.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rows = [r for r in range(g.shape[0]) if np.count_nonzero(g[r] == WALL) > g.shape[1] // 2]
    cols = [c for c in range(g.shape[1]) if np.count_nonzero(g[:, c] == WALL) > g.shape[0] // 2]
    info = {}
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            pts = [(r - r0 - 1, c - c0 - 1) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] not in (BLACK, WALL)]
            if not pts:
                continue
            color = next(int(g[r0 + 1 + r, c0 + 1 + c]) for r, c in pts)
            info[(i, j)] = (_canon(pts), color)
    groups = {}
    for room, (shape, color) in info.items():
        groups.setdefault(shape, []).append((room, color))
    for shape, rooms in groups.items():
        fill = Counter(color for _, color in rooms).most_common(1)[0][0]
        members = {room for room, _ in rooms}
        for i, j in members:
            out[rows[i] + 1 : rows[i + 1], cols[j] + 1 : cols[j + 1]] = fill
            for r in range(rows[i] + 1, rows[i + 1]):
                for c in range(cols[j] + 1, cols[j + 1]):
                    if g[r, c] not in (BLACK, WALL):
                        out[r, c] = g[r, c]
            if (i, j + 1) in members:
                out[(rows[i] + rows[i + 1]) // 2, cols[j + 1]] = BLACK
            if (i + 1, j) in members:
                out[rows[i + 1], (cols[j] + cols[j + 1]) // 2] = BLACK
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    g[0, :] = g[5, :] = g[10, :] = g[-1, :] = WALL
    g[:, 0] = g[:, 5] = g[:, 10] = g[:, -1] = WALL
    room_ids = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    shape_choices = [random.choice(SHAPES) for _ in room_ids]
    shape_choices[2] = shape_choices[0]
    shape_choices[4] = shape_choices[3]
    colors = random.sample(COLORS, len(room_ids))
    for (i, j), shape, color in zip(room_ids, shape_choices, colors):
        r0, c0 = [0, 5, 10][i], [0, 5, 10][j]
        for r, c in shape:
            g[r0 + 2 + r, c0 + 2 + c] = color
    return g
