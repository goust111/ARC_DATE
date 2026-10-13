import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
FILL = {1: 2, 2: 1, 3: 3, 4: 4}


def _wall_lines(g):
    rows = [r for r in range(g.shape[0]) if np.count_nonzero(g[r] == GREY) > g.shape[1] // 2]
    cols = [c for c in range(g.shape[1]) if np.count_nonzero(g[:, c] == GREY) > g.shape[0] // 2]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count door gaps per rectangular room, fill the room by that count, and draw center-to-door guide lines.

    Input structure:
    - Grey one-cell walls form a rectangular room lattice on black background.
    - Internal shared walls may contain one-cell door openings.
    - The outer boundary is closed and has no doors.

    Rule:
    - Each room is the rectangle between consecutive wall rows and wall columns.
    - A room with 1, 2, 3, or 4 doors is filled red, blue, green, or yellow respectively.
    - From the room center, draw a maroon orthogonal line toward every door, stopping before the wall.

    Key insight:
    - Door count determines the fill color, while door position determines the direction guide.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rows, cols = _wall_lines(g)
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i], rows[i + 1]
            c0, c1 = cols[j], cols[j + 1]
            interior = (slice(r0 + 1, r1), slice(c0 + 1, c1))
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            doors = []
            if any(g[r0, c] == BLACK for c in range(c0 + 1, c1)):
                doors.append("top")
            if any(g[r1, c] == BLACK for c in range(c0 + 1, c1)):
                doors.append("bottom")
            if any(g[r, c0] == BLACK for r in range(r0 + 1, r1)):
                doors.append("left")
            if any(g[r, c1] == BLACK for r in range(r0 + 1, r1)):
                doors.append("right")
            if doors:
                out[interior] = FILL[len(doors)]
                cr = (r0 + r1) // 2
                cc = (c0 + c1) // 2
                for d in doors:
                    if d == "top":
                        out[r0 + 1 : cr + 1, cc] = MAROON
                    elif d == "bottom":
                        out[cr : r1, cc] = MAROON
                    elif d == "left":
                        out[cr, c0 + 1 : cc + 1] = MAROON
                    else:
                        out[cr, cc:c1] = MAROON
    out[g == GREY] = GREY
    return out


def generate_input() -> np.ndarray:
    nr = random.randint(2, 3)
    nc = random.randint(2, 3)
    room_h = random.randint(3, 5)
    room_w = random.randint(3, 5)
    h = nr * room_h + nr + 1
    w = nc * room_w + nc + 1
    g = np.zeros((h, w), dtype=int)
    rows = [i * (room_h + 1) for i in range(nr + 1)]
    cols = [i * (room_w + 1) for i in range(nc + 1)]
    for r in rows:
        g[r, :] = GREY
    for c in cols:
        g[:, c] = GREY
    for i in range(1, nr):
        for j in range(nc):
            if random.random() < 0.7:
                c0, c1 = cols[j], cols[j + 1]
                c = random.randint(c0 + 1, c1 - 1)
                g[rows[i], c] = BLACK
    for i in range(nr):
        for j in range(1, nc):
            if random.random() < 0.7:
                r0, r1 = rows[i], rows[i + 1]
                r = random.randint(r0 + 1, r1 - 1)
                g[r, cols[j]] = BLACK
    return g
