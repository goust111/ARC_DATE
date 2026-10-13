import random
from collections import deque

import numpy as np


BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    out = g.copy()
    starts = [tuple(map(int, p)) for p in np.argwhere(g == ORANGE)]
    if not starts:
        return out
    traversable = (g == AZURE) | ((g != BLACK) & (g != ORANGE))
    q = deque()
    seen = set(starts)
    for sr, sc in starts:
        for dr, dc in DIR4:
            nb = (sr + dr, sc + dc)
            if 0 <= nb[0] < g.shape[0] and 0 <= nb[1] < g.shape[1] and traversable[nb]:
                q.append(nb)
                seen.add(nb)
    while q:
        r, c = q.popleft()
        if g[r, c] != BLACK:
            out[r, c] = ORANGE
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < g.shape[0] and 0 <= nb[1] < g.shape[1] and nb not in seen and traversable[nb]:
                seen.add(nb)
                q.append(nb)
    return out


def _line(g, a, b, color=AZURE):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            if g[r0, c] == BLACK:
                g[r0, c] = color
    elif c0 == c1:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            if g[r, c0] == BLACK:
                g[r, c0] = color


def _place(g, cells, top, left, color):
    for r, c in cells:
        if 0 <= top + r < g.shape[0] and 0 <= left + c < g.shape[1]:
            g[top + r, left + c] = color


def generate_input() -> np.ndarray:
    h, w = random.randint(20, 27), random.randint(24, 32)
    g = np.zeros((h, w), dtype=int)
    row = random.randint(6, h - 7)
    start_col = 0
    source = (row, start_col)
    g[source] = ORANGE
    first = (row, random.randint(5, 8))
    second = (row, random.randint(w // 2, w - 8))
    _line(g, (row, 1), first)
    _line(g, first, second)

    # A square node splits to two visible branches.
    _place(g, {(0, 0), (0, 1), (1, 0), (1, 1)}, first[0] - 1, first[1], random.choice([RED, GREEN, YELLOW]))
    up_end = random.randint(1, max(2, row - 3))
    down_end = random.randint(min(h - 2, row + 3), h - 2)
    _line(g, (row, first[1]), (up_end, first[1]))
    _line(g, (row, first[1] + 1), (down_end, first[1] + 1))

    # An L node on the main trunk has only one outgoing turn, so flood propagation matches the encoded turn.
    l_top, l_left = row - 1, second[1]
    missing = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    l_cells = {(0, 0), (0, 1), (1, 0), (1, 1)} - {missing}
    _place(g, l_cells, l_top, l_left, random.choice([BLUE, MAGENTA, YELLOW]))
    if missing in [(0, 0), (1, 0)]:
        _line(g, (row, second[1]), (row, w - 2))
    else:
        end = random.choice([2, h - 3])
        _line(g, (row, second[1]), (end, second[1]))

    # Add one unreachable wire/node island to ensure the solver keeps it unchanged.
    iso_r, iso_c = random.randint(2, h - 4), random.randint(3, w - 5)
    if abs(iso_r - row) > 3:
        _line(g, (iso_r, iso_c), (iso_r, min(w - 2, iso_c + random.randint(3, 6))))
        _place(g, {(0, 0), (0, 1), (1, 0)}, iso_r - 1, iso_c + 1, RED)
    return g
