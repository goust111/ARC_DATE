import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
AZURE = 8
PAIR_COLORS = [RED, BLUE, GREEN, YELLOW]


def _components(grid):
    seen = np.zeros_like(grid, dtype=bool)
    out = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BLACK or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append((color, cells))
    return out


def _normalize(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    return {(r - r0, c - c0) for r, c in cells}


def _shape_signature(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    norm = _normalize(cells)
    row_counts = [sum((r, c) in norm for c in range(w)) for r in range(h)]
    col_counts = [sum((r, c) in norm for r in range(h)) for c in range(w)]
    return h, w, row_counts, col_counts


def _pair_line_color(a: int, b: int) -> int:
    key = frozenset([a, b])
    if key == frozenset([RED, BLUE]):
        return MAGENTA
    if key == frozenset([RED, GREEN]):
        return YELLOW
    if key == frozenset([BLUE, GREEN]):
        return AZURE
    return GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments that complete rectangles and join each pair by an L.

    Input structure:
    - The grid contains small monochrome connected fragments of different colors.
    - Every fragment has exactly one partner of a different color such that the two
      fragments represent complementary halves of a solid rectangle split by rows or
      by columns.

    Rule:
    - Pair fragments whose normalized row profiles or column profiles complement to a
      constant rectangle width/height.
    - For each pair, connect the nearest cells by an orthogonal polyline drawn first
      horizontally then vertically.
    - The connection color depends on the unordered pair of source colors.
    - Original fragments remain unchanged.

    Key insight:
    - Pairing is driven by complementary geometry rather than proximity or identity,
      and only after that pairing does the fixed color-mixing rule determine the L's
      appearance.
    """
    comps = _components(grid)
    used = set()
    out = grid.copy()
    for i, (ca, a) in enumerate(comps):
        if i in used:
            continue
        ha, wa, ra, ca_counts = _shape_signature(a)
        partner = None
        for j in range(i + 1, len(comps)):
            if j in used:
                continue
            cb, b = comps[j]
            hb, wb, rb, cb_counts = _shape_signature(b)
            ok = False
            if ha == hb and len(ra) == len(rb):
                sums = [x + y for x, y in zip(ra, rb)]
                ok = len(set(sums)) == 1
            if wa == wb and len(ca_counts) == len(cb_counts):
                sums = [x + y for x, y in zip(ca_counts, cb_counts)]
                ok = ok or len(set(sums)) == 1
            if ok:
                partner = j
                break
        if partner is None:
            continue
        used.add(i)
        used.add(partner)
        cb, b = comps[partner]
        p1, p2 = min(((x, y) for x in a for y in b), key=lambda pair: abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]))
        line_color = _pair_line_color(ca, cb)
        r, c = p1
        tr, tc = p2
        step = 1 if tc >= c else -1
        for cc in range(c, tc + step, step):
            if out[r, cc] == BLACK:
                out[r, cc] = line_color
        step = 1 if tr >= r else -1
        for rr in range(r, tr + step, step):
            if out[rr, tc] == BLACK:
                out[rr, tc] = line_color
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        grid = np.zeros((h, w), dtype=int)
        ok = True
        for ca, cb in random.sample([(RED, BLUE), (RED, GREEN), (BLUE, GREEN), (YELLOW, BLUE)], random.randint(2, 3)):
            if random.random() < 0.5:
                rh = random.randint(3, 4)
                rw = random.randint(4, 6)
                cuts = [random.randint(1, rw - 1) for _ in range(rh)]
                y = random.randint(0, h - rh)
                x0 = random.randint(0, w - rw - 6)
                gap = random.randint(3, 5)
                for r in range(rh):
                    for c in range(cuts[r]):
                        grid[y + r, x0 + c] = ca
                    for c in range(cuts[r], rw):
                        grid[y + r, x0 + gap + c] = cb
            else:
                rh = random.randint(4, 6)
                rw = random.randint(3, 4)
                cuts = [random.randint(1, rh - 1) for _ in range(rw)]
                x = random.randint(0, w - rw)
                y0 = random.randint(0, h - rh - 6)
                gap = random.randint(3, 5)
                for c in range(rw):
                    for r in range(cuts[c]):
                        grid[y0 + r, x + c] = ca
                    for r in range(cuts[c], rh):
                        grid[y0 + gap + r, x + c] = cb
        if np.count_nonzero(grid) > 0:
            return grid
