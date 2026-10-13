import random
from collections import deque

import numpy as np


def _fill_color(area):
    if area <= 3:
        return 1
    if area <= 6:
        return 4
    if area <= 9:
        return 7
    return 6


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    out = g.copy()
    seen = np.zeros((h, w), dtype=bool)
    blockers = (g != 0)
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                touches_edge |= rr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and not blockers[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                col = _fill_color(len(comp))
                for rr, cc in comp:
                    out[rr, cc] = col
    out[g == 2] = 0
    return out


def _draw_box(g, r, c, ih, iw, crack=None):
    g[r : r + ih + 2, c] = 5
    g[r : r + ih + 2, c + iw + 1] = 5
    g[r, c : c + iw + 2] = 5
    g[r + ih + 1, c : c + iw + 2] = 5
    if crack == "v" and iw >= 3:
        cc = c + 1 + random.randrange(1, iw - 1)
        g[r + 1 : r + ih + 1, cc] = 2
    elif crack == "h" and ih >= 3:
        rr = r + 1 + random.randrange(1, ih - 1)
        g[rr, c + 1 : c + iw + 1] = 2


def generate_input() -> np.ndarray:
    g = np.zeros((14, 14), dtype=int)
    slots = [(1, 1), (1, 8), (8, 1), (8, 8), (5, 5)]
    random.shuffle(slots)
    n = random.randint(3, 5)
    crack_count = random.randint(1, min(2, n))
    for i, (r, c) in enumerate(slots[:n]):
        ih = random.randint(2, 4)
        iw = random.randint(2, 4)
        if r + ih + 2 > 14 or c + iw + 2 > 14:
            ih, iw = 3, 3
        crack = None
        if i < crack_count:
            crack = random.choice(["h", "v"])
            if crack == "h" and ih < 3:
                ih = 3
            if crack == "v" and iw < 3:
                iw = 3
        _draw_box(g, r, c, ih, iw, crack)
    return g
