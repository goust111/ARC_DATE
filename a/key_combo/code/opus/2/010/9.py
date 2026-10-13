import random
from collections import Counter, deque

import numpy as np


LINE_COLORS = {1, 3, 4, 7}


def _repair(g):
    out = g.copy()
    counts = Counter(int(x) for x in out.ravel() if int(x) in LINE_COLORS)
    for col, _ in sorted(counts.items(), key=lambda kv: (kv[1], kv[0])):
        h, w = out.shape
        for r in range(h):
            cs = [c for c in range(w) if out[r, c] == col]
            for a, b in zip(cs, cs[1:]):
                if b > a + 1 and np.all(out[r, a + 1 : b] == 0):
                    out[r, a + 1 : b] = col
        for c in range(w):
            rs = [r for r in range(h) if out[r, c] == col]
            for a, b in zip(rs, rs[1:]):
                if b > a + 1 and np.all(out[a + 1 : b, c] == 0):
                    out[a + 1 : b, c] = col
    return out


def _gray_blocks(g):
    h, w = g.shape
    blocks = []
    for r in range(h - 1):
        for c in range(w - 1):
            pts = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            if all(g[p] == 5 for p in pts):
                blocks.append(pts)
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    repaired = _repair(np.array(grid, copy=True))
    h, w = repaired.shape
    markers = [(r, c) for r in range(h) for c in range(w) if repaired[r, c] == 2]
    base = repaired.copy()
    for r, c in markers:
        base[r, c] = 0
    blocks = _gray_blocks(base)
    counts = Counter(int(x) for x in base.ravel() if int(x) in LINE_COLORS)
    out = base.copy()
    for r, c in markers:
        best = None
        for block in blocks:
            for br, bc in block:
                if r == br:
                    step = 1 if bc > c else -1
                    path = [(r, x) for x in range(c + step, bc, step)]
                    if all(base[p] == 0 for p in path):
                        stop = (r, bc - step)
                        cand = (abs(bc - c), stop)
                        best = cand if best is None or cand < best else best
                if c == bc:
                    step = 1 if br > r else -1
                    path = [(x, c) for x in range(r + step, br, step)]
                    if all(base[p] == 0 for p in path):
                        stop = (br - step, c)
                        cand = (abs(br - r), stop)
                        best = cand if best is None or cand < best else best
        if best is None:
            out[r, c] = 2
            continue
        _, (sr, sc) = best
        nearest = None
        for rr in range(h):
            for cc in range(w):
                col = int(base[rr, cc])
                if col in LINE_COLORS:
                    cand = (abs(rr - sr) + abs(cc - sc), counts[col], col)
                    nearest = cand if nearest is None or cand < nearest else nearest
        out[sr, sc] = nearest[2] if nearest else 2
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 16), dtype=int)
    g[2, 1:4] = 1
    g[2, 6:10] = 1
    g[7, 2:5] = 3
    g[7, 7:12] = 3
    g[4:7, 13] = 4
    g[9:13, 13] = 4
    if random.random() < 0.5:
        g[12, 2:5] = 7
        g[12, 7:11] = 7
    g[1:3, 13:15] = 5
    g[10:12, 1:3] = 5
    g[13:15, 10:12] = 5
    marks = [(2, 11), (10, 6), (5, 1), (13, 13), (1, 5)]
    for r, c in random.sample(marks, random.randint(3, 5)):
        if g[r, c] == 0:
            g[r, c] = 2
    return g
