import random
from collections import deque

import numpy as np


def _components(g, colors):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) not in colors:
                continue
            col = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((col, pts))
    return comps


def _legend(g):
    h, w = g.shape
    for r in (0, h - 1):
        pts = [(r, c) for c in range(w) if g[r, c] not in (0, 5)]
        if len(pts) >= 4 and max(c for _, c in pts) - min(c for _, c in pts) + 1 == len(pts):
            return pts
    for c in (0, w - 1):
        pts = [(r, c) for r in range(h) if g[r, c] not in (0, 5)]
        if len(pts) >= 4 and max(r for r, _ in pts) - min(r for r, _ in pts) + 1 == len(pts):
            return pts
    return []


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    gray = [(r, c) for r in range(h) for c in range(w) if g[r, c] == 5]
    rs = [r for r, _ in gray]
    cs = [c for _, c in gray]
    orient = "h" if len(set(rs)) == 1 else "v"
    legend = _legend(g)
    legend_set = set(legend)
    movers = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if g[r, c] not in (0, 5) and (r, c) not in legend_set]
    bar = set(gray)

    def dist_to_bar(p, mass):
        r, c = p
        return min(abs(r - rr) + abs(c - cc) for rr, cc in mass)

    movers.sort(key=lambda x: (dist_to_bar((x[0], x[1]), bar), x[0], x[1]))
    out = g.copy()
    mass = set(bar)
    placed = []
    for r, c, col in movers:
        out[r, c] = 0
        best = None
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r, c
            steps = 0
            while True:
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if (nr, nc) in mass:
                    cand = (steps, rr, cc)
                    best = cand if best is None or cand < best else best
                    break
                if out[nr, nc] != 0:
                    break
                rr, cc = nr, nc
                steps += 1
        if best is None:
            rr, cc = r, c
        else:
            _, rr, cc = best
        mass.add((rr, cc))
        placed.append((rr, cc))
    if orient == "h":
        placed.sort(key=lambda p: (p[1], min(abs(p[0] - r) for r, _ in bar)))
    else:
        placed.sort(key=lambda p: (p[0], min(abs(p[1] - c) for _, c in bar)))
    for (rr, cc), (lr, lc) in zip(placed, legend):
        out[rr, cc] = g[lr, lc]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((13, 13), dtype=int)
    horizontal = random.random() < 0.5
    n = random.randint(4, 6)
    colors = random.sample([1, 2, 3, 4, 6, 7], n)
    legend_cols = random.sample([1, 2, 3, 4, 6, 7], n)
    if horizontal:
        r, c0 = 6, 3
        g[r, c0 : c0 + n] = 5
        spots = [(r - random.choice([2, 3]), c0 + i) if i % 2 else (r + random.choice([2, 3]), c0 + i) for i in range(n)]
        for c, col in enumerate(legend_cols, 1):
            g[12, c] = col
    else:
        r0, c = 3, 6
        g[r0 : r0 + n, c] = 5
        spots = [(r0 + i, c - random.choice([2, 3])) if i % 2 else (r0 + i, c + random.choice([2, 3])) for i in range(n)]
        for r, col in enumerate(legend_cols, 1):
            g[r, 12] = col
    random.shuffle(colors)
    for (r, c), col in zip(spots, colors):
        g[r, c] = col
    return g
