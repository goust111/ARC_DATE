import random
from collections import deque

import numpy as np


MARK_COLORS = {2, 3, 4, 6, 7}


def _components(g):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) not in MARK_COLORS:
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
            comps.append({"color": col, "pts": pts, "rows": {p[0] for p in pts}, "cols": {p[1] for p in pts}})
    return [m for m in comps if len(m["pts"]) == 4 and len(m["rows"]) == 2 and len(m["cols"]) == 2]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    marks = _components(g)
    center = ((h - 1) / 2, (w - 1) / 2)
    for i, m in enumerate(marks):
        neigh = []
        for j, n in enumerate(marks):
            if i != j and (m["rows"] & n["rows"] or m["cols"] & n["cols"]):
                neigh.append(j)
        m["neigh"] = neigh
        m["A"] = len(neigh)
        rs, cs = sorted(m["rows"]), sorted(m["cols"])
        m["center_dist"] = abs((rs[0] + rs[1]) / 2 - center[0]) + abs((cs[0] + cs[1]) / 2 - center[1])
    claims = {}
    out = g.copy()
    for m in marks:
        if m["A"] == 0:
            for r, c in m["pts"]:
                out[r, c] = 5
            continue
        rows, cols = set(), set()
        if m["A"] >= 2:
            rows, cols = m["rows"], m["cols"]
        else:
            n = marks[m["neigh"][0]]
            if m["rows"] & n["rows"]:
                rows = m["rows"] & n["rows"]
            elif m["cols"] & n["cols"]:
                cols = m["cols"] & n["cols"]
        for r in rows:
            for c in range(w):
                if g[r, c] == 1:
                    key = (m["A"], -m["center_dist"])
                    if (r, c) not in claims or key > claims[(r, c)][0]:
                        claims[(r, c)] = (key, m["color"])
        for c in cols:
            for r in range(h):
                if g[r, c] == 1:
                    key = (m["A"], -m["center_dist"])
                    if (r, c) not in claims or key > claims[(r, c)][0]:
                        claims[(r, c)] = (key, m["color"])
    for (r, c), (_, col) in claims.items():
        out[r, c] = col
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((17, 17), dtype=int)
    for x in [0, 4, 8, 12, 16]:
        g[x, :] = 1
        g[:, x] = 1
    cells = [(1, 1), (1, 5), (5, 1), (9, 5), (13, 13)]
    if random.random() < 0.5:
        cells = [(1, 9), (5, 9), (1, 13), (9, 1), (13, 5)]
    colors = random.sample([2, 3, 4, 6, 7], len(cells))
    for (r, c), col in zip(cells, colors):
        g[r : r + 2, c : c + 2] = col
    return g
