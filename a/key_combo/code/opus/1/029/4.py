import itertools
import random
from collections import defaultdict

import numpy as np


def _border(r0, c0, r1, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _candidates(g, color):
    pts = [(r, c) for r, c in zip(*np.where(g == color))]
    if not pts:
        return []
    h, w = g.shape
    out = []
    for r0 in range(h):
        for r1 in range(r0 + 4, min(h, r0 + 10)):
            for c0 in range(w):
                for c1 in range(c0 + 4, min(w, c0 + 10)):
                    b = _border(r0, c0, r1, c1)
                    if all(p in b for p in pts):
                        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                        if sum(1 for p in corners if p in pts) >= 2:
                            out.append((r0, c0, r1, c1, b))
    if out:
        return out
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return [(r0, c0, r1, c1, _border(r0, c0, r1, c1))]


def _unique_toposort(colors, edges):
    indeg = {c: 0 for c in colors}
    adj = {c: set() for c in colors}
    for a, b in edges:
        if b not in adj[a]:
            adj[a].add(b)
            indeg[b] += 1
    order = []
    for _ in colors:
        ready = [c for c in colors if indeg[c] == 0 and c not in order]
        if len(ready) != 1:
            return None
        c = ready[0]
        order.append(c)
        for nb in adj[c]:
            indeg[nb] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    colors = sorted(int(c) for c in np.unique(g) if c != 0)
    cand_lists = [_candidates(g, c) for c in colors]
    best = None
    for combo in itertools.product(*cand_lists):
        rects = {c: rect for c, rect in zip(colors, combo)}
        edges = set()
        cells_to_colors = defaultdict(list)
        for c, rect in rects.items():
            for p in rect[4]:
                cells_to_colors[p].append(c)
        ok = True
        for p, owners in cells_to_colors.items():
            if len(owners) < 2:
                continue
            seen = int(g[p])
            if seen != 0:
                if seen not in owners:
                    ok = False
                    break
                for c in owners:
                    if c != seen:
                        edges.add((seen, c))
        if not ok:
            continue
        order = _unique_toposort(colors, edges)
        if order is not None:
            best = order
            break
    if best is None:
        best = colors
    return np.array(best, dtype=int).reshape(-1, 1)


def _render(rects, order):
    g = np.zeros((15, 15), dtype=int)
    for color in reversed(order):
        for r, c in _border(*rects[color]):
            g[r, c] = color
    return g


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4]
    base_rects = {
        1: (1, 1, 7, 7),
        2: (3, 4, 10, 11),
        3: (6, 2, 12, 8),
        4: (2, 8, 9, 13),
    }
    for _ in range(200):
        dr = random.randint(-1, 1)
        dc = random.randint(-1, 1)
        rects = {c: (r0 + dr, c0 + dc, r1 + dr, c1 + dc) for c, (r0, c0, r1, c1) in base_rects.items()}
        order = random.sample(colors, len(colors))
        g = _render(rects, order)
        try:
            got = list(main(g).ravel())
        except Exception:
            continue
        if got == order:
            return g
    return _render(base_rects, [1, 2, 3, 4])
