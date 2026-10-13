import random
from collections import deque

import numpy as np


def _components(g, accept):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not accept(int(g[r, c])):
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


def _norm(pts):
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((r - r0, c - c0) for r, c in pts)


def _holes(g, host_colors):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            edge = False
            border = set()
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                edge |= rr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if g[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif int(g[nr, nc]) in host_colors:
                        border.add(int(g[nr, nc]))
            if not edge and len(border) == 1:
                result.append((next(iter(border)), pts))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    comps = _components(g, lambda x: x != 0)
    templates = {_norm(pts) for col, pts in comps if col == 8}
    large_colors = {col for col, pts in comps if col in (1, 2, 3) and len(pts) >= 15}
    participants = []
    erase = []
    for col, pts in comps:
        if col in (5, 8):
            erase.extend(pts)
        elif col not in large_colors:
            erase.extend(pts)
            if _norm(pts) in templates:
                participants.append(_norm(pts))
    out = g.copy()
    for r, c in erase:
        out[r, c] = 0
    available = participants[:]
    for host, pts in _holes(g, large_colors):
        shp = _norm(pts)
        if shp in available:
            available.remove(shp)
            for r, c in pts:
                out[r, c] = host
    return out


def _stamp(g, pts, r, c, col):
    for dr, dc in pts:
        g[r + dr, c + dc] = col


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    host1, host2 = random.sample([1, 2, 3], 2)
    ar, ac = random.choice([(1, 1), (1, 2), (2, 1)])
    br, bc = random.choice([(9, 8), (9, 9), (10, 8)])
    g[ar : ar + 7, ac : ac + 7] = host1
    for dr, dc in [(2, 2), (2, 4), (3, 4), (4, 1), (4, 2)]:
        g[ar + dr, ac + dc] = 0
    g[br : br + 7, bc : bc + 8] = host2
    for dr, dc in [(2, 2), (2, 3), (3, 2), (4, 5), (5, 5), (5, 6)]:
        g[br + dr, bc + dc] = 0
    shapes = [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
    ]
    _stamp(g, shapes[0], 4, 11, 8)
    _stamp(g, shapes[1], 1, 12, 8)
    _stamp(g, shapes[2], 4, 13, 8)
    _stamp(g, shapes[3], 1, 15, 8)
    places = [(10, 1), (13, 1), (15, 4), (1, 10), (6, 11), (8, 1), (15, 1)]
    cols = [4, 7, 6, 4, 7, 6, 4]
    source_shapes = [shapes[0], shapes[1], shapes[1], shapes[2], shapes[3], shapes[3], shapes[4]]
    for shp, (r, c), col in zip(source_shapes, places, cols):
        _stamp(g, shp, r, c, col)
    for _ in range(random.randint(2, 4)):
        r, c = random.randrange(18), random.randrange(18)
        if g[r, c] == 0:
            g[r, c] = 5
    return g
