import random
from collections import deque

import numpy as np


def _components(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(pts)
    return parts


def _direction(pts):
    s = set(pts)
    ends = []
    for r, c in pts:
        deg = sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if deg == 1:
            ends.append((r, c))
    if not ends:
        ends = pts
    tip = max(ends, key=lambda p: abs(p[0] - np.mean([q[0] for q in pts])) + abs(p[1] - np.mean([q[1] for q in pts])))
    cr = np.mean([p[0] for p in pts])
    cc = np.mean([p[1] for p in pts])
    dr = tip[0] - cr
    dc = tip[1] - cc
    return (-1, 0) if abs(dr) > abs(dc) and dr < 0 else (1, 0) if abs(dr) > abs(dc) else (0, -1) if dc < 0 else (0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Replicate each directional stamp by the count of same-color singleton markers."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    for color in sorted(set(map(int, g.ravel())) - {0}):
        comps = _components(g == color)
        stamp = max(comps, key=len)
        markers = [c for c in comps if len(c) == 1]
        for r, c in sum(markers, []):
            out[r, c] = 0
        dr, dc = _direction(stamp)
        rs = [p[0] for p in stamp]
        cs = [p[1] for p in stamp]
        step = (max(rs) - min(rs) + 1) if dr else (max(cs) - min(cs) + 1)
        for k in range(1, len(markers) + 1):
            for r, c in stamp:
                rr, cc = r + dr * step * k, c + dc * step * k
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    shapes = {
        "up": [(1, 0), (1, 1), (2, 0), (2, 1), (0, 1)],
        "down": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)],
        "left": [(0, 1), (1, 1), (0, 2), (1, 2), (1, 0)],
        "right": [(0, 0), (1, 0), (0, 1), (1, 1), (1, 2)],
    }
    for color in random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 3)):
        name = random.choice(list(shapes))
        pts = shapes[name]
        h = max(p[0] for p in pts) + 1
        w = max(p[1] for p in pts) + 1
        r0 = random.randint(2, g.shape[0] - h - 6)
        c0 = random.randint(2, g.shape[1] - w - 6)
        for dr, dc in pts:
            g[r0 + dr, c0 + dc] = color
        for _ in range(random.randint(0, 3)):
            while True:
                rr, cc = random.randint(0, g.shape[0] - 1), random.randint(0, g.shape[1] - 1)
                if g[rr, cc] == 0:
                    g[rr, cc] = color
                    break
    return g
