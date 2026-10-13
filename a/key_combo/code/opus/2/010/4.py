import random
from collections import Counter, deque

import numpy as np


SQ_COLORS = {1, 2, 3, 4, 7}


def _components(g):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == 0:
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


def _is_square(pts):
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    side = r1 - r0 + 1
    return side == c1 - c0 + 1 and len(pts) == side * side, (r0, r1, c0, c1, side)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    squares = []
    for col, pts in _components(g):
        ok, box = _is_square(pts)
        if ok and col in SQ_COLORS:
            r0, r1, c0, c1, side = box
            squares.append({"color": col, "box": box, "center": ((r0 + r1) / 2, (c0 + c1) / 2)})
    counts = Counter(s["color"] for s in squares)
    owners = {}
    grid_center = ((h - 1) / 2, (w - 1) / 2)
    for idx, s in enumerate(squares):
        r0, r1, c0, c1, side = s["box"]
        layers = counts[s["color"]] - 1
        for rr in range(r0 - layers, r1 + layers + 1):
            for cc in range(c0 - layers, c1 + layers + 1):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                layer = max(r0 - rr, rr - r1, c0 - cc, cc - c1, 0)
                val = s["color"] if layer == 0 or layer % 2 == 1 else 5
                priority = (
                    side,
                    -abs(s["center"][0] - grid_center[0]) - abs(s["center"][1] - grid_center[1]),
                    -idx,
                )
                if (rr, cc) not in owners or priority > owners[(rr, cc)][0]:
                    owners[(rr, cc)] = (priority, val)
    out = np.zeros_like(g)
    for (r, c), (_, val) in owners.items():
        out[r, c] = val
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    a, b, ccol = random.sample([1, 2, 3, 4, 7], 3)
    dy, dx = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    if random.random() < 0.5:
        layout = [(1 + dy, 1 + dx, a, 3), (1, 7 + dx, a, 2), (7 + dy, 4, a, 2),
                  (10, 10, b, 2), (10, 14, b, 2), (3, 12, ccol, 2)]
    else:
        layout = [(2, 2, a, 3), (2 + dy, 8, a, 2), (7, 5 + dx, a, 2),
                  (11, 2, b, 2), (11, 7 + dx, b, 2), (12, 13, ccol, 2)]
    for r, c, col, side in layout:
        g[r : r + side, c : c + side] = col
    for _ in range(random.randint(2, 3)):
        r, c = random.randint(1, 15), random.randint(1, 15)
        col = random.choice([6, 9])
        pts = [(r, c), (r + 1, c), (r, c + 1)]
        if random.random() < 0.5:
            pts.append((r + 2, c))
        if all(0 <= rr < 18 and 0 <= cc < 18 and g[rr, cc] == 0 for rr, cc in pts):
            for rr, cc in pts:
                g[rr, cc] = col
    return g
