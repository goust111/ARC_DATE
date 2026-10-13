import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = [(r, c)]
        seen[r, c] = True
        comp = []
        while q:
            cr, cc = q.pop()
            comp.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _traverse(mask):
    pts = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    deg = {}
    for r, c in pts:
        deg[(r, c)] = sum((r + dr, c + dc) in pts for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    ends = [p for p, d in deg.items() if d == 1]
    if not ends:
        return list(pts)
    path = [ends[0]]
    prev = None
    cur = ends[0]
    while True:
        nxt = [(cur[0] + dr, cur[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nxt = [p for p in nxt if p in pts and p != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Reverse the color sequence of each path subsegment lying inside a gray U-shaped wall."""
    out = grid.copy()
    path_mask = (grid != 0) & (grid != 5)
    path = _traverse(path_mask)
    index = {p: i for i, p in enumerate(path)}
    for wall in _components(grid == 5):
        rs = [r for r, _ in wall]
        cs = [c for _, c in wall]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        inside = [p for p in path if r0 < p[0] < r1 and c0 < p[1] < c1]
        if len(inside) < 2:
            inside = [p for p in path if r0 <= p[0] <= r1 and c0 <= p[1] <= c1 and grid[p] != 5]
        if len(inside) < 2:
            continue
        inside.sort(key=lambda p: index[p])
        colors = [int(grid[p]) for p in inside][::-1]
        for p, col in zip(inside, colors):
            out[p] = col
    return out


def _u(g, r0, c0, h, w, open_side):
    if open_side != "up":
        g[r0, c0 : c0 + w] = 5
    if open_side != "down":
        g[r0 + h - 1, c0 : c0 + w] = 5
    if open_side != "left":
        g[r0 : r0 + h, c0] = 5
    if open_side != "right":
        g[r0 : r0 + h, c0 + w - 1] = 5


def generate_input() -> np.ndarray:
    g = np.zeros((15, 18), dtype=int)
    _u(g, 2, 3, 6, 5, "left")
    _u(g, 7, 10, 6, 5, "right")
    pts = [(4, c) for c in range(0, 7)] + [(r, 6) for r in range(5, 11)] + [(10, c) for c in range(7, 18)]
    cols = random.sample([1, 2, 3, 4, 7], 5)
    shift = random.randint(0, len(cols) - 1)
    for i, (r, c) in enumerate(pts):
        g[r, c] = cols[(i + shift) % len(cols)]
    return g
