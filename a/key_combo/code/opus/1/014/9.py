import random
from collections import deque
import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _solid_from_outline(comp):
    r0, c0, r1, c1 = _bbox(comp)
    rows = {}
    for r, c in comp:
        rows.setdefault(r, []).append(c)
    pts = []
    axis_left = c1
    for r, cols in rows.items():
        cr = max(cols)
        mirror = 2 * (axis_left + 1) - cr - 1
        for c in range(cr + 1, mirror):
            pts.append((r - r0, c - c0))
    if not pts:
        pts = [(r - r0, c1 + 1 - c0) for r in range(r0, r1 + 1)]
    rr0 = min(r for r, _ in pts)
    cc0 = min(c for _, c in pts)
    return [(r - rr0, c - cc0) for r, c in pts]


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros_like(grid)
    comps = _components(grid, 3)
    if not comps:
        return out
    centers = [(((_bbox(c)[0] + _bbox(c)[2]) / 2), ((_bbox(c)[1] + _bbox(c)[3]) / 2), c) for c in comps]
    row_mid = sum(x[0] for x in centers) / len(centers)
    col_mid = sum(x[1] for x in centers) / len(centers)
    for cr, cc, comp in centers:
        solid = _solid_from_outline(comp)
        sh = max(r for r, _ in solid) + 1
        sw = max(c for _, c in solid) + 1
        if cr < row_mid and cc < col_mid:
            top, left = 0, 0
        elif cr < row_mid:
            top, left = 0, w - sw
        elif cc < col_mid:
            top, left = h - sh, 0
        else:
            top, left = h - sh, w - sw
        for r, c in solid:
            out[top + r, left + c] = 3
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(15, 18)
    g = np.zeros((h, w), dtype=int)
    top = random.randint(3, 5)
    left = random.randint(3, 5)
    gap_r = random.randint(5, 7)
    gap_c = random.randint(5, 7)
    anchors = [(top, left), (top, left + gap_c), (top + gap_r, left), (top + gap_r, left + gap_c)]
    patterns = [
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)],
        [(0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0), (3, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)],
    ]
    random.shuffle(patterns)
    for (r0, c0), pat in zip(anchors, patterns):
        for r, c in pat:
            if 0 <= r0 + r < h and 0 <= c0 + c < w:
                g[r0 + r, c0 + c] = 3
    return g
