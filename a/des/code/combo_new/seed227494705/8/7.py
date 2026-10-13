import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ORDER = [1, 2, 3, 4, 7, 8, 6, 9]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _endpoints(pts):
    s = set(pts)
    ends = []
    for r, c in pts:
        deg = sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if deg == 1:
            ends.append((r, c))
    return ends


def _draw_line(out, a, b, color):
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            out[r1, c] = color
    elif c1 == c2:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            out[r, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect colored path fragments in the learned color order using grey bend points.

    Input structure:
    - Black background with separated one-pixel-wide colored path fragments.
    - Each fragment has exactly two same-color endpoints.
    - Grey singleton pixels lie between fragments as optional orthogonal bend markers.

    Rule:
    - Visit fragments in the fixed color order learned from examples.
    - For each consecutive pair, connect the nearest endpoint pair with the previous fragment's color.
    - Use a grey pixel as the bend when endpoints are not row- or column-aligned, overwriting that grey marker.

    Key insight:
    - Color order selects the sequence, nearest endpoints select the joints, and grey markers supply the turn coordinates.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    fragments = {}
    for color in ORDER:
        comps = list(_components(g == color))
        if comps:
            fragments[color] = comps[0]
    colors = [c for c in ORDER if c in fragments]
    grey_pts = [tuple(map(int, p)) for p in np.argwhere(g == GREY)]
    for c1, c2 in zip(colors, colors[1:]):
        e1 = _endpoints(fragments[c1])
        e2 = _endpoints(fragments[c2])
        if not e1 or not e2:
            continue
        a, b = min(((a, b) for a in e1 for b in e2), key=lambda pair: abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]))
        if a[0] == b[0] or a[1] == b[1]:
            _draw_line(out, a, b, c1)
        else:
            candidates = [p for p in grey_pts if (p[0] == a[0] and p[1] == b[1]) or (p[1] == a[1] and p[0] == b[0])]
            if candidates:
                bend = min(candidates, key=lambda p: abs(p[0] - a[0]) + abs(p[1] - a[1]) + abs(p[0] - b[0]) + abs(p[1] - b[1]))
            else:
                bend = (a[0], b[1])
            _draw_line(out, a, bend, c1)
            _draw_line(out, bend, b, c1)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(14, 22)
    w = rng.randint(22, 30)
    g = np.zeros((h, w), dtype=int)
    k = rng.randint(3, 5)
    colors = ORDER[:k]
    col_positions = np.linspace(2, w - 5, k).astype(int).tolist()
    rows = [rng.randint(2, h - 3) for _ in range(k)]
    for color, r, c in zip(colors, rows, col_positions):
        length = rng.randint(3, 5)
        if c + length >= w:
            c = w - length - 1
        g[r, c : c + length] = color
    for i in range(k - 1):
        a = (rows[i], col_positions[i] + int(np.count_nonzero(g[rows[i], :] == colors[i])) - 1)
        b = (rows[i + 1], col_positions[i + 1])
        if a[0] != b[0] and a[1] != b[1]:
            bend = rng.choice([(a[0], b[1]), (b[0], a[1])])
            if g[bend] == BG:
                g[bend] = GREY
    return g
