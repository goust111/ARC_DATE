import random

import numpy as np


def _neighbors(p, pts):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (r + dr, c + dc) in pts]


def _trace(pts):
    start = min(pts)
    path = [start]
    prev = None
    cur = start
    while True:
        nbs = sorted(_neighbors(cur, pts))
        nxt = nbs[0] if prev is None else [p for p in nbs if p != prev][0]
        if nxt == start:
            break
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    pts = {tuple(p) for p in np.argwhere(grid == 1)}
    path = _trace(pts)
    turns = []
    signs = []
    for i, cur in enumerate(path):
        prev = path[i - 1]
        nxt = path[(i + 1) % len(path)]
        v1 = (cur[0] - prev[0], cur[1] - prev[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        if v1 != v2:
            sign = v1[0] * v2[1] - v1[1] * v2[0]
            turns.append((cur, sign))
            signs.append(sign)
    majority = 1 if signs.count(1) >= signs.count(-1) else -1
    concave = {p for p, s in turns if s != majority}
    convex = {p for p, s in turns if s == majority}
    out = grid.copy()
    for p in convex:
        out[p] = 4
    for p in concave:
        out[p] = 2
    k = len(concave)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    for layer in range(1, k + 1):
        a, b = r0 - layer, r1 + layer
        x, y = c0 - layer, c1 + layer
        for c in range(x, y + 1):
            if 0 <= a < out.shape[0] and 0 <= c < out.shape[1] and out[a, c] == 0:
                out[a, c] = 4
            if 0 <= b < out.shape[0] and 0 <= c < out.shape[1] and out[b, c] == 0:
                out[b, c] = 4
        for r in range(a, b + 1):
            if 0 <= r < out.shape[0] and 0 <= x < out.shape[1] and out[r, x] == 0:
                out[r, x] = 4
            if 0 <= r < out.shape[0] and 0 <= y < out.shape[1] and out[r, y] == 0:
                out[r, y] = 4
    return out


def _draw_poly(grid, verts):
    for (r0, c0), (r1, c1) in zip(verts, verts[1:]):
        r, c = r0, c0
        dr = (r1 > r0) - (r1 < r0)
        dc = (c1 > c0) - (c1 < c0)
        while (r, c) != (r1, c1):
            grid[r, c] = 1
            r += dr
            c += dc
        grid[r1, c1] = 1


def generate_input() -> np.ndarray:
    size = random.randint(22, 26)
    grid = np.zeros((size, size), dtype=int)
    r0 = random.randint(4, 5)
    c0 = random.randint(4, 5)
    r1 = random.randint(r0 + 9, size - 4)
    c1 = random.randint(c0 + 10, size - 4)
    n0 = random.randint(c0 + 3, c1 - 6)
    n1 = n0 + random.randint(2, 3)
    depth = random.randint(2, 3)
    nr = r1 - depth
    verts = [(r0, c0), (r0, c1), (r1, c1), (r1, n1), (nr, n1), (nr, n0), (r1, n0), (r1, c0), (r0, c0)]
    _draw_poly(grid, verts)
    return grid
