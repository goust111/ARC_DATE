import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np


def _components(grid: np.ndarray):
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(grid[r, c]) == 0:
                continue
            color = int(grid[r, c])
            q = [(r, c)]
            visited[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if nx < 0 or nx >= H or ny < 0 or ny >= W:
                        continue
                    if visited[nx, ny] or int(grid[nx, ny]) != color:
                        continue
                    visited[nx, ny] = True
                    q.append((nx, ny))
            comps.append((color, pts))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    comps = _components(g)
    rect_by_color: Dict[int, np.ndarray] = {}
    covered = set()

    for color, pts in comps:
        if color == 0:
            continue
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if (r1 - r0 + 1) != 4 or (c1 - c0 + 1) != 4:
            continue
        if len(pts) != 12 and len(pts) != 16:
            continue

        # classify by 12-cell frame + inner black
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                is_border = (r in (r0, r1)) or (c in (c0, c1))
                has = (r, c) in set(pts)
                if is_border != has:
                    ok = False
                    break
            if not ok:
                break
        if ok and len(pts) == 12:
            local = np.zeros((4, 4), dtype=int)
            for r, c in pts:
                local[r - r0, c - c0] = color
            rect_by_color[color] = local
            for p in pts:
                covered.add(p)

    # diagonal codes among remaining non-zero cells
    remain = [(r, c, int(g[r, c])) for r in range(H) for c in range(W) if g[r, c] != 0 and (r, c) not in covered]
    best = []
    for d in {c - r for r, c, _ in remain}:
        rows = {}
        good = True
        for r, c, v in [x for x in remain if x[1] - x[0] == d]:
            if r in rows:
                good = False
                break
            rows[r] = v
        if not good:
            continue
        if not rows:
            continue
        rs = sorted(rows.keys())
        if rs != list(range(min(rs), max(rs) + 1)):
            continue
        candidate = [rows[r] for r in rs]
        if len(candidate) > len(best):
            best = candidate

    if not best:
        best = list(rect_by_color.keys())

    out = np.zeros((4, 4 * len(best)), dtype=int)
    pos = 0
    for color in best:
        if color not in rect_by_color:
            continue
        out[:, pos : pos + 4] = rect_by_color[color]
        pos += 4
    return out


def generate_input() -> np.ndarray:
    H = 20
    W = 30
    g = np.zeros((H, W), dtype=int)

    n = random.randint(3, 5)
    colors = random.sample([1, 2, 3, 4, 7, 9], n + 1)
    hollow_colors = colors[:n]
    solid_colors = colors[n : 2 * n]

    # place hollow rectangles
    for i, color in enumerate(hollow_colors):
        r0 = 8
        c0 = 2 + i * 5
        for r in range(r0, r0 + 4):
            for c in range(c0, c0 + 4):
                if r in (r0, r0 + 3) or c in (c0, c0 + 3):
                    g[r, c] = color

    # place some solid rectangles with unique colors
    for i, color in enumerate(solid_colors[: n - 1]):
        r0 = 2
        c0 = 2 + i * 5
        for r in range(r0, r0 + 4):
            for c in range(c0, c0 + 4):
                g[r, c] = color

    # encoding strip
    for i, color in enumerate(hollow_colors):
        if i < H and i < W:
            g[i, i] = color

    return g
