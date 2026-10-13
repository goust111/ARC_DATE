from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np


def _bg(grid: np.ndarray) -> int:
    v, c = np.unique(grid, return_counts=True)
    return int(v[int(np.argmax(c))])


def _components(mask: np.ndarray) -> List[Tuple[int, int]]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps: List[Tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            while q:
                cr, cc = q.popleft()
                comps.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = _bg(g)
    h, w = g.shape
    out = g.copy()
    color_map = {3: 6, 8: 4, 2: 1}
    markers = [tuple(int(x) for x in p) for p in np.argwhere(out != bg)]
    markers.sort(key=lambda p: (p[0], p[1]))

    occupied = out.copy()

    # Draw outlines first
    for r, c in markers:
        col = int(g[r, c])
        outline_col = color_map.get(col, col)
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                occupied[rr, cc] = outline_col

    # Fire rays by marker order (row-major)
    for r, c in markers:
        if (int(g[r, c]) not in color_map) or (r == 0 or c == 0 or r == h - 1 or c == w - 1):
            continue
        corners = [
            (r - 1, c - 1, -1, -1),
            (r - 1, c + 1, -1, 1),
            (r + 1, c - 1, 1, -1),
            (r + 1, c + 1, 1, 1),
        ]
        line_color = color_map[int(g[r, c])]
        for cr, cc, dr, dc in corners:
            rr, cc2 = cr, cc
            while True:
                nr, nc = rr + dr, cc2 + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if occupied[nr, nc] != bg:
                    break
                occupied[nr, nc] = line_color
                rr, cc2 = nr, nc

    return occupied


_RNG = np.random.default_rng()


def generate_input() -> np.ndarray:
    h = 15
    w = 15
    g = np.zeros((h, w), dtype=int)
    colors = [3, 8, 2]
    _RNG.shuffle(colors)
    n = int(_RNG.integers(3, 5))
    chosen = colors[:n]
    used: List[Tuple[int, int]] = []
    for col in chosen:
        for _ in range(5000):
            r = int(_RNG.integers(2, h - 2))
            c = int(_RNG.integers(2, w - 2))
            if (r, c) in used:
                continue
            ok = True
            for ur, uc in used:
                if abs(r - ur) + abs(c - uc) < 5:
                    ok = False
                    break
            if not ok:
                continue
            # keep 3x3 outline inside bounds
            if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
                continue
            g[r, c] = col
            used.append((r, c))
            break
        if len(used) < chosen.index(col) + 1:
            # unlikely fallback
            r, c = 2 + chosen.index(col), 2 + 3 * chosen.index(col)
            g[r, c] = col
            used.append((r, c))
    return g

