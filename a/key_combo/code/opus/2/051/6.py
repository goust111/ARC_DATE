from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in _DIRS:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _touches_border(comp: List[Tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = []

    marker_pos = np.argwhere(g == 5)
    if marker_pos.size == 0:
        return g.copy()
    center = tuple(int(v) for v in marker_pos[0])

    comps = _components(g != 7)
    regions = []
    for comp in comps:
        if _touches_border(comp, h, w):
            continue
        black = [p for p in comp if g[p[0], p[1]] == 0]
        if not black:
            continue
        marker_pixels = [p for p in comp if g[p[0], p[1]] not in (0, 5, 7)]
        if not marker_pixels:
            continue
        marker_color = int(g[marker_pixels[0][0], marker_pixels[0][1]])
        areas = len(black)
        cr = sum(r for r, _ in black) / len(black)
        cc = sum(c for _, c in black) / len(black)
        regions.append((areas, marker_color, cr, cc))

    if not regions:
        return g.copy()

    regions.sort(key=lambda t: np.arctan2(t[2] - center[0], t[3] - center[1]))
    N = len(regions)
    colors = [c for _, c, _, _ in regions]

    best_r = 1
    best_c = N
    for R in range(1, N + 1):
        C = (N + R - 1) // R
        if R > C:
            continue
        if R * C < best_r * best_c:
            best_r, best_c = R, C

    out_grid = np.zeros((best_r, best_c), dtype=int)
    seq = []
    for idx, color in enumerate(colors):
        seq.append(color)

    for i, color in enumerate(seq):
        r = i // best_c
        c0 = i % best_c
        if r >= best_r:
            break
        c = c0
        if r % 2 == 1:
            c = best_c - 1 - c0
        out_grid[r, c] = color
    return out_grid


def generate_input() -> np.ndarray:
    h = 15
    w = 15
    g = np.zeros((h, w), dtype=int)
    g[h // 2, w // 2] = 5

    colors = [1, 2, 3, 4, 6, 8, 9, 0, 11]
    random.shuffle(colors)
    count = random.choice([4, 6, 7])
    colors = colors[:count]

    used: set[Tuple[int, int]] = set()
    for i in range(count):
        color = colors[i]
        for _ in range(200):
            hi = random.randint(4, 6)
            wi = random.randint(4, 6)
            r0 = random.randint(1, h - hi - 2)
            c0 = random.randint(1, w - wi - 2)
            r1 = r0 + hi - 1
            c1 = c0 + wi - 1
            cells = set()
            # frame border
            for cc in range(c0, c1 + 1):
                cells.add((r0, cc))
                cells.add((r1, cc))
            for rr in range(r0, r1 + 1):
                cells.add((rr, c0))
                cells.add((rr, c1))
            if any(p in used for p in cells):
                continue
            if r0 <= h // 2 <= r1 and c0 <= w // 2 <= c1:
                continue
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    g[rr, cc] = 7
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    g[rr, cc] = 0
            mr = random.randint(r0 + 1, r1 - 1)
            mc = random.randint(c0 + 1, c1 - 1)
            g[mr, mc] = color
            used.update(cells)
            break
    return g
