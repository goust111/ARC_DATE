"""Trigger rectangle recoloring by diagonal connectors between green crosses.

Input structure:
  A black background with several green (3) cross shapes and several large solid
  rectangles in colors like blue(1), red(2), orange(7). Crosses may be on any
  diagonal lines.

Rule:
  Keep all input pixels. For every pair of green crosses whose centers are on the
  same 45-degree diagonal, if all intermediate cells are black, paint that
  diagonal segment red (2). Then, for each rectangle, if its strict interior
  contains any red pixel, recolor its strict interior to teal (8) while preserving
  the rectangle border and preserving interior red connector pixels.

Key insight:
  The drawn diagonal connectors determine a second-stage fill condition; only
  interiors pierced by these connectors are recolored.
"""

from __future__ import annotations

import random
from typing import List, Set, Tuple

import numpy as np


def _find_plus_centers(grid: np.ndarray, bg: int = 0) -> List[Tuple[int, int]]:
    H, W = grid.shape
    centers = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != 3:
                continue
            if (
                int(grid[r - 1, c]) == 3
                and int(grid[r + 1, c]) == 3
                and int(grid[r, c - 1]) == 3
                and int(grid[r, c + 1]) == 3
            ):
                centers.append((r, c))
    return centers


def _components_by_color(mask: np.ndarray) -> List[Tuple[set, Tuple[int, int, int, int], int]]:
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    out: List[Tuple[set, Tuple[int, int, int, int], int]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, c0 = min(rows), min(cols)
            r1, c1 = max(rows), max(cols)
            out.append((set(comp), (r0, c0, r1, c1), len(comp)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g.copy()
    H, W = g.shape
    bg = 0

    centers = _find_plus_centers(g, bg=bg)
    line_cells = set()
    for i, (r1, c1) in enumerate(centers):
        for j in range(i + 1, len(centers)):
            r2, c2 = centers[j]
            dr = r2 - r1
            dc = c2 - c1
            if abs(dr) != abs(dc) or dr == 0:
                continue
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1
            k = abs(dr)
            if k <= 1:
                continue
            ok = True
            path = []
            for t in range(1, k):
                rr, cc = r1 + t * step_r, c1 + t * step_c
                if g[rr, cc] != bg:
                    ok = False
                    break
                path.append((rr, cc))
            if ok:
                line_cells.update(path)

    out = g.copy()
    for r, c in line_cells:
        out[r, c] = 2

    rect_mask = (g != 0) & (g != 3)
    comps = _components_by_color(rect_mask)
    for comp, (r0, c0, r1, c1), area in comps:
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        if area != h * w:
            continue
        hit = False
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if (rr, cc) in line_cells:
                    hit = True
                    break
            if hit:
                break
        if not hit:
            continue
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if (rr, cc) not in line_cells:
                    out[rr, cc] = 8
    return out


def _draw_plus(grid: np.ndarray, r: int, c: int, color: int = 3) -> None:
    for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        grid[r + dr, c + dc] = color


def generate_input() -> np.ndarray:
    H = random.randint(16, 20)
    W = H
    g = np.zeros((H, W), dtype=np.int64)

    # place rectangles first to create background area
    rect_colors = [1, 2, 7]
    rects = []
    tries = 0
    while len(rects) < random.randint(3, 5) and tries < 400:
        tries += 1
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        r = random.randint(1, H - h - 2)
        c = random.randint(1, W - w - 2)
        rr = range(r - 1, r + h + 1)
        cc = range(c - 1, c + w + 1)
        if any(g[ri, ci] != 0 for ri in rr for ci in cc):
            continue
        color = random.choice(rect_colors)
        g[r : r + h, c : c + w] = color
        rects.append((r, c, h, w, color))

    # place pluses
    n = random.randint(3, 5)
    centers = []
    attempts = 0
    max_attempts = 2000
    while len(centers) < n and attempts < max_attempts:
        attempts += 1
        # avoid invalid ranges on small canvases
        if H <= 5 or W <= 5:
            break
        r = random.randint(2, H - 3)
        c = random.randint(2, W - 3)
        if g[r, c] != 0:
            continue
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if g[rr, cc] != 0:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        _draw_plus(g, r, c)
        centers.append((r, c))
    if len(centers) < 2:
        # make generation stable: deterministic minimal plus set
        fixed = [(3, 3), (H - 4, W - 4), (H // 2, W // 2)]
        for pr, pc in fixed:
            if 1 < pr < H - 1 and 1 < pc < W - 1 and g[pr, pc] == 0:
                if all(g[pr + dr, pc + dc] == 0 for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                    _draw_plus(g, pr, pc)
                    if (pr, pc) not in centers:
                        centers.append((pr, pc))

    # ensure at least one clear diagonal pair, and not too many crossings
    if len(centers) >= 2:
        placed = False
        for _ in range(200):
            r1, c1 = random.choice(centers)
            step = random.randint(3, 6)
            sign = random.choice([-1, 1])
            r2, c2 = r1 + step, c1 + step * sign
            if not (2 <= r2 < H - 2 and 2 <= c2 < W - 2):
                continue
            if g[r2, c2] != 0:
                continue
            _draw_plus(g, r2, c2)
            centers.append((r2, c2))
            placed = True
            # keep path-clear by forcing background on intermediate cells
            step = 1 if c2 > c1 else -1
            for t in range(1, abs(r2 - r1)):
                rr, cc = r1 + t * (1 if r2 > r1 else -1), c1 + t * step
                if 0 <= rr < H and 0 <= cc < W:
                    g[rr, cc] = 0
            break
        if not placed:
            # fallback: do not rely on clear path guarantee
            pass
    return g
