from __future__ import annotations

import numpy as np
from collections import deque
import random


BLACK = 0


def _neighbors8(r: int, c: int, H: int, W: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc


def _components(mask: np.ndarray, neigh="8"):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for rr, cc in _neighbors8(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _sort_outlines(comps):
    out = []
    for comp in comps:
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        out.append((r0, c0, comp))
    return sorted(out, key=lambda x: (x[0], x[1]))


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    H, W = grid.shape
    out = np.array(grid, copy=True)

    mask = (grid != BLACK)
    comps = _components(mask, neigh="8")
    outlines = [c for c in comps if len(c) > 0]
    if not outlines:
        return out

    sorted_outlines = _sort_outlines(outlines)
    original = {i: comp for i, (_, _, comp) in enumerate(sorted_outlines)}

    # fill in order
    for i, (_, _, comp) in enumerate(sorted_outlines):
        color = int(grid[comp[0, 0], comp[0, 1]])
        rows = sorted(set(int(r) for r in comp[:, 0]))
        for r in rows:
            cols = [int(c) for rr, c in comp if int(rr) == r]
            if len(cols) < 2:
                continue
            c0, c1 = min(cols), max(cols)
            out[r, c0 : c1 + 1] = color

    # select outline whose bbox has most unique colors after fill
    best_idx = None
    best_cnt = -1
    for i, comp in original.items():
        r0 = int(comp[:, 0].min())
        r1 = int(comp[:, 0].max())
        c0 = int(comp[:, 1].min())
        c1 = int(comp[:, 1].max())
        unique_colors = np.unique(out[r0 : r1 + 1, c0 : c1 + 1])
        n = len(unique_colors)
        if n > best_cnt:
            best_cnt = n
            best_idx = i

    if best_idx is None:
        return out
    comp = original[best_idx]
    r0 = int(comp[:, 0].min())
    r1 = int(comp[:, 0].max())
    c0 = int(comp[:, 1].min())
    c1 = int(comp[:, 1].max())
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _draw_outline(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int):
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def generate_input() -> np.ndarray:
    out = np.zeros((15, 15), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    _draw_outline(out, 1, 1, 10, 7, colors[0])
    _draw_outline(out, 4, 4, 12, 11, colors[1])
    _draw_outline(out, 2, 8, 9, 13, colors[2])
    return out
