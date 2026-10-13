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


def _ordered_path(comp: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not comp:
        return []
    comp_set = set(comp)
    deg = {p: 0 for p in comp}
    for r, c in comp:
        d = 0
        for dr, dc in _DIRS:
            if (r + dr, c + dc) in comp_set:
                d += 1
        deg[(r, c)] = d

    start = comp[0]
    for p, d in deg.items():
        if d == 1:
            start = p
            break

    path = [start]
    used = {start}
    prev = None
    cur = start
    while True:
        nxt = None
        for dr, dc in _DIRS:
            cand = (cur[0] + dr, cur[1] + dc)
            if cand in comp_set and cand != prev and cand not in used:
                nxt = cand
                break
        if nxt is None:
            break
        used.add(nxt)
        path.append(nxt)
        prev, cur = cur, nxt
        if len(path) == len(comp):
            break
    if len(path) != len(comp):
        # fallback: append remaining points in deterministic order
        rest = [p for p in comp if p not in used]
        rest.sort()
        path.extend(rest)
    return path


def _count_bends(path: List[Tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(2, len(path)):
        cur = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if cur != prev:
            turns += 1
        prev = cur
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    segments = []

    mask = g != 0
    comps = _components(mask)
    if not comps:
        return np.zeros((1, 1), dtype=int)

    comps.sort(key=lambda comp: min(c for r, c in comp))
    out_chain = np.zeros_like(g)
    col_cursor = 0
    prev_right_row = None
    segment_counts = []
    for comp in comps:
        color = int(g[comp[0][0], comp[0][1]])
        path = _ordered_path(comp)
        bends = _count_bends(path)
        segment_counts.append((color, bends))

        min_col = min(c for _, c in comp)
        left_pts = [(r, c) for r, c in comp if c == min_col]
        right_pts = [(r, c) for r, c in comp if c == max(c for _, c in comp)]
        left_row = left_pts[0][0]
        right_row = right_pts[0][0]
        shift_r = 0
        if prev_right_row is not None:
            shift_r = prev_right_row - left_row

        # keep inside.
        comp_rows = [r + shift_r for r, _ in comp]
        if min(comp_rows) < 0:
            shift_r += -min(comp_rows)
        if max(comp_rows) >= h:
            shift_r -= max(comp_rows) - (h - 1)

        comp_cols = [c - min_col for _, c in comp]
        width = max(comp_cols) + 1
        if col_cursor + width > w:
            break

        for r, c in comp:
            tr = r + shift_r
            tc = c - min_col + col_cursor
            if 0 <= tr < h and 0 <= tc < w:
                out_chain[tr, tc] = color

        prev_right_row = right_row + shift_r
        col_cursor += width + 1

    if col_cursor == 0:
        return np.zeros((1, 1), dtype=int)

    max_bends = max(b for _, b in segment_counts) if segment_counts else 0
    if max_bends == 0:
        max_bends = 1
    compact = np.zeros((max_bends, len(segment_counts)), dtype=int)
    for i, (color, bends) in enumerate(segment_counts):
        if bends <= 0:
            continue
        for rr in range(max_bends - bends, max_bends):
            compact[rr, i] = color
    return compact


def generate_input() -> np.ndarray:
    h = 7
    w = 25
    g = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3, 4, 5]
    col = 0
    for i in range(random.randint(3, 5)):
        color = colors[i % len(colors)]
        run1 = random.randint(2, 4)
        run2 = random.randint(1, 2)
        run3 = random.randint(2, 4)
        r = random.randint(1, h - 2)
        c_local = 0
        points = []
        for _ in range(run1):
            points.append((r, col + c_local))
            c_local += 1
        dir_v = random.choice([-1, 1])
        for _ in range(run2):
            r += dir_v
            if r < 0:
                r = 1
            if r >= h:
                r = h - 2
            points.append((r, col + c_local))
            c_local += 1
        dir_h = random.choice([-1, 1])
        for _ in range(run3):
            c_local += dir_h
            if c_local < 0:
                c_local = 0
            points.append((r, col + c_local))
        for rr, cc in points:
            if 0 <= rr < h and 0 <= cc < w:
                g[rr, cc] = color
        # add separator
        col = max(cc for _, cc in points) + 2
        if col + 8 >= w:
            break
    return g
