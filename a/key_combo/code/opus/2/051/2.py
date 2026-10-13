from __future__ import annotations

import random
from collections import Counter, deque
from typing import List, Sequence, Tuple

import numpy as np


COL_STAN = 9
COL_ORANGE = 7
COL_BLACK = 0
COL_CYAN = 8
COL_BG = 0
COL_GRAY = 5

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(grid: np.ndarray, value: int) -> List[List[Tuple[int, int]]]:
    mask = grid == value
    h, w = grid.shape
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


def _bbox(comp: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_comp(comp: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    m = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        m[r - r0, c - c0] = True
    return m, h, w


def _centroid(comp: Sequence[Tuple[int, int]]) -> Tuple[float, float]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return float(sum(rs)) / len(rs), float(sum(cs)) / len(cs)


def _shift_ok(mask: np.ndarray, top: int, left: int, out_shape: Tuple[int, int]) -> bool:
    h, w = out_shape
    mh, mw = mask.shape
    if top < 0 or left < 0:
        return False
    return top + mh <= h and left + mw <= w


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    stan = _components(g, COL_STAN)
    if len(stan) < 3:
        return out

    def is_ref(comp: Sequence[Tuple[int, int]]) -> bool:
        for r, c in comp:
            if c + 1 < w and g[r, c + 1] == COL_GRAY and r + 1 < h and g[r + 1, c] == COL_GRAY:
                return True
        return False

    ref = None
    for comp in stan:
        if is_ref(comp):
            ref = comp
            break
    if ref is None:
        return out

    ref_mask, _, _ = _mask_from_comp(ref)
    matches: List[List[Tuple[int, int]]] = []
    for comp in stan:
        if comp is ref:
            continue
        m, _, _ = _mask_from_comp(comp)
        if m.shape != ref_mask.shape:
            continue
        if np.array_equal(m, ref_mask):
            matches.append(comp)
    if len(matches) < 2:
        return out

    a, b = matches[0], matches[1]
    ra0, ca0 = _centroid(a)
    rb0, cb0 = _centroid(b)
    mid_r = int(round((ra0 + rb0) * 0.5))
    mid_c = int(round((ca0 + cb0) * 0.5))

    a_mask, mh, mw = _mask_from_comp(a)
    nor_mask = ~ (a_mask | _mask_from_comp(b)[0])
    start_r = int(round(mid_r - mh / 2))
    start_c = int(round(mid_c - mw / 2))

    # Place onto a copy of non-destructively painting only generated cyan pixels.
    placed = False
    for shift in range(max(h, w)):
        tr = start_r + shift
        tc = start_c
        if not _shift_ok(nor_mask, tr, tc, (h, w)):
            continue
        conflict = False
        for rr, cc in np.argwhere(nor_mask):
            rr, cc = int(rr), int(cc)
            if out[tr + rr, tc + cc] != COL_BLACK:
                conflict = True
                break
        if not conflict:
            for rr, cc in np.argwhere(nor_mask):
                rr, cc = int(rr), int(cc)
                out[tr + rr, tc + cc] = COL_CYAN
            placed = True
            break
    _ = placed
    return out


def generate_input() -> np.ndarray:
    h = 18
    w = 18
    g = np.zeros((h, w), dtype=int)

    shape = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2)]

    def place_shape(top: int, left: int) -> List[Tuple[int, int]]:
        for dr, dc in shape:
            g[top + dr, left + dc] = COL_STAN
        return [(top + dr, left + dc) for dr, dc in shape]

    # Reference pattern with L-stencil marker.
    ref_top = random.randint(1, 10)
    ref_left = random.randint(2, 12)
    r0 = ref_top
    c0 = ref_left
    place_shape(r0, c0)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    # Right gray column and bottom gray row.
    for rr in range(r0, r0 + max_r + 1):
        g[rr, c0 + max_c + 1] = COL_GRAY
    for cc in range(c0, c0 + max_c + 1):
        g[r0 + max_r + 1, cc] = COL_GRAY

    # Match candidates.
    place_shape(random.randint(8, 11), random.randint(1, 10))
    place_shape(random.randint(8, 11), random.randint(11, 14))

    distract = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    off_r = random.randint(6, 11)
    off_c = random.randint(1, 15)
    for dr, dc in distract:
        if off_r + dr < h and off_c + dc < w:
            g[off_r + dr, off_c + dc] = COL_STAN

    return g
