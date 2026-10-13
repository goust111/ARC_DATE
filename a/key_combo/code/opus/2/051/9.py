from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _components(grid: np.ndarray, val: int) -> List[List[Tuple[int, int]]]:
    mask = grid == val
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
                for dr, dc in _DIRS4:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _comp_cells(comp: List[Tuple[int, int]], g: np.ndarray, val: int) -> bool:
    if not comp:
        return False
    return all(g[r, c] == val for r, c in comp)


def _manhattan(comp: List[Tuple[int, int]], cell: Tuple[int, int]) -> int:
    return min(abs(r - cell[0]) + abs(c - cell[1]) for r, c in comp)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    gray_comps = _components(g, 5)
    red_comps = _components(g, 2)

    out = g.copy()
    for gc in gray_comps:
        if not gc:
            continue
        marker = None
        best_d = 10**9
        for rc in red_comps:
            if not rc:
                continue
            d = min(abs(r - pr) + abs(c - pc) for pr, pc in rc for r, c in gc)
            if d < best_d:
                best_d = d
                marker = rc
        if marker is None:
            continue

        marker_set = set(marker)
        if len(marker_set) == 1:
            anchor = next(iter(marker_set))
            ar, ac = anchor
            for r, c in gc:
                tr = 2 * ar - r
                tc = 2 * ac - c
                if 0 <= tr < h and 0 <= tc < w:
                    out[tr, tc] = 5
        else:
            rows = {r for r, _ in marker}
            cols = {c for _, c in marker}
            if len(rows) == 1:
                ar = next(iter(rows))
                for r, c in gc:
                    tr = 2 * ar - r
                    tc = c
                    if 0 <= tr < h and 0 <= tc < w:
                        out[tr, tc] = 5
            elif len(cols) == 1:
                ac = next(iter(cols))
                for r, c in gc:
                    tr = r
                    tc = 2 * ac - c
                    if 0 <= tr < h and 0 <= tc < w:
                        out[tr, tc] = 5

    # Remove markers.
    out[g == 2] = 0

    # Find enclosed black regions.
    black = out == 0
    visited = np.zeros_like(black, dtype=bool)
    regions: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not black[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp: List[Tuple[int, int]] = []
            touches = False
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                if x in (0, h - 1) or y in (0, w - 1):
                    touches = True
                for dr, dc in _DIRS4:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if not touches:
                regions.append(comp)

    regions.sort(key=lambda comp: len(comp))
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    for idx, comp in enumerate(regions):
        color = palette[idx % len(palette)]
        for r, c in comp:
            out[r, c] = color

    return out


def generate_input() -> np.ndarray:
    h = 16
    w = 16
    g = np.zeros((h, w), dtype=int)

    def place(coords, origin, marker, marker_type):
        ox, oy = origin
        for dr, dc in coords:
            g[ox + dr, oy + dc] = 5
        if marker_type == "line":
            for mr, mc in marker:
                g[ox + mr, oy + mc] = 2
        else:
            r, c = marker[0]
            g[ox + r, oy + c] = 2

    # Keep 2-3 sprites and randomize placement.
    sprite_count = random.randint(2, 3)
    types = ["line", "line", "point"]
    random.shuffle(types)
    types = types[:sprite_count]

    used = set()
    for marker_type in types:
        if marker_type == "line":
            gray = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (3, 2), (2, 2)]
            marker = [(1, 1), (1, 2)]
        else:
            gray = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
            marker = [(1, 1)]

        # find top-left so sprite fits and disjoint
        for _ in range(200):
            ox = random.randint(0, h - 6)
            oy = random.randint(0, w - 6)
            area = [(ox + r, oy + c) for r, c in gray]
            if any((r, c) in used for r, c in area):
                continue
            for r, c in area:
                g[r, c] = 5
            for r, c in marker:
                g[ox + r, oy + c] = 2
            used.update(area)
            used.update((ox + r, oy + c) for r, c in marker)
            break

    return g
