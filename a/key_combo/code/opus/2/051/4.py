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


def _centroid_row(comp: List[Tuple[int, int]]) -> float:
    return float(sum(r for r, _ in comp)) / len(comp)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    mono_rows = {}
    for r in range(h):
        row = g[r]
        vals = {int(v) for v in row if v != 0}
        if len(vals) == 1:
            mono_rows[r] = int(next(iter(vals)))

    if not mono_rows:
        return out

    mask = np.zeros_like(g, dtype=bool)
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v != 0 and r not in mono_rows:
                mask[r, c] = True
            elif v != 0 and r in mono_rows and v != mono_rows[r]:
                mask[r, c] = True

    mono_row_list = list(sorted(mono_rows.keys()))

    for comp in _components(mask):
        if len(comp) < 1:
            continue
        color = int(g[comp[0][0], comp[0][1]])
        cen_r = _centroid_row(comp)
        nearest = min(mono_row_list, key=lambda rr: abs(cen_r - rr))

        mirrored = []
        for r, c in comp:
            tr = 2 * nearest - r
            tc = c
            if not (0 <= tr < h and 0 <= tc < w):
                continue
            if tr in mono_rows:
                continue
            mirrored.append((tr, tc))

        orig_set = set(comp)
        for tr, tc in mirrored:
            if (tr, tc) in orig_set:
                out[tr, tc] = 7
            elif out[tr, tc] == 0:
                out[tr, tc] = color
            else:
                out[tr, tc] = 7

    return out


def _paint_path(g: np.ndarray, points: List[Tuple[int, int]], color: int) -> None:
    for r, c in points:
        g[r, c] = color


def generate_input() -> np.ndarray:
    h = 13
    w = 13
    g = np.zeros((h, w), dtype=int)

    mono_colors = [1, 2, 3, 4]
    rows = random.sample(range(2, h - 2), random.randint(2, 3))
    rows.sort()
    for i, r in enumerate(rows):
        g[r, :] = mono_colors[i % len(mono_colors)]

    sprite_colors = [6, 2, 3, 4]
    random.shuffle(sprite_colors)
    for si in range(random.randint(2, 3)):
        color = sprite_colors[si % len(sprite_colors)]
        # Keep sprite at least one row away from all mono rows.
        for _ in range(200):
            r0 = random.randint(0, h - 5)
            c0 = random.randint(0, w - 4)
            base_row = random.randint(0, h - 3)
            points = [
                (base_row, c0),
                (base_row, c0 + 1),
                (base_row + 1, c0 + 1),
                (base_row + 2, c0 + 1),
                (base_row + 2, c0 + 2),
            ]
            if any(g[r, c] != 0 for r, c in points):
                continue
            if any(p[0] in rows for p in points):
                continue
            _paint_path(g, points, color)
            break
    return g
