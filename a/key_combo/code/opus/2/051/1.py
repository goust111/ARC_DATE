from __future__ import annotations

import math
import random
from collections import deque
from typing import List, Tuple

import numpy as np


BG = 0
ORANGE = 7
MARKER = 8


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in _DIRS:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _angle_sorted_regions(
    regions: List[Tuple[Tuple[float, float], List[Tuple[int, int]], int]],
    origin: Tuple[float, float],
) -> List[Tuple[List[Tuple[int, int]], int]]:
    or_, oc = origin
    out: List[Tuple[Tuple[float, float], List[Tuple[int, int]], int]] = []
    for centroid, cells, col in regions:
        rr, cc = centroid
        angle = math.atan2(rr - or_, cc - oc)
        out.append((angle, cells, col))
    out.sort(key=lambda x: x[0])
    return [(cells, col) for _, cells, col in out]


def _cast_ray_colors(
    grid: np.ndarray, start: Tuple[int, int], dr: int, dc: int
) -> Tuple[int, int]:
    h, w = grid.shape
    r, c = start
    dist = 0
    colors = set()
    first = None
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        dist += 1
        v = int(grid[r, c])
        if v not in (BG, ORANGE, MARKER):
            colors.add(v)
            if first is None:
                first = dist
    return len(colors), (first if first is not None else 10**9)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()

    marker_pos = np.argwhere(g == MARKER)
    if marker_pos.size == 0:
        return out
    mr, mc = map(int, marker_pos[0])

    orange_mask = g == ORANGE
    if not orange_mask.any():
        return out
    orange_coords = np.argwhere(orange_mask)
    origin = (float(orange_coords[:, 0].mean()), float(orange_coords[:, 1].mean()))

    rays = {
        "up": _cast_ray_colors(g, (mr, mc), -1, 0),
        "down": _cast_ray_colors(g, (mr, mc), 1, 0),
        "left": _cast_ray_colors(g, (mr, mc), 0, -1),
        "right": _cast_ray_colors(g, (mr, mc), 0, 1),
    }
    k = max(
        range(len(rays)),
        key=lambda i: (list(rays.values())[i][0], -list(rays.values())[i][1]),
    )
    direction = list(rays.keys())[k]
    step_count = list(rays.values())[k][0]

    movable = g != BG
    movable &= g != ORANGE
    movable &= g != MARKER
    comps = _components(movable)
    if not comps:
        return out

    regions: List[Tuple[Tuple[float, float], List[Tuple[int, int]], int]] = []
    for comp in comps:
        colors = [g[r, c] for r, c in comp]
        color = max(set(colors), key=colors.count)
        if color in (BG, ORANGE, MARKER):
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        regions.append(((float(sum(rs)) / len(rs), float(sum(cs)) / len(cs)), comp, int(color)))
    if not regions:
        return out

    ordered_regions = _angle_sorted_regions(regions, origin)
    colors = [col for _, col in ordered_regions]
    if not colors:
        return out
    mapped = [colors[(i + step_count) % len(colors)] for i in range(len(colors))]

    for (cells, _), new_color in zip(ordered_regions, mapped):
        for r, c in cells:
            out[r, c] = int(new_color)

    return out


def generate_input() -> np.ndarray:
    while True:
        h = 15
        w = 15
        g = np.zeros((h, w), dtype=int)
        # Separator lines: outer frame.
        g[0, :] = ORANGE
        g[-1, :] = ORANGE
        g[:, 0] = ORANGE
        g[:, -1] = ORANGE

        # Random internal vertical/horizontal separators.
        for c in random.sample(range(2, w - 2), random.randint(1, 2)):
            g[:, c] = ORANGE
        for r in random.sample(range(2, h - 2), random.randint(1, 2)):
            g[r, :] = ORANGE

        # Fill each non-orange connected background region with a unique area color.
        palette = [1, 2, 3, 4, 5, 6]
        for i, comp in enumerate(_components(g != ORANGE)):
            if i >= len(palette):
                break
            for r, c in comp:
                g[r, c] = palette[i % len(palette)]

        marker_candidates = []
        for r, c in np.argwhere(g == ORANGE):
            r, c = int(r), int(c)
            if (r > 0 and g[r - 1, c] != ORANGE) or (
                r + 1 < h and g[r + 1, c] != ORANGE
            ) or (c > 0 and g[r, c - 1] != ORANGE) or (c + 1 < w and g[r, c + 1] != ORANGE):
                marker_candidates.append((r, c))
        if not marker_candidates:
            continue
        mr, mc = random.choice(marker_candidates)
        g[mr, mc] = MARKER
        return g
