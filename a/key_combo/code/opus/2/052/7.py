from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = [(r, c)]
            while q:
                x, y = q.pop()
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        q.append((nx, ny))
                        comp.append((nx, ny))
            comps.append(comp)
    return comps


def _max_square_candidates(region: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int, int]]:
    comp = region
    rows = [p[0] for p in comp]
    cols = [p[1] for p in comp]
    if not rows:
        return []
    comp_set = set(comp)
    rmin, cmin = min(rows), min(cols)
    rmax, cmax = max(rows), max(cols)
    max_side = min(rmax - rmin + 1, cmax - cmin + 1)

    local = np.zeros((rmax - rmin + 2, cmax - cmin + 2), dtype=int)
    for r, c in comp:
        local[r - rmin, c - cmin] = 1
    ps = local.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, side: int) -> int:
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        return int(
            ps[r1 + 1, c1 + 1]
            - (ps[r0, c1 + 1] if r0 >= 0 else 0)
            - (ps[r1 + 1, c0] if c0 >= 0 else 0)
            + (ps[r0, c0] if r0 >= 0 and c0 >= 0 else 0)
        )

    cands: list[tuple[int, int, int]] = []
    for side in range(2, max_side + 1):
        for r0 in range(rmin, rmax - side + 2):
            for c0 in range(cmin, cmax - side + 2):
                if rect_sum(r0 - rmin, c0 - cmin, side) == side * side:
                    cands.append((side, r0, c0))
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))
    return cands


def _nearest_wall_color(square: tuple[int, int, int], walls: set[tuple[int, int]], out: np.ndarray) -> int:
    side, r0, c0 = square
    r1, c1 = r0 + side - 1, c0 + side - 1
    edge: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        edge.append((r0, c))
        edge.append((r1, c))
    for r in range(r0 + 1, r1):
        edge.append((r, c0))
        edge.append((r, c1))

    bestd = 10**9
    best_pos = min(walls)
    for wr, wc in walls:
        for er, ec in edge:
            d = max(abs(er - wr), abs(ec - wc))
            if d < bestd or (d == bestd and (wr < best_pos[0] or (wr == best_pos[0] and wc < best_pos[1]))):
                bestd = d
                best_pos = (wr, wc)
    return int(out[best_pos[0], best_pos[1]])


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    h, w = out.shape
    markers = list(map(tuple, np.argwhere(out > 0)))
    markers.sort()

    walls: set[tuple[int, int]] = set(markers)
    # ray fill in scan order
    for r0, c0 in markers:
        color = int(out[r0, c0])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = r0 + dr, c0 + dc
            while 0 < r < h - 1 and 0 < c < w - 1:
                out[r, c] = color
                walls.add((r, c))
                r += dr
                c += dc

    # restore markers
    for r0, c0 in markers:
        out[r0, c0] = int(grid[r0, c0])

    free = out == 0
    comps = _components(free)
    if not comps or not walls:
        return out

    wall_list = sorted(walls)
    wall_set = set(wall_list)

    for comp in comps:
        if len(comp) < 4:
            continue
        candidates = _max_square_candidates(comp, h, w)
        occupied: set[tuple[int, int]] = set()
        for side, r0, c0 in candidates:
            overlap = False
            for r in range(r0, r0 + side):
                for c in range(c0, c0 + side):
                    if (r, c) in occupied:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                continue
            color = _nearest_wall_color((side, r0, c0), wall_set, out)
            for r in range(r0, r0 + side):
                for c in range(c0, c0 + side):
                    occupied.add((r, c))
                    out[r, c] = color

    for r0, c0 in markers:
        out[r0, c0] = int(grid[r0, c0])

    return out


def _simulate_region_count(grid: np.ndarray) -> tuple[int, int]:
    out = np.array(grid, copy=True)
    h, w = out.shape
    markers = list(map(tuple, np.argwhere(out > 0)))
    if not markers:
        return 0, 0
    walls: set[tuple[int, int]] = set(markers)
    for r0, c0 in markers:
        color = int(out[r0, c0])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = r0 + dr, c0 + dc
            while 0 < r < h - 1 and 0 < c < w - 1:
                out[r, c] = color
                walls.add((r, c))
                r += dr
                c += dc
    comps = _components(out == 0)
    if not comps:
        return 0, 0
    areas = [len(c) for c in comps]
    return len(comps), min(areas) if areas else 0


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 19
    while True:
        out = np.zeros((h, w), dtype=int)
        colors = [2, 3, 4]
        markers: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        tries = 0
        for color in colors:
            for _ in range(3):
                tries = 0
                while tries < 200:
                    tries += 1
                    r = int(rng.integers(1, h - 1))
                    c = int(rng.integers(1, w - 1))
                    if r in used_rows or c in used_cols or out[r, c] != 0:
                        continue
                    used_rows.add(r)
                    used_cols.add(c)
                    markers.append((r, c))
                    out[r, c] = color
                    break
        n_regions, min_area = _simulate_region_count(out)
        if n_regions >= 4 and min_area >= 4:
            return out
        # fallback: force a known good pattern
        if rng.random() < 0.05:
            out = np.zeros((h, w), dtype=int)
            out[4, 5] = 2
            out[13, 7] = 3
            out[8, 13] = 4
            return out

