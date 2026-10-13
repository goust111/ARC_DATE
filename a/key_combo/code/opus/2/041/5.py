from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
GRAY = Color.GREY


def main(grid: np.ndarray) -> np.ndarray:
    """Build concentric 9x9 frames from two independent ranking chains.

    Input structure: four non-touching gray 4-connected sprites with distinct areas, each with distinct bbox center columns.

    Rule: rank sprites by area descending for nesting order (largest outer), rank by bbox center column for color assignment
    [blue, red, green, yellow], then draw four centered concentric one-pixel frames in a 9x9 output.

    Key insight: geometry depth and color mapping follow independent orders; the area order cannot be inferred from position order.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")

    comps = _components(g, int(GRAY))
    if len(comps) != 4:
        return np.zeros((9, 9), dtype=int)

    areas = [len(c) for c in comps]
    if len(set(areas)) != 4:
        return np.zeros((9, 9), dtype=int)

    area_order = sorted(range(4), key=lambda i: areas[i], reverse=True)
    x_order = sorted(range(4), key=lambda i: _bbox_center(comps[i])[1])

    color_by_pos = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW)]
    sprite_colors = [None] * 4
    for rank, idx in enumerate(x_order):
        sprite_colors[idx] = color_by_pos[rank]

    out = np.zeros((9, 9), dtype=int)
    sizes = [9, 7, 5, 3]
    for k, idx in enumerate(area_order):
        color = sprite_colors[idx]
        if color is None:
            return np.zeros((9, 9), dtype=int)
        _draw_frame(out, 0, 0, sizes[k] - 1, sizes[k] - 1, color)

    return out


def _components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in dirs:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(grid[nx, ny]) == int(color):
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox_center(comp: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return ((min(rs) + max(rs)) / 2.0, (min(cs) + max(cs)) / 2.0)


def _draw_frame(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    for c in range(c0, c1 + 1):
        out[r0, c] = color
        out[r1, c] = color
    for r in range(r0 + 1, r1):
        out[r, c0] = color
        out[r, c1] = color


def generate_input() -> np.ndarray:
    h, w = 18, 18
    g = np.full((h, w), int(BLACK), dtype=int)

    # target four distinct areas
    areas = [4, 7, 11, 16]
    random.shuffle(areas)
    sprites: list[tuple[int, int, int, int]] = []

    attempts = 0
    while len(sprites) < 4 and attempts < 200:
        attempts += 1
        area = areas[len(sprites)]
        # draw irregular blob roughly within a small bbox
        hlim = min(4, area)
        r0 = random.randint(1, h - hlim - 2)
        c0 = random.randint(1, w - hlim - 2)
        cells = {(r0, c0)}
        frontier = [(r0, c0)]
        while len(cells) < area:
            cr, cc = random.choice(list(frontier))
            neigh = [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
            neigh = [(rr, cc0) for rr, cc0 in neigh if 0 < rr < h - 1 and 0 < cc0 < w - 1 and (rr, cc0) not in cells]
            if not neigh:
                break
            nr, nc = random.choice(neigh)
            cells.add((nr, nc))
            frontier.append((nr, nc))
        if len(cells) < area:
            continue
        for rr, cc in cells:
            g[rr, cc] = int(GRAY)
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        cc = (min(rs) + max(rs)) / 2
        cr = (min(cs) + max(cs)) / 2
        ok = True
        for ar, ac, ar2, ac2, _ in sprites:
            cc2 = (ac + ac2) / 2
            cr2 = (ar + ar2) / 2
            if abs(cc - cc2) < 2 and abs(cr - cr2) < 2:
                ok = False
        if not ok:
            # undo
            for rr, cc in cells:
                g[rr, cc] = int(BLACK)
            continue

        sprites.append((min(rs), min(cs), max(rs), max(cs), area))

    # fallback if not enough shapes (very unlikely)
    if len(sprites) < 4:
        for _ in range(4 - len(sprites)):
            r = random.randint(1, h - 3)
            c = random.randint(1, w - 3)
            g[r, c] = int(GRAY)

    return g
