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


def main(grid: np.ndarray) -> np.ndarray:
    """Build a dynamic color map from four top-half sprites and recolor matching markers in the lower half.

    Input structure: four 8-connected 3x3-bounded sprites in top half with colors {2,1,3,7} and lower-half markers with
    these same colors.

    Rule: order sprites by row (upper pair then lower pair) and within pair by column (left/right), map their positions to
    blue/red/green/yellow colors, erase all top-half sprites, and recolor lower-half markers according to the map.

    Key insight: the sprite atlas is only an ordering table generator; the mapping depends on sprite positions, not on fixed colors.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")
    h, w = g.shape

    sprites = _components(g)
    sprites_top = [s for s in sprites if float(s["center_r"]) < h / 2]

    # each sprite must fit in 3x3 bbox
    sprites_top = [s for s in sprites_top if _in_33_bbox(s)]
    if len(sprites_top) != 4:
        return np.zeros_like(g)

    rows = sorted(sprites_top, key=lambda s: s["center_r"])
    top2 = sorted(rows[:2], key=lambda s: s["center_c"])
    bot2 = sorted(rows[2:], key=lambda s: s["center_c"])
    ordered = top2 + bot2

    pos_colors = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW)]
    map_color: dict[int, int] = {int(s["color"]): pos_colors[i] for i, s in enumerate(ordered)}

    out = np.zeros_like(g)
    for r in range(h // 2, h):
        for c in range(w):
            v = int(g[r, c])
            if v in map_color:
                out[r, c] = int(map_color[v])

    return out


def _components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1)]
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if seen[r, c] or v == int(BLACK):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in dirs:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(grid[nx, ny]) == v:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            out.append({
                "cells": comp,
                "color": v,
                "r0": min(rs),
                "c0": min(cs),
                "r1": max(rs),
                "c1": max(cs),
                "center_r": (min(rs) + max(rs)) / 2,
                "center_c": (min(cs) + max(cs)) / 2,
            })
    return out


def _in_33_bbox(comp: dict[str, object]) -> bool:
    return (comp["r1"] - comp["r0"] <= 2) and (comp["c1"] - comp["c0"] <= 2)


def generate_input() -> np.ndarray:
    h, w = 18, 18
    g = np.full((h, w), int(BLACK), dtype=int)
    sprite_colors = [int(Color.RED), int(Color.BLUE), int(Color.GREEN), int(Color.ORANGE)]

    # two top-row sprite positions and two lower-row positions in upper half
    top_slots = [(2, 2), (2, 8)]
    lower_slots = [(4, 3), (5, 10)]
    slots = top_slots + lower_slots

    for color, (r0, c0) in zip(sprite_colors, slots):
        # draw an irregular blob in 3x3 area
        cells = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1)]
        if random.random() < 0.7:
            cells.append((r0 + 1, c0 + 1))
        if random.random() < 0.4:
            cells.append((r0 + 2, c0))
        for rr, cc in cells:
            if rr < h and cc < w:
                g[rr, cc] = color

    # lower-half markers
    for color in sprite_colors:
        n = random.randint(1, 3)
        for _ in range(n):
            rr = random.randint(h // 2, h - 1)
            cc = random.randint(0, w - 1)
            if g[rr, cc] == int(BLACK):
                g[rr, cc] = color

    return g
