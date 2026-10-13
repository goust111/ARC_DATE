import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    comps = _components(grid)
    rects = []
    red_to_rect = {}
    for idx, cells in enumerate(comps):
        vals = [int(grid[r, c]) for r, c in cells if grid[r, c] not in (0, 2)]
        if not vals:
            continue
        color = max(set(vals), key=vals.count)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        red = [(r, c) for r, c in cells if grid[r, c] == 2]
        info = {
            "cells": cells,
            "color": color,
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
            "red": red[0] if red else None,
        }
        rects.append(info)
        if red:
            red_to_rect[red[0]] = info

    reds = sorted(red_to_rect)
    visible = {}
    for a in reds:
        ar, ac = a
        for b in reds:
            if a >= b or red_to_rect[a] is red_to_rect[b]:
                continue
            br, bc = b
            ok = False
            if ar == br:
                lo, hi = sorted((ac, bc))
                ok = all(grid[ar, x] == 0 for x in range(lo + 1, hi))
            elif ac == bc:
                lo, hi = sorted((ar, br))
                ok = all(grid[y, ac] == 0 for y in range(lo + 1, hi))
            if ok:
                g = abs(ar - br) + abs(ac - bc) - 1
                visible.setdefault(a, []).append((b, g))
                visible.setdefault(b, []).append((a, g))

    paired = {}
    for a in reds:
        if a in paired:
            continue
        opts = [(b, g) for b, g in visible.get(a, []) if b not in paired]
        if len(opts) == 1:
            b, g = opts[0]
            if len([x for x, _ in visible.get(b, []) if x not in paired or x == a]) == 1:
                paired[a] = (b, g)
                paired[b] = (a, g)

    for rect in rects:
        color = rect["color"]
        r0, c0, r1, c1 = rect["bbox"]
        red = rect["red"]
        if red not in paired:
            for r, c in rect["cells"]:
                out[r, c] = 0
            continue
        g = paired[red][1]
        out[red] = color
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                lr, lc = r - (r0 + 1), c - (c0 + 1)
                out[r, c] = 0 if (lr + lc + g) % 2 == 0 else color
    return out


def generate_input() -> np.ndarray:
    h = w = 20
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 3, 4, 5, 6, 7, 8, 9], 4)
    pair_rows = random.sample([3, 10, 15], 2)
    for i, row in enumerate(pair_rows):
        rh = random.randint(4, 5)
        aw = random.randint(4, 5)
        bw = random.randint(4, 5)
        r0 = max(0, min(h - rh, row))
        gap = random.randint(3, 6)
        c0 = random.randint(0, 2)
        c1 = c0 + aw + gap
        if c1 + bw >= w:
            c1 = w - bw - 1
        color_a, color_b = colors[2 * i], colors[2 * i + 1]
        grid[r0 : r0 + rh, c0 : c0 + aw] = color_a
        grid[r0 : r0 + rh, c1 : c1 + bw] = color_b
        mr = r0 + random.randint(1, rh - 2)
        grid[mr, c0 + aw - 1] = 2
        grid[mr, c1] = 2
    return grid

