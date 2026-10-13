import random
from collections import deque
import numpy as np


def _components(grid, colors=None):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            if colors is not None and int(grid[r, c]) not in colors:
                continue
            col = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(grid[nx, ny]) == col:
                        if colors is None or col in colors:
                            seen[nx, ny] = True
                            q.append((nx, ny))
            out.append({"color": col, "cells": cells})
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    reds = [(r, c) for r in range(h) for c in range(w) if g[r, c] == 2]
    by_row = {}
    for r, c in reds:
        by_row.setdefault(r, []).append(c)
    spine_row, pair = None, None
    for r, cols in by_row.items():
        if len(cols) >= 2:
            cs = sorted(cols)
            if pair is None or cs[-1] - cs[0] > pair[1] - pair[0]:
                spine_row, pair = r, (cs[0], cs[-1])
    if pair is None:
        return g
    c1, c2 = pair
    for c in range(c1 + 1, c2):
        if g[spine_row, c] == 0:
            g[spine_row, c] = 3
    for r, c in reds:
        if r == spine_row:
            continue
        step = 1 if spine_row > r else -1
        for rr in range(r + step, spine_row, step):
            if g[rr, c] == 0:
                g[rr, c] = 3

    comps = [x for x in _components(g) if x["color"] not in (2, 3)]
    out = g.copy()
    for comp in comps:
        for r, c in comp["cells"]:
            out[r, c] = 0
    barriers = {(r, c) for r in range(h) for c in range(w) if g[r, c] == 3}
    for comp in comps:
        cells = comp["cells"]
        maxr = max(r for r, _ in cells)
        desired = len(cells)
        stop = h - 1 - maxr
        for br in sorted({r for r, c in barriers if any(cc == c for _, cc in cells)}):
            if br > maxr:
                stop = min(stop, br - 1 - maxr)
                break
        dr = max(0, min(desired, stop))
        for r, c in cells:
            out[r + dr, c] = comp["color"]
    return out


def _place(cells, top, left, color, grid):
    for r, c in cells:
        grid[top + r, left + c] = color


def generate_input() -> np.ndarray:
    h = w = 18
    for _ in range(200):
        g = np.zeros((h, w), dtype=int)
        spine_r = random.randint(10, 14)
        c1 = random.randint(2, 5)
        c2 = random.randint(12, 16)
        g[spine_r, c1] = g[spine_r, c2] = 2
        for c in random.sample(range(c1 + 1, c2), random.randint(2, 5)):
            rr = random.choice([random.randint(6, spine_r - 2), random.randint(spine_r + 1, 16)])
            g[rr, c] = 2
        shapes = [
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (2, 0)],
            [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        ]
        cols = random.sample([1, 4, 6, 7, 8, 9], 4)
        x_ranges = [(1, 3), (5, 7), (9, 11), (13, 15)]
        ok = True
        for i in range(random.randint(3, 5)):
            shape = random.choice(shapes)
            top = random.randint(1, 4)
            left = random.randint(*x_ranges[i % len(x_ranges)])
            if any(top + r >= spine_r - 1 or g[top + r, left + c] != 0 for r, c in shape):
                ok = False
                break
            _place(shape, top, left, cols[i % len(cols)], g)
        if ok:
            return g
    return g
