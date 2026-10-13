import random
from collections import deque
import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
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
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == col:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            rs = [x for x, _ in cells]; cs = [y for _, y in cells]
            comps.append({"color": col, "bbox": (min(rs), max(rs), min(cs), max(cs)), "cells": cells})
    return comps


def _contains(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return ar0 < br0 and ar1 > br1 and ac0 < bc0 and ac1 > bc1


def main(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid)
    outers = []
    for comp in comps:
        if not any(_contains(other["bbox"], comp["bbox"]) for other in comps):
            group = [x for x in comps if x is comp or _contains(comp["bbox"], x["bbox"])]
            outers.append((comp, len(group)))
    outers.sort(key=lambda x: (x[0]["bbox"][2] + x[0]["bbox"][3]) / 2)
    height = max(depth for _, depth in outers)
    out = np.zeros((height, len(outers)), dtype=int)
    for c, (comp, depth) in enumerate(outers):
        out[height - depth:, c] = comp["color"]
    return out


def _draw_layer(g, top, left, ht, wd, color, solid=False):
    if solid:
        g[top:top + ht, left:left + wd] = color
    else:
        g[top, left:left + wd] = color
        g[top + ht - 1, left:left + wd] = color
        g[top:top + ht, left] = color
        g[top:top + ht, left + wd - 1] = color


def generate_input() -> np.ndarray:
    g = np.zeros((24, 28), dtype=int)
    starts = [1, 8, 15, 22]
    n = random.randint(3, 4)
    max_depths = [4, 4, 4, 2]
    depths = [random.randint(1, max_depths[i]) for i in range(n)]
    if len(set(depths)) == 1:
        depths[0] = min(4, depths[0] + 1)
    outer_cols = random.sample([1, 2, 3, 4, 5, 6, 7], n)
    for i in range(n):
        d = depths[i]
        top = random.randint(1, 3)
        left = starts[i]
        size = 2 * d + 1
        colors = [outer_cols[i]] + random.sample([x for x in [1, 2, 3, 4, 5, 6, 7] if x != outer_cols[i]], d - 1)
        for k in range(d):
            ht = size - 2 * k
            wd = size - 2 * k
            _draw_layer(g, top + k, left + k, ht, wd, colors[k], solid=(k == d - 1))
    return g
