import random
from collections import deque
import numpy as np


DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _components(g, bg):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR8:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(g[nx, ny]) == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            comps.append({"color": color, "cells": cells, "bbox": (min(rs), min(cs), max(rs), max(cs))})
    return comps


def _contains(a, b):
    return a[0] < b[0] and a[1] < b[1] and a[2] > b[2] and a[3] > b[3]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    n = g.shape[0]
    source = g[:, :n]
    template = g[:, n:2 * n]
    bg = 5
    mask = [(r, c) for r in range(n) for c in range(n) if int(template[r, c]) != bg]
    out = np.full((n * n, n * n), bg, dtype=g.dtype)
    for r in range(n):
        for c in range(n):
            color = int(source[r, c])
            if color == bg:
                continue
            for mr, mc in mask:
                out[r * n + mr, c * n + mc] = color
    comps = _components(out, bg)
    erase = set()
    for i, small in enumerate(comps):
        for j, big in enumerate(comps):
            if i != j and small["color"] != big["color"] and _contains(big["bbox"], small["bbox"]):
                erase.add(i)
                break
    for i in erase:
        for r, c in comps[i]["cells"]:
            out[r, c] = bg
    return out


def generate_input() -> np.ndarray:
    n = random.choice([3, 4])
    g = np.full((n, 2 * n), 5, dtype=int)
    if n == 3:
        mask = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
        ring = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        center = (1, 1)
    else:
        mask = [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1), (3, 1)]
        ring = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        center = (1, 1)
    colors = random.sample([1, 2, 3, 4], 3)
    for r, c in ring:
        g[r, c] = colors[0]
    g[center] = colors[1]
    if n == 4:
        g[3, 3] = colors[2]
    for r, c in mask:
        g[r, n + c] = 7
    return g
