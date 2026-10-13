import random
from collections import deque
import numpy as np


def _components8(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            q = deque([(r, c)]); seen[r, c] = True; cells = []
            while q:
                x, y = q.popleft(); cells.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] != 0:
                            seen[nx, ny] = True; q.append((nx, ny))
            comps.append(cells)
    return comps


def _sprites(g):
    ans = []
    for cells in _components8(g):
        colors = {}
        for r, c in cells:
            colors.setdefault(int(g[r, c]), []).append((r, c))
        if len(colors) == 2 and 1 in [len(v) for v in colors.values()] and len(cells) >= 5:
            anc = next(k for k, v in colors.items() if len(v) == 1)
            body = next(k for k in colors if k != anc)
            ar, ac = colors[anc][0]
            offsets = [(r - ar, c - ac) for r, c in colors[body]] + [(0, 0)]
            ans.append({"anchor": anc, "body": body, "offsets": offsets, "cells": cells})
    return ans


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    sprites = _sprites(g)
    sprite_cells = {p for s in sprites for p in s["cells"]}
    out = g.copy()
    for p in sprite_cells:
        out[p] = 0
    for sp in sprites:
        pts = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == sp["anchor"] and (r, c) not in sprite_cells]
        if len(pts) != 2:
            continue
        pts.sort()
        p1, p2 = pts
        dr = 1
        dc = 1 if p2[1] > p1[1] else -1
        r, c = p1
        while (r, c) != p2:
            if (r, c) not in (p1, p2):
                out[r, c] = 5
            r += dr; c += dc
        for ar, ac, mirror in [(p1[0], p1[1], "none"), (p2[0], p2[1], "h" if p2[1] > p1[1] else "v")]:
            for rr, cc in sp["offsets"]:
                nr, nc = rr, cc
                if mirror == "h":
                    nc = -nc
                elif mirror == "v":
                    nr = -nr
                if 0 <= ar + nr < out.shape[0] and 0 <= ac + nc < out.shape[1]:
                    out[ar + nr, ac + nc] = sp["body"]
    return out


def _put(g, top, left, shape, body, anc):
    ar, ac = shape[0]
    for r, c in shape:
        g[top + r, left + c] = body
    g[top + ar, left + ac] = anc


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    shapes = [
        [(1, 1), (0, 1), (1, 2), (2, 1), (2, 2)],
        [(1, 1), (0, 2), (1, 0), (1, 2), (2, 2), (3, 1)],
        [(1, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    ]
    n = random.randint(2, 3)
    bodies = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    anchors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    while set(bodies) & set(anchors):
        anchors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    for i in range(n):
        _put(g, 1 + i * 5, 1, shapes[i], bodies[i], anchors[i])
        if i % 2 == random.randint(0, 1):
            p1, p2 = (3 + i * 5, 10), (6 + i * 5, 13)
        else:
            p1, p2 = (3 + i * 5, 14), (6 + i * 5, 11)
        g[p1] = anchors[i]; g[p2] = anchors[i]
    return g
