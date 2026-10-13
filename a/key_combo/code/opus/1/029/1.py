import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _norm(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    containers = []
    for comp in _components(g == 5):
        if len(comp) > 10:
            r0, c0, r1, c1 = _bbox(comp)
            holes = []
            hole_mask = np.zeros((h, w), bool)
            hole_mask[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1] == 0
            for hole in _components(hole_mask):
                holes.append({"cells": frozenset(hole), "shape": _norm(hole), "used": False})
            containers.append({"bbox": (r0, c0, r1, c1), "holes": holes})

    pieces = []
    frag_mask = (g != 0) & (g != 5)
    for comp in _components(frag_mask):
        color = int(g[comp[0]])
        r0, c0, r1, c1 = _bbox(comp)
        pieces.append({"cells": tuple(comp), "shape": _norm(comp), "color": color, "bbox": (r0, c0, r1, c1)})
    pieces.sort(key=lambda p: (-p["bbox"][0], p["bbox"][1]))

    out = g.copy()
    for p in pieces:
        for r, c in p["cells"]:
            out[r, c] = 0

    for p in pieces:
        placed = False
        cells = p["cells"]
        max_r = max(r for r, _ in cells)
        for dr in range(0, h - max_r):
            moved = frozenset((r + dr, c) for r, c in cells)
            for con in containers:
                cr0, cc0, cr1, cc1 = con["bbox"]
                if not any(cr0 <= r <= cr1 and cc0 <= c <= cc1 for r, c in moved):
                    continue
                for hole in con["holes"]:
                    if not hole["used"] and p["shape"] == hole["shape"] and moved == hole["cells"]:
                        for r, c in moved:
                            out[r, c] = p["color"]
                        hole["used"] = True
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
    return out


_SHAPES = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)},
    {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)},
]


def _put(cells, top, left):
    return {(r + top, c + left) for r, c in cells}


def generate_input() -> np.ndarray:
    h, w = 25, 15
    g = np.zeros((h, w), dtype=int)
    shapes = random.sample(_SHAPES, 4)
    holes = shapes[:3]
    rects = [(5, 1, 10, 7), (12, 4, 17, 11), (19, 2, 23, 9)]
    random.shuffle(holes)
    placed_holes = []
    preferred_lefts = [2, 7, 3]
    for idx, ((r0, c0, r1, c1), shape) in enumerate(zip(rects, holes)):
        g[r0 : r1 + 1, c0 : c1 + 1] = 5
        sh = list(shape)
        hh = max(r for r, _ in sh) + 1
        ww = max(c for _, c in sh) + 1
        top = random.randint(r0 + 1, r1 - hh)
        left = min(max(preferred_lefts[idx], c0 + 1), c1 - ww)
        placed_holes.append((shape, top, left))
        for r, c in _put(shape, top, left):
            g[r, c] = 0

    # One piece matches the upper container, one matches only after passing it, and one vanishes.
    first_shape, _, first_left = placed_holes[0]
    second_shape, _, second_left = placed_holes[1]
    miss_shape = shapes[3]
    specs = [(1, first_shape, first_left), (2, second_shape, second_left), (3, miss_shape, 11)]
    for color, shape, left in specs:
        top = random.randint(0, 2)
        for r, c in _put(shape, top, left):
            if r < 3 and c < w:
                g[r, c] = color
    return g
