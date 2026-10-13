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
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _direction(anchor, h, w):
    r, c = anchor
    if r == 0:
        return (-1, 0)
    if r == h - 1:
        return (1, 0)
    if c == 0:
        return (0, -1)
    return (0, 1)


def _distance(cells, direction, h, w):
    r0, c0, r1, c1 = _bbox(cells)
    if direction == (-1, 0):
        return r0
    if direction == (1, 0):
        return h - 1 - r1
    if direction == (0, -1):
        return c0
    return w - 1 - c1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape
    sprites = []
    for color in sorted(int(c) for c in np.unique(g) if c != 0):
        pts = [(r, c) for r, c in zip(*np.where(g == color))]
        anchors = [(r, c) for r, c in pts if r in (0, h - 1) or c in (0, w - 1)]
        if not anchors:
            continue
        anchor = anchors[0]
        cells = [p for p in pts if p != anchor]
        if not cells:
            continue
        direction = _direction(anchor, h, w)
        sprites.append(
            {
                "color": color,
                "cells": cells,
                "dir": direction,
                "dist": _distance(cells, direction, h, w),
                "bbox": _bbox(cells),
            }
        )
    sprites.sort(key=lambda s: (s["dist"], s["bbox"][0], s["bbox"][1]))

    out = np.zeros_like(g)
    occupied = set()
    for s in sprites:
        cells = set(s["cells"])
        dr, dc = s["dir"]
        while True:
            moved = {(r + dr, c + dc) for r, c in cells}
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in moved) or moved & occupied:
                break
            cells = moved
        for r, c in cells:
            out[r, c] = s["color"]
        occupied |= cells
    return out


_SHAPES = [
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2)},
]


def _can_place(g, shape, top, left):
    h, w = g.shape
    for r, c in shape:
        rr, cc = r + top, c + left
        if rr < 2 or rr >= h - 2 or cc < 2 or cc >= w - 2 or g[rr, cc] != 0:
            return False
    return True


def generate_input() -> np.ndarray:
    n = random.randint(13, 15)
    g = np.zeros((n, n), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(3, 4))
    edges = random.sample(["top", "bottom", "left", "right"], len(colors))
    for color, edge, shape in zip(colors, edges, random.sample(_SHAPES, len(colors))):
        for _ in range(100):
            top = random.randint(2, n - 5)
            left = random.randint(2, n - 5)
            if _can_place(g, shape, top, left):
                for r, c in shape:
                    g[r + top, c + left] = color
                break
        if edge == "top":
            g[0, random.randint(1, n - 2)] = color
        elif edge == "bottom":
            g[n - 1, random.randint(1, n - 2)] = color
        elif edge == "left":
            g[random.randint(1, n - 2), 0] = color
        else:
            g[random.randint(1, n - 2), n - 1] = color
    return g
