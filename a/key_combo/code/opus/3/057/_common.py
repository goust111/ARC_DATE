from collections import deque
import random

import numpy as np


def in_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def n4(r, c):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def n8(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def components(mask, conn=4):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
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
                neigh = n4(x, y) if conn == 4 else n8(x, y)
                for nr, nc in neigh:
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def bbox(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    return min(rs), min(cs), max(rs), max(cs)


def color_components(grid, colors=None, ignore=(0,), conn=4):
    vals = set(np.unique(grid).tolist()) if colors is None else set(colors)
    vals -= set(ignore)
    out = []
    for col in vals:
        for comp in components(grid == col, conn):
            out.append((int(col), comp))
    return out


def draw_cup(g, r0, c0, r1, c1, color, opening):
    if opening != "up":
        g[r0, c0:c1 + 1] = color
    if opening != "down":
        g[r1, c0:c1 + 1] = color
    if opening != "left":
        g[r0:r1 + 1, c0] = color
    if opening != "right":
        g[r0:r1 + 1, c1] = color


def detect_cups(grid, colors=None):
    cups = []
    vals = set(np.unique(grid).tolist()) if colors is None else set(colors)
    for col in vals - {0, 7, 9}:
        for comp in components(grid == col):
            if len(comp) < 5:
                continue
            r0, c0, r1, c1 = bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            top = all(grid[r0, c] == col for c in range(c0, c1 + 1))
            bottom = all(grid[r1, c] == col for c in range(c0, c1 + 1))
            left = all(grid[r, c0] == col for r in range(r0, r1 + 1))
            right = all(grid[r, c1] == col for r in range(r0, r1 + 1))
            present = {"up": top, "down": bottom, "left": left, "right": right}
            missing = [side for side, ok in present.items() if not ok]
            if len(missing) == 1 and sum(present.values()) == 3:
                cups.append({"color": col, "bbox": (r0, c0, r1, c1), "opening": missing[0]})
    return cups


def interior_of_cup(cup):
    r0, c0, r1, c1 = cup["bbox"]
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def quadrant_of(r, c, cr, cc):
    if r < cr and c < cc:
        return 0
    if r < cr and c > cc:
        return 1
    if r > cr and c < cc:
        return 2
    if r > cr and c > cc:
        return 3
    return None
