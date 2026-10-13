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


def bbox(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    return min(rs), min(cs), max(rs), max(cs)


def components(mask, conn=4):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    neigh = n4 if conn == 4 else n8
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
                for nr, nc in neigh(x, y):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def color_components(grid, colors=None, ignore=(0,), conn=4):
    vals = set(np.unique(grid).tolist()) if colors is None else set(colors)
    vals -= set(ignore)
    out = []
    for col in vals:
        for comp in components(grid == col, conn):
            out.append((int(col), comp))
    return out


def flood_exterior_not_wall(grid, wall=2):
    h, w = grid.shape
    ext = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] != wall and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] != wall and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in n4(r, c):
            if in_bounds(grid.shape, nr, nc) and grid[nr, nc] != wall and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def draw_rect_border(grid, r0, c0, r1, c1, color):
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def find_hollow_rects_by_color(grid, color):
    rects = []
    for comp in components(grid == color):
        r0, c0, r1, c1 = bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if set(comp) == border:
            rects.append((r0, c0, r1, c1))
    return rects


def place_rectangles_nonoverlap(h, w, count, min_size=4, max_size=6, margin=1):
    rects = []
    for _ in range(2000):
        if len(rects) >= count:
            break
        rh = random.randint(min_size, max_size)
        rw = random.randint(min_size, max_size)
        r0 = random.randint(margin, h - rh - margin)
        c0 = random.randint(margin, w - rw - margin)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        ok = True
        for a0, b0, a1, b1 in rects:
            if not (r1 + 2 < a0 or a1 + 2 < r0 or c1 + 2 < b0 or b1 + 2 < c0):
                ok = False
                break
        if ok:
            rects.append((r0, c0, r1, c1))
    return rects


def load_common():
    return None
