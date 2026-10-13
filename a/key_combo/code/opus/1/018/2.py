import random
from collections import deque

import numpy as np


def _rectangles(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rects = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] in (0, 8) or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [x for x, _ in cells]
            cols = [x for _, x in cells]
            rects.append({"cells": set(cells), "bbox": (min(rows), min(cols), max(rows), max(cols)), "color": color, "hits": 0})
    return rects


def _inside_rect(rects, pos):
    for i, rect in enumerate(rects):
        if pos in rect["cells"]:
            return i
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    rects = _rectangles(grid)
    cyan = sorted(map(tuple, np.argwhere(grid == 8)))
    if len(cyan) < 2:
        return out
    ends = [p for p in cyan if sum((p[0] - q[0], p[1] - q[1]) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] for q in cyan) == 1]
    if len(ends) != 2:
        ends = [cyan[0], cyan[-1]]
    e1, e2 = sorted(ends)
    dr = 1 if e2[0] > e1[0] else -1
    dc = 1 if e2[1] > e1[1] else -1
    rays = [(e1, (-dr, -dc)), (e2, (dr, dc))]
    h, w = grid.shape
    for start, (vr, vc) in rays:
        r, c = start
        for _ in range(h * w * 4):
            nr, nc = r + vr, c + vc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            ri = _inside_rect(rects, (nr, nc))
            if ri is not None:
                rects[ri]["hits"] += 1
                r0, c0, r1, c1 = rects[ri]["bbox"]
                if r < r0 or r > r1:
                    vr = -vr
                if c < c0 or c > c1:
                    vc = -vc
                continue
            r, c = nr, nc
            if out[r, c] != 8:
                out[r, c] = 3
    for rect in rects:
        if rect["hits"] == 0:
            continue
        new_color = 1 if rect["hits"] == 1 else 2
        for r, c in rect["cells"]:
            out[r, c] = new_color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    start = random.randint(9, 11)
    for k in range(4):
        grid[start + k, start + k] = 8
    colors = random.sample([1, 2, 4, 5, 6, 7, 9], 4)
    rects = [
        (start - 4, start - 4, 2, 2),
        (start + 6, start + 6, 2, 2),
        (random.randint(2, 5), random.randint(12, 15), 2, 3),
        (random.randint(12, 15), random.randint(2, 5), 3, 2),
    ]
    for (r, c, rh, cw), color in zip(rects, colors):
        grid[r : r + rh, c : c + cw] = color
    return grid

