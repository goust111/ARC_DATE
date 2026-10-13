import random
from collections import deque

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_of_color(grid: np.ndarray, color: int):
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each same-color fragment pair into a rectangle and stack the results.

    Input structure:
    - The grid is black except for monochrome 4-connected fragments.
    - Every non-black color appears in exactly two fragments.
    - The two fragments of one color align either on the same row span or the same
      column span, and when brought flush together they form a solid rectangle.

    Rule:
    - Pair fragments by color.
    - Infer each completed rectangle from the pair's total area and shared row or
      column span.
    - Sort the completed rectangles by descending area; ties break by the smallest
      input row where that color first appears.
    - Output only the stacked solid rectangles, packed tightly from bottom to top and
      horizontally centered on the original canvas width.

    Key insight:
    - The irregular pieces are not the output objects themselves: each color encodes a
      hidden rectangle split into two complementary connected fragments.
    """
    rects = []
    for color in [int(v) for v in np.unique(grid) if v != BLACK]:
        comps = _components_of_color(grid, color)
        if len(comps) != 2:
            continue
        b0 = _bbox(comps[0])
        b1 = _bbox(comps[1])
        area = len(comps[0]) + len(comps[1])
        top = min(b0[0], b1[0])
        if (b0[0], b0[2]) == (b1[0], b1[2]):
            height = b0[2] - b0[0] + 1
            width = area // height
        else:
            width = b0[3] - b0[1] + 1
            height = area // width
        rects.append((height * width, top, color, height, width))
    rects.sort(key=lambda x: (-x[0], x[1]))
    out_h = sum(h for _, _, _, h, _ in rects)
    out = np.zeros((out_h, grid.shape[1]), dtype=int)
    base = out_h
    for _, _, color, h, w in rects:
        base -= h
        c0 = (grid.shape[1] - w) // 2
        out[base : base + h, c0 : c0 + w] = color
    return out


def generate_input() -> np.ndarray:
    while True:
        n = random.randint(2, 4)
        colors = random.sample(COLORS, n)
        h = random.randint(16, 28)
        w = random.randint(16, 28)
        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        ok = True
        for color in colors:
            rh = random.randint(2, 4)
            rw = random.randint(4, 7)
            horizontal = random.random() < 0.5
            if horizontal:
                cuts = [random.randint(1, rw - 1) for _ in range(rh)]
                left = {(r, c) for r in range(rh) for c in range(cuts[r])}
                right = {(r, c) for r in range(rh) for c in range(cuts[r], rw)}
                y = random.randint(0, h - rh)
                x0 = random.randint(0, max(0, w - rw - 3))
                gap = random.randint(2, 4)
                x1 = x0 + rw + gap
                if x1 + rw > w:
                    ok = False
                    break
                for r, c in left:
                    if (y + r, x0 + c) in occupied:
                        ok = False
                    grid[y + r, x0 + c] = color
                    occupied.add((y + r, x0 + c))
                for r, c in right:
                    rr, cc = y + r, x1 + (c - cuts[r])
                    if (rr, cc) in occupied:
                        ok = False
                    grid[rr, cc] = color
                    occupied.add((rr, cc))
            else:
                rh = random.randint(4, 7)
                rw = random.randint(2, 4)
                cuts = [random.randint(1, rh - 1) for _ in range(rw)]
                top = {(r, c) for c in range(rw) for r in range(cuts[c])}
                bottom = {(r, c) for c in range(rw) for r in range(cuts[c], rh)}
                x = random.randint(0, w - rw)
                y0 = random.randint(0, max(0, h - rh - 3))
                gap = random.randint(2, 4)
                y1 = y0 + rh + gap
                if y1 + rh > h:
                    ok = False
                    break
                for r, c in top:
                    if (y0 + r, x + c) in occupied:
                        ok = False
                    grid[y0 + r, x + c] = color
                    occupied.add((y0 + r, x + c))
                for r, c in bottom:
                    rr, cc = y1 + (r - cuts[c]), x + c
                    if (rr, cc) in occupied:
                        ok = False
                    grid[rr, cc] = color
                    occupied.add((rr, cc))
        if ok and np.count_nonzero(grid) > 0:
            return grid
