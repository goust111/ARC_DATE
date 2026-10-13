import random
from collections import deque

import numpy as np


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _neighbors8(r, c, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    seeds = [(r, c, int(g[r, c])) for r in range(h) for c in range(w)
             if (r in (0, h - 1) or c in (0, w - 1)) and g[r, c] in (2, 3, 4)]

    completed = g.copy()
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if completed[r, c] == 0 and completed[r, mc] == 1:
                completed[r, c] = 1

    cavities = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if completed[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                touches_edge |= cr in (0, h - 1) or cc in (0, w - 1)
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if completed[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                cavities.append(cells)

    out = completed.copy()
    for sr, sc, color in seeds:
        wall_starts = [(nr, nc) for nr, nc in _neighbors8(sr, sc, h, w) if completed[nr, nc] == 1]
        reachable = set(wall_starts)
        q = deque(wall_starts)
        while q:
            cr, cc = q.popleft()
            for nr, nc in _neighbors8(cr, cc, h, w):
                if completed[nr, nc] == 1 and (nr, nc) not in reachable:
                    reachable.add((nr, nc))
                    q.append((nr, nc))
        for cells in cavities:
            if any((nr, nc) in reachable for cr, cc in cells for nr, nc in _neighbors4(cr, cc, h, w)):
                for cr, cc in cells:
                    out[cr, cc] = color

    for sr, sc, _ in seeds:
        out[sr, sc] = 0
    return out


def generate_input() -> np.ndarray:
    h = w = 15
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([2, 3, 4], 2)
    boxes = [((3, 5), colors[0], 0), ((9, 11), colors[1], 14)]
    if random.random() < 0.25:
        boxes = [((2, 4), colors[0], 0), ((10, 12), colors[1], 14)]

    for (r0, r1), seed_color, seed_row in boxes:
        c0, c1 = 4, 10
        g[r0, c0:c1 + 1] = 1
        g[r1, c0:c1 + 1] = 1
        g[r0:r1 + 1, c0] = 1
        g[r0:r1 + 1, c1] = 1
        lead_col = random.choice([5, 7, 9])
        if seed_row == 0:
            g[0, lead_col] = seed_color
            g[1:r0 + 1, lead_col] = 1
        else:
            g[14, lead_col] = seed_color
            g[r1:14, lead_col] = 1
        hole_rows = random.sample(range(r0 + 1, r1), min(2, r1 - r0 - 1))
        for rr in hole_rows:
            g[rr, c1] = 0

    return g
