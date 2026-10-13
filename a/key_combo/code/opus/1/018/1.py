import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc, int(grid[cr, cc])))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _fill_holes(canvas):
    h, w = canvas.shape
    out = canvas.copy()
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if out[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                touches |= cr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches:
                for cr, cc in cells:
                    out[cr, cc] = 4
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    comps = _components(grid)
    sprites = []
    for comp in comps:
        anchors = [(r, c) for r, c, v in comp if v == 5]
        if not anchors:
            continue
        ar, ac = anchors[0]
        sprites.append((ar, ac, [(r - ar, c - ac, v) for r, c, v in comp]))
    sprites.sort(key=lambda x: (x[0], x[1]))
    canvas = np.zeros((31, 31), dtype=int)
    center = (15, 15)
    for _, _, cells in sprites:
        for dr, dc, v in cells:
            rr, cc = center[0] + dr, center[1] + dc
            if 0 <= rr < 31 and 0 <= cc < 31:
                canvas[rr, cc] = v
    canvas = _fill_holes(canvas)
    canvas[canvas == 5] = 0
    pts = np.argwhere(canvas != 0)
    if len(pts) == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _place(grid, top, left, cells, color):
    for r, c, v in cells:
        grid[top + r, left + c] = 5 if v == 5 else color


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    sprites = [
        [(2, 2, 5), (1, 2, 1), (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)],
        [(2, 2, 5), (2, 3, 1), (0, 4, 1), (1, 4, 1), (2, 4, 1), (3, 4, 1), (4, 4, 1)],
        [(2, 2, 5), (3, 2, 1), (4, 0, 1), (4, 1, 1), (4, 2, 1), (4, 3, 1), (4, 4, 1)],
        [(2, 2, 5), (2, 1, 1), (0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1)],
    ]
    positions = [(1, 1), (1, 11), (11, 1), (11, 11)]
    colors = random.sample([1, 2, 3, 6], 4)
    order = list(range(4))
    random.shuffle(order)
    for idx, pos in zip(order, positions):
        _place(grid, pos[0], pos[1], sprites[idx], colors[idx])
    return grid

