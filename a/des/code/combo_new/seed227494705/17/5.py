from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
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


def _mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        m[r - r0, c - c0] = True
    return m


def _norm(mask):
    return frozenset(map(tuple, np.argwhere(mask)))


def _rotations(mask):
    return [_norm(np.rot90(mask, k)) for k in range(4)]


def main(grid: np.ndarray) -> np.ndarray:
    """Match colored keys to black lock-hole negative spaces under rotation.

    Input structure: Azure background has colored 4-connected key shapes in the
    upper half and grey solid rectangles in the lower half.  Each grey lock
    contains a black hole shape.
    Rule: Extract every black hole inside a grey lock and compare it with unused
    key masks under 0, 90, 180, or 270 degree rotation.  A matched hole is filled
    with that key's color; unmatched holes remain black.  All original keys are
    cleared to azure.
    Key insight: The object to match is the lock's negative black space, not the
    grey material, and keys may need to be mentally rotated.
    """
    out = grid.copy()
    keys = []
    for comp in _components((grid != AZURE) & (grid != GREY) & (grid != BLACK)):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 < grid.shape[0] // 2:
            color = int(grid[comp[0]])
            keys.append({"cells": comp, "mask": _mask(comp), "color": color, "used": False})
            for r, c in comp:
                out[r, c] = AZURE

    holes = []
    for grey in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(grey)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK
        for comp in _components(sub):
            cells = [(r0 + r, c0 + c) for r, c in comp]
            hr0, hc0, hr1, hc1 = _bbox(cells)
            if r0 < hr0 and hr1 < r1 and c0 < hc0 and hc1 < c1:
                holes.append({"cells": cells, "norm": _norm(_mask(cells))})

    for hole in holes:
        for key in keys:
            if key["used"]:
                continue
            if hole["norm"] in _rotations(key["mask"]):
                for r, c in hole["cells"]:
                    out[r, c] = key["color"]
                key["used"] = True
                break
    return out


def _stamp(grid, mask, top, left, color):
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = color


def generate_input() -> np.ndarray:
    grid = np.full((28, 28), AZURE, dtype=int)
    shapes = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    ]
    n = random.randint(2, 4)
    colors = random.sample([1, 2, 3, 4, 6, 7, 9], n)
    for i in range(n):
        _stamp(grid, shapes[i], 2, 2 + i * 6, colors[i])
        r0, c0 = 16, 2 + i * 6
        grid[r0 : r0 + 6, c0 : c0 + 6] = GREY
        hole = np.rot90(shapes[i], random.randint(0, 3))
        hr, hc = r0 + 2, c0 + 2
        grid[hr : hr + hole.shape[0], hc : hc + hole.shape[1]][hole] = BLACK
    if random.random() < 0.5:
        grid[3, 23:25] = MAGENTA
    return grid
