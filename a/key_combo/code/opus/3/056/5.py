import random
import numpy as np


def _bbox(coords):
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the blue/red rectangle bbox by the red inner size, truncating independently at green walls."""
    out = grid.copy()
    h, w = grid.shape
    obj = np.argwhere(np.isin(grid, [1, 2]))
    red = np.argwhere(grid == 2)
    if obj.size == 0 or red.size == 0:
        return out
    r0, c0, r1, c1 = _bbox(obj)
    ir0, ic0, ir1, ic1 = _bbox(red)
    eh, ew = ir1 - ir0 + 1, ic1 - ic0 + 1
    top, bottom = max(0, r0 - eh), min(h - 1, r1 + eh)
    left, right = max(0, c0 - ew), min(w - 1, c1 + ew)
    green = np.argwhere(grid == 3)

    changed = True
    while changed:
        old = (top, bottom, left, right)
        above = [r for r, c in green if top <= r < r0 and left <= c <= right]
        below = [r for r, c in green if r1 < r <= bottom and left <= c <= right]
        lbar = [c for r, c in green if top <= r <= bottom and left <= c < c0]
        rbar = [c for r, c in green if top <= r <= bottom and c1 < c <= right]
        if above:
            top = max(top, max(above) + 1)
        if below:
            bottom = min(bottom, min(below) - 1)
        if lbar:
            left = max(left, max(lbar) + 1)
        if rbar:
            right = min(right, min(rbar) - 1)
        changed = old != (top, bottom, left, right)

    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if out[r, c] == 0:
                out[r, c] = 2
    return out


def _frame(g, r0, c0, h, w, col):
    g[r0, c0 : c0 + w] = col
    g[r0 + h - 1, c0 : c0 + w] = col
    g[r0 : r0 + h, c0] = col
    g[r0 : r0 + h, c0 + w - 1] = col


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    r0, c0 = random.choice([(5, 5), (5, 4), (4, 5)])
    size = random.choice([5, 6])
    _frame(g, r0, c0, size, size, 1)
    g[r0 + 1:r0 + size - 1, c0 + 1:c0 + size - 1] = 2
    side = random.choice(["left_open", "right_open", "top_open", "bottom_open"])
    if side != "left_open":
        g[2:13, 2] = 3
    if side != "right_open":
        g[2:13, 12] = 3
    if side != "top_open":
        g[2, 2:13] = 3
    if side != "bottom_open":
        g[12, 2:13] = 3
    return g
