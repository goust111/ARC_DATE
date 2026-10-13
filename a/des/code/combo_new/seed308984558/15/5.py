import random

import numpy as np


def _drop(out: np.ndarray, col: int, start_row: int, color: int):
    if not (0 <= col < out.shape[1]):
        return
    r = start_row
    while r + 1 < out.shape[0] and out[r + 1, col] == 0:
        r += 1
    if out[r, col] == 0:
        out[r, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Open green valves, drain one-pixel streams downward, and keep red-valve containers unchanged."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    for r, c in zip(*np.where((g == 3) | (g == 2))):
        r, c = int(r), int(c)
        left = c
        while left - 1 >= 0 and out[r, left - 1] in (5, 2, 3):
            left -= 1
        right = c
        while right + 1 < out.shape[1] and out[r, right + 1] in (5, 2, 3):
            right += 1
        top = r
        while top - 1 >= 0 and out[top - 1, left] == 5 and out[top - 1, right] == 5:
            top -= 1
        inner = out[top : r, left + 1 : right]
        vals = [int(v) for v in inner.ravel() if int(v) not in (0, 5)]
        if g[r, c] == 2 or not vals:
            continue
        color = max(set(vals), key=vals.count)
        height = sum(np.any(inner == color, axis=1))
        out[top : r, left + 1 : right] = 0
        out[r, c] = 0
        for _ in range(height):
            _drop(out, c, r, color)
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    cols = random.sample(range(3, g.shape[1] - 3), random.randint(2, 4))
    for c in cols:
        h = random.randint(4, 6)
        w = random.choice([5, 7])
        r0 = random.randint(1, g.shape[0] - h - 2)
        c0 = c - w // 2
        c1 = c + w // 2
        g[r0 : r0 + h, c0] = 5
        g[r0 : r0 + h, c1] = 5
        g[r0 + h - 1, c0 : c1 + 1] = 5
        valve = random.choice([2, 3])
        g[r0 + h - 1, c] = valve
        color = random.choice([1, 4, 6, 7, 8, 9])
        fill_h = random.randint(1, h - 2)
        g[r0 + h - 1 - fill_h : r0 + h - 1, c0 + 1 : c1] = color
    return g
