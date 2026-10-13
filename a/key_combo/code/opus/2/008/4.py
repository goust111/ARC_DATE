import random

import numpy as np


def _cell_bounds(r, c):
    return slice(r * 4, r * 4 + 3), slice(c * 4, c * 4 + 3)


def _trace_cells(points):
    covered = set()
    for r in range(4):
        cs = [c for rr, c in points if rr == r]
        if len(cs) >= 2:
            for c in range(min(cs), max(cs) + 1):
                covered.add((r, c))
    for c in range(4):
        rs = [r for r, cc in points if cc == c]
        if len(rs) >= 2:
            for r in range(min(rs), max(rs) + 1):
                covered.add((r, c))
    return covered


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    red, green = [], []
    for r in range(4):
        for c in range(4):
            val = int(g[r * 4 + 1, c * 4 + 1])
            if val == 2:
                red.append((r, c))
            elif val == 3:
                green.append((r, c))
    green_rot = [(c, 3 - r) for r, c in green]
    red_cover = _trace_cells(red)
    green_cover = _trace_cells(green_rot)
    out = np.zeros((15, 15), dtype=int)
    out[3::4, :] = 8
    out[:, 3::4] = 8
    for r, c in red_cover | green_cover:
        rs, cs = _cell_bounds(r, c)
        color = 4 if (r, c) in red_cover and (r, c) in green_cover else 2 if (r, c) in red_cover else 3
        out[rs, cs] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    g[3::4, :] = 8
    g[:, 3::4] = 8
    while True:
        cells = [(r, c) for r in range(4) for c in range(4)]
        red = random.choice([
            [(0, 0), (0, 2), (random.randint(1, 3), random.randint(0, 3))],
            [(1, 1), (3, 1), (random.randint(0, 3), random.randint(2, 3))],
            [(2, 0), (2, 3)],
        ])
        red = list(dict.fromkeys(red))[:random.randint(2, 3)]
        used = set(red)
        green = random.sample([p for p in cells if p not in used], random.randint(2, 3))
        grot = [(c, 3 - r) for r, c in green]
        if set(grot).isdisjoint(used) and _trace_cells(grot):
            if _trace_cells(red) & _trace_cells(grot):
                break
    for r, c in red:
        g[r * 4 + 1, c * 4 + 1] = 2
    for r, c in green:
        g[r * 4 + 1, c * 4 + 1] = 3
    return g
