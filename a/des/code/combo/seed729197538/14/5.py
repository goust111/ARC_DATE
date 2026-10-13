import random

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _horizontal_gates(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    gates: dict[tuple[int, int], tuple[int, int]] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w - 2):
            a, b, d = int(grid[r, c]), int(grid[r, c + 1]), int(grid[r, c + 2])
            if a != BG and b != BG and a == d and a != b:
                gates[(r, c)] = (a, b)
                gates[(r, c + 2)] = (a, b)
    return gates


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    gates = _horizontal_gates(grid)
    for r in range(h):
        start_color = int(grid[r, 0])
        if start_color == BG:
            continue
        c = 0
        while c < w and int(grid[r, c]) == start_color:
            c += 1
        color = start_color
        seen: set[tuple[int, int, int]] = set()
        while c < w:
            state = (r, c, color)
            if state in seen:
                break
            seen.add(state)
            if (r, c) in gates and gates[(r, c)][0] == color:
                in_color, out_color = gates[(r, c)]
                if c + 2 < w and int(grid[r, c]) == in_color and int(grid[r, c + 1]) == out_color:
                    color = out_color
                    c += 3
                else:
                    color = out_color
                    c += 1
                continue
            if int(grid[r, c]) == BG:
                out[r, c] = color
            c += 1
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        h = rng.randint(7, 12)
        w = rng.randint(15, 26)
        r = rng.randint(1, h - 2)
        cols = rng.sample(COLORS, 3)
        line_color, mid1, mid2 = cols
        line_len = rng.randint(2, 4)
        g1 = rng.randint(line_len + 2, max(line_len + 3, w - 9))
        use_second = rng.random() < 0.55 and g1 + 6 < w
        grid = np.zeros((h, w), dtype=int)
        grid[r, 0:line_len] = line_color
        grid[r, g1 : g1 + 3] = [line_color, mid1, line_color]
        if use_second:
            g2 = rng.randint(g1 + 4, w - 4)
            grid[r, g2 : g2 + 3] = [mid1, mid2, mid1]
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("generate_input failed")
