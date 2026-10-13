import random

import numpy as np


BG = 0
WALL = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _channel_rows(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    rows = []
    for r in range(1, h - 1):
        wall_count = int(np.sum(grid[r - 1, 1 : w - 1] == WALL) + np.sum(grid[r + 1, 1 : w - 1] == WALL))
        if wall_count >= w - 4:
            rows.append(r)
    return rows


def _runs(vals: list[int]) -> list[tuple[int, int]]:
    out = []
    i = 0
    while i < len(vals):
        if vals[i] in (BG, WALL):
            i += 1
            continue
        j = i
        while j + 1 < len(vals) and vals[j + 1] not in (BG, WALL):
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for r in _channel_rows(grid):
        vals = [int(x) for x in grid[r]]
        runs = _runs(vals)
        if not runs:
            continue
        start, end = runs[0]
        if end - start + 1 < 2:
            continue
        signal = [int(grid[r, c]) for c in range(start, end + 1)]
        gates = [int(grid[r, a]) for a, b in runs[1:] if a == b]
        if not gates:
            continue
        for c in range(start, end + 1):
            out[r, c] = BG
        filtered = signal[:]
        for gate in gates:
            filtered = [v if v == gate else BG for v in filtered]
        final_start = w - 2 - len(filtered)
        for i, v in enumerate(filtered):
            if v != BG:
                out[r, final_start + i] = v
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        h = rng.randint(7, 11)
        w = rng.randint(15, 25)
        r = rng.randint(2, h - 3)
        grid = np.zeros((h, w), dtype=int)
        grid[r - 1, 1 : w - 1] = WALL
        grid[r + 1, 1 : w - 1] = WALL
        grid[r, 1] = WALL
        grid[r, w - 2] = WALL
        n = rng.randint(3, 5)
        seq = rng.sample(COLORS, n)
        for i, color in enumerate(seq):
            grid[r, 2 + i] = color
        keep = rng.choice(seq)
        gate_count = rng.randint(1, 3)
        first_gate = 3 + n + rng.randint(1, 3)
        if first_gate + 2 * gate_count >= w - n - 3:
            continue
        for k in range(gate_count):
            grid[r, first_gate + 2 * k] = keep
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("generate_input failed")
