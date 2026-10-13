import random

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _solid_2x2(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    out = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color != BG and np.all(grid[r : r + 2, c : c + 2] == color):
                out.append((r, c, color))
    return out


def _runs_on_line(vals: list[int], track: int) -> list[tuple[int, int, int]]:
    runs = []
    i = 0
    while i < len(vals):
        if vals[i] in (BG, track):
            i += 1
            continue
        color = vals[i]
        j = i
        while j + 1 < len(vals) and vals[j + 1] == color:
            j += 1
        runs.append((i, j, color))
        i = j + 1
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for sr, sc, color in _solid_2x2(grid):
        if int(grid[sr, 0]) == color and sc > 4:
            vals = [int(grid[sr, c]) for c in range(sc)]
            frags = _runs_on_line(vals, color)
            if not frags:
                continue
            for a, b, _frag_color in frags:
                out[sr, a : b + 1] = color
            cursor = sc - 1
            for a, b, frag_color in sorted(frags, key=lambda x: x[1], reverse=True):
                length = b - a + 1
                out[sr, cursor - length + 1 : cursor + 1] = frag_color
                cursor -= length
            return out
        if int(grid[0, sc]) == color and sr > 4:
            vals = [int(grid[r, sc]) for r in range(sr)]
            frags = _runs_on_line(vals, color)
            if not frags:
                continue
            for a, b, _frag_color in frags:
                out[a : b + 1, sc] = color
            cursor = sr - 1
            for a, b, frag_color in sorted(frags, key=lambda x: x[1], reverse=True):
                length = b - a + 1
                out[cursor - length + 1 : cursor + 1, sc] = frag_color
                cursor -= length
            return out
    return out


def _choose_positions(rng: random.Random, limit: int, count: int) -> list[int] | None:
    choices = list(range(2, limit - 3))
    rng.shuffle(choices)
    picked: list[int] = []
    for c in choices:
        if all(abs(c - p) >= 4 for p in picked):
            picked.append(c)
            if len(picked) == count:
                return sorted(picked)
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        h = rng.randint(12, 20)
        w = rng.randint(12, 20)
        track = rng.choice(COLORS)
        frag_colors = [c for c in COLORS if c != track]
        rng.shuffle(frag_colors)
        count = rng.randint(1, 3)
        grid = np.zeros((h, w), dtype=int)
        if rng.random() < 0.5:
            row = rng.randint(1, h - 3)
            sc = rng.randint(8, w - 3)
            grid[row, 0:sc] = track
            grid[row : row + 2, sc : sc + 2] = track
            positions = _choose_positions(rng, sc - 1, count)
            if positions is None:
                continue
            for pos, color in zip(positions, frag_colors):
                grid[row, pos : pos + 3] = color
        else:
            col = rng.randint(1, w - 3)
            sr = rng.randint(8, h - 3)
            grid[0:sr, col] = track
            grid[sr : sr + 2, col : col + 2] = track
            positions = _choose_positions(rng, sr - 1, count)
            if positions is None:
                continue
            for pos, color in zip(positions, frag_colors):
                grid[pos : pos + 3, col] = color
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("generate_input failed")
