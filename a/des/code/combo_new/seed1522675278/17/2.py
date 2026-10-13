import random

import numpy as np


BG = 0
GREY = 5


def _spiral_coords(size):
    center = size // 2
    coords = [(center, center)]
    r = c = center
    step = 1
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while True:
        added = False
        for i, (dr, dc) in enumerate(dirs):
            length = step if i < 2 else step + 1
            for _ in range(length):
                r += dr
                c += dc
                if 0 <= r < size and 0 <= c < size:
                    coords.append((r, c))
                    added = True
            if i == 1:
                step += 2
        if not added or len(coords) >= size * size:
            break
    uniq = []
    seen = set()
    for p in coords:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    order = [p for p in _spiral_coords(grid.shape[0]) if grid[p] != BG]
    seed_idx = [i for i, p in enumerate(order) if grid[p] not in (BG, GREY)]
    for j, idx in enumerate(seed_idx):
        r, c = order[idx]
        color = int(grid[r, c])
        end_idx = seed_idx[j + 1] if j + 1 < len(seed_idx) else len(order) - 1
        n = max(0, end_idx - idx - 1)
        dr = order[idx + 1][0] - r
        dc = order[idx + 1][1] - c
        lr, lc = -dc, dr
        rr, cc = r + lr, c + lc
        for _ in range(n):
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]) or out[rr, cc] != BG:
                break
            out[rr, cc] = color
            rr += lr
            cc += lc
    return out


def generate_input() -> np.ndarray:
    size = random.choice([13, 15, 17, 19, 21])
    grid = np.zeros((size, size), dtype=int)
    order = _spiral_coords(size)
    for r, c in order:
        grid[r, c] = GREY
    candidates = []
    for i in range(len(order) - 2):
        r, c = order[i]
        nr, nc = order[i + 1]
        dr, dc = nr - r, nc - c
        lr, lc = -dc, dr
        rr, cc = r + lr, c + lc
        if 0 <= rr < size and 0 <= cc < size and grid[rr, cc] == BG:
            candidates.append(i)
    random.shuffle(candidates)
    seeds = sorted(candidates[: random.randint(2, 4)])
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], len(seeds))
    for idx, color in zip(seeds, colors):
        r, c = order[idx]
        grid[r, c] = color
    return grid
