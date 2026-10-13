import random
from collections import Counter

import numpy as np


def _mode(values):
    vals = [int(v) for v in values if int(v) != 3]
    return Counter(vals).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    row_score = sum(Counter(int(x) for x in grid[r] if int(x) != 3).most_common(1)[0][1] for r in range(h))
    col_score = sum(Counter(int(x) for x in grid[:, c] if int(x) != 3).most_common(1)[0][1] for c in range(w))
    horizontal = row_score >= col_score
    out = np.zeros_like(grid)
    if horizontal:
        majors = [_mode(grid[r]) for r in range(h)]
        for r, color in enumerate(majors):
            out[r, :] = color
    else:
        majors = [_mode(grid[:, c]) for c in range(w)]
        for c, color in enumerate(majors):
            out[:, c] = color

    seeds = np.argwhere(grid == 3)
    r, c = map(int, seeds[0])
    dr, dc = -1, 1
    while True:
        color = majors[r] if horizontal else majors[c]
        out[r, c] = color
        nr, nc = r + dr, c + dc
        if nc < 0 or nc >= w:
            dc *= -1
            nc = c + dc
        if nr < 0:
            break
        r, c = nr, nc
    return out


def generate_input() -> np.ndarray:
    h = random.randint(10, 14)
    w = random.randint(10, 14)
    horizontal = random.choice([True, False])
    colors = random.sample([1, 2, 4, 5, 6, 7, 8, 9], random.choice([2, 3]))
    grid = np.zeros((h, w), dtype=int)
    if horizontal:
        r = 0
        stripe = 0
        while r < h:
            width = random.choice([1, 2])
            grid[r : min(h, r + width), :] = colors[stripe % len(colors)]
            r += width
            stripe += 1
    else:
        c = 0
        stripe = 0
        while c < w:
            width = random.choice([1, 2])
            grid[:, c : min(w, c + width)] = colors[stripe % len(colors)]
            c += width
            stripe += 1

    n_noise = random.randint(2, max(3, h * w // 12))
    palette = [0, 1, 2, 4, 5, 6, 7, 8, 9]
    for _ in range(n_noise):
        rr, cc = random.randrange(h), random.randrange(w)
        grid[rr, cc] = random.choice(palette)
    sr = random.randint(2, h - 1)
    sc = random.randint(1, w - 2)
    grid[sr, sc] = 3
    return grid
