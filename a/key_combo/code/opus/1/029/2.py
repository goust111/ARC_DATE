import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == 5)]
    sep = sep_rows[0] if sep_rows else g.shape[0] // 2
    top = g[:sep] != 0
    bottom = g[sep + 1 : sep + 1 + top.shape[0]] != 0
    out = np.zeros(top.shape, dtype=int)
    seeds = [(r, c) for r in range(top.shape[0]) for c in range(top.shape[1]) if top[r, c] ^ bottom[r, c]]
    for r, c in seeds:
        out[r, c] = 3
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and out[nr, nc] == 0:
                out[nr, nc] = 3
                nr += dr
                nc += dc
    return out


def _far_enough(cells, p):
    return all(abs(r - p[0]) + abs(c - p[1]) >= 3 for r, c in cells)


def generate_input() -> np.ndarray:
    h, w = 17, 9
    g = np.zeros((h, w), dtype=int)
    g[8, :] = 5
    all_cells = [(r, c) for r in range(8) for c in range(w)]
    random.shuffle(all_cells)
    seeds = []
    for p in all_cells:
        if _far_enough(seeds, p):
            seeds.append(p)
            if len(seeds) == random.randint(4, 7):
                break
    common_count = random.randint(2, 5)
    remaining = [p for p in all_cells if p not in seeds]
    common = random.sample(remaining, common_count)
    top_only = random.sample(seeds, random.randint(1, len(seeds) - 1))
    for r, c in common + top_only:
        g[r, c] = 1
    for r, c in common + [p for p in seeds if p not in top_only]:
        g[r + 9, c] = 2
    return g
