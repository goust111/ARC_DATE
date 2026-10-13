import random

import numpy as np


def _legend(grid):
    rows = []
    for r in range(grid.shape[0]):
        if grid[r, 0] != 0 and grid[r, 1] != 0:
            rows.append((int(grid[r, 0]), int(grid[r, 1])))
        elif rows:
            break
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    pairs = _legend(grid)
    legend_h = len(pairs)
    h, w = grid.shape
    in_legend = lambda r, c: r < legend_h and c < 2
    seen = set()
    for r in range(h):
        for c in range(w):
            if in_legend(r, c) or grid[r, c] == 0 or (r, c) in seen:
                continue
            color = int(grid[r, c])
            if r > 0 and c > 0 and not in_legend(r - 1, c - 1) and grid[r - 1, c - 1] == color:
                continue
            seg = []
            nr, nc = r, c
            while nr < h and nc < w and not in_legend(nr, nc) and grid[nr, nc] == color:
                seg.append((nr, nc))
                seen.add((nr, nc))
                nr += 1
                nc += 1
            if len(seg) < 2:
                continue
            for idx, (sr, sc) in enumerate(seg):
                if idx % 2 == 0:
                    continue
                cur = int(out[sr, sc])
                for a, b in pairs:
                    if cur == a:
                        cur = b
                out[sr, sc] = cur
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((14, 14), dtype=int)
    chain = random.choice([[(1, 3), (3, 6), (6, 7)], [(4, 8), (8, 2), (2, 9)], [(7, 5), (5, 1), (1, 4)]])
    for r, (a, b) in enumerate(chain):
        grid[r, 0] = a
        grid[r, 1] = b
    starts = [(5, 4), (4, 8), (8, 3), (7, 9)]
    random.shuffle(starts)
    colors = [chain[0][0], chain[1][0], random.choice([2, 3, 4, 6, 8, 9])]
    for (r, c), color in zip(starts[:3], colors):
        length = random.randint(3, min(5, 14 - max(r, c)))
        for k in range(length):
            grid[r + k, c + k] = color
    return grid

