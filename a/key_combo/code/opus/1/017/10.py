import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _turn(d, left=True):
    i = DIRS.index(d)
    return DIRS[(i - 1) % 4] if left else DIRS[(i + 1) % 4]


def _legend(grid):
    pairs = []
    for r in range(grid.shape[0]):
        if grid[r, 0] != 0 and grid[r, 1] != 0:
            pairs.append((int(grid[r, 0]), int(grid[r, 1])))
        elif pairs:
            break
    return pairs


def _next_color(color, pairs):
    cur = color
    matched = False
    for a, b in pairs:
        if cur == a:
            cur = b
            matched = True
    return cur if matched else None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    pairs = _legend(grid)
    legend_h = len(pairs)
    left_colors = {a for a, _ in pairs}
    h, w = grid.shape
    starts = []
    for r in range(legend_h, h):
        for c in range(w):
            if grid[r, c] in left_colors:
                neigh = []
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 5:
                        neigh.append((dr, dc))
                if len(neigh) == 1:
                    starts.append((r, c, int(grid[r, c]), neigh[0]))
    for sr, sc, color, d in sorted(starts):
        out[sr, sc] = 0
        pos = (sr + d[0], sc + d[1])
        turns = 0
        steps = 0
        cur = color
        while 0 <= pos[0] < h and 0 <= pos[1] < w and grid[pos] == 5 and steps < h * w * 4:
            out[pos] = cur
            nxt_color = _next_color(cur, pairs)
            if nxt_color is None:
                break
            nr, nc = pos[0] + d[0], pos[1] + d[1]
            if not (0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 5):
                turns += 1
                d = _turn(d, left=turns % 2 == 1)
                nr, nc = pos[0] + d[0], pos[1] + d[1]
                if not (0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 5):
                    break
            pos = (nr, nc)
            cur = nxt_color
            steps += 1
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    a, b, c, d = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    pairs = [(a, b), (b, c), (c, d)]
    for r, (a, b) in enumerate(pairs):
        grid[r, 0] = a
        grid[r, 1] = b
    row = random.randint(12, 14)
    col = random.randint(2, 4)
    l1 = random.randint(4, 6)
    u1 = random.randint(3, 5)
    l2 = random.randint(3, 5)
    u2 = random.randint(2, 4)
    if row - u1 - u2 < 4 or col + l1 + l2 >= 15:
        row, col, l1, u1, l2, u2 = 13, 3, 5, 4, 4, 3
    pts = [(row, col + 1), (row, col + l1), (row - u1, col + l1), (row - u1, col + l1 + l2), (row - u1 - u2, col + l1 + l2)]
    for (r1, c1), (r2, c2) in zip(pts, pts[1:]):
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                grid[r1, c] = 5
        else:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                grid[r, c1] = 5
    grid[row, col] = pairs[0][0]
    return grid
