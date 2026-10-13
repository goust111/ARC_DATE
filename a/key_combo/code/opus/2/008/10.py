import random
from collections import deque

import numpy as np


def _count_components(g, color):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if g[r, c] != color or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _complete_missing(g):
    h, w = g.shape
    best = None
    for r in range(h - 2):
        for c in range(w - 2):
            if not np.all(g[r:r + 3, c:c + 3] == 0):
                continue
            if c <= w // 2 <= c + 2:
                continue
            fill = 0
            for rr in range(r, r + 3):
                for cc in range(c, c + 3):
                    fill += int(g[rr, w - 1 - cc] != 0)
            if fill:
                best = max(best or (0, r, c), (fill, r, c))
    out = g.copy()
    if best is not None:
        _, r, c = best
        for rr in range(r, r + 3):
            for cc in range(c, c + 3):
                out[rr, cc] = g[rr, w - 1 - cc]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    completed = _complete_missing(np.array(grid, copy=True))
    candidates = []
    for color in [int(c) for c in np.unique(completed) if c != 0]:
        if _count_components(completed, color) == 1:
            candidates.append((int(np.sum(completed == color)), color))
    color = max(candidates)[1]
    return np.full((3, 3), color, dtype=int)


def generate_input() -> np.ndarray:
    g = np.zeros((11, 11), dtype=int)
    target = random.choice([1, 2, 3, 4, 7])
    others = random.sample([c for c in [1, 2, 3, 4, 7, 8, 9] if c != target], 3)
    miss_r, miss_c = random.choice([(3, 7), (4, 7), (2, 7)])
    row = miss_r + 1
    g[row, 1:7] = target
    g[row, 10] = target
    g[row - 1, 5] = target
    g[row + 1, 5] = target
    g[row - 1, 1:4] = target
    g[row + 1, 1:4] = target
    g[1, 1:4] = others[0]
    g[1, 7:10] = others[0]
    g[8, 1:3] = others[1]
    g[8, 8:10] = others[1]
    g[2:4, 0] = others[2]
    g[7:9, 10] = others[2]
    g[miss_r:miss_r + 3, miss_c:miss_c + 3] = 0
    return g
