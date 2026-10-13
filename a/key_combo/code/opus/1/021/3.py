import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            x, y = q.popleft()
            cells.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    bars = []
    for comp in _components(grid == 5):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if len(set(rs)) == 1:
            r = rs[0]
            bars.append((len(comp), r, min(cs), max(cs)))
    ranks = {bar: i + 1 for i, bar in enumerate(sorted(bars))}
    recolor = {1: 2, 2: 4, 3: 1}
    cycle = [1, 4, 3]
    for bar, rank in ranks.items():
        _, r, c0, c1 = bar
        out[r, c0 : c1 + 1] = recolor[rank]
        marker_c = c1 + 1
        out[r, marker_c] = 2
        for k, rr in enumerate(range(r + 1, grid.shape[0])):
            out[rr, marker_c] = cycle[(k + rank - 1) % 3]
    return out


def generate_input() -> np.ndarray:
    h, w = random.choice([(15, 20), (16, 21), (18, 22)])
    while True:
        grid = np.zeros((h, w), dtype=int)
        rows = sorted(random.sample(range(1, h - 7), 3))
        if min(rows[1] - rows[0], rows[2] - rows[1]) < 3:
            continue
        lengths = random.sample(range(3, 9), 3)
        cols = random.sample(range(1, w - 10), 3)
        spans = [(c, c + length - 1, c + length) for c, length in zip(cols, lengths)]
        clear = True
        for i, (_, _, marker_c) in enumerate(spans):
            for j, (bc0, bc1, _) in enumerate(spans):
                if i != j and marker_c >= bc0 and marker_c <= bc1:
                    clear = False
        if clear:
            break
    for r, length, c in zip(rows, lengths, cols):
        grid[r, c : c + length] = 5
        grid[r, c + length] = 2
    return grid
