import random
from collections import deque

import numpy as np


def _table_from_first(first):
    n = len(first)
    diag = first[0]
    prev = first[1]
    cycle = [first[i] for i in range(2, n, 2)] or [prev]
    table = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            dist = abs(r - c)
            near = min(r, c)
            if dist == 0:
                val = diag
            elif dist <= near + 1:
                val = prev
            else:
                val = cycle[((dist - near - 2) // (near + 2)) % len(cycle)]
            table[r, c] = val
    return table


def _intervals(lines, limit):
    pts = [-1] + sorted(lines) + [limit]
    return [(a + 1, b) for a, b in zip(pts, pts[1:]) if b - a > 1]


def main(grid: np.ndarray) -> np.ndarray:
    n = 0
    while n < grid.shape[1] and grid[0, n] not in (0, 5):
        n += 1
    table = _table_from_first([int(x) for x in grid[0, :n]])
    out = grid.copy()
    out[:n, :n] = 0

    gray_counts_r = [np.count_nonzero(grid[r] == 5) for r in range(grid.shape[0])]
    gray_counts_c = [np.count_nonzero(grid[:, c] == 5) for c in range(grid.shape[1])]
    row_lines = [r for r, cnt in enumerate(gray_counts_r) if cnt >= max(gray_counts_r) * 0.7 and cnt > 0]
    col_lines = [c for c, cnt in enumerate(gray_counts_c) if cnt >= max(gray_counts_c) * 0.7 and cnt > 0]
    row_cells = _intervals(row_lines, grid.shape[0])
    col_cells = _intervals(col_lines, grid.shape[1])

    for r, (r0, r1) in enumerate(row_cells[:n]):
        for c, (c0, c1) in enumerate(col_cells[:n]):
            cell = out[r0:r1, c0:c1]
            cell[cell == 5] = table[r, c]
    out[np.ix_(row_lines, range(out.shape[1]))] = 0
    out[np.ix_(range(out.shape[0]), col_lines)] = 0
    out[:n, :n] = 0
    return out


def _shape(cell):
    coords = {(random.randrange(cell), random.randrange(cell))}
    target = random.randint(2, min(5, cell * cell))
    while len(coords) < target:
        r, c = random.choice(tuple(coords))
        for nr, nc in random.sample([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)], 4):
            if 0 <= nr < cell and 0 <= nc < cell:
                coords.add((nr, nc))
                break
    return coords


def generate_input() -> np.ndarray:
    n = random.choice([4, 5])
    cell = random.choice([3, 4]) if n == 4 else 3
    first = [random.choice([1, 2, 3, 4])]
    first.append(random.choice([x for x in [1, 2, 3, 4] if x != first[0]]))
    cycle = random.sample([x for x in [1, 2, 3, 4] if x not in first], 2)
    for c in range(2, n):
        first.append(cycle[(c - 2) // 2 % len(cycle)])
    table = _table_from_first(first)

    top = left = n + 1
    gh = n * cell + n + 1
    h = top + gh
    w = left + gh
    grid = np.zeros((h, w), dtype=int)
    visible = table.copy()
    candidates = [(r, c) for r in range(1, n) for c in range(n)]
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(3, max(3, len(candidates) // 2))]:
        visible[r, c] = 0
        visible[c, r] = 0 if c != 0 else visible[c, r]
    visible[0] = table[0]
    grid[:n, :n] = visible

    for i in range(n + 1):
        grid[top + i * (cell + 1), left : left + gh] = 5
        grid[top : top + gh, left + i * (cell + 1)] = 5
    for r in range(n):
        for c in range(n):
            r0 = top + 1 + r * (cell + 1)
            c0 = left + 1 + c * (cell + 1)
            for rr, cc in _shape(cell):
                grid[r0 + rr, c0 + cc] = 5
    return grid
