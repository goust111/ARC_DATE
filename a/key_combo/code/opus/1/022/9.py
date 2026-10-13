import random
import numpy as np


def _line_positions(grid, axis):
    nonzero = np.sum(grid != 0, axis=axis)
    limit = max(3, int(max(nonzero) * 0.55))
    return [int(i) for i, n in enumerate(nonzero) if n >= limit]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    rows = _line_positions(grid, 1)
    cols = _line_positions(grid, 0)
    if len(rows) < 5 or len(cols) < 5:
        return grid.copy()
    rows, cols = rows[:5], cols[:5]
    cr, cc = rows[1] - rows[0] - 1, cols[1] - cols[0] - 1
    seed = np.zeros((4, 4), dtype=int)
    sprite = []
    for i in range(4):
        for j in range(4):
            corners = [grid[rows[i], cols[j]], grid[rows[i], cols[j + 1]],
                       grid[rows[i + 1], cols[j]], grid[rows[i + 1], cols[j + 1]]]
            if corners[0] != 0 and corners.count(corners[0]) == 4 and corners[0] != 5:
                seed[i, j] = int(corners[0])
                if seed[i, j] == 7:
                    r0, r1 = rows[i] + 1, rows[i + 1]
                    c0, c1 = cols[j] + 1, cols[j + 1]
                    sprite = [(int(r - r0), int(c - c0)) for r, c in np.argwhere(grid[r0:r1, c0:c1] == 7)]
    out = np.zeros((4 * cr, 4 * cc), dtype=int)
    for i in range(4):
        for j in range(4):
            block = out[i * cr:(i + 1) * cr, j * cc:(j + 1) * cc]
            color = seed[i, j]
            if color == 0:
                continue
            if color == 7:
                for r, c in sprite:
                    if 0 <= r < cr and 0 <= c < cc:
                        block[r, c] = 7
            else:
                block[:, :] = color
    return out


def generate_input() -> np.ndarray:
    cr = cc = random.randint(3, 4)
    h = 5 + 4 * cr
    w = 5 + 4 * cc
    grid = np.zeros((h, w), dtype=int)
    rows = [i * (cr + 1) for i in range(5)]
    cols = [i * (cc + 1) for i in range(5)]
    grid[rows, :] = 5
    grid[:, cols] = 5
    cells = [(i, j) for i in range(4) for j in range(4)]
    sprite_cell = random.choice(cells)
    active = set(random.sample([c for c in cells if c != sprite_cell], random.randint(3, 7)))
    active.add(sprite_cell)
    for i, j in active:
        color = 7 if (i, j) == sprite_cell else random.choice([1, 2, 3])
        for r, c in ((rows[i], cols[j]), (rows[i], cols[j + 1]), (rows[i + 1], cols[j]), (rows[i + 1], cols[j + 1])):
            grid[r, c] = color
    for i, j in random.sample([c for c in cells if c not in active], random.randint(2, 5)):
        colors = random.sample([1, 2, 3], 2)
        grid[rows[i], cols[j]] = colors[0]
        grid[rows[i], cols[j + 1]] = colors[1]
    si, sj = sprite_cell
    pts = random.choice([[(0, 0), (1, 0), (1, 1)], [(0, 1), (1, 1), (2, 1), (2, 0)], [(0, 0), (0, 1), (1, 1), (2, 1)]])
    off_r = random.randint(0, cr - max(r for r, _ in pts) - 1)
    off_c = random.randint(0, cc - max(c for _, c in pts) - 1)
    for dr, dc in pts:
        grid[rows[si] + 1 + off_r + dr, cols[sj] + 1 + off_c + dc] = 7
    return grid
