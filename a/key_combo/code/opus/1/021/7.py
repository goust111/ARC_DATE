import random
from collections import deque

import numpy as np


def _target_component(grid):
    mask = (grid == 3) | (grid == 2)
    seen = np.zeros(mask.shape, bool)
    best = []
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
        if len(cells) > len(best):
            best = cells
    return best


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    cells = _target_component(grid)
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    ir, ic = tuple(np.argwhere(grid == 2)[0])
    if ir == r0:
        direction = "up"
        tr0, tr1, tc0, tc1 = 0, r1, c0, c1
        line = [(r, ic) for r in range(0, r1 + 1)]
    elif ir == r1:
        direction = "down"
        tr0, tr1, tc0, tc1 = r0, grid.shape[0] - 1, c0, c1
        line = [(r, ic) for r in range(r0, grid.shape[0])]
    elif ic == c0:
        direction = "left"
        tr0, tr1, tc0, tc1 = r0, r1, 0, c1
        line = [(ir, c) for c in range(0, c1 + 1)]
    else:
        direction = "right"
        tr0, tr1, tc0, tc1 = r0, r1, c0, grid.shape[1] - 1
        line = [(ir, c) for c in range(c0, grid.shape[1])]

    markers = [tuple(p) for p in np.argwhere(grid == 9)]
    out[grid == 9] = 0
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = 3
    for p in line:
        out[p] = 2

    moving = []
    for r, c in markers:
        if tr0 <= r <= tr1 and tc0 <= c <= tc1:
            continue
        if tr0 <= r <= tr1:
            if c < tc0:
                moving.append((tc0 - c, r, c, (r, tc0), (r, tc0 - 1)))
            elif c > tc1:
                moving.append((c - tc1, r, c, (r, tc1), (r, tc1 + 1)))
        elif tc0 <= c <= tc1:
            if r < tr0:
                moving.append((tr0 - r, r, c, (tr0, c), (tr0 - 1, c)))
            elif r > tr1:
                moving.append((r - tr1, r, c, (tr1, c), (tr1 + 1, c)))
    moving.sort(key=lambda x: (x[0], x[1], x[2]))
    for i, item in enumerate(moving):
        out[item[3] if i == 0 else item[4]] = 9
    return out


def generate_input() -> np.ndarray:
    h, w = 16, 22
    grid = np.zeros((h, w), dtype=int)
    rh, rw = random.choice([(3, 4), (4, 5)])
    r0 = random.randint(5, 7)
    c0 = random.randint(8, 11)
    r1, c1 = r0 + rh - 1, c0 + rw - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 3
    direction = random.choice(["up", "down", "left", "right"])
    if direction == "up":
        grid[r0, (c0 + c1) // 2] = 2
        markers = [(r0 - 3, c0 + 1), (r1 + 3, c0 + 1), (r0 + 1, c0 - 4), (r0 + 2, c1 + 4), (r0 - 2, c1 - 1)]
    elif direction == "down":
        grid[r1, (c0 + c1) // 2] = 2
        markers = [(r1 + 3, c0 + 1), (r0 - 3, c0 + 1), (r0 + 1, c0 - 4), (r0 + 2, c1 + 4), (r1 + 2, c1 - 1)]
    elif direction == "left":
        grid[(r0 + r1) // 2, c0] = 2
        markers = [(r0 + 1, c0 - 4), (r0 + 1, c1 + 4), (r0 - 3, c0 + 1), (r1 + 3, c0 + 2), (r1 - 1, c0 - 3)]
    else:
        grid[(r0 + r1) // 2, c1] = 2
        markers = [(r0 + 1, c1 + 4), (r0 + 1, c0 - 4), (r0 - 3, c0 + 1), (r1 + 3, c0 + 2), (r1 - 1, c1 + 3)]
    for r, c in markers:
        if 0 <= r < h and 0 <= c < w and grid[r, c] == 0:
            grid[r, c] = 9
    return grid
