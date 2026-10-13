import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(comp)
    return comps


def _box(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), max(rs), min(cs), max(cs)


def _is_square(comp):
    r0, r1, c0, c1 = _box(comp)
    if r1 - r0 != c1 - c0:
        return False
    cells = set(comp)
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    blue_comps = _components(grid == 1)
    movers = [comp for comp in blue_comps if _is_square(comp)]
    blockers = set(pt for comp in blue_comps if not _is_square(comp) for pt in comp)
    reds = sorted(list(zip(*np.where(grid == 2))))
    if len(reds) != 2:
        return out
    (r0, c0), (r1, c1) = reds
    dr, dc = r1 - r0, c1 - c0

    out[(grid == 2) | (grid == 5)] = 0
    for comp in movers:
        for r, c in comp:
            out[r, c] = 0
    moved_boxes = []
    for comp in movers:
        nr_comp = []
        for r, c in comp:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = 1
                nr_comp.append((nr, nc))
        if nr_comp:
            moved_boxes.append(_box(nr_comp))

    for r0, r1, c0, c1 in moved_boxes:
        starts = [
            (r0, c0, -1, -1),
            (r0, c1, -1, 1),
            (r1, c0, 1, -1),
            (r1, c1, 1, 1),
        ]
        for sr, sc, rr, cc in starts:
            r, c = sr + rr, sc + cc
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in blockers:
                    break
                out[r, c] = 1
                r += rr
                c += cc
    return out


def _draw_square(grid, r0, c0, size):
    grid[r0:r0 + size, c0:c0 + size] = 1


def _draw_path(grid, a, b):
    (r0, c0), (r1, c1) = a, b
    grid[r0, c0] = 2
    r, c = r0, c0
    while c != c1:
        c += 1 if c1 > c else -1
        if (r, c) != b:
            grid[r, c] = 5
    while r != r1:
        r += 1 if r1 > r else -1
        if (r, c) != b:
            grid[r, c] = 5
    grid[r1, c1] = 2


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    size = random.choice([3, 4])
    sr = random.randint(1, 3)
    sc = random.randint(1, 4)
    _draw_square(grid, sr, sc, size)
    dr, dc = random.choice([(5, 5), (6, 4), (4, 6)])
    nr, nc = sr + dr, sc + dc
    obstacle = [(nr - 2, nc - 2), (nr - 2, nc - 1), (nr - 1, nc - 2)]
    for r, c in obstacle:
        if 0 <= r < 18 and 0 <= c < 18 and grid[r, c] == 0:
            grid[r, c] = 1
    if random.random() < 0.5:
        _draw_square(grid, random.randint(10, 12), random.randint(1, 3), 3)
    _draw_path(grid, (0, 10), (dr, 10 + dc))
    return grid
