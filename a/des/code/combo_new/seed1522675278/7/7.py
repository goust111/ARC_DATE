import random
from collections import defaultdict

import numpy as np


BG = 0
MIRROR = 5
WHITE = 8

REFLECT = {
    "\\": {(0, 1): (1, 0), (1, 0): (0, 1), (0, -1): (-1, 0), (-1, 0): (0, -1)},
    "/": {(0, 1): (-1, 0), (-1, 0): (0, 1), (0, -1): (1, 0), (1, 0): (0, -1)},
}


def _mirror_kind(grid, r, c):
    for dr, dc in ((1, 1), (-1, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and grid[rr, cc] == MIRROR:
            return "\\"
    return "/"


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    towers = [(r, c, int(grid[r, c])) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] not in (BG, MIRROR)]
    hits = defaultdict(set)
    for r0, c0, color in towers:
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = r0, c0
            dr, dc = d
            seen = set()
            while True:
                state = (r, c, dr, dc)
                if state in seen:
                    break
                seen.add(state)
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                    break
                if grid[nr, nc] == MIRROR:
                    kind = _mirror_kind(grid, nr, nc)
                    dr, dc = REFLECT[kind][(dr, dc)]
                    continue
                if grid[nr, nc] not in (BG, MIRROR):
                    break
                hits[(nr, nc)].add(color)
                r, c = nr, nc
    for (r, c), cols in hits.items():
        if len(cols) == 1:
            out[r, c] = next(iter(cols))
        else:
            out[r, c] = WHITE
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 9], random.randint(2, 3)):
        grid[random.randint(2, h - 3), random.randint(2, w - 3)] = color
    for _ in range(random.randint(2, 4)):
        r = random.randint(2, h - 4)
        c = random.randint(2, w - 4)
        if random.random() < 0.5:
            grid[r, c] = MIRROR
            grid[r + 1, c + 1] = MIRROR
        else:
            grid[r, c + 1] = MIRROR
            grid[r + 1, c] = MIRROR
    return grid
