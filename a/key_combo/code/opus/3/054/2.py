import random

import numpy as np


DIRS = {
    (0, 0): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (1, 1): (1, 1),
}


def _find_seed(grid):
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            if np.all(np.isin(block, [2, 6])) and np.any(block == 2):
                return r, c, block.copy()
    raise ValueError("seed block not found")


def _side_vec(dr, dc, left=True):
    if dr == dc:
        vec = (0, dc)
    else:
        vec = (dr, 0)
    if not left:
        vec = (-vec[0], -vec[1])
    return vec


def main(grid: np.ndarray) -> np.ndarray:
    """Extend magenta diagonal rays from red seed corners, detouring around green/orange bars and stopping at blue walls."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    sr, sc, seed = _find_seed(g)
    out[sr : sr + 2, sc : sc + 2] = 6

    for lr, lc in sorted(map(tuple, np.argwhere(seed == 2))):
        dr, dc = DIRS[(int(lr), int(lc))]
        r, c = sr + int(lr), sc + int(lc)
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                break
            cell = int(g[nr, nc])
            if cell == 0:
                out[nr, nc] = 6
                r, c = nr, nc
            elif cell in (3, 7):
                sv = _side_vec(dr, dc, left=(cell == 3))
                rr, cc = r + sv[0], c + sv[1]
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    break
                if g[rr, cc] == 0:
                    out[rr, cc] = 6
                r, c = rr, cc
            elif cell == 1:
                break
            else:
                break
    return out


def _bar(grid, r, c, color, horizontal):
    pts = []
    for k in (-1, 0, 1):
        rr, cc = (r, c + k) if horizontal else (r + k, c)
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            pts.append((rr, cc))
    for rr, cc in pts:
        if grid[rr, cc] == 0:
            grid[rr, cc] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    sr, sc = 9, 9
    grid[sr : sr + 2, sc : sc + 2] = 6
    corner = random.choice(list(DIRS))
    grid[sr + corner[0], sc + corner[1]] = 2
    dr, dc = DIRS[corner]
    r, c = sr + corner[0], sc + corner[1]

    hit1 = (r + 3 * dr, c + 3 * dc)
    color1 = random.choice([3, 7])
    _bar(grid, hit1[0], hit1[1], color1, horizontal=(dr == dc))
    sv = _side_vec(dr, dc, left=(color1 == 3))
    lane = (r + 2 * dr + sv[0], c + 2 * dc + sv[1])
    hit2 = (lane[0] + 4 * dr, lane[1] + 4 * dc)
    if 1 <= hit2[0] < 19 and 1 <= hit2[1] < 19:
        _bar(grid, hit2[0], hit2[1], 1, horizontal=(dr != dc))
    _bar(grid, random.choice([3, 15]), random.choice([3, 15]), random.choice([3, 7, 1]), True)
    return grid
