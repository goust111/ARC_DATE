import random
from collections import deque

import numpy as np


def _is_checker(b):
    vals = [int(x) for x in b.ravel() if x not in (0, 5)]
    if len(vals) != 9 or len(set(vals)) != 2:
        return False
    a = b[0, 0]
    other = next(x for x in set(vals) if x != a)
    for r in range(3):
        for c in range(3):
            if b[r, c] != (a if (r + c) % 2 == 0 else other):
                return False
    return True


def _rings(q):
    rings = []
    for r in range(q.shape[0] - 2):
        for c in range(q.shape[1] - 2):
            b = q[r:r + 3, c:c + 3]
            outer = [b[i, j] for i in range(3) for j in range(3) if (i, j) != (1, 1)]
            if b[1, 1] == 0 and len(set(map(int, outer))) == 1 and outer[0] not in (0, 5):
                rings.append((r, c, int(outer[0])))
    return rings


def _trace(q, start, color):
    path = [start]
    cur = start
    prev = None
    while True:
        options = []
        for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            p = (cur[0] + dr, cur[1] + dc)
            if 0 <= p[0] < q.shape[0] and 0 <= p[1] < q.shape[1] and q[p] == color and p != prev:
                options.append(p)
        if not options:
            return path
        prev, cur = cur, sorted(options)[0]
        path.append(cur)


def _representative(q):
    checker = None
    for r in range(q.shape[0] - 2):
        for c in range(q.shape[1] - 2):
            if _is_checker(q[r:r + 3, c:c + 3]):
                checker = (r, c)
                break
        if checker:
            break
    cr, cc = checker
    checker_cells = {(r, c) for r in range(cr, cr + 3) for c in range(cc, cc + 3)}
    checker_colors = set(int(q[p]) for p in checker_cells)
    guide = None
    for r, c in checker_cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if 0 <= p[0] < q.shape[0] and 0 <= p[1] < q.shape[1] and q[p] not in (0, 5) and int(q[p]) not in checker_colors:
                guide = p
                break
        if guide:
            break
    color = int(q[guide])
    endpoint = _trace(q, guide, color)[-1]
    for rr, cc, rc in _rings(q):
        ring_cells = {(r, c) for r in range(rr, rr + 3) for c in range(cc, cc + 3) if (r, c) != (rr + 1, cc + 1)}
        if any(abs(endpoint[0] - r) + abs(endpoint[1] - c) == 1 for r, c in ring_cells):
            return rc
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    gr = next(r for r in range(grid.shape[0]) if np.all(grid[r] == 5))
    gc = next(c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5))
    quads = [
        grid[:gr, :gc],
        grid[:gr, gc + 1:],
        grid[gr + 1:, :gc],
        grid[gr + 1:, gc + 1:],
    ]
    reps = [_representative(q) for q in quads]
    out = np.zeros((3, 3), dtype=int)
    out[1, :] = 5
    out[:, 1] = 5
    out[0, 0], out[0, 2], out[2, 0], out[2, 2] = reps
    return out


def _put_quadrant(q, path_color, ring_color, distractor):
    checker = np.array([[2, 3, 2], [3, 2, 3], [2, 3, 2]], dtype=int)
    q[4:7, 0:3] = checker
    for c in (3, 5, 7):
        q[5, c] = path_color
    q[4:7, 8:11] = ring_color
    q[5, 9] = 0
    q[0:3, 5:8] = distractor
    q[1, 6] = 0


def generate_input() -> np.ndarray:
    grid = np.zeros((23, 23), dtype=int)
    grid[11, :] = 5
    grid[:, 11] = 5
    origins = [(0, 0), (0, 12), (12, 0), (12, 12)]
    path_colors = random.sample([4, 6, 7, 8], 4)
    ring_colors = random.sample([1, 2, 3, 9], 4)
    distractors = random.sample([1, 2, 3, 9], 4)
    for (r, c), pc, rc, dc in zip(origins, path_colors, ring_colors, distractors):
        _put_quadrant(grid[r:r + 11, c:c + 11], pc, rc, dc)
    return grid
