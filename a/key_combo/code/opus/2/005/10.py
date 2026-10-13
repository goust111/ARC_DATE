import random

import numpy as np


def _key(arr):
    return (arr.shape, tuple(arr.astype(int).ravel()))


def _cw(arr, k):
    return np.rot90(arr, -k)


def _find_rects(grid):
    pts = [tuple(map(int, p)) for p in np.argwhere(grid == 6)]
    pset = set(pts)
    rects = []
    used = set()
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    candidates = []
    for r1 in rows:
        for r2 in rows:
            if r2 <= r1:
                continue
            for c1 in cols:
                for c2 in cols:
                    if c2 <= c1:
                        continue
                    corners = {(r1, c1), (r1, c2), (r2, c1), (r2, c2)}
                    if corners <= pset and (r2 - r1) == (c2 - c1):
                        area = (r2 - r1) * (c2 - c1)
                        candidates.append((area, (r1, c1, r2, c2), corners))
    for _, rect, corners in sorted(candidates):
        if not (corners & used):
            rects.append(rect)
            used |= corners
            if len(rects) == 3:
                break
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    rects = _find_rects(grid)
    tiles = []
    for r1, c1, r2, c2 in rects:
        tiles.append(grid[r1 + 1:r2, c1 + 1:c2].copy())
    if not tiles:
        return np.zeros((1, 1), dtype=int)
    base = tiles[0]
    rots = [_cw(base, k) for k in range(4)]
    by_angle = {}
    for tile in tiles:
        for k, r in enumerate(rots):
            if tile.shape == r.shape and np.array_equal(tile, r):
                by_angle[k] = tile
                break
    missing = next(k for k in range(4) if k not in by_angle)
    by_angle[missing] = rots[missing]
    h, w = grid.shape
    orange = np.argwhere(grid == 7)
    if len(orange):
        pr, pc = map(int, orange[0])
    else:
        pr, pc = 0, 0
    placeholder_q = (0 if pr < h / 2 else 1, 0 if pc < w / 2 else 1)
    default_q = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    q_to_angle = {q: a for a, q in default_q.items()}
    placement = dict(default_q)
    old_q = default_q[missing]
    displaced = q_to_angle[placeholder_q]
    placement[missing] = placeholder_q
    placement[displaced] = old_q
    n = base.shape[0]
    out = np.zeros((2 * n, 2 * n), dtype=int)
    for angle, q in placement.items():
        r0, c0 = q[0] * n, q[1] * n
        out[r0:r0 + n, c0:c0 + n] = by_angle[angle]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    base = np.array([[2, 2, 0, 0],
                     [0, 2, 0, 0],
                     [0, 2, 2, 2],
                     [0, 0, 0, 2]], dtype=int)
    missing = random.randrange(4)
    angles = [a for a in range(4) if a != missing]
    random.shuffle(angles)
    rects = [(0, 0), (0, 9), (9, 3)]
    for (r0, c0), angle in zip(rects, angles):
        tile = _cw(base, angle)
        grid[r0, c0] = grid[r0, c0 + 5] = grid[r0 + 5, c0] = grid[r0 + 5, c0 + 5] = 6
        grid[r0 + 1:r0 + 5, c0 + 1:c0 + 5] = tile
    quadrants = [(6, 1), (6, 15), (15, 1), (15, 15)]
    pr, pc = random.choice(quadrants)
    grid[pr, pc] = 7
    return grid
