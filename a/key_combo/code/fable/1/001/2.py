import random
from collections import deque

import numpy as np


BG = 0
WALL = 5


def _diag_pairings(points):
    points = tuple(sorted(points))

    def rec(rest):
        if not rest:
            yield []
            return
        first = rest[0]
        for i in range(1, len(rest)):
            other = rest[i]
            if first[0] == other[0] or first[1] == other[1]:
                next_rest = rest[1:i] + rest[i + 1 :]
                for tail in rec(next_rest):
                    yield [(first, other)] + tail

    return list(rec(points))


def _draw_line(out, a, b):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            out[r0, c] = WALL
    else:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            out[r, c0] = WALL


def _flood_open(mask_wall, start):
    h, w = mask_wall.shape
    q = deque([start])
    seen = {start}
    touches = start[0] in (0, h - 1) or start[1] in (0, w - 1)
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask_wall[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                touches = touches or nr in (0, h - 1) or nc in (0, w - 1)
                q.append((nr, nc))
    return seen, touches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair grey endpoints into straight wall segments, then fill only the seed
    regions enclosed by the newly drawn wall topology.
    """
    g = np.asarray(grid, dtype=int)
    grey = [(int(r), int(c)) for r, c in zip(*np.where(g == WALL))]
    pairings = _diag_pairings(grey)
    assert len(pairings) == 1, "expected a unique grey-point pairing"

    out = g.copy()
    for a, b in pairings[0]:
        _draw_line(out, a, b)

    seeds = [(int(r), int(c), int(g[r, c])) for r, c in zip(*np.where((g != BG) & (g != WALL)))]
    wall_mask = out == WALL
    decisions = []
    for r, c, color in seeds:
        region, touches = _flood_open(wall_mask, (r, c))
        decisions.append((region, touches, (r, c), color))

    for region, touches, seed, color in decisions:
        if touches:
            out[seed] = BG
        else:
            for r, c in region:
                out[r, c] = color
    return out


def _place_wall_points(grid, rt, rb, cl, cr):
    pts = [
        (rt, cl), (rt, cr),
        (rb, cl + 1), (rb, cr + 1),
        (rt - 1, cl + 2), (rb + 1, cl + 2),
        (rt - 2, cr - 2), (rb + 2, cr - 2),
    ]
    for r, c in pts:
        if not (1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in pts:
        grid[r, c] = WALL
    return True


def _gen_attempt():
    h = random.randint(18, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    rt = random.randint(4, min(7, h - 10))
    rb = random.randint(rt + 5, min(h - 4, rt + 9))
    cl = random.randint(2, 5)
    cr = random.randint(max(cl + 7, 10), w - 4)
    if not _place_wall_points(grid, rt, rb, cl, cr):
        return None
    if len(_diag_pairings([(int(r), int(c)) for r, c in zip(*np.where(grid == WALL))])) != 1:
        return None

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
    inside_cells = [(r, c) for r in range(rt + 1, rb) for c in range(cl + 3, cr - 2)]
    r, c = random.choice(inside_cells)
    grid[r, c] = colors[0]
    for color in colors[1:]:
        for _ in range(100):
            r = random.choice(list(range(1, rt - 2)) + list(range(rb + 3, h - 1)))
            c = random.randint(1, w - 2)
            if grid[r, c] == BG:
                grid[r, c] = color
                break
    try:
        out = main(grid)
    except Exception:
        return None
    if not np.any(out == colors[0]) or any(np.any(out == color) for color in colors[1:]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
