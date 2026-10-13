import random

import numpy as np


BG = 0
PATH = 5
CYCLE = [2, 7, 4, 3, 1, 6]


def _neighbors(pos):
    r, c = pos
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _right_turn(d1, d2):
    return d1[0] * d2[1] - d1[1] * d2[0] < 0


def _trace_path(mask: np.ndarray, start):
    h, w = mask.shape
    prev = None
    cur = start
    order = [start]
    while True:
        nxts = []
        for nr, nc in _neighbors(cur):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) != prev:
                nxts.append((nr, nc))
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("path is not a simple chain")
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the signal along the grey path and shift its color at each left or right turn.

    Input structure:
    - The background is black and a one-cell-wide orthogonal polyline is drawn mostly in grey.
    - Exactly one endpoint of that path is a colored start pixel from a fixed cyclic palette.
    - The path may contain several 90-degree bends but no branches.

    Rule:
    - Follow the path from the colored endpoint to the other end, repainting every grey path cell.
    - Keep the current signal color unchanged across straight segments.
    - At each bend, advance one step in the color cycle on a right turn and move one step backward on a left turn; the corner cell itself uses the updated color.

    Key insight:
    - The path geometry carries the instructions: whether a bend is left-handed or right-handed determines how the signal color changes.
    """
    g = np.asarray(grid, dtype=int)
    starts = np.argwhere((g != BG) & (g != PATH))
    if len(starts) != 1:
        raise ValueError("expected exactly one colored start")
    start = tuple(map(int, starts[0]))
    color = int(g[start])
    idx = CYCLE.index(color)
    mask = (g == PATH) | ((np.indices(g.shape)[0] == start[0]) & (np.indices(g.shape)[1] == start[1]))
    order = _trace_path(mask, start)
    out = g.copy()
    out[start] = CYCLE[idx]

    for i in range(1, len(order)):
        cur = order[i]
        if i < len(order) - 1:
            prev = order[i - 1]
            nxt = order[i + 1]
            d1 = (cur[0] - prev[0], cur[1] - prev[1])
            d2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            if d1 != d2:
                if _right_turn(d1, d2):
                    idx = (idx + 1) % len(CYCLE)
                else:
                    idx = (idx - 1) % len(CYCLE)
        out[cur] = CYCLE[idx]
    return out


def _make_path(rng: random.Random):
    max_steps = rng.randint(3, 6)
    direction = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
    dirs = [direction]
    for _ in range(max_steps - 1):
        turns = [(-direction[1], direction[0]), (direction[1], -direction[0])]
        if rng.random() < 0.35:
            nxt = direction
        else:
            nxt = turns[rng.randrange(2)]
        dirs.append(nxt)
        direction = nxt

    cells = [(0, 0)]
    used = {(0, 0)}
    cur = (0, 0)
    for d in dirs:
        length = rng.randint(2, 5)
        for _ in range(length):
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if nxt in used:
                return None
            adjacent = 0
            for nr, nc in _neighbors(nxt):
                if (nr, nc) in used:
                    adjacent += 1
            if adjacent != 1:
                return None
            cur = nxt
            used.add(cur)
            cells.append(cur)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    h = max_r - min_r + 5
    w = max_c - min_c + 5
    if h > 30 or w > 30:
        return None

    shifted = [(r - min_r + 2, c - min_c + 2) for r, c in cells]
    return h, w, shifted


def _gen_attempt(rng: random.Random):
    spec = _make_path(rng)
    if spec is None:
        return None
    h, w, cells = spec
    grid = np.full((h, w), BG, dtype=int)
    start_color = rng.choice(CYCLE)
    for r, c in cells[1:]:
        grid[r, c] = PATH
    grid[cells[0]] = start_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
