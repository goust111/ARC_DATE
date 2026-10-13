import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    blue = np.argwhere(grid == 1)
    top, left = blue.min(axis=0)
    bottom, right = blue.max(axis=0)
    top = int(top)
    left = int(left)
    bottom = int(bottom)
    right = int(right)

    red_rows = [(int(np.sum(grid[r, :] == 2)), r) for r in range(grid.shape[0])]
    red_cols = [(int(np.sum(grid[:, c] == 2)), c) for c in range(grid.shape[1])]
    cr = max(red_rows)[1]
    cc = max(red_cols)[1]
    corners = [(top, left), (top, right), (bottom, left), (bottom, right)]
    order = {corners[0]: 0, corners[1]: 1, corners[2]: 2, corners[3]: 3}
    target = min(corners, key=lambda p: (abs(cr - p[0]) + abs(cc - p[1]), order[p]))

    out[out == 2] = 0
    out[top, left:right + 1] = 1
    out[bottom, left:right + 1] = 1
    out[top:bottom + 1, left] = 1
    out[top:bottom + 1, right] = 1
    out[grid == 7] = 7

    tr, tc = target
    if target == (top, left):
        external = [(-1, 0), (0, -1)]
        internal = [(top + 1, left + 1, 0, 1), (top + 1, left + 1, 1, 0)]
    elif target == (top, right):
        external = [(-1, 0), (0, 1)]
        internal = [(top + 1, right - 1, 0, -1), (top + 1, right - 1, 1, 0)]
    elif target == (bottom, left):
        external = [(1, 0), (0, -1)]
        internal = [(bottom - 1, left + 1, 0, 1), (bottom - 1, left + 1, -1, 0)]
    else:
        external = [(1, 0), (0, 1)]
        internal = [(bottom - 1, right - 1, 0, -1), (bottom - 1, right - 1, -1, 0)]

    for dr, dc in external:
        r, c = tr, tc
        while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = 2
            r += dr
            c += dc
    for r, c, dr, dc in internal:
        while top < r < bottom and left < c < right:
            if grid[r, c] == 7:
                break
            out[r, c] = 2
            r += dr
            c += dc
    out[grid == 7] = 7
    out[tr, tc] = 2
    return out


def _draw_frame(grid, top, left, bottom, right):
    grid[top, left:right + 1] = 1
    grid[bottom, left:right + 1] = 1
    grid[top:bottom + 1, left] = 1
    grid[top:bottom + 1, right] = 1


def generate_input() -> np.ndarray:
    rng = random.Random()
    size = rng.choice([15, 17, 19])
    grid = np.zeros((size, size), dtype=int)
    fh = rng.choice([7, 8])
    fw = rng.choice([7, 8])
    top = rng.randint(2, size - fh - 2)
    left = rng.randint(2, size - fw - 2)
    bottom = top + fh - 1
    right = left + fw - 1
    _draw_frame(grid, top, left, bottom, right)

    corner = rng.choice([(top, left), (top, right), (bottom, left), (bottom, right)])
    if corner == (top, left):
        cross = (rng.randint(0, top - 1), rng.randint(0, left - 1))
        paths = [(top + 1, left + 1, 0, 1), (top + 1, left + 1, 1, 0)]
    elif corner == (top, right):
        cross = (rng.randint(0, top - 1), rng.randint(right + 1, size - 1))
        paths = [(top + 1, right - 1, 0, -1), (top + 1, right - 1, 1, 0)]
    elif corner == (bottom, left):
        cross = (rng.randint(bottom + 1, size - 1), rng.randint(0, left - 1))
        paths = [(bottom - 1, left + 1, 0, 1), (bottom - 1, left + 1, -1, 0)]
    else:
        cross = (rng.randint(bottom + 1, size - 1), rng.randint(right + 1, size - 1))
        paths = [(bottom - 1, right - 1, 0, -1), (bottom - 1, right - 1, -1, 0)]

    blockers = set()
    for r, c, dr, dc in paths:
        cells = []
        rr, cc = r, c
        while top < rr < bottom and left < cc < right:
            cells.append((rr, cc))
            rr += dr
            cc += dc
        if cells:
            blockers.add(rng.choice(cells))
    all_inside = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]
    for p in rng.sample(all_inside, rng.randint(0, 2)):
        blockers.add(p)
    for r, c in blockers:
        grid[r, c] = 7

    cr, cc = cross
    grid[cr, :] = 2
    grid[:, cc] = 2
    return grid
