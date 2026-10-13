import random

import numpy as np


def _ranges(n, divs):
    starts = [0] + [d + 1 for d in divs]
    ends = [d - 1 for d in divs] + [n - 1]
    return list(zip(starts, ends))


def _crop_mask(mask):
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1], (r0, c0)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    div_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == 5)]
    div_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    row_ranges = _ranges(grid.shape[0], div_rows)
    col_ranges = _ranges(grid.shape[1], div_cols)
    seed = None
    seed_pos = None
    seed_off = None
    for i, (r0, r1) in enumerate(row_ranges):
        for j, (c0, c1) in enumerate(col_ranges):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = cell == 3
            crop, off = _crop_mask(mask)
            if not np.array_equal(crop, np.fliplr(crop)):
                seed = crop
                seed_pos = (i, j)
                seed_off = off
    sr, sc = seed_pos
    for i in range(sr + 1, len(row_ranges)):
        r0, r1 = row_ranges[i]
        c0, c1 = col_ranges[sc]
        out[r0 : r1 + 1, c0 : c1 + 1][out[r0 : r1 + 1, c0 : c1 + 1] == 3] = 0
        pat = np.flipud(seed) if (i - sr) % 2 == 1 else seed
        ro, co = seed_off
        for rr in range(pat.shape[0]):
            for cc in range(pat.shape[1]):
                if pat[rr, cc]:
                    out[r0 + ro + rr, c0 + co + cc] = 3
    return out


def generate_input() -> np.ndarray:
    cell = 5
    n = cell * 3 + 2
    grid = np.zeros((n, n), dtype=int)
    divs = [cell, cell * 2 + 1]
    grid[divs, :] = 5
    grid[:, divs] = 5
    sym = [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]]),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]]),
    ]
    asym = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]]),
        np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]]),
        np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),
    ]
    bad = (random.randrange(2), random.randrange(3))
    row_ranges = _ranges(n, divs)
    col_ranges = _ranges(n, divs)
    for i, (r0, _) in enumerate(row_ranges):
        for j, (c0, _) in enumerate(col_ranges):
            pat = random.choice(asym if (i, j) == bad else sym) * 3
            grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = pat
    return grid
