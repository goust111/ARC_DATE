import math
import random

import numpy as np


SEED_COLORS = (1, 2, 3, 4, 7)


def _separator_indices(grid, color=5):
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
    return rows, cols


def _cell_bounds(seps, idx):
    return seps[idx] + 1, seps[idx + 1]


def _build_active(seed_mask, n):
    m = n * n
    active = np.zeros((m, m), dtype=bool)
    for i in range(n):
        for j in range(n):
            if seed_mask[i, j]:
                active[i * n : (i + 1) * n, j * n : (j + 1) * n] = True
    return active


def _nearest_seed_color(cell_r, cell_c, seeds):
    best = None
    for sr, sc, color in seeds:
        key = (abs(cell_r - sr) + abs(cell_c - sc), sc, sr, color)
        if best is None or key < best[0]:
            best = (key, color)
    return best[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Recover a small minority-cell blueprint, scale it up to decide which cells are active, and then extend each active cell's color straight downward.

    Input Model:
    Gray separator lines define a regular cell lattice; only a few seed cells contain a single colored pixel, all at the same relative offset inside their cells.

    Algorithm:
    Read the seed pattern from the top-left blueprint block, expand each blueprint 1-cell into an n-by-n active cell block, assign colors to newly active cells by nearest seed, then grow every active pixel down its exact pixel column while skipping separator rows.

    Output:
    Return the grid with the original separators intact, inactive cells left empty, and active columns painted downward until another colored pixel blocks them.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    sep_rows, sep_cols = _separator_indices(grid)
    cell_h = sep_rows[1] - sep_rows[0] - 1
    m = len(sep_rows) - 1
    n = int(round(math.sqrt(m)))
    seeds = []
    rel = None
    seed_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        r0, r1 = _cell_bounds(sep_rows, i)
        for j in range(n):
            c0, c1 = _cell_bounds(sep_cols, j)
            cell = grid[r0:r1, c0:c1]
            pts = np.argwhere((cell != 0) & (cell != 5))
            if pts.size == 0:
                continue
            rr, cc = pts[0]
            rel = (int(rr), int(cc))
            color = int(cell[rr, cc])
            seed_mask[i, j] = True
            seeds.append((i, j, color))
    active = _build_active(seed_mask, n)
    for cell_r in range(active.shape[0]):
        r0, r1 = _cell_bounds(sep_rows, cell_r)
        for cell_c in range(active.shape[1]):
            c0, c1 = _cell_bounds(sep_cols, cell_c)
            cell = out[r0:r1, c0:c1]
            pts = np.argwhere((cell != 0) & (cell != 5))
            if not active[cell_r, cell_c]:
                continue
            if pts.size:
                rr, cc = map(int, pts[0])
                color = int(cell[rr, cc])
            else:
                rr, cc = rel
                color = _nearest_seed_color(cell_r, cell_c, seeds)
                out[r0 + rr, c0 + cc] = color
            pr = r0 + rr
            pc = c0 + cc
            for rr2 in range(pr + 1, out.shape[0]):
                if out[rr2, pc] == 5:
                    continue
                if out[rr2, pc] == 0:
                    out[rr2, pc] = color
                else:
                    break
    return out


def _generate_case(rng: random.Random):
    n = 3
    m = n * n
    cell = 2
    side = m * cell + (m + 1)
    grid = np.zeros((side, side), dtype=int)
    for k in range(m + 1):
        pos = k * (cell + 1)
        grid[pos, :] = 5
        grid[:, pos] = 5
    rp = rng.randint(0, cell - 1)
    cp = rng.randint(0, cell - 1)
    count = rng.randint(2, 4)
    seed_positions = rng.sample([(i, j) for i in range(n) for j in range(n)], count)
    colors = rng.sample(SEED_COLORS, count)
    for (i, j), color in zip(seed_positions, colors):
        r0 = 1 + i * (cell + 1)
        c0 = 1 + j * (cell + 1)
        grid[r0 + rp, c0 + cp] = color
    seed_mask = np.zeros((n, n), dtype=bool)
    seeds = []
    for (i, j), color in zip(seed_positions, colors):
        seed_mask[i, j] = True
        seeds.append((i, j, color))
    return grid, main(grid)


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]
