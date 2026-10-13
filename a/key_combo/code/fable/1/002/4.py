import random

import numpy as np


BG = 0


def _periodic_positions(a: int, b: int, limit: int):
    p = abs(a - b)
    assert p > 1
    start = a % p
    return [x for x in range(limit) if x % p == start]


def _interval_index(lines, x):
    for i in range(len(lines) - 1):
        if lines[i] < x < lines[i + 1]:
            return i
    return None


def _cell_slices(lines):
    return [(lines[i] + 1, lines[i + 1]) for i in range(len(lines) - 1) if lines[i] + 1 < lines[i + 1]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer invisible periodic grid lines from two top ticks and two left ticks.
    Fill lattice cells in coarse cell space, and complete both boundary tick
    sequences.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    top_pts = [(int(c), int(g[0, c])) for c in range(1, w) if g[0, c] != BG]
    left_pts = [(int(r), int(g[r, 0])) for r in range(1, h) if g[r, 0] != BG]
    assert len(top_pts) == 2 and len(left_pts) == 2
    top_color = top_pts[0][1]
    left_color = left_pts[0][1]
    assert all(v == top_color for _, v in top_pts) and all(v == left_color for _, v in left_pts)
    cols = _periodic_positions(top_pts[0][0], top_pts[1][0], w)
    rows = _periodic_positions(left_pts[0][0], left_pts[1][0], h)
    assert len(cols) >= 3 and len(rows) >= 3

    seeds = {}
    for r, c in zip(*np.where(g != BG)):
        r, c = int(r), int(c)
        v = int(g[r, c])
        if r == 0 or c == 0:
            continue
        if r in rows or c in cols:
            continue
        seeds.setdefault(v, []).append((r, c))

    out = np.zeros_like(g)
    for c in cols:
        out[0, c] = top_color
    for r in rows:
        out[r, 0] = left_color

    row_cells = _cell_slices(rows)
    col_cells = _cell_slices(cols)
    for color, pts in seeds.items():
        assert len(pts) in (1, 2)
        coarse = []
        for r, c in pts:
            ri = _interval_index(rows, r)
            ci = _interval_index(cols, c)
            assert ri is not None and ci is not None
            coarse.append((ri, ci))
        rlo = min(r for r, _ in coarse)
        rhi = max(r for r, _ in coarse)
        clo = min(c for _, c in coarse)
        chi = max(c for _, c in coarse)
        for ri in range(rlo, rhi + 1):
            for ci in range(clo, chi + 1):
                r0, r1 = row_cells[ri]
                c0, c1 = col_cells[ci]
                out[r0:r1, c0:c1] = color
    return out


def _gen_attempt():
    h, w = random.randint(18, 27), random.randint(18, 27)
    grid = np.zeros((h, w), dtype=int)
    p_col = random.choice([4, 5, 6])
    p_row = random.choice([4, 5, 6])
    c0 = random.randint(1, p_col - 1)
    r0 = random.randint(1, p_row - 1)
    cols = [c for c in range(w) if c % p_col == c0]
    rows = [r for r in range(h) if r % p_row == r0]
    if len(cols) < 4 or len(rows) < 4:
        return None
    top_color, left_color = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 2)
    for c in random.sample(cols, 2):
        grid[0, c] = top_color
    for r in random.sample(rows, 2):
        grid[r, 0] = left_color

    row_cells = _cell_slices(rows)
    col_cells = _cell_slices(cols)
    if len(row_cells) < 3 or len(col_cells) < 3:
        return None
    seed_colors = [c for c in range(1, 10) if c not in (top_color, left_color)]
    random.shuffle(seed_colors)
    specs = []
    for color in seed_colors[: random.randint(4, 6)]:
        count = 2 if random.random() < 0.65 else 1
        cells = random.sample([(r, c) for r in range(len(row_cells)) for c in range(len(col_cells))], count)
        specs.append((color, cells))
    used = set()
    for color, cells in specs:
        for ri, ci in cells:
            r0s, r1s = row_cells[ri]
            c0s, c1s = col_cells[ci]
            for _ in range(50):
                p = (random.randint(r0s, r1s - 1), random.randint(c0s, c1s - 1))
                if p not in used:
                    used.add(p)
                    grid[p] = color
                    break
            else:
                return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
