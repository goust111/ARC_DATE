import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
AXIS = 3
YELLOW = 4


def _find_axes(g: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == AXIS)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == AXIS)]
    assert len(rows) == 1 and len(cols) == 1, "expected one green cross"
    return rows[0], cols[0]


def _diag_path(p, q):
    r0, c0 = p
    r1, c1 = q
    assert abs(r1 - r0) == abs(c1 - c0) and r0 != r1
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    return [(r0 + k * dr, c0 + k * dc) for k in range(abs(r1 - r0) + 1)]


def _line_cells(g: np.ndarray) -> set[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in zip(*np.where(g == BLUE))]
    groups = {}
    for r, c in pts:
        groups.setdefault(("d", r - c), []).append((r, c))
        groups.setdefault(("s", r + c), []).append((r, c))
    cells = set()
    for (kind, _), group in groups.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda p: (p[0], p[1]))
        p, q = group[0], group[-1]
        if abs(p[0] - q[0]) == abs(p[1] - q[1]):
            cells.update(_diag_path(p, q))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw blue diagonal connections and mirrored red markers, coloring their intersections yellow.

    Input structure:
    - A black grid contains one full green row and one full green column as reflection axes.
    - Blue endpoint pixels occur in diagonal-aligned pairs away from the axes.
    - Red marker pixels are sparse and lie off the axes in one quadrant.

    Rule:
    - Connect each diagonal group of blue endpoint pixels by filling the segment between its extremes.
    - Reflect every red marker across the green row, across the green column, and across both axes.
    - Red marks overwrite black or green cells, but any red mark that lies on a blue connection becomes yellow.
    - Keep the original green cross and pure blue line cells otherwise unchanged.

    Key insight:
    - Yellow marks are the spatial intersection of two independently constructed structures: diagonal blue lines and axis-reflected red markers.
    """
    g = np.asarray(grid, dtype=int)
    ar, ac = _find_axes(g)
    line = _line_cells(g)
    out = g.copy()
    for r, c in line:
        if out[r, c] in (BG, BLUE):
            out[r, c] = BLUE

    red = [(int(r), int(c)) for r, c in zip(*np.where(g == RED))]
    for r, c in red:
        for rr, cc in ((r, c), (2 * ar - r, c), (r, 2 * ac - c), (2 * ar - r, 2 * ac - c)):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = YELLOW if (rr, cc) in line else RED
    return out


def _free(grid, r, c):
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == BG


def _place_blue_pair(grid, through=None, used_diagonals=None):
    used_diagonals = used_diagonals if used_diagonals is not None else set()
    h, w = grid.shape
    for _ in range(200):
        slope = random.choice([1, -1])
        if through is None:
            r0 = random.randint(1, h - 2)
            c0 = random.randint(1, w - 2)
        else:
            r0, c0 = through
        k1 = random.randint(2, 4)
        k2 = random.randint(2, 4)
        p = (r0 - k1, c0 - slope * k1)
        q = (r0 + k2, c0 + slope * k2)
        diag_key = ("d", p[0] - p[1]) if slope == 1 else ("s", p[0] + p[1])
        if diag_key in used_diagonals:
            continue
        path = _diag_path(p, q) if 0 <= p[0] < h and 0 <= p[1] < w and 0 <= q[0] < h and 0 <= q[1] < w else []
        if not path:
            continue
        if any(grid[r, c] != BG for r, c in (p, q)):
            continue
        if any(grid[r, c] == AXIS for r, c in path):
            continue
        grid[p] = BLUE
        grid[q] = BLUE
        used_diagonals.add(diag_key)
        return True
    return False


def _gen_attempt():
    h = w = random.randint(13, 18)
    ar = random.randint(5, h - 6)
    ac = random.randint(5, w - 6)
    grid = np.zeros((h, w), dtype=int)
    grid[ar, :] = AXIS
    grid[:, ac] = AXIS

    qname = random.choice(["TL", "TR", "BL", "BR"])
    rows = range(1, ar) if qname in ("TL", "TR") else range(ar + 1, h - 1)
    cols = range(1, ac) if qname in ("TL", "BL") else range(ac + 1, w - 1)
    red_cells = random.sample([(r, c) for r in rows for c in cols], random.randint(2, 4))
    for r, c in red_cells:
        grid[r, c] = RED

    target_red = random.choice(red_cells)
    target = random.choice(
        [
            target_red,
            (2 * ar - target_red[0], target_red[1]),
            (target_red[0], 2 * ac - target_red[1]),
            (2 * ar - target_red[0], 2 * ac - target_red[1]),
        ]
    )
    if not (1 <= target[0] < h - 1 and 1 <= target[1] < w - 1):
        return None
    used_diagonals = set()
    if not _place_blue_pair(grid, target, used_diagonals):
        return None
    for _ in range(random.randint(1, 2)):
        _place_blue_pair(grid, used_diagonals=used_diagonals)

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
