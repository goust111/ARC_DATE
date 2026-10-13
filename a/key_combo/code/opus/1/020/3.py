import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _turn(direction, amount):
    idx = DIRS.index(direction)
    return DIRS[(idx + amount) % 4]


def _dominant_rows(grid):
    colors = []
    for r in range(grid.shape[0]):
        vals = [int(x) for x in grid[r] if x not in (0, 8)]
        if not vals:
            colors.append(0)
        else:
            colors.append(max(set(vals), key=vals.count))
    return colors


def _cross_center_and_tips(grid):
    pts = set(zip(*np.where(grid == 8)))
    if not pts:
        return None, []
    center = max(pts, key=lambda p: sum((p[0] + dr, p[1] + dc) in pts for dr, dc in DIRS))
    tips = [p for p in pts if sum((p[0] + dr, p[1] + dc) in pts for dr, dc in DIRS) == 1]
    return center, tips


def _path_from_tip(tip, center, occupied, h, w):
    dr = 0 if tip[0] == center[0] else (1 if tip[0] > center[0] else -1)
    dc = 0 if tip[1] == center[1] else (1 if tip[1] > center[1] else -1)
    outward = (dr, dc)
    heading = _turn(outward, 1)
    pos = tip
    path = []
    local = set()
    for _ in range(h * w):
        moved = False
        for turn_amount in (1, 0, -1):
            ndir = _turn(heading, turn_amount)
            step = 1 if turn_amount != 0 else 2
            cells = []
            rr, cc = pos
            ok = True
            for _s in range(step):
                rr += ndir[0]
                cc += ndir[1]
                if not (0 <= rr < h and 0 <= cc < w) or (rr, cc) in occupied or (rr, cc) in local:
                    ok = False
                    break
                cells.append((rr, cc))
            if not ok:
                continue
            for cell in cells:
                prev = pos if cell == cells[0] else cells[cells.index(cell) - 1]
                for ar, ac in DIRS:
                    nb = (cell[0] + ar, cell[1] + ac)
                    if nb in occupied or nb in local:
                        if nb != prev:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                for cell in cells:
                    local.add(cell)
                    path.append(cell)
                pos = cells[-1]
                heading = ndir
                moved = True
                break
        if not moved:
            break
    occupied.update(local)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Sample cleaned horizontal stripe colors along spiral paths grown from a cyan cross.

    Input structure: Noisy horizontal stripes encode a dominant color for each
    row, and a short cyan cross supplies four arm tips near the grid center.
    Rule: Compute each row's dominant nonblack, noncyan color. From every cross
    arm tip, grow a turn-priority spiral path; each visited cell is colored by
    the dominant color of its row. All original stripes, noise, and cyan pixels
    are removed.
    Key insight: The stripes are a row-indexed palette, while the cross-derived
    spiral supplies only the geometry to sample that palette.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    row_colors = _dominant_rows(grid)
    center, tips = _cross_center_and_tips(grid)
    out = np.zeros_like(grid)
    if center is None:
        return out
    occupied = set(zip(*np.where(grid == 8)))
    for tip in sorted(tips):
        for r, c in _path_from_tip(tip, center, occupied, h, w):
            out[r, c] = row_colors[r]
    return out


def generate_input() -> np.ndarray:
    n = random.choice([15, 17, 19])
    colors = random.sample([1, 2, 3, 4, 6], random.randint(4, 5))
    grid = np.zeros((n, n), dtype=int)
    for r in range(n):
        base = random.choice(colors)
        grid[r, :] = base
        for c in random.sample(range(n), random.randint(1, 3)):
            grid[r, c] = random.choice(colors + [0])
    center = n // 2 + random.choice([-1, 0, 1])
    arm = random.choice([1, 2])
    cr = cc = center
    grid[cr, cc] = 8
    for d in range(1, arm + 1):
        grid[cr - d, cc] = 8
        grid[cr + d, cc] = 8
        grid[cr, cc - d] = 8
        grid[cr, cc + d] = 8
    return grid
