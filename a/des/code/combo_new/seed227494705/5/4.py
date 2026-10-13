import random
import numpy as np


BG = 0
GREY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _center(cells):
    best = None
    best_deg = -1
    s = set(cells)
    for r, c in cells:
        deg = sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if deg > best_deg:
            best_deg = deg
            best = (r, c)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Connect paired same-color arrows and mark channel crossings grey.

    Input structure: Each foreground color forms exactly two T-like arrow
    markers on a black background. The pair is aligned in a row or column and
    points toward each other.
    Rule: For each same-color pair, fill the one-pixel channel between the two
    arrow centers. Channels paint black cells in the pair color; a black cell
    belonging to two or more channels becomes grey. Original arrow pixels keep
    their colors.
    Key insight: Arrow shape supplies the center and direction, color supplies
    the pair, and channel overlaps use a special mixing color.
    """
    out = grid.copy()
    original = grid != BG
    claims = {}
    for col in [int(x) for x in np.unique(grid) if x != BG]:
        comps = _components(grid == col)
        if len(comps) != 2:
            continue
        a, b = [_center(c) for c in comps]
        cells = []
        if a[0] == b[0]:
            r = a[0]
            for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
                cells.append((r, c))
        elif a[1] == b[1]:
            c = a[1]
            for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
                cells.append((r, c))
        for p in cells:
            if not original[p]:
                claims.setdefault(p, set()).add(col)
    for (r, c), cols in claims.items():
        out[r, c] = GREY if len(cols) > 1 else next(iter(cols))
    return out


def _draw_arrow(grid, r, c, direction, color):
    dr, dc = direction
    grid[r, c] = color
    for k in range(1, 3):
        grid[r + dr * k, c + dc * k] = color
    tip = (r + dr * 2, c + dc * 2)
    if dr:
        grid[tip[0], tip[1] - 1] = color
        grid[tip[0], tip[1] + 1] = color
    else:
        grid[tip[0] - 1, tip[1]] = color
        grid[tip[0] + 1, tip[1]] = color


def generate_input() -> np.ndarray:
    h = w = 25
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    pairs = [((8, 4), (8, 20), (0, 1), (0, -1)),
             ((4, 12), (20, 12), (1, 0), (-1, 0)),
             ((16, 3), (16, 18), (0, 1), (0, -1)),
             ((3, 18), (18, 18), (1, 0), (-1, 0))]
    for col, (a, b, da, db) in zip(colors, pairs):
        _draw_arrow(grid, a[0], a[1], da, col)
        _draw_arrow(grid, b[0], b[1], db, col)
    return grid
