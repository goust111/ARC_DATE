import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
RAIL_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _rail_at(grid, r, c):
    col = int(grid[r, c])
    if col in (BG, GREY, MAROON):
        return None
    coords = np.argwhere(grid == col)
    if len(coords) < 2:
        return None
    rows = set(int(x) for x in coords[:, 0])
    cols = set(int(x) for x in coords[:, 1])
    if len(rows) == 1:
        return col, "h", next(iter(rows))
    if len(cols) == 1:
        return col, "v", next(iter(cols))
    return None


def _marker(grid):
    coords = np.argwhere(grid == MAROON)
    if len(coords) == 0:
        return None
    r, c = coords[0]
    return int(r), int(c)


def main(grid: np.ndarray) -> np.ndarray:
    mark = _marker(grid)
    if mark is None:
        return grid.copy()
    mr, mc = mark
    out = grid.copy()
    for cells in _components(grid, GREY):
        rail = None
        for r, c in cells:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    rail = _rail_at(grid, nr, nc)
                    if rail is not None:
                        break
            if rail is not None:
                break
        if rail is None:
            continue
        color, orient, _line = rail
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if orient == "h":
            shift = mc - max(cs) if max(cs) < mc else mc - min(cs)
            moved = [(r, c + shift) for r, c in cells]
        else:
            shift = mr - max(rs) if max(rs) < mr else mr - min(rs)
            moved = [(r + shift, c) for r, c in cells]
        for r, c in cells:
            out[r, c] = BG
        for r, c in moved:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = color
    return out


def _draw_case(grid, color, side):
    h, w = grid.shape
    mr, mc = _marker(grid)
    if side == "left":
        row = mr
        grid[row, 0:mc] = color
        base = [(row - 1, random.randint(1, max(1, mc // 2)))]
    elif side == "right":
        row = mr
        grid[row, mc + 1 : w] = color
        base = [(row + 1, random.randint(mc + 2, w - 2))]
    elif side == "top":
        col = mc
        grid[0:mr, col] = color
        base = [(random.randint(1, max(1, mr // 2)), col - 1)]
    else:
        col = mc
        grid[mr + 1 : h, col] = color
        base = [(random.randint(mr + 2, h - 2), col + 1)]
    r, c = base[0]
    shape = [(r, c)]
    if random.random() < 0.7:
        shape.append((r, c + 1 if c + 1 < w - 1 else c - 1))
    if random.random() < 0.5:
        shape.append((r + 1 if r + 1 < h - 1 else r - 1, c))
    for rr, cc in shape:
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
            grid[rr, cc] = GREY


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(18, 32)
        w = random.randint(18, 32)
        grid = np.zeros((h, w), dtype=int)
        mr = random.randint(7, h - 8)
        mc = random.randint(7, w - 8)
        grid[mr, mc] = MAROON
        sides = random.sample(["left", "right", "top", "bottom"], random.randint(1, 3))
        colors = random.sample(RAIL_COLORS, len(sides))
        for color, side in zip(colors, sides):
            _draw_case(grid, color, side)
        if np.any(grid == GREY) and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate seed794177829/3/8 input")
