import random
from collections import defaultdict

import numpy as np


BG = 8
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 9]


def _find_line(grid):
    row_hits = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == RED)]
    if row_hits:
        return "h", row_hits[0]
    col_hits = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]
    if col_hits:
        return "v", col_hits[0]
    raise ValueError("no full red reference line found")


def _boundary(cells, h, w):
    cells = set(cells)
    out = []
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or (nr, nc) not in cells:
                out.append((r, c))
                break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    orient, line = _find_line(grid)
    out = grid.copy()

    by_color = defaultdict(list)
    for r, c in np.argwhere((grid != BG) & (grid != RED)):
        by_color[int(grid[r, c])].append((int(r), int(c)))

    for color, cells in by_color.items():
        if orient == "h":
            neg = [(r, c) for r, c in cells if r < line]
            pos = [(r, c) for r, c in cells if r > line]
            adjacent = any(abs(r - line) == 1 for r, _ in cells)
        else:
            neg = [(r, c) for r, c in cells if c < line]
            pos = [(r, c) for r, c in cells if c > line]
            adjacent = any(abs(c - line) == 1 for _, c in cells)

        if neg and pos:
            doomed = neg if len(neg) < len(pos) else pos
            for r, c in doomed:
                out[r, c] = BG
        elif adjacent:
            for r, c in _boundary(cells, h, w):
                out[r, c] = RED

    if orient == "h":
        out[line, :] = RED
    else:
        out[:, line] = RED
    return out


def _place(grid, cells, color):
    h, w = grid.shape
    for r, c in cells:
        if r < 0 or r >= h or c < 0 or c >= w or grid[r, c] != BG:
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _rect_cells(r0, c0, height, width):
    return [(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)]


def _notched_rect(r0, c0, height, width):
    cells = set(_rect_cells(r0, c0, height, width))
    if height >= 3 and width >= 3:
        cells.discard((r0, c0))
        cells.discard((r0 + height - 1, c0 + width - 1))
    return sorted(cells)


def _side_rect_h(line, side, adj, h, w, rng):
    height = rng.randint(2, 4)
    width = rng.randint(3, 6)
    c0 = rng.randint(1, w - width - 2)
    if side < 0:
        max_bottom = line - 1 if adj else line - 3
        if max_bottom - height + 1 < 1:
            return None
        r0 = rng.randint(1, max_bottom - height + 1)
    else:
        min_top = line + 1 if adj else line + 3
        if min_top + height > h - 1:
            return None
        r0 = rng.randint(min_top, h - height - 1)
    cells = _notched_rect(r0, c0, height, width) if rng.random() < 0.35 else _rect_cells(r0, c0, height, width)
    return cells


def _side_rect_v(line, side, adj, h, w, rng):
    height = rng.randint(3, 6)
    width = rng.randint(2, 4)
    r0 = rng.randint(1, h - height - 2)
    if side < 0:
        max_right = line - 1 if adj else line - 3
        if max_right - width + 1 < 1:
            return None
        c0 = rng.randint(1, max_right - width + 1)
    else:
        min_left = line + 1 if adj else line + 3
        if min_left + width > w - 1:
            return None
        c0 = rng.randint(min_left, w - width - 1)
    cells = _notched_rect(r0, c0, height, width) if rng.random() < 0.35 else _rect_cells(r0, c0, height, width)
    return cells


def _cross_h(line, h, w, rng):
    width = rng.randint(3, 6)
    c0 = rng.randint(1, w - width - 2)
    top_h = rng.randint(1, min(4, line - 1))
    bot_h = rng.randint(1, min(4, h - line - 2))
    if top_h == bot_h:
        if bot_h < min(4, h - line - 2):
            bot_h += 1
        elif top_h > 1:
            top_h -= 1
    if top_h == bot_h:
        return None
    cells = _rect_cells(line - top_h, c0, top_h, width)
    cells += _rect_cells(line + 1, c0, bot_h, width)
    return cells


def _cross_v(line, h, w, rng):
    height = rng.randint(3, 6)
    r0 = rng.randint(1, h - height - 2)
    left_w = rng.randint(1, min(4, line - 1))
    right_w = rng.randint(1, min(4, w - line - 2))
    if left_w == right_w:
        if right_w < min(4, w - line - 2):
            right_w += 1
        elif left_w > 1:
            left_w -= 1
    if left_w == right_w:
        return None
    cells = _rect_cells(r0, line - left_w, height, left_w)
    cells += _rect_cells(r0, line + 1, height, right_w)
    return cells


def _attempt(rng):
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    grid = np.full((h, w), BG, dtype=int)
    orient = rng.choice(["h", "v"])
    if orient == "h":
        line = rng.randint(5, h - 6)
        grid[line, :] = RED
    else:
        line = rng.randint(5, w - 6)
        grid[:, line] = RED

    colors = PALETTE[:]
    rng.shuffle(colors)
    specs = [
        ("far", rng.choice([-1, 1])),
        ("touch", rng.choice([-1, 1])),
        ("cross", 0),
    ]
    for _ in range(rng.randint(1, 3)):
        specs.append((rng.choice(["far", "touch", "cross"]), rng.choice([-1, 1])))

    for kind, side in specs:
        color = colors.pop()
        placed = False
        for _ in range(80):
            if orient == "h":
                if kind == "cross":
                    cells = _cross_h(line, h, w, rng)
                else:
                    cells = _side_rect_h(line, side, kind == "touch", h, w, rng)
            else:
                if kind == "cross":
                    cells = _cross_v(line, h, w, rng)
                else:
                    cells = _side_rect_v(line, side, kind == "touch", h, w, rng)
            if cells is not None and _place(grid, cells, color):
                placed = True
                break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
