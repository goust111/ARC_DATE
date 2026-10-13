import random

import numpy as np


def _rectangles(grid):
    rects = []
    for color in [int(x) for x in np.unique(grid) if x not in (0, 2, 7)]:
        pts = list(zip(*np.where(grid == color)))
        if len(pts) != 4:
            continue
        rows = sorted({r for r, _ in pts})
        cols = sorted({c for _, c in pts})
        if len(rows) == 2 and len(cols) == 2:
            corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
            if set(pts) == corners:
                rects.append({"color": color, "box": (rows[0], rows[1], cols[0], cols[1]),
                              "center": ((rows[0] + rows[1]) // 2, (cols[0] + cols[1]) // 2),
                              "corners": corners})
    return rects


def _clear_line(a, b, state, corner_cells, ignore):
    ar, ac = a
    br, bc = b
    if ar != br and ac != bc:
        return False
    dr = 0 if ar == br else (1 if br > ar else -1)
    dc = 0 if ac == bc else (1 if bc > ac else -1)
    r, c = ar + dr, ac + dc
    while (r, c) != (br, bc):
        if (r, c) not in corner_cells and (r, c) not in ignore and state[r, c] != 0:
            return False
        r += dr
        c += dc
    return True


def _line_cells(a, b):
    ar, ac = a
    br, bc = b
    dr = 0 if ar == br else (1 if br > ar else -1)
    dc = 0 if ac == bc else (1 if bc > ac else -1)
    cells = []
    r, c = ar, ac
    while (r, c) != (br, bc):
        cells.append((r, c))
        r += dr
        c += dc
    cells.append((br, bc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Greedily visit corner-defined rooms, filling each visited room into a new obstacle.

    Input structure: A red 2x2 start block and several colored sets of four
    corner markers define empty axis-aligned rectangular rooms.
    Rule: From the current point, choose the nearest unvisited room center that
    shares a row or column and has unobstructed line of sight. Paint the corridor
    through black cells orange, fill the reached room interior with its marker
    color, and repeat from that center. Filled rooms obstruct later visibility.
    Key insight: The traversal graph changes after every visit because a newly
    filled room becomes a blocker for subsequent line-of-sight choices.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    start = np.argwhere(grid == 2)
    current = tuple(start.min(axis=0)) if len(start) else (0, 0)
    rects = _rectangles(grid)
    corner_cells = set().union(*(r["corners"] for r in rects)) if rects else set()
    unvisited = set(range(len(rects)))
    current_ignore = set(zip(*np.where(grid == 2)))
    while True:
        choices = []
        for idx in unvisited:
            rect = rects[idx]
            if _clear_line(current, rect["center"], out, corner_cells, current_ignore):
                dist = abs(current[0] - rect["center"][0]) + abs(current[1] - rect["center"][1])
                choices.append((dist, rect["center"][0], rect["center"][1], idx))
        if not choices:
            break
        _, _, _, idx = min(choices)
        rect = rects[idx]
        for r, c in _line_cells(current, rect["center"]):
            if out[r, c] == 0:
                out[r, c] = 7
        r0, r1, c0, c1 = rect["box"]
        filled = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) not in rect["corners"]:
                    out[r, c] = rect["color"]
                    filled.add((r, c))
        current = rect["center"]
        current_ignore = filled
        unvisited.remove(idx)
    return out


def _draw_corners(grid, box, color):
    r0, r1, c0, c1 = box
    for p in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
        grid[p] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    grid[1:3, 1:3] = 2
    boxes = [
        (0, 2, 6, 8),
        (5, 9, 5, 9),
        (5, 9, 12, 16),
        (12, 16, 12, 16),
        (12, 16, 5, 9),
    ]
    colors = random.sample([1, 3, 4, 6, 8], len(boxes))
    for box, color in zip(boxes, colors):
        _draw_corners(grid, box, color)
    return grid

