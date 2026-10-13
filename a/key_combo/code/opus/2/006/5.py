import random

import numpy as np


def _intervals(is_separator):
    intervals = []
    start = None
    for i, sep in enumerate(is_separator):
        if sep:
            if start is not None:
                intervals.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        intervals.append((start, len(is_separator)))
    return intervals


def _spiral(rows, cols, start):
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = [start]
    seen = {start}
    y, x = start
    d = 0
    while True:
        moved = False
        for _ in range(4):
            dy, dx = dirs[d]
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and (ny, nx) not in seen:
                y, x = ny, nx
                seen.add((y, x))
                visited.append((y, x))
                moved = True
                break
            d = (d + 1) % 4
        if not moved:
            break
    return visited


def main(grid: np.ndarray) -> np.ndarray:
    row_sep = [np.all(row == 1) for row in grid]
    col_sep = [np.all(grid[:, c] == 1) for c in range(grid.shape[1])]
    row_ranges = _intervals(row_sep)
    col_ranges = _intervals(col_sep)
    cells = {}
    source = None
    marker = None
    motif = []
    for cr, (r1, r2) in enumerate(row_ranges):
        for cc, (c1, c2) in enumerate(col_ranges):
            cells[(cr, cc)] = (r1, r2, c1, c2)
            sub = grid[r1:r2, c1:c2]
            reds = list(zip(*np.where(sub == 2)))
            yellows = list(zip(*np.where(sub == 4)))
            if reds:
                source = (cr, cc)
                motif = [(int(r), int(c)) for r, c in reds]
            if yellows:
                marker = (cr, cc)
    out = grid.copy()
    out[out == 4] = 0
    for cr, cc in _spiral(len(row_ranges), len(col_ranges), marker):
        r1, r2, c1, c2 = cells[(cr, cc)]
        for dr, dc in motif:
            y, x = r1 + dr, c1 + dc
            if y < r2 and x < c2:
                out[y, x] = 2
    if source is not None:
        r1, r2, c1, c2 = cells[source]
        for dr, dc in motif:
            if r1 + dr < r2 and c1 + dc < c2:
                out[r1 + dr, c1 + dc] = 2
    return out


def generate_input() -> np.ndarray:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    cell = random.randint(2, 4)
    h = rows * cell + rows - 1
    w = cols * cell + cols - 1
    grid = np.zeros((h, w), dtype=int)
    for r in range(cell, h, cell + 1):
        grid[r, :] = 1
    for c in range(cell, w, cell + 1):
        grid[:, c] = 1
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    marker_cell = random.choice(corners)
    possible_sources = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != marker_cell]
    source_cell = random.choice(possible_sources)
    count = random.randint(2, min(4, cell * cell))
    shape = [(random.randrange(cell), random.randrange(cell))]
    while len(shape) < count:
        y, x = random.choice(shape)
        candidates = []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < cell and 0 <= nx < cell and (ny, nx) not in shape:
                candidates.append((ny, nx))
        if candidates:
            shape.append(random.choice(candidates))
    sr, sc = source_cell
    for dr, dc in shape:
        grid[sr * (cell + 1) + dr, sc * (cell + 1) + dc] = 2
    mr, mc = marker_cell
    grid[mr * (cell + 1) + random.randrange(cell), mc * (cell + 1) + random.randrange(cell)] = 4
    return grid
