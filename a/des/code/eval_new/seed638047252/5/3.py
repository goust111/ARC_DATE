import random
from collections import Counter, deque

import numpy as np


GREY = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_legend(grid, bg):
    h, w = grid.shape
    candidates = []
    for r in [0, h - 1]:
        for c in range(w - 5):
            vals = [int(grid[r, c + i]) for i in range(6)]
            if all(v not in (bg, GREY) for v in vals) and len(set(vals)) == 6:
                candidates.append(([(r, c + i) for i in range(6)], vals))
    for c in [0, w - 1]:
        for r in range(h - 5):
            vals = [int(grid[r + i, c]) for i in range(6)]
            if all(v not in (bg, GREY) for v in vals) and len(set(vals)) == 6:
                candidates.append(([(r + i, c) for i in range(6)], vals))
    if not candidates:
        return [], set()
    cells, vals = candidates[0]
    return vals, set(cells)


def _colored_path_cells(grid, bg, legend_cells):
    h, w = grid.shape
    path = {(r, c) for r in range(h) for c in range(w) if grid[r, c] == GREY}
    for r in range(h):
        for c in range(w):
            if (r, c) in legend_cells or grid[r, c] in (bg, GREY):
                continue
            grey_neighbors = 0
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == GREY:
                    grey_neighbors += 1
            if grey_neighbors >= 2:
                path.add((r, c))
    return path


def _components(cells):
    remaining = set(cells)
    comps = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in DIRS:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _find_seed_blocks(grid, bg, legend_cells):
    h, w = grid.shape
    blocks = []
    seen = set()
    for r in range(h - 1):
        for c in range(w - 1):
            cells = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if cells & legend_cells or cells & seen:
                continue
            vals = [int(grid[x, y]) for x, y in cells]
            if len(set(vals)) == 1 and vals[0] not in (bg, GREY):
                blocks.append((cells, vals[0]))
                seen |= cells
    return blocks


def _degree(cell, comp):
    r, c = cell
    return sum((r + dr, c + dc) in comp for dr, dc in DIRS)


def _adjacent_seed(endpoint, blocks):
    er, ec = endpoint
    for cells, color in blocks:
        if any(abs(er - r) + abs(ec - c) == 1 for r, c in cells):
            return color
    return None


def _slot_from_endpoint(grid, bg, endpoint, neighbor):
    h, w = grid.shape
    er, ec = endpoint
    nr, nc = neighbor
    odr, odc = er - nr, ec - nc
    candidates = []
    if odc == 1:
        candidates = [(er - 1, ec + 1), (er, ec + 1)]
    elif odc == -1:
        candidates = [(er - 1, ec - 2), (er, ec - 2)]
    elif odr == 1:
        candidates = [(er + 1, ec - 1), (er + 1, ec)]
    elif odr == -1:
        candidates = [(er - 2, ec - 1), (er - 2, ec)]
    for r0, c0 in candidates:
        cells = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
        if all(0 <= r < h and 0 <= c < w and grid[r, c] == bg for r, c in cells):
            return cells
    return None


def _ordered_path(comp, start):
    endpoints = [cell for cell in comp if _degree(cell, comp) <= 1]
    order = [start]
    prev = None
    cur = start
    while True:
        nbrs = []
        r, c = cur
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if nxt in comp and nxt != prev:
                nbrs.append(nxt)
        if not nbrs:
            break
        prev, cur = cur, nbrs[0]
        order.append(cur)
        if cur in endpoints and cur != start:
            break
    return order


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    bg = _background(grid)
    ring, legend_cells = _find_legend(grid, bg)
    if not ring:
        return out
    ring_index = {color: i for i, color in enumerate(ring)}
    path_cells = _colored_path_cells(grid, bg, legend_cells)
    seed_blocks = _find_seed_blocks(grid, bg, legend_cells)

    for comp in _components(path_cells):
        endpoints = [cell for cell in comp if _degree(cell, comp) <= 1]
        if len(endpoints) < 2:
            continue
        start = None
        seed_color = None
        for endpoint in endpoints:
            color = _adjacent_seed(endpoint, seed_blocks)
            if color is not None:
                start = endpoint
                seed_color = color
                break
        if start is None or seed_color not in ring_index:
            continue
        order = _ordered_path(comp, start)
        end = order[-1]
        er, ec = end
        prev = order[-2] if len(order) > 1 else start
        slot = _slot_from_endpoint(grid, bg, end, prev)
        if slot is None:
            continue
        filters = sum(1 for r, c in order if grid[r, c] not in (bg, GREY))
        final_color = ring[(ring_index[seed_color] + filters) % len(ring)]
        for cell in slot:
            out[cell] = final_color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(15, 24)
    w = random.randint(16, 26)
    grid = np.zeros((h, w), dtype=int)
    ring = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 6)
    start_col = random.randint(1, w - 7)
    for i, color in enumerate(ring):
        grid[0, start_col + i] = color

    possible_rows = list(range(4, h - 2, 4))
    random.shuffle(possible_rows)
    pipe_count = random.randint(2, min(4, len(possible_rows)))
    seed_colors = random.choices(ring, k=pipe_count)

    for row, seed_color in zip(possible_rows[:pipe_count], seed_colors):
        if row + 1 >= h:
            continue
        grid[row - 1 : row + 1, 1:3] = seed_color
        end_col = w - 4
        grid[row, 3 : end_col + 1] = GREY
        filter_count = random.randint(1, min(4, max(1, end_col - 5)))
        choices = list(range(5, end_col - 1))
        random.shuffle(choices)
        placed = []
        for c in choices:
            if all(abs(c - old) > 1 for old in placed):
                placed.append(c)
            if len(placed) == filter_count:
                break
        for c in placed:
            grid[row, c] = random.choice([x for x in ring if x != seed_color])
    return grid
