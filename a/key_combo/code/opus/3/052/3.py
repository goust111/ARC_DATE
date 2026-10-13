from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), min(cs), max(rs), max(cs)


def _find_table(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    magenta = list(map(tuple, np.argwhere(grid == 6)))
    if not magenta:
        return None
    return _bbox(magenta)


def _complete_table(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    width = inner.shape[1]
    cycle: list[int | None] = [None] * width
    for rr in range(inner.shape[0]):
        for cc in range(inner.shape[1]):
            color = int(inner[rr, cc])
            if color == 0:
                continue
            phase = cc if rr <= cc else cc + 1
            cycle[phase % width] = color
    palette = [1, 2, 3, 4, 7, 8, 9, 5]
    used = {c for c in cycle if c is not None}
    for i, value in enumerate(cycle):
        if value is not None:
            continue
        for color in palette:
            if color not in used:
                cycle[i] = color
                used.add(color)
                break
    return [int(c) for c in cycle]


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the lower-right color table and use its first row to recolor hub-adjacent branches by contact count."""
    out = np.array(grid, copy=True)
    table = _find_table(grid)
    lookup = [1, 2]
    table_cells: set[tuple[int, int]] = set()
    if table is not None:
        r0, c0, r1, c1 = table
        lookup = _complete_table(grid, table)
        cycle = lookup
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                phase = (cc - c0 - 1) if rr <= r0 + 1 + (cc - c0 - 1) else (cc - c0)
                if out[rr, cc] == 0:
                    out[rr, cc] = cycle[phase % len(cycle)]
        table_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    grey = (grid == 5).copy()
    for r, c in table_cells:
        grey[r, c] = False
    hub = None
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grey[r : r + 3, c : c + 3]):
                hub = (r, c)
                break
        if hub:
            break
    if hub is None:
        return out

    hr, hc = hub
    hub_cells = {(r, c) for r in range(hr, hr + 3) for c in range(hc, hc + 3)}
    branch_mask = grey.copy()
    for r, c in hub_cells:
        branch_mask[r, c] = False
    for comp in _components(branch_mask):
        k = 0
        for r, c in comp:
            if any((r + dr, c + dc) in hub_cells for dr, dc in DIRS):
                k += 1
        if k == 0:
            continue
        color = lookup[(k - 1) % len(lookup)]
        for r, c in comp:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    hr, hc = 8, 8
    grid[hr : hr + 3, hc : hc + 3] = 5
    palette = random.sample([1, 2, 3, 4], 4)
    branches = [
        ([(hr - i, hc + 1) for i in range(1, random.choice([4, 5]))], (hr - 4, hc + 2), palette[0]),
        ([(hr + 1, hc + 3 + i) for i in range(random.choice([4, 5]))], (hr + 2, hc + 7), palette[1]),
        ([(hr + 3 + i, hc + 1) for i in range(random.choice([3, 4]))], (hr + 7, hc), palette[2]),
        ([(hr + 1, hc - 1 - i) for i in range(random.choice([4, 5]))], (hr, hc - 5), palette[3]),
    ]
    for cells, seed, color in branches[: random.randint(3, 4)]:
        for r, c in cells:
            grid[r, c] = 5
        grid[seed] = color
    r0, c0 = 15, 15
    grid[r0 : r0 + 4, c0] = 6
    grid[r0 : r0 + 4, c0 + 3] = 6
    grid[r0, c0 : c0 + 4] = 6
    grid[r0 + 3, c0 : c0 + 4] = 6
    table_vals = random.sample([1, 2, 3, 4], 2)
    grid[r0 + 1, c0 + 1] = table_vals[0]
    grid[r0 + 1, c0 + 2] = table_vals[1]
    grid[r0 + 2, c0 + 1] = table_vals[1]
    return grid
