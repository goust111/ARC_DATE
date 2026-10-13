import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _regions(grid):
    gray_rows = [i for i in range(grid.shape[0]) if np.all(grid[i, :] == 5)]
    gray_cols = [j for j in range(grid.shape[1]) if np.all(grid[:, j] == 5)]
    row_bounds = [-1] + gray_rows + [grid.shape[0]]
    col_bounds = [-1] + gray_cols + [grid.shape[1]]
    regs = []
    for mr in range(len(row_bounds) - 1):
        for mc in range(len(col_bounds) - 1):
            r0, r1 = row_bounds[mr] + 1, row_bounds[mr + 1]
            c0, c1 = col_bounds[mc] + 1, col_bounds[mc + 1]
            regs.append((mr, mc, slice(r0, r1), slice(c0, c1)))
    return regs


def main(grid: np.ndarray) -> np.ndarray:
    """Keep the largest patterned macro-cell as template and fill empty cells by distance-ranked rotations."""
    out = grid.copy()
    regs = _regions(grid)
    filled = []
    for mr, mc, rs, cs in regs:
        region = grid[rs, cs]
        count = int(np.sum((region != 0) & (region != 5)))
        if count:
            filled.append((count, mr, mc, rs, cs))
    template = max(filled, key=lambda x: x[0])
    _, tr, tc, trs, tcs = template
    template_region = grid[trs, tcs].copy()
    for _, mr, mc, rs, cs in filled:
        if (mr, mc) != (tr, tc):
            mask = (out[rs, cs] != 0) & (out[rs, cs] != 5)
            out[rs, cs][mask] = 0
    occupied = {(mr, mc) for _, mr, mc, _, _ in filled}
    empty = [(abs(mr - tr) + abs(mc - tc), mr, mc, rs, cs) for mr, mc, rs, cs in regs if (mr, mc) not in occupied]
    empty.sort(key=lambda x: (x[0], x[1], x[2]))
    for rank, (_, _, _, rs, cs) in enumerate(empty, start=1):
        rotated = np.rot90(template_region, -rank % 4)
        out[rs, cs] = rotated
    return out


def _shape(cell, area, color):
    mask = np.zeros((cell, cell), dtype=bool)
    r = c = cell // 2
    mask[r, c] = True
    frontier = [(r, c)]
    while int(mask.sum()) < area:
        rr, cc = random.choice(frontier)
        nbrs = [(rr + dr, cc + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < cell and 0 <= nc < cell and not mask[nr, nc]:
                mask[nr, nc] = True
                frontier.append((nr, nc))
                break
    arr = np.zeros((cell, cell), dtype=int)
    arr[mask] = color
    return arr


def generate_input() -> np.ndarray:
    cell = random.choice([4, 5])
    size = 3 * cell + 2
    grid = np.zeros((size, size), dtype=int)
    for k in (cell, 2 * cell + 1):
        grid[k, :] = 5
        grid[:, k] = 5
    regs = _regions(grid)
    chosen = random.sample(regs, 3)
    areas = random.sample(range(4, min(11, cell * cell)), 3)
    if len(set(areas)) < 3:
        areas = [4, 6, 9]
    colors = random.sample([1, 2, 3, 4, 6, 8, 9], 3)
    for (_, _, rs, cs), area, color in zip(chosen, areas, colors):
        grid[rs, cs] = _shape(cell, area, color)
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert isinstance(result, np.ndarray)
