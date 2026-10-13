import random
from collections import deque

import numpy as np


def _components(grid, r_start=0):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(r_start, h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc, int(grid[cr, cc])))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _palette(grid):
    rows = []
    r = 0
    while r < grid.shape[0] and np.any(grid[r] != 0):
        rows.append(r)
        r += 1
    cols = []
    for c in range(grid.shape[1]):
        vals = [int(grid[r, c]) for r in rows if grid[r, c] != 0]
        if vals:
            cols.append(vals[0])
    return cols, (max(rows) + 1 if rows else 0)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    pal, start = _palette(grid)
    comps = _components(grid, start)
    anchor = next(comp for comp in comps if any(v == 4 for _, _, v in comp))
    pieces = sorted([c for c in comps if c is not anchor], key=lambda x: (min(r for r, _, _ in x), min(c for _, c, _ in x)))
    state = np.zeros_like(grid)
    for r, c, v in anchor:
        state[r, c] = v
    used = set()
    remaining = pieces[:]
    k = 0
    while remaining and k < len(pal):
        placed = None
        reds = sorted((r, c) for r, c in zip(*np.where(state == 2)) if (r, c) not in used)
        for ar, ac in reds:
            for piece in remaining:
                for pr, pc, pv in piece:
                    if pv != 2:
                        continue
                    dr, dc = ar - pr, ac - pc
                    shifted = [(r + dr, c + dc, v) for r, c, v in piece]
                    if any(r < 0 or r >= h or c < 0 or c >= w for r, c, _ in shifted):
                        continue
                    if all(((r, c) == (ar, ac) and v == 2) or state[r, c] == 0 for r, c, v in shifted):
                        placed = (piece, shifted, (ar, ac))
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            break
        piece, shifted, joint = placed
        for r, c, v in shifted:
            state[r, c] = pal[k] if v == 5 else v
        used.add(joint)
        remaining.remove(piece)
        k += 1
    state[state == 2] = 0
    return state


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 22), dtype=int)
    colors = random.sample([1, 3, 6, 7, 8, 9], 3)
    for i, color in enumerate(colors):
        grid[0:2, 1 + i * 2] = color
    ar, ac = 8, 9
    grid[ar : ar + 2, ac : ac + 2] = 4
    grid[ar + 1, ac - 1] = 2
    grid[ar + 1, ac + 2] = 2
    grid[ar - 1, ac] = 2
    shapes = [
        [(0, 1, 2), (1, 0, 5), (1, 1, 5), (2, 1, 5)],
        [(0, 0, 2), (0, 1, 5), (1, 1, 5), (1, 2, 5)],
        [(0, 2, 2), (1, 0, 5), (1, 1, 5), (1, 2, 5)],
    ]
    for cells, (top, left) in zip(shapes, [(3, 2), (14, 3), (4, 17)]):
        for r, c, v in cells:
            grid[top + r + random.randint(0, 1), left + c] = v
    return grid

