import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
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
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _shear(cells, k):
    minr = min(r for r, _, _ in cells)
    return [(r, c + (k - 1) * (r - minr), v) for r, c, v in cells]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    comps = _components(grid)
    anchor = next(comp for comp in comps if any(v == 4 for _, _, v in comp))
    pieces = sorted([c for c in comps if c is not anchor], key=lambda x: (min(r for r, _, _ in x), min(c for _, c, _ in x)))
    state = np.zeros_like(grid)
    for r, c, v in anchor:
        state[r, c] = v
    used = set()
    remaining = pieces[:]
    k = 1
    while remaining:
        placed = None
        reds = sorted((r, c) for r, c in zip(*np.where(state == 2)) if (r, c) not in used)
        for ar, ac in reds:
            for piece in remaining:
                sheared = _shear(piece, k)
                for pr, pc, pv in sheared:
                    if pv != 2:
                        continue
                    dr, dc = ar - pr, ac - pc
                    shifted = [(r + dr, c + dc, v) for r, c, v in sheared]
                    if any(r < 0 or r >= h or c < 0 or c >= w for r, c, _ in shifted):
                        continue
                    ok = True
                    for r, c, v in shifted:
                        if (r, c) == (ar, ac) and v == 2:
                            continue
                        if state[r, c] != 0 and v != 0:
                            ok = False
                            break
                    if ok:
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
            if v:
                state[r, c] = v
        used.add(joint)
        remaining.remove(piece)
        k += 1
    state[state == 2] = 0
    return state


def _put(grid, top, left, cells, color):
    for r, c, v in cells:
        grid[top + r, left + c] = color if v == 1 else v


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    ar, ac = 9, 9
    grid[ar : ar + 3, ac : ac + 3] = 4
    for r, c in [(ar + 1, ac - 1), (ar + 1, ac + 3), (ar - 1, ac + 1)]:
        grid[r, c] = 2
    grid[ar + 1, ac] = 1
    shapes = [
        [(0, 1, 2), (1, 0, 1), (1, 1, 1), (2, 1, 1)],
        [(0, 0, 2), (0, 1, 1), (1, 1, 1), (1, 2, 1), (2, 2, 1)],
        [(0, 2, 2), (1, 0, 1), (1, 1, 1), (1, 2, 1), (2, 0, 1)],
    ]
    positions = [(2, 2), (15, 3), (3, 16)]
    for cells, pos, color in zip(shapes, positions, random.sample([3, 5, 6, 7, 8, 9], 3)):
        _put(grid, pos[0] + random.randint(0, 1), pos[1] + random.randint(0, 1), cells, color)
    return grid

