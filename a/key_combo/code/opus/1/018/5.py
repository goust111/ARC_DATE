import random
from collections import deque

import numpy as np


DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    h, w = grid.shape
    seeds = [(r, c, int(grid[r, c])) for r, c in zip(*np.where((grid != 0) & (grid != 3)))]
    r = c = 0
    d = 0
    for _ in range(h * w * 4):
        out[r, c] = 3
        moved = False
        for _turns in range(4):
            dr, dc = DIRS[d]
            nr, nc = r + dr, c + dc
            nnr, nnc = r + 2 * dr, c + 2 * dc
            blocked = not (0 <= nr < h and 0 <= nc < w) or out[nr, nc] == 3 or grid[nr, nc] not in (0, 3)
            if 0 <= nnr < h and 0 <= nnc < w and out[nnr, nnc] == 3:
                blocked = True
            if blocked:
                d = (d + 1) % 4
            else:
                r, c = nr, nc
                moved = True
                break
        if not moved:
            break
    seen = np.zeros((h, w), dtype=bool)
    for sr, sc, scolor in seeds:
        out[sr, sc] = scolor
    for rr in range(h):
        for cc in range(w):
            if out[rr, cc] != 0 or seen[rr, cc]:
                continue
            q = deque([(rr, cc)])
            seen[rr, cc] = True
            cells = []
            owners = set()
            while q:
                cr, cl = q.popleft()
                cells.append((cr, cl))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cl + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if out[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    else:
                        for sr, sc, scolor in seeds:
                            if (nr, nc) == (sr, sc):
                                owners.add(scolor)
            if len(owners) == 1:
                color = next(iter(owners))
                for cr, cl in cells:
                    out[cr, cl] = color
    return out


def generate_input() -> np.ndarray:
    n = random.choice([12, 13, 14])
    grid = np.zeros((n, n), dtype=int)
    colors = random.sample([1, 2, 4, 6, 7], random.randint(3, 5))
    taken = set()
    for color in colors:
        for _ in range(100):
            r, c = random.randint(2, n - 3), random.randint(2, n - 3)
            if (r, c) not in taken:
                taken.add((r, c))
                grid[r, c] = color
                break
    return grid

