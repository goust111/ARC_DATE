import random
from collections import deque

import numpy as np


def _fill_stamp_holes(block):
    b = block.copy()
    n, m = b.shape
    seen = np.zeros((n, m), dtype=bool)
    for r in range(n):
        for c in range(m):
            if b[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            edge = False
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                edge |= rr in (0, n - 1) or cc in (0, m - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < n and 0 <= nc < m and b[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not edge and len(pts) in (1, 2, 3):
                col = {1: 4, 2: 7, 3: 6}[len(pts)]
                for rr, cc in pts:
                    b[rr, cc] = col
    return b


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    n, m = g.shape
    out = np.zeros((n * n, m * m), dtype=int)
    stamp = _fill_stamp_holes(g)
    for r in range(n):
        for c in range(m):
            rr, cc = r * n, c * m
            val = int(g[r, c])
            if val == 2:
                out[rr : rr + n, cc : cc + m] = stamp
            elif val != 0:
                out[rr : rr + n, cc : cc + m] = val
    return out


def generate_input() -> np.ndarray:
    wall = random.choice([1, 3])
    deco = 3 if wall == 1 else 1
    g = np.array(
        [
            [2, wall, wall, wall, wall],
            [wall, 0, wall, 0, wall],
            [wall, wall, wall, 0, wall],
            [wall, 0, 0, wall, wall],
            [wall, wall, wall, wall, 2],
        ],
        dtype=int,
    )
    for r, c in random.sample([(0, 2), (2, 0), (2, 2), (4, 2), (2, 4)], random.randint(0, 3)):
        g[r, c] = deco
    for r, c in random.sample([(0, 4), (4, 0), (0, 1), (4, 3)], random.randint(1, 2)):
        g[r, c] = 2
    if random.random() < 0.5:
        g = np.rot90(g, random.randrange(4))
    return g
