import random
from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Spread each seed block across empty blocks with tie-breaks by seed count."""
    g = np.array(grid, dtype=int)
    rows = list(np.where(np.all(g == 5, axis=1))[0])
    cols = list(np.where(np.all(g == 5, axis=0))[0])
    rr = [(rows[i] + 1, rows[i + 1] - 1) for i in range(len(rows) - 1)]
    cc = [(cols[i] + 1, cols[i + 1] - 1) for i in range(len(cols) - 1)]
    seeds = {}
    owners = {}
    q = deque()
    for i, (r0, r1) in enumerate(rr):
        for j, (c0, c1) in enumerate(cc):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = [int(v) for v in patch.ravel() if int(v) not in (0, 5)]
            if colors:
                color = max(set(colors), key=colors.count)
                cnt = colors.count(color)
                seeds[(i, j)] = (color, cnt)
                owners[(i, j)] = color
                q.append((0, -cnt, i, j, color, cnt))
    best = {}
    while q:
        dist, negcnt, i, j, color, cnt = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if (ni, nj) not in owners and 0 <= ni < len(rr) and 0 <= nj < len(cc):
                key = (ni, nj)
                cand = (dist + 1, negcnt, color, cnt)
                if key not in best or cand < best[key]:
                    best[key] = cand
                    owners[key] = color
                    q.append((dist + 1, negcnt, ni, nj, color, cnt))
    out = g.copy()
    for (i, j), color in owners.items():
        if (i, j) in seeds:
            continue
        r0, r1 = rr[i]
        c0, c1 = cc[j]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def generate_input() -> np.ndarray:
    rh, rw = random.randint(2, 3), random.randint(2, 3)
    rows, cols = random.randint(3, 4), random.randint(3, 4)
    h = rows * rh + rows + 1
    w = cols * rw + cols + 1
    g = np.zeros((h, w), dtype=int)
    g[:: rh + 1, :] = 5
    g[:, :: rw + 1] = 5
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    cells = random.sample([(i, j) for i in range(rows) for j in range(cols)], len(colors))
    for color, (i, j) in zip(colors, cells):
        r0 = i * (rh + 1) + 1
        c0 = j * (rw + 1) + 1
        for _ in range(random.randint(1, min(4, rh * rw))):
            g[random.randint(r0, r0 + rh - 1), random.randint(c0, c0 + rw - 1)] = color
    return g
