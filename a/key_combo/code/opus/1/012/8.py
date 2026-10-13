import random
from collections import deque

import numpy as np


def _trace(start, shape):
    h, w = shape
    r, c = start
    dc = 1
    path = []
    while 0 <= r < h:
        path.append((r, c))
        nr = r - 1
        if nr < 0:
            break
        nc = c + dc
        if nc < 0 or nc >= w:
            dc *= -1
            nc = c + dc
        r, c = nr, nc
    return path


def _gray_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 5 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 5 and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    starts = [tuple(p) for p in np.argwhere(grid == 9)]
    if not starts:
        return out
    path = _trace(starts[0], grid.shape)
    path_set = set(path)
    containers = []
    for comp in _gray_components(grid):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        charged = [(r, c) for r, c in comp if (r, c) in path_set]
        containers.append((len(charged), r0, r1, c0, c1, set(charged)))
    for r, c in path:
        out[r, c] = 9
    for e, r0, r1, c0, c1, charged in sorted(containers):
        snapshot = np.full((r1 - r0 + 1, c1 - c0 + 1), 5, dtype=int)
        for r, c in charged:
            snapshot[r - r0, c - c0] = 7
        out[r0:r1 + 1, c0:c1 + 1] = 0
        nr0, nr1 = r0 + e, r1 + e
        if nr1 < out.shape[0]:
            out[nr0:nr1 + 1, c0:c1 + 1] = snapshot
    for r, c in path:
        out[r, c] = 9
    return out


def _rects_overlap(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def generate_input() -> np.ndarray:
    h = w = 18
    for _ in range(200):
        grid = np.zeros((h, w), dtype=int)
        start = (h - 1, random.randint(1, 5))
        path = _trace(start, grid.shape)
        grid[start] = 9
        choices = [p for p in path if 3 <= p[0] <= 13 and 2 <= p[1] <= 15]
        random.shuffle(choices)
        rects = []
        for p in choices[:8]:
            ih, iw = random.choice([(2, 2), (2, 3), (3, 3)])
            r0 = p[0] - random.randrange(ih)
            c0 = p[1] - random.randrange(iw)
            r1, c1 = r0 + ih - 1, c0 + iw - 1
            if r0 < 2 or c0 < 1 or r1 > 13 or c1 > 16:
                continue
            rect = (r0 - 1, r1 + 1, c0 - 1, c1 + 1)
            if any(_rects_overlap(rect, old) for old in rects):
                continue
            rects.append(rect)
            grid[r0:r1 + 1, c0:c1 + 1] = 5
            if len(rects) == random.randint(2, 3):
                break
        hits = []
        finals = []
        ok = len(rects) >= 2
        for r0, r1, c0, c1 in rects:
            inner = (r0 + 1, r1 - 1, c0 + 1, c1 - 1)
            e = sum(1 for p in path if inner[0] <= p[0] <= inner[1] and inner[2] <= p[1] <= inner[3])
            hits.append(e)
            finals.append((r0 + e, r1 + e, c0, c1))
        ok = ok and len(set(hits)) == len(hits) and sum(e > 0 for e in hits) >= 2
        ok = ok and all(fin[1] < h for fin in finals)
        ok = ok and all(not _rects_overlap(finals[i], finals[j]) for i in range(len(finals)) for j in range(i + 1, len(finals)))
        if ok:
            return grid
    grid = np.zeros((18, 18), dtype=int)
    grid[17, 1] = 9
    grid[11:13, 7:9] = 5
    grid[6:9, 12:15] = 5
    return grid
