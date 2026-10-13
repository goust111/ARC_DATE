import random
from collections import deque

import numpy as np


DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _black_region_size(grid, start):
    h, w = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    seeds = [(r, c, int(grid[r, c])) for r, c in zip(*np.where((grid != 0) & (grid != 5) & (grid != 7)))]
    traces = []
    all_gray = set(map(tuple, np.argwhere(grid == 5)))
    for sr, sc, color in seeds:
        neigh = [(sr + dr, sc + dc) for dr, dc in DIRS if (sr + dr, sc + dc) in all_gray]
        if not neigh:
            continue
        path = []
        prev = (sr, sc)
        cur = neigh[0]
        while cur in all_gray and cur not in path:
            path.append(cur)
            nxts = [(cur[0] + dr, cur[1] + dc) for dr, dc in DIRS if (cur[0] + dr, cur[1] + dc) in all_gray and (cur[0] + dr, cur[1] + dc) != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
        end = path[-1]
        blacks = [(end[0] + dr, end[1] + dc) for dr, dc in DIRS if 0 <= end[0] + dr < h and 0 <= end[1] + dc < w and grid[end[0] + dr, end[1] + dc] == 0]
        if not blacks:
            continue
        target = max(blacks, key=lambda p: _black_region_size(grid, p))
        traces.append((sr, sc, color, path, target))
    out = np.array(grid, copy=True)
    for sr, sc, _, path, _ in traces:
        out[sr, sc] = 0
        for cell in path:
            out[cell] = 0
    areas = []
    for _, _, color, _, target in traces:
        q = deque([target])
        seen = {target}
        while q:
            r, c = q.popleft()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        for r, c in seen:
            out[r, c] = color
        areas.append((len(seen), color, seen))
    if areas:
        _, _, best = max(areas, key=lambda x: x[0])
        for r, c in best:
            out[r, c] = 8
    return out


def _room(grid, r0, c0, rh, cw):
    grid[r0 : r0 + rh, c0] = 7
    grid[r0 : r0 + rh, c0 + cw - 1] = 7
    grid[r0, c0 : c0 + cw] = 7
    grid[r0 + rh - 1, c0 : c0 + cw] = 7


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    rooms = [(1, 1, 6, 6), (1, 9, 7, 6), (9, 4, 6, 8)]
    colors = random.sample([1, 2, 3, 4], 3)
    for (r0, c0, rh, cw), color in zip(rooms, colors):
        _room(grid, r0, c0, rh, cw)
        sr, sc = r0 + 1, c0 + 1
        grid[sr, sc] = color
        length = random.randint(1, max(1, cw - 4))
        for k in range(1, length + 1):
            grid[sr, sc + k] = 5
    return grid

