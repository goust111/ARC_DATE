import random
from collections import deque

import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r, c in zip(*np.where(grid == color)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for comp in _components(grid, 2):
        cols = {c for _, c in comp}
        if len(cols) != 1:
            continue
        rows = sorted(r for r, _ in comp)
        if len(rows) != rows[-1] - rows[0] + 1:
            continue
        keep = (len(rows) + 1) // 2
        c = next(iter(cols))
        for r in rows[keep:]:
            out[r, c] = 8

    seeds = [(r, c, int(grid[r, c])) for r, c in zip(*np.where((grid != 0) & (grid != 2) & (grid != 5))) if int(grid[r, c]) != 8]
    q = deque()
    best = {}
    ties = set()
    for r, c, color in seeds:
        q.append((r, c, color, 0))
        best[(r, c)] = (0, color)

    while q:
        r, c, color, dist = q.popleft()
        if (r, c) in ties:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or out[nr, nc] in (5, 2):
                continue
            nd = dist + 1
            old = best.get((nr, nc))
            if old is None:
                best[(nr, nc)] = (nd, color)
                q.append((nr, nc, color, nd))
            elif old[0] == nd and old[1] != color:
                ties.add((nr, nc))

    for (r, c), (_, color) in best.items():
        if (r, c) in ties and out[r, c] == 8:
            continue
        if out[r, c] in (0, 8) or grid[r, c] == color:
            out[r, c] = color
    return out


def _draw_c(grid, r0, c0, height, width):
    grid[r0, c0:c0 + width] = 5
    grid[r0 + height - 1, c0:c0 + width] = 5
    grid[r0:r0 + height, c0] = 5


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 22), dtype=int)
    colors = [1, 3, 4]
    for i, cbar in enumerate([6, 12, 18]):
        top = random.randint(1, 3)
        height = random.randint(5, 10)
        grid[top:top + height, cbar] = 2
        r0 = min(max(top + random.randint(0, 2), 1), 9)
        _draw_c(grid, r0, cbar - 4, 5, 4)
        grid[r0 + 2, cbar - 2] = colors[i]
    return grid
