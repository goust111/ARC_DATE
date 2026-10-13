import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            x, y = q.popleft()
            cells.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(cells)
    return comps


def _frames(grid):
    frames = []
    for comp in _components(grid == 5):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            frames.append((r0, r1, c0, c1, (r1 - r0 - 1) * (c1 - c0 - 1)))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    cycle = [2, 1, 3]
    for rank, frame in enumerate(sorted(_frames(grid), key=lambda x: x[4]), start=1):
        r0, r1, c0, c1, _ = frame
        for c in range(c0 + 1, c1):
            color = cycle[((c - c0 - 1) + rank - 1) % 3]
            for r in range(r0 + 1, r1):
                if out[r, c] == 0:
                    out[r, c] = color
    return out


def _draw(grid, r0, c0, ih, iw):
    r1, c1 = r0 + ih + 1, c0 + iw + 1
    grid[r0 : r1 + 1, c0] = 5
    grid[r0 : r1 + 1, c1] = 5
    grid[r0, c0 : c1 + 1] = 5
    grid[r1, c0 : c1 + 1] = 5
    return r0, r1, c0, c1


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    specs = [(1, 1, 3, 3), (1, 10, 4, 3), (11, 4, 5, 4)]
    random.shuffle(specs)
    frames = [_draw(grid, *spec) for spec in specs]
    ranked = sorted(frames, key=lambda f: (f[1] - f[0] - 1) * (f[3] - f[2] - 1))
    cycle = [2, 1, 3]
    for rank, (r0, r1, c0, c1) in enumerate(ranked, start=1):
        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        random.shuffle(cells)
        keep = set(cells[: max(2, len(cells) // 2)])
        for r, c in keep:
            grid[r, c] = cycle[((c - c0 - 1) + rank - 1) % 3]
    return grid
