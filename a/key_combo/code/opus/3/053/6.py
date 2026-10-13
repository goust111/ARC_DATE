import random
from collections import deque

import numpy as np


BG = 7
CORNER = 4
WALL = 6
SOURCE_COLORS = [1, 2, 3, 8, 9]


def _rectangles(grid):
    ys = {(int(r), int(c)) for r, c in np.argwhere(grid == CORNER)}
    rows = sorted({r for r, _ in ys})
    cols = sorted({c for _, c in ys})
    rects = []
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                    if corners <= ys and r1 > r0 + 1 and c1 > c0 + 1:
                        rects.append((r0, c0, r1, c1))
    rects.sort(key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]), b))
    chosen = []
    for rect in rects:
        r0, c0, r1, c1 = rect
        interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
        if all(not (interior & {(r, c) for r in range(a + 1, b) for c in range(d + 1, e)}) for a, d, b, e in chosen):
            chosen.append(rect)
    return chosen


def _reachable(grid, source, rect):
    h, w = grid.shape
    target = {(r, c) for r in range(rect[0] + 1, rect[2]) for c in range(rect[1] + 1, rect[3])}
    q = deque([source])
    seen = {source}
    while q:
        r, c = q.popleft()
        if (r, c) in target:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                if grid[nr, nc] == BG or (nr, nc) in target:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each yellow-corner rectangle using the nearest reachable colored source, with unique source use."""
    out = grid.copy()
    rects = _rectangles(grid)
    sources = []
    for r, c in np.argwhere(np.isin(grid, SOURCE_COLORS)):
        sources.append((int(r), int(c), int(grid[r, c])))
    pairs = []
    for ri, rect in enumerate(rects):
        center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
        for si, (sr, sc, color) in enumerate(sources):
            if _reachable(grid, (sr, sc), rect):
                dist = abs(sr - center[0]) + abs(sc - center[1])
                pairs.append((dist, sr, sc, ri, si, color))
    pairs.sort()
    used_rects = set()
    used_sources = set()
    assignments = {}
    for _, _, _, ri, si, color in pairs:
        if ri not in used_rects and si not in used_sources:
            used_rects.add(ri)
            used_sources.add(si)
            assignments[ri] = (si, color)
    for ri, (si, color) in assignments.items():
        r0, c0, r1, c1 = rects[ri]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
        sr, sc, _ = sources[si]
        out[sr, sc] = BG
    return out


def generate_input() -> np.ndarray:
    grid = np.full((15, 15), BG, dtype=int)
    rects = random.sample([(1, 1, 5, 5), (1, 8, 5, 13), (8, 1, 13, 5), (8, 8, 13, 13)], random.choice([2, 3]))
    for r0, c0, r1, c1 in rects:
        for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            grid[p] = CORNER
    spots = [(0, 7), (7, 0), (7, 14), (14, 7), (3, 7), (11, 7)]
    for color, (r, c) in zip(random.sample(SOURCE_COLORS, random.randint(4, 5)), spots):
        grid[r, c] = color
    if random.random() < 0.8:
        grid[6, 2:8] = WALL
    if random.random() < 0.8:
        grid[2:8, 6] = WALL
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert isinstance(result, np.ndarray)
