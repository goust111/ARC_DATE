import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and mask[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid != 0)
    squares = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 == c1 - c0 and r1 > r0:
            reds = int(np.sum(grid[r0:r1 + 1, c0:c1 + 1] == 2))
            squares.append((reds, r0, c0, r1, c1))
    if not squares:
        return np.array(grid, copy=True)
    _, r0, c0, r1, c1 = min(squares)
    out = np.array(grid[r0:r1 + 1, c0:c1 + 1], copy=True)
    n = out.shape[0]
    reds = list(map(tuple, np.argwhere(out == 2)))
    for r, c in reds:
        pts = [(c, n - 1 - r), (n - 1 - r, n - 1 - c), (n - 1 - c, r)]
        for rr, cc in pts:
            out[rr, cc] = 2
    return out


def generate_input() -> np.ndarray:
    h = w = 18
    g = np.zeros((h, w), dtype=int)
    side = random.randint(5, 7)
    positions = [(1, 1), (1, 10), (10, 1), (10, 10)]
    random.shuffle(positions)
    colors = random.sample([1, 3, 4, 6], random.randint(3, 4))
    counts = random.sample([1, 2, 3, 4], len(colors))
    if min(counts) != counts[0]:
        counts[counts.index(min(counts))], counts[0] = counts[0], min(counts)
    for (r0, c0), color, count in zip(positions, colors, counts):
        if r0 + side > h or c0 + side > w:
            continue
        g[r0:r0 + side, c0:c0 + side] = color
        cells = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
        for r, c in random.sample(cells, count):
            g[r0 + r, c0 + c] = 2
    return g
