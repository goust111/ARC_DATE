import random
from collections import deque

import numpy as np


BG = 0
MIX = 3


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cur)
    return out


def _arrow_info(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    pts = set(comp)
    if len(set(cols)) == 2:
        cmax = max(cols)
        head = next(p for p in comp if p[1] == cmax)
        return head, (0, 1)
    if len(set(rows)) == 2:
        rmax = max(rows)
        head = next(p for p in comp if p[0] == rmax)
        return head, (1, 0)
    return comp[0], (0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    hits = {}
    arrow_cells = set()
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            if len(comp) != 3:
                continue
            head, d = _arrow_info(comp)
            arrow_cells |= set(comp)
            r, c = head[0] + d[0], head[1] + d[1]
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and (r, c) not in arrow_cells:
                hits.setdefault((r, c), set()).add(color)
                r += d[0]
                c += d[1]
    for (r, c), cols in hits.items():
        if len(cols) == 1 and out[r, c] == BG:
            out[r, c] = next(iter(cols))
        elif len(cols) > 1 and out[r, c] == BG:
            out[r, c] = MIX
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    cols = random.sample([1, 2], 2)
    specs = [((0, 1), np.array([(0, 1), (-1, 0), (1, 0)])), ((1, 0), np.array([(1, 0), (0, -1), (0, 1)]))]
    for color in cols:
        for _ in range(random.randint(2, 3)):
            d, pts = random.choice(specs)
            r = random.randint(2, h - 3)
            c = random.randint(2, w - 3)
            for dr, dc in pts:
                grid[r + dr, c + dc] = color
    return grid
