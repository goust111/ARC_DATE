import random
from collections import deque

import numpy as np


BG = 0
TRACK = 5
MARK = 2


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


def _track_order(r0, r1, c0, c1):
    order = []
    for c in range(c0, c1 + 1):
        order.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        order.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        order.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        order.append((r, c0))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    for comp in _components((grid == TRACK) | ((grid != BG) & (grid != MARK))):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        order = _track_order(r0, r1, c0, c1)
        mover = next((p for p in order if grid[p] not in (TRACK, BG, MARK)), None)
        if mover is None:
            continue
        color = int(grid[mover])
        n = int(np.sum(grid[r0 + 1 : r1, c0 + 1 : c1] == MARK))
        idx = order.index(mover)
        out[mover] = TRACK
        out[order[(idx + n) % len(order)]] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 3, 4, 6, 7, 8, 9], random.randint(2, 3)):
        rh = random.randint(4, 7)
        rw = random.randint(4, 7)
        top = random.randint(1, h - rh - 2)
        left = random.randint(1, w - rw - 2)
        grid[top : top + rh, left] = TRACK
        grid[top : top + rh, left + rw - 1] = TRACK
        grid[top, left : left + rw] = TRACK
        grid[top + rh - 1, left : left + rw] = TRACK
        order = _track_order(top, top + rh - 1, left, left + rw - 1)
        mover = random.choice(order)
        grid[mover] = color
        for _ in range(random.randint(1, 4)):
            mr = random.randint(top + 1, top + rh - 2)
            mc = random.randint(left + 1, left + rw - 2)
            grid[mr, mc] = MARK
    return grid
