import random
from collections import deque

import numpy as np


def _component(grid, color):
    pts = list(zip(*np.where(grid == color)))
    if not pts:
        return []
    start = pts[0]
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in seen and grid[nr, nc] == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return list(seen)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    gray = _component(grid, 5)
    if not gray:
        return out
    r0, r1 = min(r for r, _ in gray), max(r for r, _ in gray)
    c0, c1 = min(c for _, c in gray), max(c for _, c in gray)
    ph, pw = r1 - r0 + 1, c1 - c0 + 1
    protected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    colors = {}
    for r, c in zip(*np.where((grid != 0) & (grid != 5))):
        key = (r % ph, c % pw)
        colors.setdefault(key, int(grid[r, c]))
    for r, c in zip(*np.where(grid == 0)):
        if (r, c) in protected:
            continue
        key = (r % ph, c % pw)
        if key in colors:
            out[r, c] = colors[key]
    return out


def _make_object(h, w):
    obj = np.zeros((h, w), dtype=int)
    if random.random() < 0.5:
        obj[h // 2, :] = 5
        obj[:, w // 2] = 5
    else:
        obj[0, :] = obj[-1, :] = 5
        obj[:, 0] = obj[:, -1] = 5
    return obj


def generate_input() -> np.ndarray:
    h, w = 18, 18
    ph, pw = random.choice([(3, 4), (4, 5), (5, 3), (5, 4)])
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.choice([2, 3]))
    tile = np.zeros((ph, pw), dtype=int)
    for r in range(ph):
        for c in range(pw):
            tile[r, c] = colors[(r * 2 + c) % len(colors)]
    grid = np.fromfunction(lambda r, c: tile[r.astype(int) % ph, c.astype(int) % pw], (h, w), dtype=int)
    r0 = random.randint(5, 9)
    c0 = random.randint(4, 9)
    obj = _make_object(ph, pw)
    grid[r0:r0 + ph, c0:c0 + pw] = 0
    grid[r0:r0 + ph, c0:c0 + pw][obj == 5] = 5
    missing = set()
    while len(missing) < random.randint(5, 10):
        r, c = random.randrange(h), random.randrange(w)
        if r0 <= r < r0 + ph and c0 <= c < c0 + pw:
            continue
        missing.add((r, c))
    for r, c in missing:
        grid[r, c] = 0
    return grid.astype(int)
