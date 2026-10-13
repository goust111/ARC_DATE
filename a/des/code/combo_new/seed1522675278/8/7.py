import random
from collections import defaultdict, deque

import numpy as np


BG = 5


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


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    by_color = defaultdict(list)
    for color in [c for c in np.unique(grid) if c != BG]:
        by_color[color] = _components(grid == color)
    for color, comps in by_color.items():
        if len(comps) != 2:
            continue
        comps = sorted(comps, key=len)
        small, large = comps[0], comps[1]
        sr = min(r for r, _ in small)
        sc = min(c for _, c in small)
        lr = min(r for r, _ in large)
        lc = min(c for _, c in large)
        small_set = {(r - sr, c - sc) for r, c in small}
        extra = [(r - lr, c - lc) for r, c in large if (r - lr, c - lc) not in small_set]
        same_row = abs((min(r for r, _ in comps[0]) + max(r for r, _ in comps[0])) // 2 - (min(r for r, _ in comps[1]) + max(r for r, _ in comps[1])) // 2) <= 1
        for rr, cc in extra:
            gr, gc = lr + rr, lc + cc
            if same_row:
                target = max(c for _, c in small) if min(c for _, c in small) > gc else min(c for _, c in small)
                step = 1 if target > gc else -1
                for c in range(gc, target + step, step):
                    out[gr, c] = color
            else:
                target = max(r for r, _ in small) if min(r for r, _ in small) > gr else min(r for r, _ in small)
                step = 1 if target > gr else -1
                for r in range(gr, target + step, step):
                    out[r, gc] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.full((h, w), BG, dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3)):
        horiz = random.choice([True, False])
        shape = np.array([[1, 1], [1, 0]])
        bigger = shape.copy()
        bigger[1, 1] = 1
        if horiz:
            r = random.randint(1, h - 4)
            c1 = random.randint(1, w // 2 - 3)
            c2 = random.randint(w // 2 + 1, w - 4)
            for rr in range(2):
                for cc in range(2):
                    if shape[rr, cc]:
                        grid[r + rr, c1 + cc] = color
                    if bigger[rr, cc]:
                        grid[r + rr, c2 + cc] = color
        else:
            c = random.randint(1, w - 4)
            r1 = random.randint(1, h // 2 - 3)
            r2 = random.randint(h // 2 + 1, h - 4)
            for rr in range(2):
                for cc in range(2):
                    if shape[rr, cc]:
                        grid[r1 + rr, c + cc] = color
                    if bigger[rr, cc]:
                        grid[r2 + rr, c + cc] = color
    return grid
