import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                            seen[xx, yy] = True
                            q.append((xx, yy))
                comps.append(comp)
    return comps


def _palettes(grid):
    h, w = grid.shape
    high = []
    wide = []
    for comp in _components(grid == 6):
        if len(comp) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(comp)
        if c0 == c1 and abs(r0 - r1) == 1:
            row = min(r0, r1)
            c = c0 + 1
            while c < w and grid[row, c] == 0:
                c += 1
            while c < w and grid[row, c] not in (0, 5, 6):
                high.append(int(grid[row, c]))
                c += 1
        if r0 == r1 and abs(c0 - c1) == 1:
            row = r0
            c = max(c0, c1) + 1
            while c < w and grid[row, c] == 0:
                c += 1
            while c < w and grid[row, c] not in (0, 5, 6):
                wide.append(int(grid[row, c]))
                c += 1
    return high, wide


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    high_pal, wide_pal = _palettes(grid)
    high = []
    wide = []
    for comp in _components(grid == 5):
        pts = np.array(comp)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1) or r1 - r0 < 2 or c1 - c0 < 2:
            continue
        item = (int(r0), int(c0), int(r1), int(c1))
        if r1 - r0 > c1 - c0:
            high.append(item)
        elif c1 - c0 > r1 - r0:
            wide.append(item)
    high.sort(key=lambda x: (x[1], x[0]))
    wide.sort(key=lambda x: (x[0], x[1]))
    for rects, pal in ((high, high_pal), (wide, wide_pal)):
        for (r0, c0, r1, c1), color in zip(rects, pal):
            out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def generate_input() -> np.ndarray:
    h, w = 18, 22
    grid = np.zeros((h, w), dtype=int)
    high_n = random.randint(2, 3)
    wide_n = random.randint(2, 4)
    rects = []
    attempts = 0
    while len(rects) < high_n + wide_n and attempts < 1000:
        attempts += 1
        tall = len([r for r in rects if r[4]]) < high_n
        if tall:
            rh, rw = random.choice([(5, 3), (6, 3), (7, 4)])
        else:
            rh, rw = random.choice([(3, 5), (3, 6), (4, 7)])
        r0 = random.randint(1, 12 - rh)
        c0 = random.randint(1, w - rw - 2)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        if all(r1 + 1 < a or b + 1 < r0 or c1 + 1 < c or d + 1 < c0 for a, c, b, d, _ in rects):
            rects.append((r0, c0, r1, c1, tall))
            grid[r0 : r1 + 1, c0 : c1 + 1] = 5
    high_colors = random.sample([1, 2, 3, 4, 7, 8, 9], high_n)
    wide_colors = random.sample([1, 2, 3, 4, 7, 8, 9], wide_n)
    grid[15, 0] = grid[16, 0] = 6
    for i, color in enumerate(high_colors):
        grid[15, 2 + i] = color
    grid[17, 0] = grid[17, 1] = 6
    for i, color in enumerate(wide_colors):
        grid[17, 3 + i] = color
    return grid
