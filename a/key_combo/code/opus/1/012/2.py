import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _h_symmetric(comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    pts = {(r - r0, c - c0) for r, c in comp}
    h = r1 - r0 + 1
    return {(h - 1 - r, c) for r, c in pts} == pts


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    for color in sorted(set(np.unique(grid)) - {0}):
        comps = _components(grid == color)
        if len(comps) != 2:
            continue
        comps = sorted(comps, key=lambda comp: sum(r for r, _ in comp) / len(comp))
        upper, lower = comps
        sym = _h_symmetric(upper)
        uc = sum(c for _, c in upper) / len(upper)
        lc = sum(c for _, c in lower) / len(lower)
        ur0, ur1 = min(r for r, _ in upper), max(r for r, _ in upper)
        uc0, uc1 = min(c for _, c in upper), max(c for _, c in upper)
        lr0, lr1 = min(r for r, _ in lower), max(r for r, _ in lower)
        lc0, lc1 = min(c for _, c in lower), max(c for _, c in lower)
        if lc > uc:
            start, end = (ur1, uc1), (lr0, lc0)
            step = (1, 1)
        else:
            start, end = (ur1, uc0), (lr0, lc1)
            step = (1, -1)
        r, c = start[0] + step[0], start[1] + step[1]
        draw = 1 if sym else 2
        while (r, c) != end and 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = draw
            r += step[0]
            c += step[1]
    return out


def _put(grid, top, left, cells, color):
    for r, c in cells:
        grid[top + r, left + c] = color


def generate_input() -> np.ndarray:
    size = random.randint(15, 18)
    grid = np.zeros((size, size), dtype=int)
    sym = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
    asym = [(0, 0), (1, 0), (1, 1), (2, 1)]
    colors = random.sample([3, 4, 6, 7, 8, 9], random.randint(3, 4))
    shapes = [sym, asym] + [random.choice([sym, asym]) for _ in range(len(colors) - 2)]
    for color, shape in zip(colors, shapes):
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        for _attempt in range(200):
            d = random.randint(3, 5)
            direction = random.choice([-1, 1])
            ur = random.randint(0, size - 1 - h - d - h)
            if direction == 1:
                uc = random.randint(0, size - 1 - w - d - w)
                lr, lc = ur + h - 1 + d, uc + w - 1 + d
                start, end = (ur + h - 1, uc + w - 1), (lr, lc)
            else:
                uc = random.randint(d + w, size - 1 - w)
                lr, lc = ur + h - 1 + d, uc - d - (w - 1)
                start, end = (ur + h - 1, uc), (lr, lc + w - 1)
            cells1 = [(ur + r, uc + c) for r, c in shape]
            cells2 = [(lr + r, lc + c) for r, c in shape]
            if not all(0 <= r < size and 0 <= c < size and grid[r, c] == 0 for r, c in cells1 + cells2):
                continue
            sr = 1
            sc = 1 if direction == 1 else -1
            line = []
            r, c = start[0] + sr, start[1] + sc
            while (r, c) != end:
                line.append((r, c))
                r += sr
                c += sc
            if not all(0 <= r < size and 0 <= c < size and grid[r, c] == 0 for r, c in line):
                continue
            _put(grid, ur, uc, shape, color)
            _put(grid, lr, lc, shape, color)
            break
    used_colors = set(colors)
    for color in random.sample([c for c in range(1, 10) if c not in used_colors], random.randint(1, 3)):
        for _ in range(50):
            r, c = random.randrange(size), random.randrange(size)
            if grid[r, c] != 0:
                continue
            if all(not (0 <= r + dr < size and 0 <= c + dc < size and grid[r + dr, c + dc] != 0)
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                grid[r, c] = color
                break
    return grid
