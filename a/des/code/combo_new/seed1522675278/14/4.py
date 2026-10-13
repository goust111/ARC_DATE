import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
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
            out.append(comp)
    return out


def _trace_path(comp):
    cells = set(comp)
    ends = []
    for cell in cells:
        deg = sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if deg <= 1:
            ends.append(cell)
    start = min(ends) if ends else min(comp)
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in ((cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]), (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)) if n in cells and n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    corner_templates = {}
    for color in [v for v in np.unique(grid) if v != BG]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            if min(rows) == 0 and min(cols) == 0 or min(rows) == 0 and max(cols) == w - 1 or max(rows) == h - 1 and min(cols) == 0 or max(rows) == h - 1 and max(cols) == w - 1:
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                corner_templates[color] = (r0, c0, grid[r0 : r1 + 1, c0 : c1 + 1].copy())
    out = grid.copy()
    for color in corner_templates:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            if min(rows) == 0 or max(rows) == h - 1 or min(cols) == 0 or max(cols) == w - 1:
                continue
            path = _trace_path(comp)
            _, _, templ = corner_templates[color]
            step = templ.shape[1]
            ch, cw = templ.shape
            cr, cc = ch // 2, cw // 2
            for idx in range(0, len(path), step):
                pr, pc = path[idx]
                top, left = pr - cr, pc - cc
                if top < 0 or left < 0 or top + ch > h or left + cw > w:
                    continue
                patch = out[top : top + ch, left : left + cw]
                out[top : top + ch, left : left + cw] = np.where(templ != BG, templ, patch)
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((23, 23), dtype=int)
    colors = random.sample([1, 2, 3, 4], 4)
    patterns = []
    for color in colors:
        arr = np.zeros((3, 3), dtype=int)
        arr[1, :] = color
        arr[:, 1] = color
        arr[random.choice([0, 2]), random.choice([0, 2])] = color
        patterns.append(arr)
    grid[:3, :3] = patterns[0]
    grid[:3, -3:] = patterns[1]
    grid[-3:, :3] = patterns[2]
    grid[-3:, -3:] = patterns[3]
    occupied = grid != BG
    for color in random.sample(colors, random.randint(2, 4)):
        horiz = random.choice([True, False])
        if horiz:
            r = random.randint(6, 16)
            c0 = random.randint(5, 8)
            length = random.randint(6, 10)
            for c in range(c0, c0 + length):
                grid[r, c] = color
            occupied[r, c0 : c0 + length] = True
        else:
            c = random.randint(6, 16)
            r0 = random.randint(5, 8)
            length = random.randint(6, 10)
            for r in range(r0, r0 + length):
                grid[r, c] = color
            occupied[r0 : r0 + length, c] = True
    return grid
