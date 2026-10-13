import random
from collections import defaultdict, deque

import numpy as np


BG = 0


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
    out = np.zeros_like(grid)
    bins = {}
    pieces = defaultdict(list)
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if (r1 - r0 + 1) >= 3 and len(comp) >= 2 * (r1 - r0 + 1) + (c1 - c0 + 1) - 2:
                bins[color] = (r0, r1, c0, c1)
                out[r0 : r1 + 1, c0] = color
                out[r1, c0 : c1 + 1] = color
                out[r0 : r1 + 1, c1] = color
            else:
                pieces[color].append(comp)
    for color, comps in pieces.items():
        if color not in bins:
            continue
        r0, r1, c0, c1 = bins[color]
        inner_w = c1 - c0 - 1
        cur_bottom = r1 - 1
        masks = []
        for comp in sorted(comps, key=len):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            br0, bc0 = min(rows), min(cols)
            mask = np.zeros((max(rows) - br0 + 1, max(cols) - bc0 + 1), dtype=int)
            for r, c in comp:
                mask[r - br0, c - bc0] = color
            masks.append(mask)
        for mask in masks:
            top = cur_bottom - mask.shape[0] + 1
            left = c0 + 1 + (inner_w - mask.shape[1]) // 2
            for rr in range(mask.shape[0]):
                for cc in range(mask.shape[1]):
                    if mask[rr, cc]:
                        out[top + rr, left + cc] = color
            cur_bottom = top - 1
    return out


def generate_input() -> np.ndarray:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    x = 1
    for color in colors:
        bh = random.randint(6, 9)
        bw = random.randint(5, 7)
        if x + bw + 2 >= w:
            break
        top = h - bh - 1
        grid[top : top + bh, x] = color
        grid[top + bh - 1, x : x + bw] = color
        grid[top : top + bh, x + bw - 1] = color
        for _ in range(random.randint(1, 3)):
            ph, pw = random.randint(1, 2), random.randint(1, 2)
            pr = random.randint(1, top - 3)
            pc = random.randint(0, w - pw - 1)
            grid[pr : pr + ph, pc : pc + pw] = color
        x += bw + 4
    return grid
