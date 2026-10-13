import random
from collections import deque

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
    out = grid.copy()
    frame = None
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if len(comp) == 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
                frame = (color, r0, r1, c0, c1)
                break
    if frame is None:
        return out
    fcolor, r0, r1, c0, c1 = frame
    pieces = []
    for color in [c for c in np.unique(grid) if c not in (BG, fcolor)]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            pieces.append((min(rows), min(cols), comp))
    out[(out != fcolor) & (out != BG)] = BG
    for _, _, comp in sorted(pieces):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        pr0, pr1, pc0, pc1 = min(rows), max(rows), min(cols), max(cols)
        dr = dc = 0
        if pr1 < r0:
            dr = r0 + 1 - pr0
        elif pr0 > r1:
            dr = r1 - 1 - pr1
        elif pc1 < c0:
            dc = c0 + 1 - pc0
        else:
            dc = c1 - 1 - pc1
        for r, c in comp:
            nr, nc = r + dr, c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = fcolor
                out[r, c] = BG
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    top, left = 6, 6
    fh, fw = random.randint(6, 10), random.randint(6, 10)
    grid[top : top + fh, left] = color
    grid[top : top + fh, left + fw - 1] = color
    grid[top, left : left + fw] = color
    grid[top + fh - 1, left : left + fw] = color
    for _ in range(random.randint(2, 4)):
        pcolor = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c != color])
        ph, pw = random.randint(2, 3), random.randint(2, 3)
        side = random.choice(["t", "b", "l", "r"])
        if side == "t":
            r, c = random.randint(1, top - ph - 1), random.randint(left, left + fw - pw)
        elif side == "b":
            lo = top + fh + 1
            hi = h - ph - 1
            if lo > hi:
                continue
            r, c = random.randint(lo, hi), random.randint(left, left + fw - pw)
        elif side == "l":
            r, c = random.randint(top, top + fh - ph), random.randint(1, left - pw - 1)
        else:
            hi = w - pw - 1
            lo = left + fw + 1
            if lo > hi:
                continue
            r, c = random.randint(top, top + fh - ph), random.randint(lo, hi)
        grid[r : r + ph, c : c + pw] = pcolor
    return grid
