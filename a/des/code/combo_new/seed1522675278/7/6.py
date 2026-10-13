import random
from collections import deque

import numpy as np


BG = 0
DIV = 1
TRAY = 5


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
    split = int(np.where(np.all(grid == DIV, axis=1))[0][0])
    tray = next(comp for comp in _components(grid == TRAY) if len(comp) > 10)
    rows = [r for r, _ in tray]
    cols = [c for _, c in tray]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    out[:split, :] = BG
    shapes = []
    for color in [c for c in np.unique(grid[:split]) if c != BG]:
        for comp in _components(grid[:split] == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            br0, bc0 = min(rs), min(cs)
            mask = np.zeros((max(rs) - br0 + 1, max(cs) - bc0 + 1), dtype=int)
            for r, c in comp:
                mask[r - br0, c - bc0] = color
            shapes.append((len(comp), bc0, mask))
    shapes.sort(key=lambda x: (x[0], x[1]))
    cr, cc = r0 + 1, c0 + 1
    row_h = 0
    for _, _, mask in shapes:
        if cc + mask.shape[1] > c1:
            cr += row_h + 1
            cc = c0 + 1
            row_h = 0
        for rr in range(mask.shape[0]):
            for cc2 in range(mask.shape[1]):
                if mask[rr, cc2] and cr + rr < out.shape[0] and cc + cc2 < out.shape[1]:
                    out[cr + rr, cc + cc2] = mask[rr, cc2]
        cc += mask.shape[1] + 1
        row_h = max(row_h, mask.shape[0])
    return out


def generate_input() -> np.ndarray:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    split = h // 2
    grid[split, :] = DIV
    colors = random.sample([2, 3, 4, 6, 7, 8, 9], random.randint(3, 5))
    x = 1
    for color in colors:
        ph, pw = random.randint(1, 3), random.randint(1, 3)
        grid[random.randint(1, split - ph - 1) : random.randint(1, split - ph - 1) + ph, x : x + pw] = color
        x += pw + 2
    top, left = split + 2, 2
    th, tw = h - split - 4, w - 4
    grid[top : top + th, left] = TRAY
    grid[top : top + th, left + tw - 1] = TRAY
    grid[top, left : left + tw] = TRAY
    grid[top + th - 1, left : left + tw] = TRAY
    return grid
