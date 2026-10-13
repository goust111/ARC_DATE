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
    comps = _components(grid != BG)
    src = max(comps, key=len)
    rows = [r for r, _ in src]
    cols = [c for _, c in src]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    corners = {
        int(block[0, 0]): block,
        int(block[0, -1]): np.fliplr(block),
        int(block[-1, 0]): np.flipud(block),
        int(block[-1, -1]): np.rot90(block, 2),
    }
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if (r, c) in src or color not in corners:
                continue
            pat = corners[color]
            for rr in range(pat.shape[0]):
                for cc in range(pat.shape[1]):
                    nr, nc = r + rr, c + cc
                    if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and pat[rr, cc] != BG:
                        out[nr, nc] = pat[rr, cc]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4], 4)
    top, left = 2, 2
    block = np.array([[colors[0], 7, colors[1]], [8, 6, 9], [colors[2], 4, colors[3]]])
    grid[top : top + 3, left : left + 3] = block
    for color in colors:
        r = random.randint(8, h - 4)
        c = random.randint(8, w - 4)
        grid[r, c] = color
    return grid
