import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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
    for color in [c for c in np.unique(grid) if c not in (BG, GREY)]:
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if len(comp) != 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            seed_colors = [c for c in np.unique(inner) if c not in (GREY, color, BG)]
            if not seed_colors:
                continue
            seed = seed_colors[0]
            pts = np.argwhere(inner == seed)
            sr0, sc0 = pts.min(axis=0)
            sr1, sc1 = pts.max(axis=0)
            mask = inner[sr0 : sr1 + 1, sc0 : sc1 + 1] == seed
            hh, ww = inner.shape
            fill = inner.copy()
            for rr in range(0, hh, mask.shape[0]):
                for cc in range(0, ww, mask.shape[1]):
                    for a in range(mask.shape[0]):
                        for b in range(mask.shape[1]):
                            if rr + a < hh and cc + b < ww and mask[a, b]:
                                fill[rr + a, cc + b] = color
            out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4)):
        ih = random.randint(4, 7)
        iw = random.randint(4, 7)
        top = random.randint(1, h - ih - 3)
        left = random.randint(1, w - iw - 3)
        grid[top : top + ih + 2, left] = color
        grid[top : top + ih + 2, left + iw + 1] = color
        grid[top, left : left + iw + 2] = color
        grid[top + ih + 1, left : left + iw + 2] = color
        grid[top + 1 : top + ih + 1, left + 1 : left + iw + 1] = GREY
        seed_color = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c != color])
        sh, sw = random.randint(1, 2), random.randint(1, 2)
        sr = random.randint(top + 1, top + ih - sh + 1)
        sc = random.randint(left + 1, left + iw - sw + 1)
        for rr in range(sr, sr + sh):
            for cc in range(sc, sc + sw):
                if random.random() < 0.7:
                    grid[rr, cc] = seed_color
        if not np.any(grid[top + 1 : top + ih + 1, left + 1 : left + iw + 1] == seed_color):
            grid[sr, sc] = seed_color
    return grid
