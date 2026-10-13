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
    signs = {}
    used = set()
    for comp in _components(grid != BG):
        if len(comp) == 2:
            a, b = comp
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                signs[a] = b
                signs[b] = a
                used |= set(comp)
    entries = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == BG or (r, c) in used:
                continue
            if r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1):
                if r == 0:
                    d = (1, 0)
                elif r == grid.shape[0] - 1:
                    d = (-1, 0)
                elif c == 0:
                    d = (0, 1)
                else:
                    d = (0, -1)
                entries.append(((r, c), d, int(grid[r, c])))
    for (r, c), d, color in entries:
        r += d[0]
        c += d[1]
        cur = color
        seen = set()
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            state = (r, c, d[0], d[1], cur)
            if state in seen:
                break
            seen.add(state)
            if (r, c) in signs and grid[r, c] == cur:
                rr, cc = signs[(r, c)]
                d = (rr - r, cc - c)
                cur = int(grid[rr, cc])
                r, c = rr + d[0], cc + d[1]
                continue
            if grid[r, c] == BG:
                out[r, c] = cur
            r += d[0]
            c += d[1]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    for color in colors:
        side = random.choice(["t", "b", "l", "r"])
        if side == "t":
            c = random.randint(2, w - 3)
            grid[0, c] = color
            r, c0 = random.randint(3, h - 4), c
            next_color = random.choice([x for x in colors if x != color])
            grid[r, c0] = color
            grid[r, c0 + 1] = next_color
        elif side == "b":
            c = random.randint(2, w - 3)
            grid[h - 1, c] = color
            r, c0 = random.randint(2, h - 5), c
            next_color = random.choice([x for x in colors if x != color])
            grid[r, c0] = next_color
            grid[r, c0 + 1] = color
        elif side == "l":
            r = random.randint(2, h - 3)
            grid[r, 0] = color
            r0, c = r, random.randint(3, w - 4)
            next_color = random.choice([x for x in colors if x != color])
            grid[r0, c] = color
            grid[r0 + 1, c] = next_color
        else:
            r = random.randint(2, h - 3)
            grid[r, w - 1] = color
            r0, c = r, random.randint(2, w - 5)
            next_color = random.choice([x for x in colors if x != color])
            grid[r0, c] = next_color
            grid[r0 + 1, c] = color
    return grid
