import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SHADOW = {1: 8, 2: 9, 3: 4, 4: 6, 6: 7, 7: 2, 8: 1, 9: 3}


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
    objects = []
    indicators = set()
    for color in [c for c in np.unique(grid) if c not in (BG, GREY)]:
        for comp in _components(grid == color):
            objects.append((color, comp))
    object_cells = {p for _, comp in objects for p in comp}
    for color, comp in objects:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if (r0 - 1, (c0 + c1) // 2) in [(r, c) for r, c in zip(*np.where(grid == GREY))]:
            d = (-1, 0)
            edge = [(r0, c) for c in range(c0, c1 + 1)]
            indicators.add((r0 - 1, (c0 + c1) // 2))
        elif (r1 + 1, (c0 + c1) // 2) in [(r, c) for r, c in zip(*np.where(grid == GREY))]:
            d = (1, 0)
            edge = [(r1, c) for c in range(c0, c1 + 1)]
            indicators.add((r1 + 1, (c0 + c1) // 2))
        elif ((r0 + r1) // 2, c0 - 1) in [(r, c) for r, c in zip(*np.where(grid == GREY))]:
            d = (0, -1)
            edge = [(r, c0) for r in range(r0, r1 + 1)]
            indicators.add(((r0 + r1) // 2, c0 - 1))
        else:
            d = (0, 1)
            edge = [(r, c1) for r in range(r0, r1 + 1)]
            indicators.add(((r0 + r1) // 2, c1 + 1))
        scolor = SHADOW.get(color, color)
        for sr, sc in edge:
            r, c = sr + d[0], sc + d[1]
            while 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and (r, c) not in object_cells:
                if out[r, c] == BG:
                    out[r, c] = scolor
                r += d[0]
                c += d[1]
    for r, c in indicators:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = BG
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4], random.randint(2, 4)):
        oh = random.randint(2, 4)
        ow = random.randint(2, 4)
        top = random.randint(2, h - oh - 3)
        left = random.randint(2, w - ow - 3)
        grid[top : top + oh, left : left + ow] = color
        side = random.choice(["u", "d", "l", "r"])
        if side == "u":
            grid[top - 1, left + ow // 2] = GREY
        elif side == "d":
            grid[top + oh, left + ow // 2] = GREY
        elif side == "l":
            grid[top + oh // 2, left - 1] = GREY
        else:
            grid[top + oh // 2, left + ow] = GREY
    return grid
