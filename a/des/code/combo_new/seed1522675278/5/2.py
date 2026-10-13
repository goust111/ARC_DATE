import random
from collections import defaultdict, deque

import numpy as np


BG = 0


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
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cur)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    by_color = defaultdict(list)
    for color in [c for c in np.unique(grid) if c != BG]:
        for comp in _components(grid == color):
            by_color[color].append(comp)
    for color, comps in by_color.items():
        if len(comps) != 2:
            continue
        horiz = vert = None
        for comp in comps:
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if len(rows) == 1:
                horiz = comp
            elif len(cols) == 1:
                vert = comp
        if horiz is None or vert is None:
            continue
        r = horiz[0][0]
        c = vert[0][1]
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    used = set()
    for color in colors:
        while True:
            row = random.randint(2, h - 3)
            col0 = random.randint(1, w - 7)
            lh = random.randint(2, 5)
            col = random.randint(col0 + 2, min(w - 3, col0 + 6))
            row0 = random.randint(1, h - 7)
            lv = random.randint(2, 5)
            if abs(row - row0) <= lv and abs(col - col0) <= lh:
                continue
            horiz = {(row, c) for c in range(col0, col0 + lh)}
            vert = {(r, col) for r in range(row0, row0 + lv)}
            plus = {(row + dr, col + dc) for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))}
            pts = horiz | vert | plus
            if any(not (0 <= r < h and 0 <= c < w) for r, c in pts):
                continue
            if pts & used:
                continue
            for r, c in horiz | vert:
                grid[r, c] = color
            used |= pts
            break
    return grid
