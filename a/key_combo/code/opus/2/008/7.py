import random
from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    gap = int(np.where(g[0] == 0)[0][0])
    seen = np.zeros((h, w), dtype=bool)
    rects = []
    for r in range(1, h):
        for c in range(w):
            if g[r, c] == 0 or seen[r, c]:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 1 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
            if area == len(cells):
                rects.append((len(cells), color))
    fill = min(rects)[1]
    palette = g[0].copy()
    palette[gap] = fill
    out = np.zeros_like(g)
    out[0] = palette
    row = 1
    while row < h and 2 * row - 1 < 2 * w:
        left = gap - row + 1
        right = gap + row - 1
        for c in range(max(0, left), min(w - 1, right) + 1):
            out[row, c] = palette[c]
        if right - left + 1 >= w:
            break
        row += 1
    return out


def _put_rect(g, r, c, hh, ww, color):
    if np.any(g[r:r + hh, c:c + ww] != 0):
        return False
    g[r:r + hh, c:c + ww] = color
    return True


def generate_input() -> np.ndarray:
    h, w = 12, 5
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 7], w)
    gap = random.randrange(w)
    missing = colors[gap]
    row = colors[:]
    row[gap] = 0
    g[0] = row
    rect_colors = [missing] + random.sample([c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c != missing], 3)
    areas = [4, 6, 8, 9]
    random.shuffle(rect_colors[1:])
    specs = [(1, 0, 2, 2, rect_colors[0]), (4, 0, 2, 3, rect_colors[1]),
             (7, 0, 2, 4, rect_colors[2]), (9, 2, 3, 3, rect_colors[3])]
    random.shuffle(specs)
    for i, (_, _, hh, ww, color) in enumerate(specs):
        if i == 0:
            hh, ww, color = 2, 2, missing
        rr = 1 + i * 2
        if rr + hh <= h:
            _put_rect(g, rr, random.randint(0, w - ww), hh, ww, color)
    if random.random() < 0.8:
        g[h - 1, 0] = random.choice([6, 8, 9])
        g[h - 2, 1] = g[h - 1, 0]
    return g
