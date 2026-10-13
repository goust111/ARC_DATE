import random
from collections import deque

import numpy as np


def _components(g):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
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
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    hollow = 0
    glyph = None
    for color, cells in _components(g):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 == 2 and c1 - c0 == 2:
            block = g[r0:r1 + 1, c0:c1 + 1]
            outer = [block[0, 0], block[0, 1], block[0, 2], block[1, 0],
                     block[1, 2], block[2, 0], block[2, 1], block[2, 2]]
            if block[1, 1] == 0 and all(x == color for x in outer):
                hollow += 1
            else:
                glyph = block.copy()
    n = hollow
    tile = np.where(glyph == 0, 7, 0)
    out = np.tile(tile, (n, n))
    start = (3 * n - 3) // 2
    mask = glyph != 0
    out[start:start + 3, start:start + 3][mask] = 9
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((14, 14), dtype=int)
    n = random.randint(2, 4)
    anchors = [(0, 0), (0, 5), (0, 10), (5, 0), (5, 5), (5, 10), (10, 0), (10, 5), (10, 10)]
    random.shuffle(anchors)
    colors = random.sample([1, 2, 3, 4], n)
    for (r, c), color in zip(anchors[:n], colors):
        g[r:r + 3, c:c + 3] = color
        g[r + 1, c + 1] = 0
    glyphs = [
        np.array([[0, 7, 0], [7, 7, 7], [0, 7, 0]]),
        np.array([[7, 0, 7], [7, 7, 7], [7, 0, 7]]),
        np.array([[7, 7, 7], [0, 7, 0], [7, 7, 7]]),
        np.array([[7, 7, 0], [0, 7, 0], [0, 7, 7]]),
    ]
    r, c = anchors[n]
    g[r:r + 3, c:c + 3] = random.choice(glyphs)
    return g
