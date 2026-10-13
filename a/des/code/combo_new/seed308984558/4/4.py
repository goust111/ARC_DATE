import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read bridge directions and colors from the key grid, then bridge the matching large object pairs.

    Input structure:
    - A grey-framed key grid in one corner contains color pairs with a bridge color between them.
    - Elsewhere, each keyed color appears as two larger disconnected objects.
    - Pair orientation in the key decides whether the large objects are connected horizontally or vertically.

    Rule:
    - Decode each keyed color's bridge orientation and bridge color from the key.
    - Find the two large objects of that color.
    - Draw a straight bridge through background between their facing edges using the keyed bridge color.

    Key insight:
    - The key grid is not a miniature copy of the scene; it is a symbolic legend that separately encodes bridge direction and paint color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    frame = max(_components(g == FRAME), key=len)
    r0, c0, r1, c1 = _bbox(frame)
    key = g[r0 + 1 : r1, c0 + 1 : c1]
    info = {}
    for color in [v for v in np.unique(key) if v not in (BG, FRAME)]:
        pts = list(zip(*np.where(key == int(color)), strict=False))
        if len(pts) != 2:
            continue
        (a0, b0), (a1, b1) = pts
        if a0 == a1:
            bridge_color = int(key[a0, (b0 + b1) // 2])
            info[int(color)] = ("h", bridge_color)
        elif b0 == b1:
            bridge_color = int(key[(a0 + a1) // 2, b0])
            info[int(color)] = ("v", bridge_color)
    for color, (axis, bcol) in info.items():
        comps = [cells for cells in _components(g == color) if not (r0 <= cells[0][0] <= r1 and c0 <= cells[0][1] <= c1)]
        if len(comps) != 2:
            continue
        a, b = comps
        a0, a1, a2, a3 = _bbox(a)
        b0, b1, b2, b3 = _bbox(b)
        if axis == "h":
            left, right = sorted([(_bbox(a), a), (_bbox(b), b)], key=lambda x: x[0][1])
            row = (max(left[0][0], right[0][0]) + min(left[0][2], right[0][2])) // 2
            for c in range(left[0][3] + 1, right[0][1]):
                out[row, c] = bcol
        else:
            top, bottom = sorted([(_bbox(a), a), (_bbox(b), b)], key=lambda x: x[0][0])
            col = (max(top[0][1], bottom[0][1]) + min(top[0][3], bottom[0][3])) // 2
            for r in range(top[0][2] + 1, bottom[0][0]):
                out[r, col] = bcol
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(20, 28), random.randint(20, 28)
    g = np.zeros((h, w), dtype=int)
    g[0:5, 0] = g[0:5, 4] = FRAME
    g[0, 0:5] = g[4, 0:5] = FRAME
    entries = [(1, "h", 6), (2, "v", 7)]
    for idx, (color, axis, bcol) in enumerate(entries):
        if axis == "h":
            rr = 1 + idx * 2
            g[rr, 1], g[rr, 3], g[rr, 2] = color, color, bcol
        else:
            cc = 1 + idx * 2
            g[1, cc], g[3, cc], g[2, cc] = color, color, bcol
    for color, axis, _ in entries:
        if axis == "h":
            r = random.randint(6, h - 4)
            c1 = random.randint(1, w // 3)
            c2 = random.randint(c1 + 5, w - 4)
            g[r : r + 2, c1 : c1 + 2] = color
            g[r : r + 2, c2 : c2 + 2] = color
        else:
            c = random.randint(6, w - 4)
            r1 = random.randint(1, h // 3)
            r2 = random.randint(r1 + 5, h - 4)
            g[r1 : r1 + 2, c : c + 2] = color
            g[r2 : r2 + 2, c : c + 2] = color
    return g
