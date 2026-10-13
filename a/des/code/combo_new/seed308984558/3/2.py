import random
from collections import deque

import numpy as np


BG = 0
LEG1 = 2
LEG2 = 1


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
    Connect each same-color pair on its larger separation axis and color the stitch from the pair-size comparison legend.

    Input structure:
    - A black background contains exactly two legend pixels in one corner and several paired monochrome objects.
    - Each non-legend color appears in exactly two disconnected components.
    - Pairs are separated more along one axis than the other.

    Rule:
    - Remove the two-pixel legend from the output.
    - For each same-color pair, connect the midpoints of the opposing bbox faces along the axis with larger separation.
    - Use the first legend color if the earlier object on that axis is larger, the second if the later object is larger, otherwise use the pair's own color.

    Key insight:
    - The corner legend encodes the comparison colors, while the actual connector direction comes from geometry and the chosen color comes from component-size ordering.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    out[0, 0] = out[0, 1] = BG
    for color in [v for v in np.unique(g) if v not in (BG, LEG1, LEG2)]:
        comps = _components(g == int(color))
        if len(comps) != 2:
            continue
        a, b = comps
        a0, a1, a2, a3 = _bbox(a)
        b0, b1, b2, b3 = _bbox(b)
        row_gap = max(0, max(b0 - a2 - 1, a0 - b2 - 1))
        col_gap = max(0, max(b1 - a3 - 1, a1 - b3 - 1))
        if row_gap >= col_gap:
            first, second = sorted([(_bbox(a), a), (_bbox(b), b)])
            color2 = LEG1 if len(first[1]) > len(second[1]) else LEG2 if len(first[1]) < len(second[1]) else int(color)
            row0 = first[0][2] + 1
            row1 = second[0][0] - 1
            col = (max(first[0][1], second[0][1]) + min(first[0][3], second[0][3])) // 2
            for r in range(row0, row1 + 1):
                out[r, col] = color2
        else:
            first, second = sorted([(_bbox(a), a), (_bbox(b), b)], key=lambda x: x[0][1])
            color2 = LEG1 if len(first[1]) > len(second[1]) else LEG2 if len(first[1]) < len(second[1]) else int(color)
            col0 = first[0][3] + 1
            col1 = second[0][1] - 1
            row = (max(first[0][0], second[0][0]) + min(first[0][2], second[0][2])) // 2
            for c in range(col0, col1 + 1):
                out[row, c] = color2
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    g[0, 0], g[0, 1] = LEG1, LEG2
    for color in random.sample([3, 4, 5, 6, 7, 8, 9], random.randint(2, 3)):
        h1, w1 = random.randint(2, 3), random.randint(2, 3)
        h2, w2 = random.randint(2, 3), random.randint(2, 3)
        if random.choice([True, False]):
            r1, c1 = random.randint(2, h - h1 - h2 - 4), random.randint(1, w - max(w1, w2) - 1)
            r2, c2 = r1 + h1 + random.randint(2, 4), c1 + random.randint(-1, 1)
        else:
            r1, c1 = random.randint(2, h - max(h1, h2) - 1), random.randint(1, w - w1 - w2 - 4)
            r2, c2 = r1 + random.randint(-1, 1), c1 + w1 + random.randint(2, 4)
        if r2 < 0 or c2 < 0 or r2 + h2 > h or c2 + w2 > w:
            continue
        if np.any(g[r1 : r1 + h1, c1 : c1 + w1]) or np.any(g[r2 : r2 + h2, c2 : c2 + w2]):
            continue
        g[r1 : r1 + h1, c1 : c1 + w1] = color
        g[r2 : r2 + h2, c2 : c2 + w2] = color
    return g
