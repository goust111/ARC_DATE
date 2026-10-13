import random
from collections import deque

import numpy as np

BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS4:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    comps = _components(g != BG)
    for comp in comps:
        if len(comp) != 5:
            continue
        r0, r1, c0, c1 = _bbox(comp)
        if (r1 - r0 + 1) != 3 or (c1 - c0 + 1) != 3:
            continue
        color = int(g[comp[0]])

        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        k = sum(1 for p in corners if p in comp)

        if k == 1:
            patch = np.array(g[r0 : r1 + 1, c0 : c1 + 1], copy=True)
            patch = np.fliplr(patch)
            g[r0 : r1 + 1, c0 : c1 + 1] = patch
        elif k == 2:
            patch = np.array(g[r0 : r1 + 1, c0 : c1 + 1], copy=True)
            patch = np.flipud(patch)
            g[r0 : r1 + 1, c0 : c1 + 1] = patch
        elif k == 3:
            for rr, cc in comp:
                g[rr, cc] = BG
        elif k == 4:
            # wipe original component area
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    g[rr, cc] = BG
            for rr, cc in comp:
                dr, dc = rr - r0, cc - c0
                nr, nc = r0 + 2 * dr, c0 + 2 * dc
                if nr + 1 >= h or nc + 1 >= w:
                    continue
                g[nr, nc] = color
                g[nr + 1, nc] = color
                g[nr, nc + 1] = color
                g[nr + 1, nc + 1] = color

    return g


def _draw_shape(grid: np.ndarray, top: int, left: int, shape, color: int):
    for dr, dc in shape:
        rr, cc = top + dr, left + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            grid[rr, cc] = color


def generate_input() -> np.ndarray:
    h = w = 16
    g = np.zeros((h, w), dtype=int)

    shapes = {
        1: {(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)},
        2: {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)},
        3: {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
        4: {(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)},
    }
    colors = random.sample([1, 2, 3, 4, 7], 4)
    anchors = [(0, 0), (0, 8), (8, 0), (8, 8)]
    random.shuffle(anchors)

    for k, anchor in zip((1, 2, 3, 4), anchors):
        color = colors[k - 1]
        top, left = anchor
        _draw_shape(g, top, left, shapes[k], color)

    return g
