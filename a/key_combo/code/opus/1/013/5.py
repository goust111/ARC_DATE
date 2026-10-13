import itertools
import random
from collections import deque
import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    gray = _components(g, 5)
    if not gray:
        return g
    legend = max(gray, key=lambda comp: (len(comp), _bbox(comp)[1]))
    lr0, lc0, lr1, lc1 = _bbox(legend)
    blocks = [comp for comp in gray if comp is not legend]

    markers = []
    for r in range(lr0, lr1 + 1):
        for c in range(lc0, lc1 + 1):
            if g[r, c] in (1, 2, 3):
                markers.append((r, c, int(g[r, c])))
    if not markers or not blocks:
        return g

    br0 = min(_bbox(b)[0] for b in blocks)
    bc0 = min(_bbox(b)[1] for b in blocks)
    br1 = max(_bbox(b)[2] for b in blocks)
    bc1 = max(_bbox(b)[3] for b in blocks)
    block_pos = []
    for b in blocks:
        r0, c0, r1, c1 = _bbox(b)
        rx = ((c0 + c1) / 2 - bc0) / max(1, bc1 - bc0)
        ry = ((r0 + r1) / 2 - br0) / max(1, br1 - br0)
        block_pos.append((rx, ry))
    marker_pos = [((c - lc0) / max(1, lc1 - lc0), (r - lr0) / max(1, lr1 - lr0), col) for r, c, col in markers]

    best = None
    for perm in itertools.permutations(range(len(marker_pos)), len(blocks)):
        score = sum((block_pos[i][0] - marker_pos[j][0]) ** 2 + (block_pos[i][1] - marker_pos[j][1]) ** 2 for i, j in enumerate(perm))
        if best is None or score < best[0]:
            best = (score, perm)

    out = np.array(g, copy=True)
    out[lr0:lr1 + 1, lc0:lc1 + 1] = 0
    for comp in blocks:
        for r, c in comp:
            out[r, c] = 0

    moving = []
    for i, comp in enumerate(blocks):
        color = marker_pos[best[1][i]][2]
        moving.append((color, list(comp)))

    occupied = out != 0
    directions = {2: (0, -1), 1: (0, 1), 3: (1, 0)}
    for color in (2, 1, 3):
        same = [(col, coords) for col, coords in moving if col == color]
        dr, dc = directions[color]
        if color == 2:
            same.sort(key=lambda item: min(c for _, c in item[1]))
        elif color == 1:
            same.sort(key=lambda item: -max(c for _, c in item[1]))
        else:
            same.sort(key=lambda item: -max(r for r, _ in item[1]))
        for _, coords in same:
            coords = list(coords)
            while True:
                nxt = [(r + dr, c + dc) for r, c in coords]
                if any(not (0 <= r < h and 0 <= c < w) or occupied[r, c] for r, c in nxt):
                    break
                coords = nxt
            for r, c in coords:
                out[r, c] = color
                occupied[r, c] = True
    return out


def generate_input() -> np.ndarray:
    h = w = 17
    g = np.zeros((h, w), dtype=int)
    g[:, 0] = 2
    g[:, -1] = 1
    g[-1, :] = 3
    lr0, lc0 = 1, 12
    g[lr0:lr0 + 4, lc0:lc0 + 4] = 5
    marker_cells = random.sample([(1, 1), (1, 2), (2, 1), (2, 2)], random.randint(3, 4))
    colors = [2, 1, 3] + ([random.choice([1, 2, 3])] if len(marker_cells) == 4 else [])
    random.shuffle(colors)
    for (rr, cc), color in zip(marker_cells, colors):
        g[lr0 + rr, lc0 + cc] = color

    region = (3, 2, 12, 10)
    shapes = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
    ]
    for i, (rr, cc) in enumerate(marker_cells):
        ry = rr / 3
        rx = cc / 3
        r0 = int(region[0] + ry * (region[2] - region[0])) + random.randint(-1, 1)
        c0 = int(region[1] + rx * (region[3] - region[1])) + random.randint(-1, 1)
        shape = shapes[i]
        for r, c in shape:
            if 1 <= r0 + r < h - 1 and 1 <= c0 + c < w - 1 and g[r0 + r, c0 + c] == 0:
                g[r0 + r, c0 + c] = 5
    return g
