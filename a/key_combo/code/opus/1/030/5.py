import random
from collections import Counter, deque

import numpy as np


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
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(cells)
    return out


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _orange_frame(grid):
    cells = list(zip(*np.where(grid == 7)))
    r0, c0, r1, c1 = _bbox(cells)
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    r0, c0, r1, c1 = _orange_frame(g)
    counts = Counter(int(g[r, c]) for r in range(r0 + 1, r1 - 1) for c in range(c0 + 1, c1 - 1) if g[r, c] not in (0, 7))
    win = max(counts, key=lambda k: (counts[k], -k))

    tile = None
    h, w = g.shape
    for r in range(h - 2):
        for c in range(w - 2):
            block = g[r:r + 3, c:c + 3]
            if np.all(block != 0) and set(np.unique(block)).issubset({4, 5, 9}):
                tile = block.copy()
                break
        if tile is not None:
            break
    if tile is None:
        tile_cells = [(r, c) for r, c in zip(*np.where(np.isin(g, [4, 9]))) if not (r0 <= r < r1 and c0 <= c < c1)]
        tr0, tc0, tr1, tc1 = _bbox(tile_cells)
        tile = g[tr0:tr1, tc0:tc1].copy()

    tile[tile == 5] = win

    comps = _components(g == 5)
    ctrl = max(comps, key=len)
    cr0, cc0, cr1, cc1 = _bbox(ctrl)
    sprite = g[cr0:cr1, cc0:cc1]
    out = np.zeros((sprite.shape[0] * 3, sprite.shape[1] * 3), dtype=g.dtype)
    for r in range(sprite.shape[0]):
        for c in range(sprite.shape[1]):
            if sprite[r, c] == 5:
                out[r * 3:(r + 1) * 3, c * 3:(c + 1) * 3] = tile
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((14, 14), dtype=int)

    fr, fc = random.choice([(0, 0), (0, 1), (1, 0)])
    fh, fw = random.choice([(5, 6), (6, 5), (6, 6)])
    g[fr:fr + fh, fc] = 7
    g[fr:fr + fh, fc + fw - 1] = 7
    g[fr, fc:fc + fw] = 7
    g[fr + fh - 1, fc:fc + fw] = 7

    colors = [1, 2, 3]
    win = random.choice(colors)
    counts = {win: random.randint(4, 6)}
    for col in colors:
        if col != win:
            counts[col] = random.randint(1, counts[win] - 1)
    inside = [(r, c) for r in range(fr + 1, fr + fh - 1) for c in range(fc + 1, fc + fw - 1)]
    random.shuffle(inside)
    i = 0
    for col in colors:
        for _ in range(counts[col]):
            if i < len(inside):
                r, c = inside[i]
                g[r, c] = col
                i += 1

    tile = np.array(
        random.choice([
            [[5, 4, 9], [4, 5, 4], [9, 4, 5]],
            [[4, 5, 4], [9, 9, 5], [5, 4, 9]],
            [[9, 4, 5], [5, 4, 9], [4, 5, 4]],
        ]),
        dtype=int,
    )
    tr, tc = random.choice([(10, 10), (10, 0), (0, 10)])
    g[tr:tr + 3, tc:tc + 3] = tile

    shapes = [
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 1), (3, 0), (3, 1)],
    ]
    shape = random.choice(shapes)
    sr, sc = random.choice([(8, 1), (8, 4), (5, 8)])
    if tr == 10 and tc == 10:
        sr, sc = random.choice([(8, 1), (8, 4)])
    for dr, dc in shape:
        g[sr + dr, sc + dc] = 5
    return g
