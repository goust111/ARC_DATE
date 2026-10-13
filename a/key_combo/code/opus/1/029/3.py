import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _norm(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _crop_color(cells, color):
    cells = _norm(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    a = np.zeros((h, w), dtype=int)
    for r, c in cells:
        a[r, c] = color
    return a


def _minimal_tiles(a_cells, b_cells):
    tiles = []
    best = None
    a = set(a_cells)
    b = set(b_cells)
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            moved_b = {(r + dr, c + dc) for r, c in b}
            if a & moved_b:
                continue
            all_cells = a | moved_b
            r0 = min(r for r, _ in all_cells)
            c0 = min(c for _, c in all_cells)
            r1 = max(r for r, _ in all_cells)
            c1 = max(c for _, c in all_cells)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or area < best:
                best = area
                tiles = []
            if area == best:
                tile = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
                for r, c in a:
                    tile[r - r0, c - c0] = 1
                for r, c in moved_b:
                    tile[r - r0, c - c0] = 2
                if not any(np.array_equal(tile, t) for t in tiles):
                    tiles.append(tile)
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == 5)]
    sep = sep_rows[0]
    top = g[:sep]
    lower = g[sep + 1 :]
    comps1 = _components(top == 1)
    comps2 = _components(top == 2)
    a_cells = _norm(comps1[0])
    b_cells = _norm(comps2[0])
    candidates = _minimal_tiles(a_cells, b_cells)

    good = []
    for tile in candidates:
        th, tw = tile.shape
        ok = True
        for r in range(lower.shape[0]):
            for c in range(lower.shape[1]):
                if lower[r, c] != 0 and lower[r, c] != tile[r % th, c % tw]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            good.append(tile)
    tile = good[0] if good else candidates[0]
    out = np.zeros_like(lower)
    th, tw = tile.shape
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[r, c] = tile[r % th, c % tw]
    return out


_SHAPES = [
    ({(0, 0), (1, 0), (1, 1)}, {(0, 1), (1, 1), (2, 0), (2, 1)}),
    ({(0, 0), (0, 1), (1, 0), (2, 0)}, {(0, 1), (1, 1), (1, 2)}),
    ({(0, 1), (1, 0), (1, 1)}, {(0, 0), (1, 0), (2, 0), (2, 1)}),
    ({(0, 0), (1, 0), (2, 0), (2, 1)}, {(0, 1), (1, 1), (1, 2), (2, 2)}),
]


def generate_input() -> np.ndarray:
    h, w = 18, 12
    for _ in range(200):
        a_cells, b_cells = random.choice(_SHAPES)
        candidates = _minimal_tiles(_norm(a_cells), _norm(b_cells))
        candidates = [t for t in candidates if 3 <= t.shape[0] <= 4 and 3 <= t.shape[1] <= 4]
        if not candidates:
            continue
        tile = random.choice(candidates)
        g = np.zeros((h, w), dtype=int)
        # Draw separated top fragments.
        ar = random.randint(0, 1)
        br = random.randint(0, 1)
        for r, c in a_cells:
            g[r + ar, c + 1] = 1
        for r, c in b_cells:
            g[r + br, c + 7] = 2
        g[4, :] = 5
        for r in range(5, h):
            for c in range(w):
                g[r, c] = tile[(r - 5) % tile.shape[0], c % tile.shape[1]]
        for _ in range(random.randint(1, 2)):
            oh = random.randint(2, 4)
            ow = random.randint(2, 4)
            rr = random.randint(5, h - oh)
            cc = random.randint(0, w - ow)
            g[rr : rr + oh, cc : cc + ow] = 0
        return g
    return generate_input()
