import random
from collections import deque
import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_color(g, color, region=None):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    rows = range(h) if region is None else range(region[0], region[2] + 1)
    cols = range(w) if region is None else range(region[1], region[3] + 1)
    for r in rows:
        for c in cols:
            if seen[r, c] or int(g[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if nx in rows and ny in cols and not seen[nx, ny] and int(g[nx, ny]) == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(cells):
    br, bc, _, _ = _bbox(cells)
    return sorted((r - br, c - bc) for r, c in cells)


def _rotations(cells):
    pts = _norm(cells)
    variants = []
    cur = pts
    for _ in range(4):
        normed = tuple(_norm(cur))
        if normed not in variants:
            variants.append(normed)
        cur = [(c, -r) for r, c in cur]
    return [list(v) for v in variants]


def _find_palette(g):
    h, w = g.shape
    cols = [c for c in range(w) if int(g[h - 1, c]) != 0]
    if len(cols) >= 3:
        return [int(g[h - 1, c]) for c in cols[:3]]
    return [2, 3, 4]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    blue = [(r, c) for r in range(h) for c in range(w) if int(g[r, c]) == 1]
    if not blue:
        return g
    bb = _bbox(blue)
    pockets = [set(cells) for cells in _components_color(g, 0, bb) if any(r == bb[2] for r, _ in cells)]
    candidates = [cells for cells in _components_color(g, 5)]
    palette = _find_palette(g)
    out = g.copy()
    out[h - 1, :] = 0
    matches = []
    used_pockets = set()
    for ci, cand in enumerate(candidates):
        found = []
        for pi, pocket in enumerate(pockets):
            for shape in _rotations(cand):
                maxr = max(r for r, _ in shape)
                maxc = max(c for _, c in shape)
                prs = [r for r, _ in pocket]
                pcs = [c for _, c in pocket]
                for ar in range(min(prs) - maxr, max(prs) + 1):
                    for ac in range(min(pcs) - maxc, max(pcs) + 1):
                        placed = {(ar + r, ac + c) for r, c in shape}
                        if placed and placed <= pocket:
                            found.append((pi, placed))
        if found:
            found.sort(key=lambda x: (x[0], min(r for r, _ in x[1]), min(c for _, c in x[1])))
            for pi, placed in found:
                if pi not in used_pockets:
                    used_pockets.add(pi)
                    matches.append((pi, placed))
                    break
    for pi, seed in matches:
        pocket = pockets[pi]
        layer = set(seed)
        seen = set(seed)
        k = 0
        while layer:
            for r, c in layer:
                out[r, c] = palette[k % len(palette)]
            nxt = set()
            for r, c in layer:
                for dr, dc in DIR4:
                    npix = (r + dr, c + dc)
                    if npix in pocket and npix not in seen:
                        nxt.add(npix)
            seen |= nxt
            layer = nxt
            k += 1
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    g[0:9, 1:17] = 1
    pockets = [
        [(5, 3), (6, 3), (7, 3), (8, 3), (7, 4), (8, 4)],
        [(5, 8), (6, 8), (6, 9), (7, 9), (8, 9), (8, 8)],
        [(4, 13), (5, 13), (6, 13), (7, 13), (8, 13), (6, 14), (8, 14)],
    ]
    random.shuffle(pockets)
    for pocket in pockets:
        for r, c in pocket:
            g[r, c] = 0
    candidate_shapes = [
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)],
    ]
    bases = [(11, 2), (11, 8), (11, 13)]
    for shape, (br, bc) in zip(candidate_shapes, bases):
        if random.choice([True, False]):
            shape = [(c, -r) for r, c in shape]
            shape = _norm(shape)
        for r, c in shape:
            g[br + r, bc + c] = 5
    start = random.randint(1, 14)
    g[17, start:start + 3] = [2, 3, 4]
    return g
