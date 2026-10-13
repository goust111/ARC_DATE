import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = [(r, c)]
        seen[r, c] = True
        comp = []
        while q:
            cr, cc = q.pop()
            comp.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _intervals(lines, n):
    cuts = [-1] + sorted(lines) + [n]
    return [(cuts[i] + 1, cuts[i + 1] - 1) for i in range(len(cuts) - 1) if cuts[i] + 1 <= cuts[i + 1] - 1]


def main(grid: np.ndarray) -> np.ndarray:
    """Use blue rectangle centers to draw magenta separators, then tile the red pattern into every partition cell."""
    h, w = grid.shape
    out = grid.copy()
    centers_r, centers_c = [], []
    for comp in _components(grid == 1):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        centers_r.append((min(rs) + max(rs)) // 2)
        centers_c.append((min(cs) + max(cs)) // 2)
    centers_r = sorted(set(centers_r))
    centers_c = sorted(set(centers_c))
    for r in centers_r:
        for c in range(w):
            if out[r, c] != 1:
                out[r, c] = 6
    for c in centers_c:
        for r in range(h):
            if out[r, c] != 1:
                out[r, c] = 6
    row_cells = _intervals(centers_r, h)
    col_cells = _intervals(centers_c, w)
    red = [(int(r), int(c)) for r, c in np.argwhere(grid == 2)]
    if not red:
        return out
    src = None
    for ri, (r0, r1) in enumerate(row_cells):
        for ci, (c0, c1) in enumerate(col_cells):
            inside = [(r, c) for r, c in red if r0 <= r <= r1 and c0 <= c <= c1]
            if inside:
                src = (ri, ci, r0, c0, [(r - r0, c - c0) for r, c in inside])
                break
        if src:
            break
    if src is None:
        return out
    _, _, _, _, rel = src
    for r0, r1 in row_cells:
        for c0, c1 in col_cells:
            for dr, dc in rel:
                rr, cc = r0 + dr, c0 + dc
                if rr <= r1 and cc <= c1 and out[rr, cc] == 8:
                    out[rr, cc] = 2
    return out


def _rect(g, r0, c0, h, w):
    g[r0, c0 : c0 + w] = 1
    g[r0 + h - 1, c0 : c0 + w] = 1
    g[r0 : r0 + h, c0] = 1
    g[r0 : r0 + h, c0 + w - 1] = 1


def generate_input() -> np.ndarray:
    g = np.full((16, 16), 8, dtype=int)
    _rect(g, random.choice([1, 2]), 1, random.choice([3, 5]), 3)
    _rect(g, 9, random.choice([9, 10]), 5, random.choice([3, 5]))
    if random.random() < 0.5:
        _rect(g, 3, 11, 3, 3)
    pattern = random.choice([
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ])
    base = random.choice([(5, 5), (6, 4), (5, 6)])
    for dr, dc in pattern:
        g[base[0] + dr, base[1] + dc] = 2
    return g
