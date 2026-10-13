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


def _cw(v):
    dr, dc = v
    return dc, -dr


def _dist_to_edge(shape, p, v):
    h, w = shape
    r, c = p
    dr, dc = v
    if dr < 0:
        return r
    if dr > 0:
        return h - 1 - r
    if dc < 0:
        return c
    return w - 1 - c


def _fold_one(grid, comp, out):
    cells = sorted(comp)
    grays = [p for p in cells if grid[p] == 5]
    blues = [p for p in cells if grid[p] == 1]
    if len(grays) != 1 or len(blues) != 1:
        return
    gray, hinge = grays[0], blues[0]
    rs = {r for r, _ in cells}
    cs = {c for _, c in cells}
    if len(rs) == 1:
        ordered = sorted(cells, key=lambda p: p[1])
    elif len(cs) == 1:
        ordered = sorted(cells, key=lambda p: p[0])
    else:
        return
    gi = ordered.index(gray)
    hi = ordered.index(hinge)
    if gi < hi:
        free = ordered[hi + 1 :]
        base_v = (ordered[hi + 1][0] - hinge[0], ordered[hi + 1][1] - hinge[1]) if free else (0, 0)
    else:
        free = list(reversed(ordered[:hi]))
        base_v = (ordered[hi - 1][0] - hinge[0], ordered[hi - 1][1] - hinge[1]) if free else (0, 0)
    if not free:
        return
    cw = _cw(base_v)
    ccw = (-cw[0], -cw[1])
    fold_v = cw if _dist_to_edge(grid.shape, hinge, cw) >= _dist_to_edge(grid.shape, hinge, ccw) else ccw
    for p in free:
        out[p] = 0
    for k in range(1, len(free) + 1):
        nr, nc = hinge[0] + fold_v[0] * k, hinge[1] + fold_v[1] * k
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and out[nr, nc] == 0:
            out[nr, nc] = 2


def main(grid: np.ndarray) -> np.ndarray:
    """Fold each hinged rod at its blue marker, then connect outermost aligned gray endpoints with orange spans."""
    out = grid.copy()
    mask = np.isin(grid, [1, 2, 5])
    for comp in _components(mask):
        if any(grid[p] == 1 for p in comp) and any(grid[p] == 5 for p in comp):
            _fold_one(grid, comp, out)

    gray = [(int(r), int(c)) for r, c in np.argwhere(out == 5)]
    by_row = {}
    by_col = {}
    for r, c in gray:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)
    for r, cols in by_row.items():
        if len(cols) >= 2:
            a, b = min(cols), max(cols)
            for c in range(a + 1, b):
                if out[r, c] == 0:
                    out[r, c] = 7
    for c, rows in by_col.items():
        if len(rows) >= 2:
            a, b = min(rows), max(rows)
            for r in range(a + 1, b):
                if out[r, c] == 0:
                    out[r, c] = 7
    return out


def _place_rod(g, start, v, length, hinge_i):
    r, c = start
    for i in range(length):
        g[r + v[0] * i, c + v[1] * i] = 2
    g[r, c] = 5
    g[r + v[0] * hinge_i, c + v[1] * hinge_i] = 1


def generate_input() -> np.ndarray:
    h, w = 12, 15
    g = np.zeros((h, w), dtype=int)
    candidates = [
        ((2, 1), (0, 1), random.choice([5, 6]), random.choice([2, 3])),
        ((2, 10), (1, 0), random.choice([5, 6]), 2),
        ((8, 3), (0, 1), random.choice([5, 6]), 2),
        ((1, 2), (1, 0), random.choice([6, 7]), 3),
        ((7, 8), (0, 1), random.choice([5, 6]), 2),
        ((10, 2), (0, 1), 5, random.choice([2, 3])),
        ((1, 12), (1, 0), 5, 2),
    ]
    random.shuffle(candidates)
    for args in candidates[:random.randint(3, 4)]:
        _place_rod(g, *args)
    return g
