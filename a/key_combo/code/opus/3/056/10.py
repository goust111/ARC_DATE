import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
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
        comps.append(comp)
    return comps


def _cw(v):
    dr, dc = v
    return dc, -dr


def _dist(shape, p, v):
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


def _fold(grid, comp, out):
    cells = sorted(comp)
    blues = [p for p in cells if grid[p] == 1]
    if len(blues) != 1:
        return
    hinge = blues[0]
    rs = {r for r, _ in cells}
    cs = {c for _, c in cells}
    if len(rs) == 1:
        ordered = sorted(cells, key=lambda p: p[1])
    elif len(cs) == 1:
        ordered = sorted(cells, key=lambda p: p[0])
    else:
        return
    endpoints = [ordered[0], ordered[-1]]
    fixed = None
    for ep in endpoints:
        r, c = ep
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 5:
                fixed = ep
    if fixed is None:
        return
    fi = ordered.index(fixed)
    hi = ordered.index(hinge)
    if fi < hi:
        free = ordered[hi + 1 :]
        base_v = (ordered[hi + 1][0] - hinge[0], ordered[hi + 1][1] - hinge[1]) if free else (0, 0)
    else:
        free = list(reversed(ordered[:hi]))
        base_v = (ordered[hi - 1][0] - hinge[0], ordered[hi - 1][1] - hinge[1]) if free else (0, 0)
    if not free:
        return
    cw = _cw(base_v)
    ccw = (-cw[0], -cw[1])
    fold_v = cw if _dist(grid.shape, hinge, cw) >= _dist(grid.shape, hinge, ccw) else ccw
    for p in free:
        out[p] = 0
    for k in range(1, len(free) + 1):
        nr, nc = hinge[0] + fold_v[0] * k, hinge[1] + fold_v[1] * k
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and out[nr, nc] == 0:
            out[nr, nc] = 2


def main(grid: np.ndarray) -> np.ndarray:
    """Use adjacent gray markers to choose each rod fixed end, fold the opposite tail, then remove the markers."""
    out = grid.copy()
    for comp in _components(np.isin(grid, [1, 2])):
        _fold(grid, comp, out)
    out[out == 5] = 0
    return out


def _rod(g, start, v, length, hinge_i, mark_side):
    r, c = start
    cells = []
    for i in range(length):
        p = (r + v[0] * i, c + v[1] * i)
        cells.append(p)
        g[p] = 2
    g[cells[hinge_i]] = 1
    ep = cells[0] if mark_side == 0 else cells[-1]
    for dr, dc in ((-v[1], v[0]), (v[1], -v[0]), (-v[0], -v[1]), (v[0], v[1])):
        mr, mc = ep[0] + dr, ep[1] + dc
        if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and g[mr, mc] == 0:
            g[mr, mc] = 5
            return


def generate_input() -> np.ndarray:
    g = np.zeros((12, 12), dtype=int)
    candidates = [
        ((2, 1), (0, 1), 6, random.choice([2, 3]), 0),
        ((6, 9), (1, 0), 5, 2, 1),
        ((9, 2), (0, 1), 5, random.choice([2, 3]), 0),
        ((1, 3), (1, 0), 6, 3, 0),
        ((4, 5), (0, 1), 6, 2, 1),
        ((1, 10), (1, 0), 5, 2, random.choice([0, 1])),
    ]
    random.shuffle(candidates)
    for args in candidates[:random.randint(2, 3)]:
        _rod(g, *args)
    return g
