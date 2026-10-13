import random
import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _fg(grid):
    return np.isin(grid, [2, 5])


def _neighbors(mask, r, c):
    h, w = mask.shape
    return [(r + dr, c + dc) for dr, dc in DIRS if 0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]]


def main(grid: np.ndarray) -> np.ndarray:
    """Symmetrically complete tabs and notches, then recolor maximal solid 3x3 marked windows."""
    out = grid.copy()
    mask = _fg(grid)
    h, w = grid.shape
    remove = []
    add = []
    fill_notches = []

    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        ns = _neighbors(mask, r, c)
        if len(ns) == 1:
            nr, nc = ns[0]
            dr, dc = r - nr, c - nc
            if dr == 0:
                cols = [cc for rr, cc in np.argwhere(mask) if rr == r]
                remove.append((r, max(cols) if dc < 0 else min(cols)))
            else:
                rows = [rr for rr, cc in np.argwhere(mask) if cc == c]
                remove.append((max(rows) if dr < 0 else min(rows), c))

    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            ns = _neighbors(mask, r, c)
            if len(ns) == 3:
                miss = None
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                        miss = (dr, dc)
                        break
                if miss is None:
                    continue
                fill_notches.append((r, c))
                dr, dc = miss
                if dr == 0:
                    cols = [cc for rr, cc in np.argwhere(mask) if rr == r]
                    if cols:
                        edge = max(cols) if dc < 0 else min(cols)
                        add.append((r, edge + (1 if dc < 0 else -1)))
                else:
                    rows = [rr for rr, cc in np.argwhere(mask) if cc == c]
                    if rows:
                        edge = max(rows) if dr < 0 else min(rows)
                        add.append((edge + (1 if dr < 0 else -1), c))

    for r, c in remove:
        if 0 <= r < h and 0 <= c < w and out[r, c] == 5:
            out[r, c] = 0
    for r, c in fill_notches + add:
        if 0 <= r < h and 0 <= c < w and out[r, c] == 0:
            out[r, c] = 5

    candidates = []
    solid = np.isin(out, [2, 5])
    for r in range(h - 2):
        for c in range(w - 2):
            win = out[r : r + 3, c : c + 3]
            if solid[r : r + 3, c : c + 3].all() and (win == 2).any() and out[r + 1, c + 1] == 5:
                reds = frozenset((r + rr, c + cc) for rr, cc in zip(*np.where(win == 2)))
                candidates.append((r, c, reds))
    selected = []
    for cand in candidates:
        reds = cand[2]
        if not any(reds < other[2] for other in candidates):
            selected.append(cand)
    for r, c, _ in selected:
        for rr in range(r, r + 3):
            for cc in range(c, c + 3):
                if out[rr, cc] == 5:
                    out[rr, cc] = 4 if (rr, cc) == (r + 1, c + 1) else 7
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((14, 14), dtype=int)
    ro, co = random.choice([(3, 3), (2, 3), (3, 2)])
    g[ro:ro + 7, co:co + 8] = 5
    for r, c in random.sample([(ro + 2, co + 3), (ro + 3, co + 4), (ro + 1, co + 2), (ro + 4, co + 5)], 2):
        g[r, c] = 2
    for r, c in random.sample([(ro + 1, co + 2), (ro + 5, co + 6), (ro + 2, co + 5)], 2):
        g[r, c] = 0
    if random.random() < 0.5:
        g[max(0, ro - 1), co + 5] = 5
        g[ro + 7, co + 2] = 5
    else:
        g[ro + 3, max(0, co - 1)] = 5
        g[ro + 4, min(13, co + 8)] = 5
    return g
