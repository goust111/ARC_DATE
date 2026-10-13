import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mask(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r1, c1 = min(rs), min(cs)
    arr = np.zeros((max(rs) - r1 + 1, max(cs) - c1 + 1), dtype=bool)
    for r, c in cells:
        arr[r - r1, c - c1] = True
    return arr


def _key(arr):
    return (arr.shape, tuple(arr.astype(int).ravel()))


def _cw(arr, k):
    return np.rot90(arr, -k)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    guide = None
    for r in range(h):
        for c in range(w - 3):
            vals = list(out[r, c:c + 4])
            if all(v not in (0, 5) for v in vals):
                left = c == 0 or out[r, c - 1] in (0, 5)
                right = c + 4 == w or out[r, c + 4] in (0, 5)
                if left and right:
                    guide = vals
                    break
        if guide is not None:
            break
    if guide is None:
        return out

    comps = _components(out == 5)
    if not comps:
        return out
    masks = [_mask(comp) for comp in comps]
    all_rots = []
    for m in masks:
        all_rots.extend(_cw(m, k) for k in range(4))
    t0 = min(all_rots, key=_key)
    rots = [_cw(t0, k) for k in range(4)]

    for comp, m in zip(comps, masks):
        angle = 0
        for k, rmask in enumerate(rots):
            if m.shape == rmask.shape and np.array_equal(m, rmask):
                angle = k
                break
        for r, c in comp:
            out[r, c] = guide[angle]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    grid = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3, 4]
    random.shuffle(colors)
    c0 = random.randint(0, w - 4)
    grid[0, c0:c0 + 4] = colors
    base = np.array([[1, 1, 0],
                     [0, 1, 0],
                     [0, 1, 1]], dtype=bool)
    angles = random.sample(range(4), random.randint(3, 4))
    if len(angles) < 5 and random.random() < 0.5:
        angles.append(random.choice(angles))
    positions = []
    for k in angles:
        m = _cw(base, k)
        mh, mw = m.shape
        placed = False
        for _ in range(200):
            r = random.randint(2, h - mh - 1)
            c = random.randint(0, w - mw)
            if r == 0:
                continue
            area = grid[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(w, c + mw + 1)]
            if np.all(area == 0):
                positions.append((r, c, m))
                grid[r:r + mh, c:c + mw][m] = 5
                placed = True
                break
        if not placed:
            break
    return grid
