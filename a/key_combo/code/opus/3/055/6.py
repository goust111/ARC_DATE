from collections import deque
import random

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
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


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(mask):
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0:r1 + 1, c0:c1 + 1].astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Merge rectangle marker views, match each blue placeholder to a missing signature, and recolor it."""
    rects = []
    for comp in _components((grid != 0) & (grid != 1)):
        vals = {int(grid[p]) for p in comp}
        if 5 not in vals:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        marker_vals = [int(x) for x in np.unique(sub) if x not in (0, 5)]
        if len(marker_vals) == 1:
            rects.append((r0, c0, sub, marker_vals[0]))
    rects.sort()
    template = np.zeros(rects[0][2].shape, dtype=bool)
    for _, _, sub, _ in rects:
        template |= (sub != 5) & (sub != 0)
    signatures = []
    for _, _, sub, color in rects:
        signatures.append((_norm(template & (sub == 5)), color))
    out = grid.copy()
    for comp in _components(grid == 1):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        norm = _norm(mask)
        for sig, color in signatures:
            if sig.shape == norm.shape and np.array_equal(sig, norm):
                for p in comp:
                    out[p] = color
                break
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    starts = [(1, 1), (1, 7), (1, 13)]
    marker_sets = [
        [(0, 0), (0, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(2, 1), (2, 0), (1, 0)],
    ]
    random.shuffle(marker_sets)
    colors = random.sample([2, 3, 7, 4, 6], 3)
    for (r0, c0), pts, color in zip(starts, marker_sets, colors):
        grid[r0:r0 + 4, c0:c0 + 4] = 5
        for dr, dc in pts:
            grid[r0 + dr, c0 + dc] = color
    template = np.zeros((4, 4), dtype=bool)
    for pts in marker_sets:
        for p in pts:
            template[p] = True
    for (r0, c0), pts in zip([(11, 1), (11, 7), (11, 13)], marker_sets):
        missing = template.copy()
        for p in pts:
            missing[p] = False
        for rr, cc in zip(*np.where(missing)):
            grid[r0 + rr, c0 + cc] = 1
    return grid
