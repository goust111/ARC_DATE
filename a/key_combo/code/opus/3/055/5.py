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


def _repair_template(template):
    out = template.copy()
    n = out.shape[0]
    for r in range(n):
        for c in range(n):
            orbit = {(r, c), (c, n - 1 - r), (n - 1 - r, n - 1 - c), (n - 1 - c, r)}
            if any(out[p] == 5 for p in orbit):
                for p in orbit:
                    if out[p] == 4:
                        out[p] = 5
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the C4-symmetric gray template and stamp it at every green control point with baseline rows."""
    coords = np.argwhere((grid == 5) | (grid == 4))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    template = _repair_template(grid[r0:r1 + 1, c0:c1 + 1])
    n = template.shape[0]
    half = n // 2
    controls = [(int(r), int(c)) for r, c in np.argwhere(grid == 3) if not (r0 <= r <= r1 and c0 <= c <= c1)]
    controls.sort(key=lambda p: (p[1], p[0]))
    out = np.zeros_like(grid)
    for cr, cc in controls:
        for tr in range(n):
            for tc in range(n):
                if template[tr, tc] == 5:
                    rr, col = cr + tr - half, cc + tc - half
                    if 0 <= rr < out.shape[0] and 0 <= col < out.shape[1]:
                        out[rr, col] = 5
        out[cr, cc] = 3
    for cr, cc in controls:
        br = cr + half + 1
        if 0 <= br < out.shape[0]:
            out[br, out[br] == 0] = 5
            lo, hi = max(0, cc - half), min(out.shape[1], cc + half + 1)
            out[br, lo:hi] = 3
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    tmpl = np.array(
        [
            [0, 5, 5, 5, 0],
            [5, 0, 5, 0, 5],
            [5, 5, 5, 5, 5],
            [5, 0, 5, 0, 5],
            [0, 5, 5, 5, 0],
        ],
        dtype=int,
    )
    n = tmpl.shape[0]
    orbits = []
    used = set()
    for r, c in zip(*np.where(tmpl == 5)):
        if (r, c) in used:
            continue
        orbit = {(r, c), (c, n - 1 - r), (n - 1 - r, n - 1 - c), (n - 1 - c, r)}
        orbit = sorted(p for p in orbit if tmpl[p] == 5)
        used.update(orbit)
        if len(orbit) > 1:
            orbits.append(orbit)
    for orbit in random.sample(orbits, random.randint(2, min(4, len(orbits)))):
        for p in random.sample(orbit, random.randint(1, len(orbit) - 1)):
            tmpl[p] = 4
    grid[1:6, 1:6] = tmpl
    controls = [(8, 8), (8, 14), (14, 8), (13, 14), (10, 11)]
    for p in random.sample(controls, random.randint(2, 4)):
        grid[p] = 3
    return grid
