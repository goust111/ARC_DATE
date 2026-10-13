import random
from collections import deque

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
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _count_holes(crop, color):
    obj = crop == color
    bg = ~obj
    seen = np.zeros(bg.shape, dtype=bool)
    holes = 0
    h, w = bg.shape
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """Repair yellow mirror-orbit pixels, count each repaired object's holes, and recolor by the hole count."""
    out = grid.copy()
    for comp in _components(grid != 0):
        vals = {int(grid[r, c]) for r, c in comp if grid[r, c] not in (0, 4)}
        if not vals:
            continue
        color = vals.pop()
        r0, c0, r1, c1 = _bbox(comp)
        hh, ww = r1 - r0, c1 - c0
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                orbit = {
                    (rr, cc),
                    (r1 - (rr - r0), cc),
                    (rr, c1 - (cc - c0)),
                    (r1 - (rr - r0), c1 - (cc - c0)),
                }
                orbit = {(r, c) for r, c in orbit if r0 <= r <= r1 and c0 <= c <= c1}
                orbit_colors = [int(out[r, c]) for r, c in orbit if out[r, c] not in (0, 4)]
                if orbit_colors:
                    fill = orbit_colors[0]
                    for r, c in orbit:
                        if out[r, c] == 4:
                            out[r, c] = fill
        crop = out[r0 : r1 + 1, c0 : c1 + 1]
        holes = _count_holes(crop, color)
        new_color = {1: 1, 2: 2, 3: 3}.get(holes, color)
        obj = crop == color
        crop[obj] = new_color
        out[r0 : r1 + 1, c0 : c1 + 1] = crop
    return out


def _orbit_cells(shape, r, c):
    h, w = shape
    return {
        (r, c),
        (h - 1 - r, c),
        (r, w - 1 - c),
        (h - 1 - r, w - 1 - c),
    }


def _damage_template(mask, color):
    arr = np.zeros(mask.shape, dtype=int)
    used = set()
    for r, c in zip(*np.where(mask)):
        if (r, c) in used:
            continue
        orbit = _orbit_cells(mask.shape, r, c)
        obj_orbit = sorted(p for p in orbit if mask[p])
        used.update(obj_orbit)
        keep = random.choice(obj_orbit)
        for p in obj_orbit:
            arr[p] = color if p == keep else 4
    return arr


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    shapes = []
    one = np.ones((5, 5), dtype=bool)
    one[1:4, 1:4] = False
    shapes.append((one, random.choice([0, 1, 2]), random.choice([0, 1, 2]), 1))
    two = np.ones((5, 7), dtype=bool)
    two[2, 2] = False
    two[2, 4] = False
    shapes.append((two, random.choice([0, 1, 2]), random.choice([7, 8]), 2))
    three = np.ones((7, 7), dtype=bool)
    three[3, 1] = False
    three[3, 3] = False
    three[3, 5] = False
    shapes.append((three, random.choice([7, 8]), random.choice([3, 4, 5]), 3))
    colors = random.sample([1, 2, 3, 7], 3)
    random.shuffle(shapes)
    for (mask, r0, c0, _), color in zip(shapes, colors):
        damaged = _damage_template(mask, color)
        h, w = damaged.shape
        grid[r0 : r0 + h, c0 : c0 + w] = damaged
    return grid
