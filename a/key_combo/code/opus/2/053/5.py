from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            pts: List[Tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_border_rect(grid: np.ndarray, coords: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2:
        return False, (r0, c0, r1, c1)
    expected = 2 * (h + w - 2)
    if coords.shape[0] != expected:
        return False, (r0, c0, r1, c1)
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False, (r0, c0, r1, c1)
    return True, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = 0
    red = 2
    h, w = g.shape
    out = g.copy()

    rects = []
    for comp in _components(g == red):
        ok, bb = _is_border_rect(g, comp)
        if ok:
            rects.append(bb)

    if not rects:
        return out

    # Depth toggle on odd nesting depth
    for r in range(h):
        for c in range(w):
            v = int(out[r, c])
            if v in (bg, red):
                continue
            depth = 0
            for r0, c0, r1, c1 in rects:
                if r0 < r < r1 and c0 < c < c1:
                    depth += 1
            if depth % 2 == 1:
                out[r, c] = 3 if v == 1 else 1
    return out


_RNG = np.random.default_rng()


def generate_input() -> np.ndarray:
    h = 12
    w = 12
    g = np.zeros((h, w), dtype=int)

    # Build 2~4 nested rectangular frames (simplified: 3 nested)
    n = int(_RNG.integers(2, 4))
    margin_top = int(_RNG.integers(1, 2))
    margin_left = int(_RNG.integers(1, 2))
    margin_b = int(_RNG.integers(1, 2))
    margin_r = int(_RNG.integers(1, 2))

    r0, c0, r1, c1 = margin_top, margin_left, h - 1 - margin_b, w - 1 - margin_r
    rects = []
    for _ in range(n):
        rects.append((r0, c0, r1, c1))
        r0 += 2
        c0 += 2
        r1 -= 2
        c1 -= 2
        if r1 - r0 < 2 or c1 - c0 < 2:
            break

    for r0, c0, r1, c1 in rects:
        g[r0, c0:c1 + 1] = 2
        g[r1, c0:c1 + 1] = 2
        g[r0:r1 + 1, c0] = 2
        g[r0:r1 + 1, c1] = 2

    # Fill blue/green points in each depth region
    cells_by_depth: dict[int, list[Tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                continue
            depth = 0
            for rr0, cc0, rr1, cc1 in rects:
                if rr0 < r < rr1 and cc0 < c < cc1:
                    depth += 1
            cells_by_depth.setdefault(depth, []).append((r, c))

    for depth, cells in cells_by_depth.items():
        if not cells:
            continue
        # avoid adjacency to red
        valid = []
        for r, c in cells:
            ok = True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == 2:
                    ok = False
                    break
            if ok:
                valid.append((r, c))
        if not valid:
            continue
        _RNG.shuffle(valid)
        npts = min(5, len(valid))
        for i in range(npts):
            r, c = valid[i]
            g[r, c] = int(_RNG.choice([1, 3]))

    # guarantee all depths 0,1,2 have content if available in grid size
    max_depth = len(rects)
    for d in range(max_depth + 1):
        if d not in cells_by_depth or not cells_by_depth[d]:
            continue
        if not any((g[r, c] in (1, 3)) for r, c in cells_by_depth[d]):
            rr, cc = cells_by_depth[d][0]
            g[rr, cc] = 1

    return g
