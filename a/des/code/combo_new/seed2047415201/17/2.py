from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
GREY = 5
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in points]
    cs = [c for _r, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored marker out of its grey object along the marker's quadrant direction.

    Input structure:
    - Azure is the background.
    - Each 4-connected foreground object is Grey except for exactly one colored marker pixel inside it.
    - The marker is neither Grey nor Azure and lies in one of the object's four bounding-box quadrants.

    Rule:
    - Replace each marker with Grey.
    - Use the marker's relative quadrant to choose a diagonal direction away from the object.
    - Starting just outside the corresponding object corner, draw a ray in the marker color whose length equals the
      object's foreground area, clipped by grid bounds.

    Key insight: The marker color supplies the ray color, its position supplies direction, and object area supplies length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for comp in _components(g != AZURE):
        markers = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) not in (AZURE, GREY)]
        if len(markers) != 1:
            continue
        mr, mc, color = markers[0]
        minr, minc, maxr, maxc = _bbox(comp)
        dr = -1 if (mr - minr) * 2 < (maxr - minr + 1) else 1
        dc = -1 if (mc - minc) * 2 < (maxc - minc + 1) else 1
        out[mr, mc] = GREY
        rr = (minr - 1) if dr < 0 else (maxr + 1)
        cc = (minc - 1) if dc < 0 else (maxc + 1)
        for _ in range(len(comp)):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, hh: int, ww: int) -> bool:
    h, w = occupied.shape
    r0p = max(0, r0 - 2)
    c0p = max(0, c0 - 2)
    r1p = min(h, r0 + hh + 2)
    c1p = min(w, c0 + ww + 2)
    return not bool(occupied[r0p:r1p, c0p:c1p].any())


def generate_input() -> np.ndarray:
    rows = random.randint(14, 26)
    cols = random.randint(14, 26)
    grid = np.full((rows, cols), AZURE, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    colors = random.sample(MARKER_COLORS, k=random.randint(2, 4))

    for color in colors:
        hh = random.randint(3, 5)
        ww = random.randint(3, 6)
        placed = False
        for _attempt in range(120):
            if rows - hh <= 4 or cols - ww <= 4:
                break
            r0 = random.randint(2, rows - hh - 2)
            c0 = random.randint(2, cols - ww - 2)
            if not _can_place(occupied, r0, c0, hh, ww):
                continue
            grid[r0 : r0 + hh, c0 : c0 + ww] = GREY
            quadrant = random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
            mr_rows = range(r0, r0 + max(1, hh // 2)) if quadrant[0] < 0 else range(r0 + hh // 2, r0 + hh)
            mc_cols = range(c0, c0 + max(1, ww // 2)) if quadrant[1] < 0 else range(c0 + ww // 2, c0 + ww)
            mr = random.choice(list(mr_rows))
            mc = random.choice(list(mc_cols))
            grid[mr, mc] = int(color)
            occupied[r0 : r0 + hh, c0 : c0 + ww] = True
            placed = True
            break
        if not placed:
            continue
    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
