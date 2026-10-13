from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RECT_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]


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
    Compress each solid rectangle into a short-side square plus a one-pixel tail along the removed dimension.

    Input structure:
    - Azure is the background.
    - Foreground consists of separated solid monochrome rectangles.
    - Each rectangle is non-square, either wider than tall or taller than wide.

    Rule:
    - Replace every rectangle by an SxS square where S is the original shorter side.
    - For a horizontal rectangle, place the square against the left side and draw a centered horizontal tail to the right edge.
    - For a vertical rectangle, place the square against the top side and draw a centered vertical tail to the bottom edge.

    Key insight: The output keeps the short dimension as a square and summarizes the long dimension as a tail.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, AZURE)
    for comp in _components(g != AZURE):
        minr, minc, maxr, maxc = _bbox(comp)
        color = int(g[comp[0]])
        hh = maxr - minr + 1
        ww = maxc - minc + 1
        if hh == ww:
            for r, c in comp:
                out[r, c] = color
            continue
        side = min(hh, ww)
        out[minr : minr + side, minc : minc + side] = color
        if ww > hh:
            tail_r = minr + side // 2
            out[tail_r, minc + side : maxc + 1] = color
        else:
            tail_c = minc + side // 2
            out[minr + side : maxr + 1, tail_c] = color
    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, hh: int, ww: int) -> bool:
    h, w = occupied.shape
    r0p = max(0, r0 - 1)
    c0p = max(0, c0 - 1)
    r1p = min(h, r0 + hh + 1)
    c1p = min(w, c0 + ww + 1)
    return not bool(occupied[r0p:r1p, c0p:c1p].any())


def generate_input() -> np.ndarray:
    rows = random.randint(14, 26)
    cols = random.randint(14, 26)
    grid = np.full((rows, cols), AZURE, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    colors = random.sample(RECT_COLORS, k=random.randint(2, 5))
    orientations = [True, False] + [random.choice([True, False]) for _ in range(len(colors))]

    for color, horizontal in zip(colors, orientations):
        if horizontal:
            hh = random.randint(2, 5)
            ww = random.randint(hh + 2, min(9, hh + 6))
        else:
            ww = random.randint(2, 5)
            hh = random.randint(ww + 2, min(9, ww + 6))
        for _attempt in range(100):
            if rows - hh <= 1 or cols - ww <= 1:
                break
            r0 = random.randint(0, rows - hh)
            c0 = random.randint(0, cols - ww)
            if not _can_place(occupied, r0, c0, hh, ww):
                continue
            grid[r0 : r0 + hh, c0 : c0 + ww] = int(color)
            occupied[r0 : r0 + hh, c0 : c0 + ww] = True
            break
    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
