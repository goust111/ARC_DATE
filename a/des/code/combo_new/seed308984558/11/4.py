from __future__ import annotations

import random

import numpy as np
from arc_utils import components_mask


BG = 0
ORANGE = 7
MAROON = 9
LEGEND = [2, 1, 3, 4, 8]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each orange-enclosed region by counting maroon markers and indexing the bottom legend.

    Input structure:
    - Black background with one-pixel-wide orange closed borders around regions.
    - Each region contains between one and five maroon marker pixels.
    - The bottom row is a five-color legend with black gaps.

    Rule:
    - Find every enclosed non-orange region above the legend row.
    - Count its maroon marker pixels; use that count as a 1-based index into the legend colors from left to right.
    - Recolor the black pixels of that region with the selected legend color.
    - Keep maroon markers, orange borders, and the legend unchanged.

    Key insight:
    - The region fill color comes from a count-to-color lookup on the bottom border, not from any color already inside the region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    work = (g[:-1] != ORANGE)
    for comp in components_mask(work, connectivity=4):
        rows = comp[:, 0]
        cols = comp[:, 1]
        if np.any(rows == 0) or np.any(cols == 0) or np.any(rows == g.shape[0] - 2) or np.any(cols == g.shape[1] - 1):
            continue
        count = 0
        for r, c in comp:
            if int(g[r, c]) == MAROON:
                count += 1
        fill = LEGEND[max(0, min(4, count - 1))]
        for r, c in comp:
            if int(g[r, c]) == BG:
                out[r, c] = fill
    return out


def _draw_rect(g: np.ndarray, r0: int, c0: int, h: int, w: int, markers: int) -> None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    g[r0, c0 : c1 + 1] = ORANGE
    g[r1, c0 : c1 + 1] = ORANGE
    g[r0 : r1 + 1, c0] = ORANGE
    g[r0 : r1 + 1, c1] = ORANGE
    markers = min(markers, max(1, (h - 2) * (w - 2)))
    placed = set()
    while len(placed) < markers:
        rr = random.randint(r0 + 1, r1 - 1)
        cc = random.randint(c0 + 1, c1 - 1)
        placed.add((rr, cc))
    for rr, cc in placed:
        g[rr, cc] = MAROON


def generate_input() -> np.ndarray:
    h = random.randint(14, 22)
    w = random.randint(16, 26)
    g = np.zeros((h, w), dtype=int)
    for i, color in enumerate(LEGEND):
        g[-1, 2 * i] = color
    used: list[tuple[int, int, int, int]] = []
    for _ in range(random.randint(1, 3)):
        for _ in range(100):
            rh = random.randint(4, 7)
            rw = random.randint(4, 8)
            r0 = random.randint(1, h - rh - 3)
            c0 = random.randint(1, w - rw - 2)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if all(box[2] < ur0 - 1 or box[0] > ur1 + 1 or box[3] < uc0 - 1 or box[1] > uc1 + 1 for ur0, uc0, ur1, uc1 in used):
                used.append(box)
                _draw_rect(g, r0, c0, rh, rw, random.randint(1, 5))
                break
    return g
