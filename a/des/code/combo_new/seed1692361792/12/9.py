from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _color_components(grid: np.ndarray, target: int) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != target:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == target:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            parts.append(coords)
    return parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each rectangle's hidden-hole location into a sorted report panel on the right.

    Input structure:
    - The background is black.
    - A full-height grey vertical divider splits the grid into a left canvas and a right report panel.
    - The left canvas contains several non-overlapping solid colored rectangles, each with exactly one internal black hole pixel.
    - The right report panel starts empty and has one column reserved for each rectangle.

    Rule:
    - For every colored rectangle on the left, recover its bounding box and find the unique black pixel inside that box.
    - Record the hole's absolute row, absolute column, and the rectangle's color.
    - Sort the rectangles by their hole column from left to right.
    - Clear the report panel and place one colored pixel per rectangle at the same row as its hole, using report columns in sorted order.

    Key insight:
    - The black defect inside each rectangle is the data to extract, and the right panel is a compact ordered report keyed by defect column rather than by rectangle identity.
    """
    g = np.asarray(grid, dtype=int)
    divider_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == DIVIDER)]
    divider = divider_cols[0]
    left = g[:, :divider]
    out = g.copy()
    out[:, divider + 1 :] = BG

    records: list[tuple[int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(left) if int(v) not in (BG, DIVIDER)):
        comps = _color_components(left, color)
        assert len(comps) == 1
        comp = comps[0]
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        holes = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(left[r, c]) == BG]
        assert len(holes) == 1
        hole_r, hole_c = holes[0]
        records.append((hole_c, hole_r, color))

    records.sort(key=lambda item: (item[0], item[1]))
    for idx, (_hole_c, hole_r, color) in enumerate(records):
        out[hole_r, divider + 1 + idx] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        rect_count = rng.randint(2, 5)
        report_w = rect_count
        h = rng.randint(10, 18)
        left_w = rng.randint(max(8, rect_count + 4), 18)
        if left_w + 1 + report_w > 30:
            left_w = 30 - 1 - report_w
        if left_w < 8:
            continue
        grid = np.zeros((h, left_w + 1 + report_w), dtype=int)
        grid[:, left_w] = DIVIDER
        occ = np.zeros((h, left_w), dtype=bool)
        used_hole_cols: set[int] = set()
        colors = rng.sample(COLORS, rect_count)
        ok = True

        for color in colors:
            placed = False
            for _try in range(200):
                rh = rng.randint(3, min(6, h - 1))
                rw = rng.randint(3, min(6, left_w - 1))
                top = rng.randint(0, h - rh)
                left = rng.randint(0, left_w - rw)
                rs = max(0, top - 1)
                re = min(h, top + rh + 1)
                cs = max(0, left - 1)
                ce = min(left_w, left + rw + 1)
                if occ[rs:re, cs:ce].any():
                    continue
                hole_r = rng.randint(top + 1, top + rh - 2)
                hole_c = rng.randint(left + 1, left + rw - 2)
                if hole_c in used_hole_cols:
                    continue
                used_hole_cols.add(hole_c)
                grid[top : top + rh, left : left + rw] = color
                grid[hole_r, hole_c] = BG
                occ[top : top + rh, left : left + rw] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid

    raise RuntimeError("failed to generate input")
