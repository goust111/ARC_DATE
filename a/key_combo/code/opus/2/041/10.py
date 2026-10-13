from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """Compress filled rectangles into a 5x3 color grid, then reverse each column around its middle row.

    Input structure: 15 pure rectangles (5 rows x 3 cols), each at least 3x3, with small noise and no other large rectangles.

    Rule: keep only solid filled rectangles, derive row and column bands by bbox overlap grouping, build a 5x3 matrix by placement,
    then swap rows (1,2) and (4,5) independently in each column.

    Key insight: rectangle-to-color-grid compression must be done first; only then does the column-wise local reversal apply.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")

    rects = _find_rectangles(g)
    if len(rects) < 15:
        return np.zeros((5, 3), dtype=int)

    rows = [(r0, r1) for _, r0, c0, r1, c1 in rects for _ in [0]]
    cols = [(c0, c1) for _, r0, c0, r1, c1 in rects for _ in [0]]

    row_groups = _group_intervals(rows)
    col_groups = _group_intervals(cols)

    if len(row_groups) != 5 or len(col_groups) != 3:
        return np.zeros((5, 3), dtype=int)

    # sort groups by span
    row_groups = sorted(row_groups, key=lambda g: min(s[0] for s in g))
    col_groups = sorted(col_groups, key=lambda g: min(s[0] for s in g))

    out = np.zeros((5, 3), dtype=int)
    for color, r0, c0, r1, c1 in rects:
        rr = _interval_group_index(row_groups, (r0, r1))
        cc = _interval_group_index(col_groups, (c0, c1))
        if 0 <= rr < 5 and 0 <= cc < 3:
            out[rr, cc] = int(color)

    # local reverse per column (rows 0<->1, 3<->4)
    for c in range(3):
        out[:, c] = _reverse_column(out[:, c])

    return out


def _find_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(BLACK) or visited[r, c]:
                continue
            # flood component 4-neigh
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = [(r, c)]
            while stack:
                x, y = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and int(g[nx, ny]) == color:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
                        comp.append((nx, ny))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
                continue
            if len(comp) != area:
                continue
            rects.append((color, r0, c0, r1, c1))

    # filter non-maximal duplicates by overlap if necessary
    out: list[tuple[int, int, int, int, int]] = []
    for i, item in enumerate(rects):
        c, r0, c0, r1, c1 = item
        keep = True
        for j, other in enumerate(rects):
            if i == j:
                continue
            co, r0o, c0o, r1o, c1o = other
            if r0o <= r0 and c0o <= c0 and r1o >= r1 and c1o >= c1:
                if (r1o - r0o + 1) * (c1o - c0o + 1) > (r1 - r0 + 1) * (c1 - c0 + 1):
                    keep = False
                    break
        if keep:
            out.append((c, r0, c0, r1, c1))

    return out


def _group_intervals(intervals: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    for seg in sorted(intervals, key=lambda t: (t[0], t[1])):
        placed = False
        for g in groups:
            g0 = min(s[0] for s in g)
            g1 = max(s[1] for s in g)
            if not (seg[1] < g0 or seg[0] > g1):
                g.append(seg)
                placed = True
                break
        if not placed:
            groups.append([seg])
    return groups


def _interval_group_index(groups: list[list[tuple[int, int]]], seg: tuple[int, int]) -> int:
    for i, g in enumerate(groups):
        g0 = min(s[0] for s in g)
        g1 = max(s[1] for s in g)
        if not (seg[1] < g0 or seg[0] > g1):
            return i
    return 0


def _reverse_column(col: np.ndarray) -> np.ndarray:
    out = col.copy()
    out[0], out[1] = out[1], out[0]
    out[3], out[4] = out[4], out[3]
    return out


def generate_input() -> np.ndarray:
    h, w = 20, 20
    g = np.full((h, w), int(BLACK), dtype=int)

    colors = [1, 2, 3, 4, 6, 7, 9, 5, 8]

    # define a 5x3 lattice with small random heights/widths
    row_starts = [1, 5, 9, 13, 17]
    row_h = [3, 3, 3, 3, 3]
    col_starts = [1, 7, 13]
    col_w = [3, 3, 3]

    for ri in range(5):
        r0 = row_starts[ri]
        h0 = random.randint(3, 4)
        r1 = min(h - 1, r0 + h0 - 1)
        for ci in range(3):
            c0 = col_starts[ci]
            w0 = random.randint(3, 4)
            c1 = min(w - 1, c0 + w0 - 1)
            color = int(random.choice(colors))
            g[r0 : r1 + 1, c0 : c1 + 1] = color

    # add tiny noise that is not large rectangles
    for _ in range(25):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        if g[r, c] == int(BLACK) and random.random() < 0.5:
            g[r, c] = int(random.choice(colors))

    return g
