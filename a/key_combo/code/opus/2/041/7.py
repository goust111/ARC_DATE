from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
BLUE = Color.BLUE
GRAY = Color.GREY
RED = Color.RED
GREEN = Color.GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """Build parallel diagonal guides from gray markers and fill top-row markers into the induced diagonal container.

    Input structure: one blue main diagonal, two gray U-shapes on opposite sides, and red/green markers on top row.

    Rule: find the blue diagonal line, find each gray guide component and take its farthest cell from the line as seed,
    draw a parallel guide line through each farthest seed, erase gray markers, then fall top-row markers downward to fill
    the diagonal region between adjacent guide lines with marker color.

    Key insight: gray guides are not output, they define new guide lines; region assignment depends on line order, not on explicit
    container tiles.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")
    h, w = g.shape

    k0 = _find_blue_diagonal_constant(g)
    if k0 is None:
        return g.copy()

    line_constants = [k0]
    guide_comps = _find_gray_components(g)
    for comp in guide_comps:
        if not comp:
            continue
        fr, fc = _farthest_from_line(comp, k0)
        line_constants.append(fr - fc)

    line_constants = sorted(set(line_constants))
    if len(line_constants) < 3:
        return g.copy()

    out = g.copy()
    # draw guide lines (original + new)

    for k in line_constants:
        _draw_blue_line(out, k)

    # erase all gray
    out[g == int(GRAY)] = int(BLACK)

    markers = [(0, int(c), int(out[0, c])) for c in range(w) if int(out[0, c]) in {int(RED), int(GREEN)}]
    if not markers:
        return out

    # classify regions by d = r-c intervals
    ds = sorted(line_constants)
    for r, c, col in markers:
        d = -c
        lo, hi = _interval_for_d(ds, d)
        for rr in range(0, h):
            for cc in range(0, w):
                if out[rr, cc] != int(BLACK):
                    continue
                dv = rr - cc
                if lo <= dv <= hi:
                    out[rr, cc] = col
    return out


def _find_blue_diagonal_constant(g: np.ndarray) -> int | None:
    rr, cc = np.where(g == int(BLUE))
    if not len(rr):
        return None
    vals = rr - cc
    # main diagonal family gives one dominant constant
    vals_int = list(map(int, vals))
    best = max((vals_int.count(v), v) for v in set(vals_int))[1]
    return int(best)


def _find_gray_components(g: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(GRAY):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in dirs:
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(g[nx, ny]) == int(GRAY):
                        seen[nx, ny] = True
                        q.append((nx, ny))
            if 2 <= len(comp) <= 6:
                comps.append(comp)
    return comps


def _farthest_from_line(comp: list[tuple[int, int]], k: int) -> tuple[int, int]:
    best = None
    best_d = -1
    for r, c in comp:
        d = abs((r - c) - k)
        if d > best_d:
            best_d = d
            best = (r, c)
    return best if best is not None else comp[0]


def _draw_blue_line(grid: np.ndarray, k: int) -> None:
    h, w = grid.shape
    for r in range(h):
        c = r - k
        if 0 <= c < w:
            grid[r, c] = int(BLUE)


def _interval_for_d(ds: list[int], d: int) -> tuple[int, int]:
    ds_sorted = sorted(ds)
    if d <= ds_sorted[0]:
        return (-10**6, ds_sorted[0])
    if d >= ds_sorted[-1]:
        return (ds_sorted[-1], 10**6)
    for i in range(len(ds_sorted) - 1):
        if ds_sorted[i] <= d <= ds_sorted[i + 1]:
            return (ds_sorted[i], ds_sorted[i + 1])
    return (ds_sorted[0], ds_sorted[-1])


def generate_input() -> np.ndarray:
    h, w = 16, 14
    g = np.full((h, w), int(BLACK), dtype=int)

    # choose main diagonal
    c0 = random.randint(1, w - 5)
    k0 = -c0
    for r in range(h):
        c = r - k0
        if 0 <= c < w:
            g[r, c] = int(BLUE)

    # choose parallel diagonal constants and far points near each
    d1 = k0 - 2
    d2 = k0 + 3
    far_points = []
    for d in (d1, d2):
        rr = random.randint(3, h - 4)
        cc = np.clip(rr - d, 0, w - 1)
        far_points.append((rr, cc))

    for idx, (fr, fc) in enumerate(far_points):
        # make triangle with 4-6 cells, farthest point selected as (fr,fc)
        near_c = fr - k0
        if 0 <= near_c < w:
            base_r, base_c = fr, near_c
        else:
            base_r, base_c = fr, min(w - 1, max(0, fc + 1))

        cells = {(fr, fc), (base_r, base_c)}
        # connect with tiny diagonal strip
        r, c = base_r, base_c
        while (r, c) != (fr, fc):
            if r < fr:
                r += 1
            elif r > fr:
                r -= 1
            if c < fc:
                c += 1
            elif c > fc:
                c -= 1
            cells.add((r, c))
            if len(cells) > 5:
                break
        while len(cells) < random.randint(4, 6):
            r, c = next(iter(cells))
            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    cells.add((nr, nc))
                    break

        for rr, cc in cells:
            g[rr, cc] = int(GRAY)

    # markers in top row in two distinct regions between diagonal lines
    c1 = max(0, 0 - min(d1, k0, d2) - 1)
    c2 = min(w - 1, w - 1)
    xs = [-(k0), -d1, -d2]
    xs.sort()
    m1 = int(np.clip(random.randint(min(xs), (xs[1] if len(xs) > 1 else xs[0]),), 0, w - 2))
    m2 = int(np.clip(random.randint((xs[1] if len(xs) > 1 else xs[0]), max(xs),), 1, w - 1))
    if m1 == m2:
        m2 = min(w - 1, m1 + 1)
    g[0, m1] = int(RED)
    g[0, m2] = int(GREEN)

    return g
