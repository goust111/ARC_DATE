import random
from collections import Counter, deque

import numpy as np


BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask & ~seen)):
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _draw_line(g, a, b, color):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            g[r0, c] = color
    elif c0 == c1:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            g[r, c0] = color


def _endpoints(comp):
    s = set(comp)
    ends = []
    for r, c in comp:
        deg = sum((r + dr, c + dc) in s for dr, dc in DIR4)
        if deg <= 1:
            ends.append((r, c))
    return ends if ends else comp[:1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    out = g.copy()
    colors = [c for c in set(g.ravel()) if c not in (BLACK, GREY, MAROON)]

    for color in colors:
        comps = _components(g == color)
        if len(comps) < 2:
            continue
        ends = [p for comp in comps for p in _endpoints(comp)]
        used = set()
        changed = True
        while changed:
            changed = False
            best = None
            for i, a in enumerate(ends):
                if i in used:
                    continue
                for j, b in enumerate(ends):
                    if i >= j or j in used:
                        continue
                    if a[0] == b[0] or a[1] == b[1]:
                        dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
                        if best is None or dist < best[0]:
                            best = (dist, i, j, a, b)
            if best is not None:
                _, i, j, a, b = best
                _draw_line(out, a, b, color)
                used.add(i)
                used.add(j)
                changed = True

    grey_rows = [r for r in range(g.shape[0]) if np.count_nonzero(g[r] == GREY) > g.shape[1] // 2]
    grey_cols = [c for c in range(g.shape[1]) if np.count_nonzero(g[:, c] == GREY) > g.shape[0] // 2]
    row_bounds = [-1] + grey_rows + [g.shape[0]]
    col_bounds = [-1] + grey_cols + [g.shape[1]]

    for mr, mc in zip(*np.where(g == MAROON)):
        rlo = max(b for b in row_bounds if b < mr) + 1
        rhi = min(b for b in row_bounds if b > mr) - 1
        clo = max(b for b in col_bounds if b < mc) + 1
        chi = min(b for b in col_bounds if b > mc) - 1
        seen = set()
        for r in range(max(0, rlo - 1), min(g.shape[0], rhi + 2)):
            for c in range(max(0, clo - 1), min(g.shape[1], chi + 2)):
                v = int(out[r, c])
                if v not in (BLACK, GREY, MAROON):
                    seen.add(v)
        if len(seen) == 1:
            out[mr, mc] = next(iter(seen))
        elif len(seen) > 1:
            out[mr, mc] = ORANGE
    return out


def _draw_grid(g, rows, cols):
    for r in rows:
        g[r, :] = GREY
    for c in cols:
        g[:, c] = GREY
    for r0, r1 in zip([-1] + rows, rows + [g.shape[0]]):
        for c0, c1 in zip([-1] + cols, cols + [g.shape[1]]):
            rr = min(r1 - 1, r0 + 2)
            cc = min(c1 - 1, c0 + 2)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                g[rr, cc] = MAROON


def _scatter_segments(g, color, points):
    for a, b in zip(points, points[1:]):
        if a[0] == b[0]:
            r = a[0]
            lo, hi = sorted([a[1], b[1]])
            cuts = sorted(random.sample(range(lo + 1, hi), min(2, max(0, hi - lo - 1))))
            start = lo
            for cut in cuts + [hi + 1]:
                if cut - start >= 2:
                    g[r, start:cut] = color
                start = cut + 1
        else:
            c = a[1]
            lo, hi = sorted([a[0], b[0]])
            cuts = sorted(random.sample(range(lo + 1, hi), min(2, max(0, hi - lo - 1))))
            start = lo
            for cut in cuts + [hi + 1]:
                if cut - start >= 2:
                    g[start:cut, c] = color
                start = cut + 1


def generate_input() -> np.ndarray:
    h, w = random.randint(22, 28), random.randint(26, 34)
    g = np.zeros((h, w), dtype=int)
    r_first = random.randint(7, h - 12)
    rows = [r_first, random.randint(r_first + 4, h - 6)]
    c_first = random.randint(8, w - 15)
    cols = [c_first, random.randint(c_first + 5, w - 7)]
    _draw_grid(g, rows, cols)
    colors = random.sample([RED, BLUE, GREEN, YELLOW], random.randint(2, 4))
    lanes = random.sample(range(2, h - 2), len(colors))
    for color, r in zip(colors, lanes):
        c0 = random.randint(1, max(2, cols[0] - 2))
        c1 = random.randint(cols[0] + 1, cols[1] - 1)
        c2 = random.randint(cols[1] + 1, w - 2)
        pts = [(r, c0), (r, c1), (r, c2)]
        if random.random() < 0.45:
            bend = random.randint(2, h - 3)
            pts = [(r, c0), (r, c1), (bend, c1), (bend, c2)]
        _scatter_segments(g, color, pts)
    return g
