from __future__ import annotations
from collections import Counter, defaultdict
from math import gcd
import numpy as np
from arc_utils import get_bg


def gcd_diffs(pos) -> int:
    """Local helper (only used by this solver): GCD of successive differences of sorted positions."""
    pos = sorted(int(x) for x in pos)
    d = 0
    for a, b in zip(pos, pos[1:]):
        d = int(gcd(int(d), int(b - a)))
    return int(d)

def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic progressions along rows/columns where a color repeats.

    Input structure:
    - bg = most common color. Sparse non-bg dots of several colors.
    - Some colors appear ≥2 times in a single row or column.

    Rule:
    - For each color repeating in a row/col, the step size is the GCD of gaps between its dots.
    - If a companion color on the same line aligns to the same modular class, use it as fill color; otherwise fill with the dot color.
    - Paint bg cells along the progression between extreme dots. If row vs. col evidence ties, apply both.

    Key insight: GCD of inter-dot gaps reveals the hidden period; a companion color sharing the same modular alignment becomes the paint color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    # Positions by color (global).
    pos_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v != int(bg):
                pos_by_color[v].append((int(r), int(c)))

    def process_row(r: int, marker: int) -> None:
        pos = [int(c) for rr, c in pos_by_color[int(marker)] if int(rr) == int(r)]
        if len(pos) < 2:
            return
        d = int(gcd_diffs(pos))
        if d == 0:
            return
        offset = int(sorted(pos)[0] % d)

        others: dict[int, list[int]] = defaultdict(list)
        for c in range(W):
            v = int(out[int(r), int(c)])
            if v != int(bg) and v != int(marker):
                others[int(v)].append(int(c))

        paint = int(marker)
        if others:
            cand = [(len(cs), int(col)) for col, cs in others.items() if any((int(cc) % d) == offset for cc in cs)]
            if cand:
                cand.sort()
                paint = int(cand[0][1])

        if paint == int(marker):
            lo, hi = 0, int(W - 1)
        else:
            u = pos + others[int(paint)]
            lo, hi = int(min(u)), int(max(u))

        for c in range(int(lo), int(hi) + 1):
            if (int(c) % d) != int(offset):
                continue
            cur = int(out[int(r), int(c)])
            if cur == int(bg) or cur == int(marker):
                out[int(r), int(c)] = int(paint)

    def process_col(c: int, marker: int) -> None:
        pos = [int(r) for r, cc in pos_by_color[int(marker)] if int(cc) == int(c)]
        if len(pos) < 2:
            return
        d = int(gcd_diffs(pos))
        if d == 0:
            return
        offset = int(sorted(pos)[0] % d)

        others: dict[int, list[int]] = defaultdict(list)
        for r in range(H):
            v = int(out[int(r), int(c)])
            if v != int(bg) and v != int(marker):
                others[int(v)].append(int(r))

        paint = int(marker)
        if others:
            cand = [(len(rs), int(col)) for col, rs in others.items() if any((int(rr) % d) == offset for rr in rs)]
            if cand:
                cand.sort()
                paint = int(cand[0][1])

        if paint == int(marker):
            lo, hi = 0, int(H - 1)
        else:
            u = pos + others[int(paint)]
            lo, hi = int(min(u)), int(max(u))

        for r in range(int(lo), int(hi) + 1):
            if (int(r) % d) != int(offset):
                continue
            cur = int(out[int(r), int(c)])
            if cur == int(bg) or cur == int(marker):
                out[int(r), int(c)] = int(paint)

    for col, pts in pos_by_color.items():
        row_counts = Counter([int(r) for r, _c in pts])
        col_counts = Counter([int(c) for _r, c in pts])
        rows_dup = [int(r) for r, v in row_counts.items() if int(v) >= 2]
        cols_dup = [int(c) for c, v in col_counts.items() if int(v) >= 2]
        if not rows_dup and not cols_dup:
            continue
        if len(rows_dup) > len(cols_dup):
            for r in rows_dup:
                process_row(int(r), int(col))
        elif len(cols_dup) > len(rows_dup):
            for c in cols_dup:
                process_col(int(c), int(col))
        else:
            for r in rows_dup:
                process_row(int(r), int(col))
            for c in cols_dup:
                process_col(int(c), int(col))

    return out

