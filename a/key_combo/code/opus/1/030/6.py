import math
import random

import numpy as np


def _gcd_diffs(values):
    vals = sorted(set(values))
    diffs = [b - a for i, a in enumerate(vals) for b in vals[i + 1:] if b > a]
    if not diffs:
        return 0
    g = diffs[0]
    for d in diffs[1:]:
        g = math.gcd(g, d)
    return g


def _series(vals, step):
    vals = sorted(set(vals))
    if step <= 0:
        return vals
    return list(range(vals[0], vals[-1] + 1, step))


def _draw_stair(out, r, c):
    h, w = out.shape
    for dirs in [[(-1, 0), (0, 1)], [(1, 0), (0, -1)]]:
        x, y = r, c
        k = 0
        while True:
            dr, dc = dirs[k % 2]
            moved = False
            for _ in range(2):
                x += dr
                y += dc
                if 0 <= x < h and 0 <= y < w:
                    if out[x, y] != 8:
                        out[x, y] = 5
                    moved = True
            if not moved or not (0 <= x < h and 0 <= y < w):
                break
            k += 1


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    anchors = list(zip(*np.where(out == 8)))
    if not anchors:
        return out
    rows = [r for r, _ in anchors]
    cols = [c for _, c in anchors]
    rg = _gcd_diffs(rows)
    cg = _gcd_diffs(cols)
    full_rows = _series(rows, rg)
    full_cols = _series(cols, cg)
    existing = set(anchors)
    for r in full_rows:
        for c in full_cols:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and (r, c) not in existing:
                out[r, c] = 8
                _draw_stair(out, r, c)
    for r, c in anchors:
        out[r, c] = 8
    return out


def generate_input() -> np.ndarray:
    h = random.choice([19, 21, 22, 23])
    w = random.choice([19, 21, 22, 23])
    g = np.zeros((h, w), dtype=int)
    rs = random.choice([[3, 8, 13], [4, 9, 14], [2, 7, 12, 17]])
    cs = random.choice([[4, 10, 16], [3, 9, 15], [2, 8, 14]])
    rs = [r for r in rs if r < h - 2]
    cs = [c for c in cs if c < w - 2]
    all_pts = [(r, c) for r in rs for c in cs]
    removable = [p for p in all_pts if len([q for q in all_pts if q[0] == p[0]]) > 1 and len([q for q in all_pts if q[1] == p[1]]) > 1]
    random.shuffle(removable)
    missing_count = random.randint(1, min(3, max(1, len(removable) // 2)))
    missing = set(removable[:missing_count])
    for r, c in all_pts:
        if (r, c) in missing:
            continue
        g[r, c] = 8
        _draw_stair(g, r, c)
        g[r, c] = 8
    return g
