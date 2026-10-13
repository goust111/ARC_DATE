from __future__ import annotations

import random
from typing import List

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    sep_cols = [c for c in range(w) if all(int(v) == 1 for v in g[:, c])]
    if len(sep_cols) < 2:
        return g.copy()
    sep_cols.sort()
    l = sep_cols[0]
    m = sep_cols[1]
    if l < 1 or m <= l + 1 or m >= w - 1:
        return g.copy()

    # Legend order by row in first zone.
    legend_cols = range(0, l)
    priority: List[int] = []
    for r in range(h):
        for c in legend_cols:
            v = int(g[r, c])
            if v != 0 and v != 1 and v not in priority:
                priority.append(v)
    pos = {v: i for i, v in enumerate(priority)}

    panel_w = m - l - 1
    panel_h = h
    out = np.zeros((panel_h, panel_w), dtype=int)
    for r in range(panel_h):
        for c in range(panel_w):
            lv = int(g[r, l + 1 + c])
            rv = int(g[r, m + 1 + c])
            if lv == 0 and rv == 0:
                out[r, c] = 0
            elif lv != 0 and rv != 0:
                leftp = pos.get(lv, len(pos) + 5)
                rightp = pos.get(rv, len(pos) + 6)
                out[r, c] = lv if leftp < rightp else rv
            else:
                out[r, c] = lv if lv != 0 else rv
    return out


def generate_input() -> np.ndarray:
    h = 12
    w = 25
    g = np.zeros((h, w), dtype=int)
    legend_end = 4
    sep1 = 6
    sep2 = 14
    for r in range(h):
        g[r, sep1] = 1
        g[r, sep2] = 1

    palette = [1, 2, 3, 4]
    random.shuffle(palette)
    use = palette[:3]
    for i, color in enumerate(use):
        g[i % h, i] = color

    pw = sep2 - sep1 - 1
    for r in range(h):
        for c in range(pw):
            rr = random.randint(0, h - 1)
            cc = random.randint(0, pw - 1)
            g[rr, sep1 + 1 + cc] = random.choice([0] + use)

    # Ensure same-size right panel with controlled conflicts.
    for i, color in enumerate(use):
        rr = random.randint(0, h - 1)
        cc = random.randint(0, pw - 1)
        g[rr, sep2 + 1 + cc] = color

    return g
