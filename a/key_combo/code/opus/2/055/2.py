import random
from typing import Dict

import numpy as np


def _cross_lines(g: np.ndarray):
    rows = np.where(np.all(g == 5, axis=1))[0]
    cols = np.where(np.all(g == 5, axis=0))[0]
    H, W = g.shape
    r0 = int(rows[0]) if len(rows) else H // 2
    c0 = int(cols[0]) if len(cols) else W // 2
    return r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    cr, cc = _cross_lines(g)

    colors = {}
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v in (0, 5) or r == cr or c == cc:
                continue
            colors[v] = colors.get(v, 0) + 1

    if not colors:
        return g.copy()

    ordered = sorted(colors.items(), key=lambda kv: kv[1])
    mode: Dict[int, str] = {}
    mode[ordered[-1][0]] = "both"
    mode[ordered[0][0]] = "row"
    if len(ordered) >= 3:
        mode[ordered[1][0]] = "col"
    elif len(ordered) == 2:
        mode[ordered[1][0]] = "col"

    out = g.copy()
    for color, _cnt in ordered:
        pos = np.argwhere(g == color)
        for r, c in pos:
            out[r, c] = color

        t = mode[color]
        if t == "both":
            for r, c in pos:
                rr = 2 * cr - r
                cc2 = 2 * cc - c
                if 0 <= rr < H:
                    out[rr, c] = color
                if 0 <= cc2 < W:
                    out[r, cc2] = color
                if 0 <= rr < H and 0 <= cc2 < W:
                    out[rr, cc2] = color
        elif t == "row":
            for r, c in pos:
                rr = 2 * cr - r
                if 0 <= rr < H:
                    out[rr, c] = color
        else:  # col
            for r, c in pos:
                cc2 = 2 * cc - c
                if 0 <= cc2 < W:
                    out[r, cc2] = color

    return out


def generate_input() -> np.ndarray:
    H = random.choice([15, 17, 19])
    W = H
    g = np.zeros((H, W), dtype=int)

    cr = H // 2
    cc = W // 2
    g[cr, :] = 5
    g[:, cc] = 5

    colors = random.sample([1, 2, 3, 4, 7], random.randint(2, 3))
    counts = random.sample(range(2, 9), len(colors))

    cells = [(r, c) for r in range(H) for c in range(W) if r != cr and c != cc]
    random.shuffle(cells)
    for col, cnt in zip(colors, sorted(counts, reverse=True)):
        placed = 0
        it = 0
        while placed < cnt and it < len(cells):
            r, c = cells[it]
            it += 1
            if g[r, c] == 0:
                g[r, c] = col
                placed += 1
    return g
