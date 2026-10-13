from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    colors = sorted(set(map(int, g.flatten())))
    bg = 0
    non_bg = [c for c in colors if c != bg]
    bar_h_col = None
    bar_v_col = None

    for col in non_bg:
        rows = np.where(np.sum(g == col, axis=1) == w)[0]
        if len(rows) == 2 and rows[1] - rows[0] == 1:
            bar_h_col = col
            rh = rows
            break
    for col in non_bg:
        cols = np.where(np.sum(g == col, axis=0) == h)[0]
        if len(cols) == 2 and cols[1] - cols[0] == 1:
            bar_v_col = col
            cv = cols
            break

    if bar_h_col is None or bar_v_col is None:
        return g.copy()

    out = g.copy()
    r0, r1 = int(rh[0]), int(rh[1])
    c0, c1 = int(cv[0]), int(cv[1])

    # Step 1: crossing swap
    out[r0 : r1 + 1, c0 : c1 + 1] = 2

    # Markers
    markers = np.argwhere(out == 3).tolist()
    markers = [(int(r), int(c)) for r, c in sorted(markers, key=lambda p: (p[0], p[1]))]
    if len(markers) >= 2:
        dr = markers[-1][0] - markers[-2][0]
        dc = markers[-1][1] - markers[-2][1]
    else:
        return out

    if dr == 0 and dc == 0:
        return out

    # Step 3: extrapolate from last marker
    r, c = markers[-1]
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if out[r, c] == 0:
            out[r, c] = 4

    return out


_RNG = np.random.default_rng()


def _place_bar(grid: np.ndarray, color: int, is_row: bool, start: int, span_len: int) -> None:
    h, w = grid.shape
    if is_row:
        grid[start : start + 2, :] = color
    else:
        grid[:, start : start + 2] = color


def generate_input() -> np.ndarray:
    h = 15
    w = 15
    g = np.zeros((h, w), dtype=int)

    # Horizontal blue bar (2 rows)
    rh = int(_RNG.integers(1, h - 3))
    _place_bar(g, 1, True, rh, 2)

    # Vertical red bar (2 cols)
    cv = int(_RNG.integers(1, w - 3))
    _place_bar(g, 2, False, cv, 2)

    # Ensure cross currently blue before operation
    g[rh : rh + 2, cv : cv + 2] = 1

    # Three evenly spaced markers on arithmetic line
    mode = int(_RNG.integers(0, 3))
    if mode == 0:  # horizontal
        max_step = max(2, (w - 1) // 2)
        step = int(_RNG.integers(2, max_step + 1))
        for _ in range(200):
            row = int(_RNG.integers(0, h))
            c0 = int(_RNG.integers(0, w - 2 * step))
            pts = [(row, c0), (row, c0 + step), (row, c0 + 2 * step)]
            if all(0 <= r < h and 0 <= c < w and g[r, c] == 0 for r, c in pts):
                break
        else:
            pts = [(0, 0), (0, step), (0, 2 * step)]
    elif mode == 1:  # vertical
        max_step = max(2, (h - 1) // 2)
        step = int(_RNG.integers(2, max_step + 1))
        for _ in range(200):
            col = int(_RNG.integers(0, w))
            r0 = int(_RNG.integers(0, h - 2 * step))
            pts = [(r0, col), (r0 + step, col), (r0 + 2 * step, col)]
            if all(0 <= r < h and 0 <= c < w and g[r, c] == 0 for r, c in pts):
                break
        else:
            pts = [(0, 0), (step, 0), (2 * step, 0)]
    else:  # diagonal
        step = int(_RNG.integers(1, 3))
        dr = int(_RNG.choice([-1, 1]))
        dc = int(_RNG.choice([-1, 1]))
        pts = []
        for _ in range(200):
            r0 = int(_RNG.integers(0, h))
            c0 = int(_RNG.integers(0, w))
            pts = [(r0, c0), (r0 + dr * step, c0 + dc * step), (r0 + dr * 2 * step, c0 + dc * 2 * step)]
            if all(0 <= r < h and 0 <= c < w and g[r, c] == 0 for r, c in pts):
                break
        if not all(0 <= r < h and 0 <= c < w and g[r, c] == 0 for r, c in pts):
            pts = [(7, 7), (7 + dr * step, 7 + dc * step), (7 + dr * 2 * step, 7 + dc * 2 * step)]

    for r, c in pts:
        if 0 <= r < h and 0 <= c < w:
            g[r, c] = 3

    return g
