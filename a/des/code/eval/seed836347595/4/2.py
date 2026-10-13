from __future__ import annotations

import heapq
import random

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _move_width(pipe: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = pipe.shape
    if dr != 0:
        return 2 if ((c > 0 and pipe[r, c - 1]) or (c + 1 < w and pipe[r, c + 1])) else 1
    return 2 if ((r > 0 and pipe[r - 1, c]) or (r + 1 < h and pipe[r + 1, c])) else 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    pipe = g == GREY
    seed_cells = [(int(g[r, c]), r, c) for r in range(h) for c in range(w) if int(g[r, c]) not in (BG, GREY)]
    assert seed_cells, "expected colored pipe-end seeds"

    out = g.copy()
    best: dict[tuple[int, int], tuple[int, int, int]] = {}
    heap: list[tuple[int, int, int, int, int, int]] = []
    for color, r, c in seed_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and pipe[nr, nc]:
                dr, dc = nr - r, nc - c
                width = _move_width(pipe, nr, nc, dr, dc)
                heapq.heappush(heap, (-width, 1, color, nr, nc, color))

    while heap:
        neg_width, dist, tie_color, r, c, color = heapq.heappop(heap)
        key = (neg_width, dist, tie_color)
        old = best.get((r, c))
        if old is not None and old <= key:
            continue
        best[(r, c)] = key
        out[r, c] = color
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and pipe[nr, nc]:
                dr, dc = nr - r, nc - c
                width = _move_width(pipe, nr, nc, dr, dc)
                heapq.heappush(heap, (-width, dist + 1, color, nr, nc, color))

    assert all(pos in best for pos in map(tuple, np.argwhere(pipe))), "all grey pipe cells must be colored"
    return out


def _draw_cross(h: int, w: int, hr: int, hc0: int, hc1: int, hw: int, vc: int, vr0: int, vr1: int, vw: int) -> np.ndarray:
    g = np.full((h, w), BG, dtype=int)
    g[hr : hr + hw, hc0 : hc1 + 1] = GREY
    g[vr0 : vr1 + 1, vc : vc + vw] = GREY
    return g


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    hw = random.choice((1, 2))
    vw = random.choice((1, 2))
    hr_hi = h - hw - 4
    vc_hi = w - vw - 4
    if hr_hi < 3 or vc_hi < 3:
        return None
    hr = random.randint(3, hr_hi)
    vc = random.randint(3, vc_hi)
    left = random.randint(1, max(1, vc - 3))
    right = random.randint(vc + vw + 2, w - 2)
    top = random.randint(1, max(1, hr - 3))
    bottom = random.randint(hr + hw + 2, h - 2)
    g = _draw_cross(h, w, hr, left, right, hw, vc, top, bottom, vw)

    colors = random.sample(SEED_COLORS, 4)
    endpoints = [
        (hr, left, colors[0]),
        (hr + hw - 1, right, colors[1]),
        (top, vc, colors[2]),
        (bottom, vc + vw - 1, colors[3]),
    ]
    for r, c, color in endpoints:
        if g[r, c] != GREY:
            return None
        g[r, c] = color

    try:
        out = main(g)
    except AssertionError:
        return None
    if np.any(out == GREY):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid input")
