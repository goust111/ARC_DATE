from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

BLACK = 0
TEAL = 8


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _scan_line(arr: np.ndarray, idx: int) -> int:
    # length of contiguous TEAL run around idx on row/col 0-run
    # This helper assumes `arr` is 1D with TEAL values and zeros elsewhere.
    n = int(arr.size)
    l = idx
    while l - 1 >= 0 and arr[l - 1] == TEAL:
        l -= 1
    r = idx
    while r + 1 < n and arr[r + 1] == TEAL:
        r += 1
    return int(r - l + 1)


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    h, w = out.shape
    markers = []
    colors = []

    for r in range(h):
        for c in range(w):
            v = int(out[r, c])
            if v != BLACK and v != TEAL:
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    markers.append((r, c, v))

    projections = []
    for r, c, col in markers:
        if r in (0, h - 1):
            dr = 1 if r == 0 else -1
            rr = r + dr
            dist = 0
            while 0 <= rr < h and out[rr, c] == BLACK:
                rr += dr
                dist += 1
            if not (0 <= rr < h):
                continue
            if out[rr, c] != TEAL:
                continue
            row = int(rr)
            base_len = _scan_line(out[row, :], c)
            layers = dist // 2
            if layers <= 0:
                continue
            projections.append((layers, row, c, col, "h", base_len))
        else:
            dc = 1 if c == 0 else -1
            cc = c + dc
            dist = 0
            while 0 <= cc < w and out[r, cc] == BLACK:
                cc += dc
                dist += 1
            if not (0 <= cc < w):
                continue
            if out[r, cc] != TEAL:
                continue
            col_idx = int(cc)
            base_len = _scan_line(out[:, col_idx], r)
            layers = dist // 2
            if layers <= 0:
                continue
            projections.append((layers, r, col_idx, col, "v", base_len))

    projections.sort(key=lambda x: (-x[0], x[1], x[2]))
    prio = np.full(out.shape, -1, dtype=int)

    for layers, p0, p1, col, orient, base_len in projections:
        for t in range(layers):
            run_len = max(0, base_len - 2 * t)
            if run_len <= 0:
                break
            if orient == "h":
                # expand vertically from collision row with horizontal shrinking run
                rr = p0 - t
                if 0 <= rr < h:
                    half = run_len // 2
                    lo = p1 - half
                    hi = p1 + (run_len - 1 - half)
                    for cc in range(lo, hi + 1):
                        if 0 <= cc < w and out[rr, cc] == BLACK and prio[rr, cc] < layers:
                            out[rr, cc] = col
                            prio[rr, cc] = layers
                rr = p0 + t
                if 0 <= rr < h:
                    half = run_len // 2
                    lo = p1 - half
                    hi = p1 + (run_len - 1 - half)
                    for cc in range(lo, hi + 1):
                        if 0 <= cc < w and out[rr, cc] == BLACK and prio[rr, cc] < layers:
                            out[rr, cc] = col
                            prio[rr, cc] = layers
            else:
                # expand horizontally from collision column with vertical shrinking run
                cc = p1 - t
                if 0 <= cc < w:
                    half = run_len // 2
                    lo = p0 - half
                    hi = p0 + (run_len - 1 - half)
                    for rr in range(lo, hi + 1):
                        if 0 <= rr < h and out[rr, cc] == BLACK and prio[rr, cc] < layers:
                            out[rr, cc] = col
                            prio[rr, cc] = layers
                cc = p1 + t
                if 0 <= cc < w:
                    half = run_len // 2
                    lo = p0 - half
                    hi = p0 + (run_len - 1 - half)
                    for rr in range(lo, hi + 1):
                        if 0 <= rr < h and out[rr, cc] == BLACK and prio[rr, cc] < layers:
                            out[rr, cc] = col
                            prio[rr, cc] = layers

    return out


def _rect_perimeter_mask(h: int, w: int, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[r0, c0 : c1 + 1] = True
    m[r1, c0 : c1 + 1] = True
    m[r0 : r1 + 1, c0] = True
    m[r0 : r1 + 1, c1] = True
    return m


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 17
    out = np.zeros((h, w), dtype=int)

    top = int(rng.integers(4, 6))
    left = int(rng.integers(3, 5))
    obj_h = int(rng.integers(8, 11))
    obj_w = int(rng.integers(8, 11))
    obj_h = min(obj_h, h - top - 2)
    obj_w = min(obj_w, w - left - 2)
    obj_r1 = top + obj_h - 1
    obj_c1 = left + obj_w - 1
    out[top : obj_r1 + 1, left : obj_c1 + 1] = TEAL

    marker_colors = [2, 3, 4, 5, 7, 6]
    rng.shuffle(marker_colors)
    n_markers = int(rng.integers(3, 6))
    # Always put markers on border where the ray can hit the target object.
    left_d = left - 1
    right_d = w - 1 - obj_c1 - 1
    top_d = top - 1
    bot_d = h - 1 - obj_r1 - 1
    candidates = []
    if 3 <= left_d <= 8:
        candidates.append("L")
    if 3 <= right_d <= 8:
        candidates.append("R")
    if 3 <= top_d <= 8:
        candidates.append("T")
    if 3 <= bot_d <= 8:
        candidates.append("B")
    if not candidates:
        candidates = ["L", "R", "T", "B"]

    used_colors = marker_colors[:n_markers]
    used_rows: set[int] = set()
    for i in range(n_markers):
        side = candidates[i % len(candidates)]
        col = rng.integers(0, w)
        if side == "L":
            col = 0
            r = int(rng.integers(top, obj_r1 + 1))
            while r in used_rows:
                r = int(rng.integers(top, obj_r1 + 1))
            used_rows.add(r)
            out[r, col] = used_colors[i]
        elif side == "R":
            col = w - 1
            r = int(rng.integers(top, obj_r1 + 1))
            while r in used_rows:
                r = int(rng.integers(top, obj_r1 + 1))
            used_rows.add(r)
            out[r, col] = used_colors[i]
        elif side == "T":
            r = 0
            c = int(rng.integers(left, obj_c1 + 1))
            out[r, c] = used_colors[i]
        else:
            r = h - 1
            c = int(rng.integers(left, obj_c1 + 1))
            out[r, c] = used_colors[i]

    return out

