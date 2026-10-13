from __future__ import annotations

from itertools import product
from typing import Tuple

import numpy as np


def _best_period_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = mask.shape
    fg = np.argwhere(mask == 1)
    if fg.size == 0:
        return 1, 1, 0, 0
    max_dy = max(2, h // 2)
    max_dx = max(2, w // 2)
    best = (-1.0, h * w + 1, 1, 1, 0, 0)
    obs_rows = max(2, int(round(h * 0.6)))

    for dy in range(2, max_dy + 1):
        for dx in range(2, max_dx + 1):
            for pr in range(dy):
                for pc in range(dx):
                    template = np.zeros((dy, dx), dtype=bool)
                    for r, c in fg:
                        if r >= obs_rows:
                            continue
                        template[(r - pr) % dy, (c - pc) % dx] = True
                    if not template.any():
                        continue
                    rr, cc = np.indices((obs_rows, w))
                    pred = template[(rr - pr) % dy, (cc - pc) % dx]
                    obs = mask[:obs_rows] == 1
                    score = float(np.mean(pred == obs))
                    area = dy * dx
                    if (score > best[0]) or (abs(score - best[0]) < 1e-9 and area < best[1]):
                        best = (score, area, dy, dx, pr, pc)

    if best[0] < 0:
        return 1, 1, 0, 0
    _, _, dy, dx, pr, pc = best
    return dy, dx, pr, pc


def _complete_panel(panel: np.ndarray, dy: int, dx: int, pr: int, pc: int) -> np.ndarray:
    h, w = panel.shape
    out = np.zeros((h, w), dtype=int)
    rr, cc = np.indices((h, w))
    repeat = ((rr - pr) % dy == 0) & ((cc - pc) % dx == 0)
    # Build full periodic residue map
    full = np.zeros((dy, dx), dtype=bool)
    for r, c in np.argwhere(panel == 1):
        full[(r - pr) % dy, (c - pc) % dx] = True
    out = full[(rr - pr) % dy, (cc - pc) % dx].astype(int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    sep_cols = [i for i in range(1, w - 1) if np.all(g[:, i] == 5)]
    if not sep_cols:
        return g.copy()
    sep = int(np.median(sep_cols))

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    if left.size == 0 or right.size == 0:
        return g.copy()

    left_mask = (left == 2).astype(int)
    right_mask = (right == 1).astype(int)

    dy1, dx1, pr1, pc1 = _best_period_from_mask(left_mask)
    dy2, dx2, pr2, pc2 = _best_period_from_mask(right_mask)

    left_full = _complete_panel(left_mask, dy1, dx1, pr1, pc1)
    right_full = _complete_panel(right_mask, dy2, dx2, pr2, pc2)

    out = np.zeros_like(left)
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            if left_full[r, c] and right_full[r, c]:
                out[r, c] = 4
            elif left_full[r, c]:
                out[r, c] = 2
            elif right_full[r, c]:
                out[r, c] = 1
            else:
                out[r, c] = 0
    return out


_RNG = np.random.default_rng()


def _build_periodic_panel(h: int, w: int, fg_color: int, dy: int, dx: int, shift_r: int, shift_c: int) -> np.ndarray:
    # Build a small base tile then make a full periodic field, then mask
    base = np.zeros((dy, dx), dtype=int)
    for r in range(dy):
        for c in range(dx):
            if _RNG.random() < 0.35:
                base[r, c] = fg_color

    rr, cc = np.indices((h, w))
    full = ((rr - shift_r) % dy < dy) & ((cc - shift_c) % dx < dx)
    tiled = full.astype(int)
    tiled[(rr - shift_r) % dy, (cc - shift_c) % dx] = base[(rr[(rr - shift_r) % dy < dy] - shift_r) % dy, :]
    # Simpler stable fill:
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = base[(r - shift_r) % dy, (c - shift_c) % dx]
    # Reveal only upper window, lower rows remain bg (as in generation)
    return out


def generate_input() -> np.ndarray:
    h = int(_RNG.integers(13, 16))
    w = 27
    sep = 13

    left_w = right_w = sep
    d1_r = int(_RNG.integers(2, min(4, left_w) + 1))
    d1_c = int(_RNG.integers(2, min(5, h - 2) + 1))
    d2_r = int(_RNG.integers(2, min(4, left_w) + 1))
    d2_c = int(_RNG.integers(2, min(5, h - 2) + 1))
    if (d2_r, d2_c) == (d1_r, d1_c):
        d2_c = max(2, min(5, d1_c + 1))

    sr1 = int(_RNG.integers(0, d1_r))
    sc1 = int(_RNG.integers(0, d1_c))
    sr2 = int(_RNG.integers(0, d2_r))
    sc2 = int(_RNG.integers(0, d2_c))

    left = np.zeros((h, left_w), dtype=int)
    right = np.zeros((h, right_w), dtype=int)

    left_base = np.zeros((d1_r, d1_c), dtype=int)
    right_base = np.zeros((d2_r, d2_c), dtype=int)
    for r in range(d1_r):
        for c in range(d1_c):
            if _RNG.random() < 0.35:
                left_base[r, c] = 1
    for r in range(d2_r):
        for c in range(d2_c):
            if _RNG.random() < 0.35:
                right_base[r, c] = 1
    if not left_base.any():
        left_base[0, 0] = 1
    if not right_base.any():
        right_base[0, 0] = 1
    if np.array_equal(left_base, right_base) and d1_r == d2_r and d1_c == d2_c and (d1_r == 1 or d1_c == 1):
        right_base[0, min(1, right_base.shape[1] - 1)] = 0 if right_base.shape[1] > 1 else right_base[0, 0]

    rr, cc = np.indices((h, left_w))
    for r in range(h):
        for c in range(left_w):
            if left_base[(r + sr1) % d1_r, (c + sc1) % d1_c]:
                left[r, c] = 2
    for r in range(h):
        for c in range(right_w):
            if right_base[(r + sr2) % d2_r, (c + sc2) % d2_c]:
                right[r, c] = 1

    # only upper 40%~60% visible
    visible_rows = int(max(5, round(h * _RNG.uniform(0.4, 0.6))))
    left[visible_rows:] = 0
    right[visible_rows:] = 0

    sep_col = np.full((h, 1), 5, dtype=int)
    return np.concatenate((left, sep_col, right), axis=1)

