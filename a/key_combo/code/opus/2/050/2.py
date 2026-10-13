"""Build an output by combining quadrant counts with a unique sprite exemplar.

Input structure:
  An even-sized square black-background grid split by a full row/column divider
  into four quadrants. The upper quadrants contain sparse non-black marks.
  The lower-right quadrant contains several same-size small sprite blocks; most
  share one pattern and one sprite is the unique anomalous pattern.

Rule:
  Output height is the count of non-black pixels in the upper-left quadrant and
  output width is the count of non-black pixels in the upper-right quadrant.
  Select the unique sprite pattern in the lower-right quadrant among same-size
  sprite objects and tile this pattern from the output top-left to fill the
  output grid. Other inputs are not directly drawn in output.

Key insight:
  The size of the output and the patterned content come from orthogonal sources:
  one pair of quadrants encodes geometry, the other encodes the anomalous
  template.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def _find_division(grid: np.ndarray, bg: int = 0) -> Tuple[int, int]:
    H, W = grid.shape
    row_scores = [int((row != bg).sum()) for row in grid]
    row = int(np.argmax(row_scores))
    col_scores = [(grid[:, c] != bg).sum() for c in range(W)]
    col = int(np.argmax(col_scores))
    return row, col


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and not visited[nr, nc]
                        and mask[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _pattern_key(grid: np.ndarray, cells: List[Tuple[int, int]]) -> Tuple[Tuple[int, ...], int, int, int, int]:
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return tuple(int(x) for x in patch.flatten()), r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=False)
    if g.ndim != 2 or min(g.shape) == 0:
        return g.copy()
    H, W = g.shape
    r_line, c_line = _find_division(g)

    ul = g[:r_line, :c_line]
    ur = g[:r_line, c_line + 1 :]
    h_out = int((ul != 0).sum())
    w_out = int((ur != 0).sum())
    h_out = max(1, min(30, h_out))
    w_out = max(1, min(30, w_out))

    q4 = g[r_line + 1 :, c_line + 1 :]
    excl = np.zeros_like(q4, dtype=bool)
    excl[:, :] = q4 != 0
    comps = _components(excl)
    if not comps:
        return np.zeros((h_out, w_out), dtype=np.int64)

    patt_sizes: Dict[Tuple[int, int], List[dict]] = {}
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, c0 = min(rs), min(cs)
        r1, c1 = max(rs), max(cs)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        key_patch = tuple(int(x) for x in q4[r0 : r1 + 1, c0 : c1 + 1].flatten())
        key = (h, w, key_patch)
        patt_sizes.setdefault((h, w), []).append({"key": key, "data": q4[r0 : r1 + 1, c0 : c1 + 1]})

    best_pattern = None
    for k, lst in patt_sizes.items():
        if not lst:
            continue
        freq = Counter(d["key"] for d in lst)
        unique_keys = [t for t, c in freq.items() if c == 1]
        if len(unique_keys) == 1:
            up = unique_keys[0]
            for d in lst:
                if d["key"] == up:
                    best_pattern = d["data"]
                    break
            break

    if best_pattern is None:
        # Fallback: choose the least frequent pattern among same-size groups
        candidates = []
        for lst in patt_sizes.values():
            freq = Counter(d["key"] for d in lst)
            for d in lst:
                candidates.append((freq[d["key"]], d["data"]))
        candidates.sort(key=lambda t: t[0])
        best_pattern = candidates[0][1] if candidates else np.zeros((1, 1), dtype=np.int64)

    ph, pw = best_pattern.shape
    out = np.zeros((h_out, w_out), dtype=np.int64)
    for r in range(0, h_out, ph):
        for c in range(0, w_out, pw):
            h2 = min(ph, h_out - r)
            w2 = min(pw, w_out - c)
            out[r : r + h2, c : c + w2] = best_pattern[:h2, :w2]
    return out


def _place_sprite(grid: np.ndarray, top: int, left: int, pattern: np.ndarray) -> None:
    h, w = pattern.shape
    grid[top : top + h, left : left + w] = pattern


def _make_pattern(size: Tuple[int, int]) -> np.ndarray:
    h, w = size
    c1 = random.choice([1, 2, 3, 8, 9, 7])
    c2 = random.choice([1, 2, 3, 8, 9, 7, c1])
    while c2 == c1:
        c2 = random.choice([1, 2, 3, 8, 9, 7, c1])
    if random.random() < 0.65:
        return np.full((h, w), c1, dtype=np.int64)
    patch = np.full((h, w), c1, dtype=np.int64)
    n = random.randint(1, max(1, h * w // 3))
    cells = random.sample([(r, c) for r in range(h) for c in range(w)], n)
    for rr, cc in cells:
        patch[rr, cc] = c2
    return patch


def generate_input() -> np.ndarray:
    H = random.choice([14, 16])
    W = H
    line = 5
    grid = np.zeros((H, W), dtype=np.int64)
    r_line = H // 2
    c_line = W // 2
    grid[r_line, :] = line
    grid[:, c_line] = line

    # upper quadrants: sparse markers
    for r in range(0, r_line):
        for c in range(0, c_line):
            if random.random() < 0.03:
                grid[r, c] = random.choice([2, 3, 4, 7, 8, 9, 1])
        for c in range(c_line + 1, W):
            if random.random() < 0.03:
                grid[r, c] = random.choice([2, 3, 4, 7, 8, 9, 1])

    # lower-right sprites
    size = random.choice([(2, 2), (3, 3)])
    sr, sc = size
    start_r0 = r_line + 1
    start_c0 = c_line + 1
    avail_r = list(range(start_r0, H - sr + 1))
    avail_c = list(range(start_c0, W - sc + 1))
    n = random.randint(4, 6)
    n_rep = random.randint(2, n - 1)
    base_pattern = _make_pattern(size)
    uniq_pattern = base_pattern.copy()
    rr, cc = random.randint(0, sr - 1), random.randint(0, sc - 1)
    uniq_pattern[rr, cc] = random.choice([x for x in [1, 2, 3, 4, 7, 8, 9] if x != uniq_pattern[rr, cc]])

    def can_place(tr: int, tc: int) -> bool:
        if np.any(grid[tr - 1 : tr + sr + 1, tc - 1 : tc + sc + 1] != 0):
            # enforce one-cell spacing inside quadrant
            return False
        return True

    used = []
    # place repeated sprites
    for _ in range(n_rep):
        placed = False
        for _ in range(200):
            tr = random.choice(avail_r)
            tc = random.choice(avail_c)
            if can_place(tr, tc):
                _place_sprite(grid, tr, tc, base_pattern)
                used.append((tr, tc))
                placed = True
                break
        if not placed:
            tr = random.choice(avail_r)
            tc = random.choice(avail_c)
            _place_sprite(grid, tr, tc, base_pattern)

    # place unique outlier
    placed = False
    for _ in range(300):
        tr = random.choice(avail_r)
        tc = random.choice(avail_c)
        if (tr, tc) in used and can_place(tr, tc):
            uniq_pattern = base_pattern.copy()
            rr, cc = random.randint(0, sr - 1), random.randint(0, sc - 1)
            uniq_pattern[rr, cc] = random.choice([x for x in [1, 2, 3, 4, 7, 8, 9] if x != uniq_pattern[rr, cc]])
            _place_sprite(grid, tr, tc, uniq_pattern)
            used.append((tr, tc))
            placed = True
            break
    if not placed and used:
        tr, tc = used[0]
        _place_sprite(grid, tr, tc, uniq_pattern)
    return grid
