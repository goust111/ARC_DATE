from __future__ import annotations

import random
from collections import Counter

import numpy as np

import os
import sys
from pathlib import Path


_arc_utils_path = None
for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    for _cand_dir in (_p / "new" / "solvers", _p / "solvers"):
        _cand = _cand_dir / "arc_utils.py"
        if _cand.exists():
            _arc_utils_path = str(_cand_dir)
            break
    if _arc_utils_path is not None:
        break
if _arc_utils_path and _arc_utils_path not in sys.path:
    sys.path.insert(0, _arc_utils_path)

from arc_utils import get_bg



def _is_separator_row(row: np.ndarray, bg: int) -> bool:
    return int(row[0]) != bg and np.all(row == row[0])


def _is_separator_col(col: np.ndarray, bg: int) -> bool:
    return int(col[0]) != bg and np.all(col == col[0])


def _bands(mask: np.ndarray) -> list[tuple[int, int]]:
    bands = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        start = i
        while i < n and not mask[i]:
            i += 1
        bands.append((start, i - 1))
    return bands


def _majority_color(block: np.ndarray, bg: int) -> int:
    cnt = Counter(int(v) for v in block.ravel() if int(v) != bg)
    if not cnt:
        return 0
    return int(cnt.most_common(1)[0][0])


def _infer_cycle(compact: np.ndarray) -> tuple[int, list[int]]:
    R, C = compact.shape
    states = [int(v) for v in compact.ravel() if v != 0]
    if not states:
        raise ValueError("no clues")

    best_period = None
    best_map = None
    best_coverage = -1

    max_p = max(R + C, 1)
    for p in range(1, max_p + 1):
        mapping: dict[int, int] = {}
        valid = True
        for r in range(R):
            for c in range(C):
                color = int(compact[r, c])
                if color == 0:
                    continue
                k = (r + c) % p
                if k in mapping and mapping[k] != color:
                    valid = False
                    break
                mapping[k] = color
            if not valid:
                break

        if not valid:
            continue

        coverage = len(mapping)
        if coverage == p:
            best_period = p
            best_map = mapping
            break

        if coverage > best_coverage:
            best_period = p
            best_map = mapping
            best_coverage = coverage

    if best_period is None or best_map is None:
        raise ValueError("period inference failed")

    if len(best_map) < best_period:
        fallback = next(iter(best_map.values()))
        for k in range(best_period):
            best_map.setdefault(k, fallback)

    cycle = [best_map[k] for k in range(best_period)]
    return int(best_period), cycle


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    Rg, Cg = g.shape
    sep_color = 5

    row_sep = np.array([_is_separator_row(g[r], bg) for r in range(Rg)])
    col_sep = np.array([_is_separator_col(g[:, c], bg) for c in range(Cg)])

    row_bands = _bands(row_sep)
    col_bands = _bands(col_sep)
    if not row_bands or not col_bands:
        return g.copy()

    for r in np.where(row_sep)[0]:
        sep_color = int(g[r, 0])
        break

    compact = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            compact[i, j] = _majority_color(block, bg)

    period, cycle = _infer_cycle(compact)
    out = np.full((len(row_bands) + 2, len(col_bands) + 2), sep_color, dtype=int)
    for r in range(len(row_bands)):
        for c in range(len(col_bands)):
            out[r + 1, c + 1] = cycle[(r + c) % period]
    return out


def _random_block_patch(colors: list[int], majority_color: int) -> np.ndarray:
    h = 3
    w = 3
    total = h * w
    min_major = total // 2 + 1
    major_count = random.randint(min_major, total)
    block = np.full((h, w), majority_color, dtype=int)

    if major_count < total:
        other = [c for c in colors if c != majority_color]
        coords = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(coords)
        for rr, cc in coords[: total - major_count]:
            block[rr, cc] = random.choice(other)
    return block


def generate_input() -> np.ndarray:
    bg = 0
    colors = [1, 2, 3, 4, 6]

    for _ in range(200):
        R = random.randint(3, 5)
        C = random.randint(4, 6)
        cycle_len = random.randint(2, min(4, R + C))
        cycle = random.sample(colors, cycle_len)

        h = 3
        w = 3
        H = R * (h + 1) + 1
        W = C * (w + 1) + 1
        if H > 30 or W > 30:
            continue

        g = np.full((H, W), bg, dtype=int)
        for r in range(H):
            if r % (h + 1) == 0:
                g[r, :] = 5
        for c in range(W):
            if c % (w + 1) == 0:
                g[:, c] = 5

        all_cells = [(r, c) for r in range(R) for c in range(C)]
        blanks = set(random.sample(all_cells, random.randint(2, 4)))

        # avoid full blank row/col in block lattice
        row_counts = [0] * R
        col_counts = [0] * C
        for rr, cc in blanks:
            row_counts[rr] += 1
            col_counts[cc] += 1
        if any(v == C for v in row_counts) or any(v == R for v in col_counts):
            continue

        ok = True
        for rb in range(R):
            for cb in range(C):
                r0 = rb * (h + 1) + 1
                c0 = cb * (w + 1) + 1
                if (rb, cb) in blanks:
                    continue

                col = cycle[(rb + cb) % cycle_len]
                block = _random_block_patch(colors, col)
                g[r0 : r0 + h, c0 : c0 + w] = block

        try:
            out = main(g)
            if out.ndim != 2:
                continue
        except Exception:
            ok = False

        if ok:
            return g

    raise RuntimeError("failed to generate valid puzzle-2 input")
