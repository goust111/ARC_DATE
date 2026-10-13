from __future__ import annotations

import random
from collections import Counter, defaultdict

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



def _parse_palette(row: np.ndarray) -> list[int]:
    palette = []
    c = 0
    while c + 1 < len(row):
        if row[c] == row[c + 1]:
            palette.append(int(row[c]))
            c += 2
        else:
            c += 1
    return palette


def _find_bar_rows(grid: np.ndarray, bg: int) -> list[int]:
    bars = []
    for r in range(1, grid.shape[0]):
        row = grid[r]
        if row[0] == bg:
            continue
        if np.all(row == row[0]):
            bars.append(int(r))
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    bg = int(get_bg(g))

    palette = _parse_palette(g[0])
    if not palette:
        return g.copy()

    bars = _find_bar_rows(g, bg)
    colors_to_row = {}
    for r in bars:
        colors_to_row[int(g[r, 0])] = r
    if len(colors_to_row) != 3:
        return g.copy()

    out = g.copy()
    counts = {row: 0 for row in colors_to_row.values()}
    moved_by_row: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # erase all candidate pixels first, keep palette row and bars unchanged for now
    for r in range(1, H):
        for c in range(W):
            if r in colors_to_row.values():
                continue
            if g[r, c] != bg:
                out[r, c] = bg

    for r in range(1, H):
        for c in range(W):
            color = int(g[r, c])
            if color == bg:
                continue
            if color not in colors_to_row:
                continue
            if r in colors_to_row.values():
                continue
            bar_row = colors_to_row[color]
            if r < bar_row:
                dest = bar_row - 1
            else:
                dest = bar_row + 1
            if not (0 <= dest < H):
                continue
            # ensure no additional bar is crossed in vertical path
            blocked = False
            step = 1 if dest > r else -1
            rr = r + step
            while rr != dest + step:
                if rr in colors_to_row.values() and rr != bar_row:
                    blocked = True
                    break
                rr += step
            if blocked:
                continue
            out[dest, c] = color
            counts[bar_row] += 1
            moved_by_row[bar_row].append((dest, c))

    # recolor each bar and its attached pixels
    for row, color in colors_to_row.items():
        cnt = counts[row]
        if cnt <= 0 or cnt > len(palette):
            continue
        new_col = palette[cnt - 1]
        out[row, :] = new_col
        for rr, cc in moved_by_row[row]:
            out[rr, cc] = new_col

    out[0, :] = g[0, :]
    return out


def _clear_row(arr: np.ndarray, r: int, c0: int, c1: int, value: int) -> None:
    for c in range(c0, c1):
        arr[r, c] = value


def generate_input() -> np.ndarray:
    H = 14
    W = 14
    bg = 0

    for _ in range(200):
        palette_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 4))
        row0 = np.full(W, bg, dtype=int)
        p = 0
        for col in palette_colors:
            if p + 1 >= W:
                break
            row0[p] = col
            row0[p + 1] = col
            p += 2

        if p < W:
            # keep exact pairs, leave remaining as bg
            for k in range(p, W):
                row0[k] = bg

        bar_colors = random.sample([c for c in [1, 2, 7, 3, 4, 6, 8, 9] if c not in palette_colors], 3)
        bar_rows = random.sample(range(1, H - 1), 3)
        if any(v == 0 for v in bar_colors):
            continue

        if len(set(bar_rows)) != 3:
            continue
        bar_rows = sorted(bar_rows)

        g = np.full((H, W), bg, dtype=int)
        g[0, :] = row0
        for row, col in zip(bar_rows, bar_colors):
            g[row, :] = col

        # matched pixels near each bar, unique counts from 1..len(palette)
        counts = random.sample(range(1, len(palette_colors) + 1), 3)
        for row, color, cnt in zip(bar_rows, bar_colors, counts):
            for _ in range(cnt):
                col = random.randint(0, W - 1)
                candidate_row = row - 1 if row > 0 else row + 1
                if not (0 <= candidate_row < H):
                    continue
                if g[candidate_row, col] != bg:
                    col = (col + 1) % W
                g[candidate_row, col] = color

        # unmatched noise pixels
        noise_colors = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in set(bar_colors)]
        for _ in range(random.randint(1, 3)):
            rr = random.randint(2, H - 2)
            cc = random.randint(0, W - 1)
            if g[rr, cc] == bg:
                g[rr, cc] = random.choice(noise_colors)

        try:
            _ = main(g)
            return g
        except Exception:
            continue

    raise RuntimeError("failed to generate valid puzzle-6 input")
