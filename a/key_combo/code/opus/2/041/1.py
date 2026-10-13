from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
GREEN = Color.GREEN
BROWN = Color.MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each gate-separated segment of one-pixel channel by inferring its own periodic pattern.

    Input structure: mostly black background, one green channel line (horizontal or vertical), 2-3 gate motifs
    "\ 9-green-9 splitting the line into independent segments, and segments with black holes plus pattern colors.

    Rule: detect channel orientation and gate centers, split the channel into segments, infer a 1D repeating tile
    for each segment from non-black cells only, then fill black holes in that segment preserving phase.

    Key insight: gate motifs are not path endpoints; they are segment boundaries, so each segment needs independent
    period inference.
    """

    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D ndarray")
    out = g.copy()

    h, w = g.shape
    channel_mask = g == int(GREEN)
    if not np.any(channel_mask):
        return out

    # Detect dominant channel orientation by channel extent.
    rows, cols = np.where(channel_mask)
    row_span = int(rows.max() - rows.min()) if rows.size else 0
    col_span = int(cols.max() - cols.min()) if cols.size else 0

    gates = _find_gates(g, int(GREEN), int(BROWN))
    if row_span >= col_span:
        axis = "h"
        # choose the channel row with most green pixels
        row_counts = [(int((g[r] == int(GREEN)).sum()), r) for r in range(h)]
        _, row = max(row_counts)
        row = int(row)
        row_gates = sorted({c for gr, c in gates if gr == row})
        green_cols = sorted(np.where(g[row] == int(GREEN))[0])
        if not green_cols:
            return out
        segs = _split_indices(green_cols, row_gates, int(min(green_cols)), int(max(green_cols)))
        for lo, hi in segs:
            _fill_1d_segment(out, [(row, c) for c in range(lo, hi + 1)], horizontal=True)
    else:
        axis = "v"
        # choose the channel column with most green pixels
        col_counts = [(int((g[:, c] == int(GREEN)).sum()), c) for c in range(w)]
        _, col = max(col_counts)
        col = int(col)
        col_gates = sorted({r for r, gc in gates if gc == col})
        green_rows = sorted(np.where(g[:, col] == int(GREEN))[0])
        if not green_rows:
            return out
        segs = _split_indices(green_rows, col_gates, int(min(green_rows)), int(max(green_rows)))
        for lo, hi in segs:
            _fill_1d_segment(out, [(r, col) for r in range(lo, hi + 1)], horizontal=False)

    return out


def _find_gates(grid: np.ndarray, green: int, brown: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    out: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != green:
                continue
            if int(grid[r - 1, c]) == brown and int(grid[r + 1, c]) == brown:
                out.add((r, c))
            if int(grid[r, c - 1]) == brown and int(grid[r, c + 1]) == brown:
                out.add((r, c))
    return out


def _split_indices(
    axis_coords: list[int], gate_coords: list[int], start: int, end: int
) -> list[tuple[int, int]]:
    if not axis_coords:
        return []
    gate_coords = [c for c in sorted(gate_coords) if start <= c <= end]
    if not gate_coords:
        return [(start, end)]

    segments: list[tuple[int, int]] = []
    prev = start
    for c in gate_coords:
        if prev <= c - 1:
            segments.append((prev, c - 1))
        prev = c + 1
    if prev <= end:
        segments.append((prev, end))
    return segments


def _infer_period_1d(values: list[int]) -> list[int]:
    known = [(i, v) for i, v in enumerate(values) if v != int(BLACK)]
    n = len(values)
    if n == 0:
        return [int(GREEN)]
    if not known:
        return [int(GREEN)]

    for p in range(1, n + 1):
        ok = True
        tile = [None] * p
        for i, v in known:
            k = i % p
            if tile[k] is None:
                tile[k] = v
            elif tile[k] != v:
                ok = False
                break
        if not ok:
            continue
        fill = next(v for v in tile if v is not None)
        for i, v in enumerate(tile):
            if v is None:
                tile[i] = fill
        return list(map(int, tile))
    return [values[known[0][1]]]


def _fill_1d_segment(out: np.ndarray, cells: list[tuple[int, int]], horizontal: bool) -> None:
    if not cells:
        return
    values = [int(out[r, c]) for r, c in cells]
    # Replace marker black holes from within only.
    tile = _infer_period_1d(values)
    p = len(tile)
    for i, (r, c) in enumerate(cells):
        if int(out[r, c]) == int(BLACK):
            out[r, c] = int(tile[i % p])


def generate_input() -> np.ndarray:
    h = random.randint(11, 12)
    w = random.randint(14, 18)
    g = np.full((h, w), int(BLACK), dtype=int)

    axis = random.choice(["h", "v"])
    if axis == "h":
        row = random.randint(2, h - 3)
        start = random.randint(1, 2)
        end = random.randint(start + 6, w - 3)
        n_gates = random.randint(2, 3)
        # draw base green channel
        for c in range(start, end + 1):
            g[row, c] = int(GREEN)

        gate_candidates = [c for c in range(start + 2, end - 2) if c + 1 < w - 1]
        random.shuffle(gate_candidates)
        chosen = sorted(gate_candidates[:n_gates])
        for c in chosen:
            g[row - 1, c] = int(BROWN)
            g[row, c] = int(GREEN)
            g[row + 1, c] = int(BROWN)

        segments = _split_indices(list(range(start, end + 1)), chosen, start, end)
        colors = [1, 2, 3, 4, 6, 7, 9]
        for lo, hi in segments:
            if lo > hi:
                continue
            length = hi - lo + 1
            p = random.randint(1, min(4, max(1, length)))
            tile = [random.choice(colors) for _ in range(p)]
            for off, c in enumerate(range(lo, hi + 1)):
                if random.random() < 0.18:
                    continue
                g[row, c] = int(tile[off % p])
    else:
        col = random.randint(2, w - 3)
        start = random.randint(1, 2)
        end = random.randint(start + 6, h - 3)
        n_gates = random.randint(2, 3)
        for r in range(start, end + 1):
            g[r, col] = int(GREEN)

        gate_candidates = [r for r in range(start + 2, end - 2) if r + 1 < h - 1]
        random.shuffle(gate_candidates)
        chosen = sorted(gate_candidates[:n_gates])
        for r in chosen:
            g[r, col - 1] = int(BROWN)
            g[r, col] = int(GREEN)
            g[r, col + 1] = int(BROWN)

        segments = _split_indices(list(range(start, end + 1)), chosen, start, end)
        colors = [1, 2, 3, 4, 6, 7, 9]
        for lo, hi in segments:
            if lo > hi:
                continue
            length = hi - lo + 1
            p = random.randint(1, min(4, max(1, length)))
            tile = [random.choice(colors) for _ in range(p)]
            for off, r in enumerate(range(lo, hi + 1)):
                if random.random() < 0.18:
                    continue
                g[r, col] = int(tile[off % p])

    return g
