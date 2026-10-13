from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
NON_YELLOW_NON_BLACK = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _compressed_axes(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep_rows = np.all(grid == BLACK, axis=1)
    sep_cols = np.all(grid == BLACK, axis=0)
    keep_rows = np.flatnonzero(~sep_rows)
    keep_cols = np.flatnonzero(~sep_cols)
    assert keep_rows.size > 0 and keep_cols.size > 0, "expected non-separator rows and cols"
    return keep_rows, keep_cols


def _diagonal_segments(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    segments: list[list[tuple[int, int]]] = []
    for start in sorted(coords):
        r, c = start
        if (r - 1, c - 1) in coords:
            continue
        seg: list[tuple[int, int]] = []
        while (r, c) in coords:
            seg.append((r, c))
            r += 1
            c += 1
        segments.append(seg)
    return segments


def _transform_compressed(grid: np.ndarray) -> np.ndarray:
    bg = _dominant_color(grid)
    out = np.array(grid, copy=True)
    fg_mask = grid != bg
    for seg in _diagonal_segments(fg_mask):
        for idx, (r, c) in enumerate(seg):
            if idx % 2 == 1:
                out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove full black separators, alternate each compressed diagonal, then project the recoloring back.

    Input structure:
    - The canvas has a uniform non-black background plus several full black separator rows and columns.
    - Away from those separators, isolated foreground pixels occupy some of the remaining row/column intersections.
    - In the compressed grid formed by deleting every all-black row and column, the foreground pixels form one or more maximal down-right diagonal segments.

    Rule:
    - Delete all pure-black rows and pure-black columns to obtain the compressed grid.
    - For every maximal foreground segment connected by repeated `(r+1, c+1)` steps in that compressed grid, count cells from the top-left endpoint.
    - Keep even-indexed cells in their original color and recolor odd-indexed cells to yellow, then place the compressed result back at the original non-separator row/column intersections.

    Key insight:
    - The diagonal adjacency lives in separator-compressed coordinates, not in the irregular original spacing, so the recoloring must be decided after removing the full-span black rows and columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    keep_rows, keep_cols = _compressed_axes(g)
    compressed = g[np.ix_(keep_rows, keep_cols)]
    recolored = _transform_compressed(compressed)

    out = np.array(g, copy=True)
    out[np.ix_(keep_rows, keep_cols)] = recolored
    return out


def _diag_cells(h: int, w: int, delta: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        c = r + delta
        if 0 <= c < w:
            cells.append((r, c))
    return cells


def _sample_segments(h: int, w: int) -> list[list[tuple[int, int]]] | None:
    diag_ids = list(range(-(h - 1), w))
    random.shuffle(diag_ids)

    chosen: list[int] = []
    for delta in diag_ids:
        if all(abs(delta - prev) >= 2 for prev in chosen):
            chosen.append(delta)
        if len(chosen) >= 4:
            break

    if len(chosen) < 2:
        return None

    n_segments = random.randint(2, min(4, len(chosen)))
    chosen = random.sample(chosen, n_segments)

    long_segment_count = 0
    segments: list[list[tuple[int, int]]] = []
    for idx, delta in enumerate(chosen):
        cells = _diag_cells(h, w, delta)
        if not cells:
            return None
        max_len = min(5, len(cells))
        min_len = 2 if idx == 0 and max_len >= 2 else 1
        if min_len > max_len:
            return None
        length = random.randint(min_len, max_len)
        if length >= 2:
            long_segment_count += 1
        start = random.randint(0, len(cells) - length)
        segments.append(cells[start : start + length])

    if long_segment_count == 0:
        return None
    return segments


def _sample_spacing(n: int, limit: int) -> tuple[int, list[int], int] | None:
    for _ in range(200):
        top = random.randint(0, 2)
        bottom = random.randint(0, 2)
        between = [random.randint(0, 2) for _ in range(n - 1)]
        if sum(between) == 0:
            continue
        if n >= 3 and not any(gap == 0 for gap in between):
            continue
        size = top + bottom + n + sum(between)
        if size <= limit:
            return top, between, bottom
    return None


def _inflate(compressed: np.ndarray) -> np.ndarray | None:
    h, w = compressed.shape
    row_spacing = _sample_spacing(h, 30)
    col_spacing = _sample_spacing(w, 30)
    if row_spacing is None or col_spacing is None:
        return None

    top_rows, row_gaps, bottom_rows = row_spacing
    left_cols, col_gaps, right_cols = col_spacing

    row_map: list[int] = []
    cur_r = top_rows
    for r in range(h):
        row_map.append(cur_r)
        cur_r += 1
        if r < h - 1:
            cur_r += row_gaps[r]
    out_h = cur_r + bottom_rows

    col_map: list[int] = []
    cur_c = left_cols
    for c in range(w):
        col_map.append(cur_c)
        cur_c += 1
        if c < w - 1:
            cur_c += col_gaps[c]
    out_w = cur_c + right_cols

    grid = np.full((out_h, out_w), BLACK, dtype=int)
    for r in range(h):
        for c in range(w):
            grid[row_map[r], col_map[c]] = int(compressed[r, c])
    return grid


def _build_compressed() -> np.ndarray | None:
    h = random.randint(6, 10)
    w = random.randint(6, 10)
    bg, fg = random.sample(NON_YELLOW_NON_BLACK, 2)

    segments = _sample_segments(h, w)
    if segments is None:
        return None

    compressed = np.full((h, w), bg, dtype=int)
    for seg in segments:
        for r, c in seg:
            compressed[r, c] = fg

    fg_count = int(np.sum(compressed == fg))
    if fg_count < 4 or fg_count > (h * w) // 3:
        return None
    return compressed


def _gen_attempt() -> np.ndarray | None:
    compressed = _build_compressed()
    if compressed is None:
        return None

    grid = _inflate(compressed)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    keep_rows, keep_cols = _compressed_axes(grid)
    recolored = out[np.ix_(keep_rows, keep_cols)]
    if np.any(recolored == YELLOW):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
