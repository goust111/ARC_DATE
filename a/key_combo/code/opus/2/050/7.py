"""Use the unique monocolor unit rectangle to recolor other unit interiors.

Input structure:
  A 3x3 grid partition created by magenta(6) lines. Each cell contains a large
  mostly-solid rectangle, with noise inside eight cells and one cell remaining fully
  monocolor (the anomaly).

Rule:
  Identify the 3x3 partition, find the unique rectangle whose rectangle pixels are
  all the same non-line color C, then recolor strict interiors of rectangles in all
  other cells with C while preserving their borders. Clear the anomalous cell's
  rectangle region to black and keep partition lines unchanged.

Key insight:
  The anomaly supplies a single color parameter that is consumed to update the other
  cells instead of being emitted directly.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


def _detect_divider_lines(grid: np.ndarray, line_color: int = 6) -> Tuple[List[int], List[int]]:
    H, W = grid.shape
    rows = [r for r in range(H) if (grid[r, :] == line_color).sum() > W * 0.75]
    cols = [c for c in range(W) if (grid[:, c] == line_color).sum() > H * 0.75]
    if len(rows) >= 2:
        rows = rows[:2]
    if len(cols) >= 2:
        cols = cols[:2]
    return rows, cols


def _cell_rectangles_in_cell(
    cell: np.ndarray,
) -> Tuple[Tuple[int, int, int, int, int] | None, np.ndarray]:
    mask = cell != 6
    if not np.any(mask):
        return None, cell
    rs, cs = np.where(mask)
    r0, c0 = int(rs.min()), int(cs.min())
    r1, c1 = int(rs.max()), int(cs.max())
    sub = cell[r0 : r1 + 1, c0 : c1 + 1]
    colors = set(int(x) for x in sub.flatten() if int(x) != 0 and int(x) != 6)
    if not colors:
        return None, cell
    return (r0, c0, r1, c1, len(colors)), sub


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g.copy()
    H, W = g.shape
    rows, cols = _detect_divider_lines(g, line_color=6)
    if len(rows) < 2 or len(cols) < 2:
        return g.copy()
    r0, r1 = rows
    c0, c1 = cols

    rows_cells = [(0, r0), (r0 + 1, r1), (r1 + 1, H)]
    cols_cells = [(0, c0), (c0 + 1, c1), (c1 + 1, W)]

    cell_info = []
    for ri, (ra, rb) in enumerate(rows_cells):
        for ci, (ca, cb) in enumerate(cols_cells):
            cell = g[ra:rb, ca:cb]
            info, sub = _cell_rectangles_in_cell(cell)
            cell_info.append((ri, ci, ra, rb, ca, cb, info, sub))

    anomaly_idx = None
    anomaly_color = None
    for i, (_, _, _, _, _, _, info, sub) in enumerate(cell_info):
        if info is None:
            continue
        _, _, _, _, n_colors = info
        if n_colors == 1:
            if anomaly_idx is None:
                anomaly_idx = i
                anomaly_color = int(
                    min({int(x) for x in sub.flatten() if int(x) not in (0, 6)})
                )
    if anomaly_idx is None or anomaly_color is None:
        return g.copy()

    for i, (ri, ci, ra, rb, ca, cb, info, sub) in enumerate(cell_info):
        if info is None:
            continue
        r0, c0, r1, c1, _ = info
        r00, c00 = ra + r0, ca + c0
        r11, c11 = ra + r1, ca + c1
        if i == anomaly_idx:
            # clear anomalous unit rectangle
            g[r00 : r11 + 1, c00 : c11 + 1] = 0
            continue
        if anomaly_color is None or anomaly_color == 6:
            continue
        # recolor interior only
        if r11 - r00 >= 2 and c11 - c00 >= 2:
            g[r00 + 1 : r11, c00 + 1 : c11] = anomaly_color
    return g


def generate_input() -> np.ndarray:
    cell = 5
    H = 3 * cell + 2
    W = 3 * cell + 2
    g = np.zeros((H, W), dtype=np.int64)
    line = 6
    r_lines = [cell, 2 * cell + 1]
    c_lines = [cell, 2 * cell + 1]
    for r in r_lines:
        g[r, :] = line
    for c in c_lines:
        g[:, c] = line

    rect_colors = [1, 2, 3, 4, 7, 8, 9]
    anomaly = random.randint(0, 8)
    cell_idx = 0
    for ri, (ra, rb) in enumerate([(0, cell), (cell + 1, 2 * cell + 1), (2 * cell + 2, H)]):
        for ci, (ca, cb) in enumerate([(0, cell), (cell + 1, 2 * cell + 1), (2 * cell + 2, W)]):
            h = random.randint(4, 5)
            w = random.randint(4, 5)
            top_min, top_max = ra, rb - h
            left_min, left_max = ca, cb - w
            if top_max < top_min:
                top = top_min
            else:
                top = random.randint(top_min, top_max)
            if left_max < left_min:
                left = left_min
            else:
                left = random.randint(left_min, left_max)
            color = random.choice([x for x in rect_colors if x != (anomaly + 1) % 10])
            g[top : top + h, left : left + w] = color

            # noise for non-anomalous cells
            if cell_idx != anomaly:
                noise_n = random.randint(1, 3)
                for _ in range(noise_n):
                    rr = random.randint(top + 1, top + h - 2)
                    cc = random.randint(left + 1, left + w - 2)
                    g[rr, cc] = random.choice([x for x in rect_colors if x != color])
            cell_idx += 1
    return g
