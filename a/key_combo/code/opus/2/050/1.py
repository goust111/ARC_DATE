"""Project segment markers to a priority-ordered vertical band stack.

Input structure:
  A black background grid with a solid top gray bar and 3-5 horizontal
  color-1/2/3 segments below it. Each segment is one pixel thick, has
  1-3 yellow (4) marker pixels, and may include marker-replaced columns.

Rule:
  For each segment, paint the columns from the row above the segment up to
  the row below the top gray bar with the segment's base color, then paint each
  marker column in that band to yellow (4). Segments are drawn in ascending
  marker-count order (stable tie-break by row), and later draws overwrite earlier
  bands. Keep the original top gray row and segments unchanged.

Key insight:
  Marker count controls global painting order while marker columns control the
  inner stripe pattern within each painted band.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


def _top_gray_row(grid: np.ndarray) -> int:
    H, W = grid.shape
    gray_counts = [(grid[r] == 5).sum() for r in range(H)]
    if not gray_counts:
        return 0
    return int(np.argmax(gray_counts))


def _extract_segments(grid: np.ndarray, gray_row: int, bg: int = 0) -> List[dict]:
    H, W = grid.shape
    segments: List[dict] = []
    for r in range(gray_row + 1, H):
        c = 0
        while c < W:
            v = int(grid[r, c])
            if v in (bg, 5):
                c += 1
                continue
            s = c
            while c < W and int(grid[r, c]) not in (bg, 5):
                c += 1
            e = c
            run = grid[r, s:e]
            if run.size < 2:
                continue
            non_marker = run[(run != 0) & (run != 4) & (run != 5)]
            if non_marker.size == 0:
                continue
            base_colors = set(int(x) for x in non_marker)
            if len(base_colors) != 1:
                continue
            base = int(non_marker[0])
            marker_positions = [s + int(i) for i, x in enumerate(run) if int(x) == 4]
            if len(marker_positions) == 0:
                continue
            segments.append(
                {
                    "row": r,
                    "c0": s,
                    "c1": e,
                    "base": base,
                    "markers": marker_positions,
                    "count": len(marker_positions),
                }
            )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g.copy()
    H, W = g.shape
    bg = 0
    gray_row = _top_gray_row(g)
    out = g.copy()
    segments = _extract_segments(g, gray_row, bg=bg)
    segments.sort(key=lambda s: (s["count"], s["row"], s["c0"]))
    for seg in segments:
        r1 = seg["row"]
        if r1 <= gray_row + 1:
            continue
        c0, c1 = seg["c0"], seg["c1"]
        base = seg["base"]
        out[gray_row + 1 : r1, c0:c1] = base
        for c in seg["markers"]:
            if c0 <= c < c1 and c < W:
                out[gray_row + 1 : r1, c] = 4
    return out


def _rand_segment(grid: np.ndarray, row: int, colors: List[int]) -> None:
    H, W = grid.shape
    for _ in range(200):
        length = random.randint(4, min(8, W))
        start = random.randint(0, max(0, W - length))
        c0 = start
        c1 = start + length
        if c0 < 1 or c1 > W - 1:
            continue
        base = random.choice(colors)
        marker_max = min(3, length)
        k = random.randint(1, marker_max)
        pos = random.sample(range(c0, c1), k)
        if row <= 0 or row >= H:
            continue
        row_values = [int(grid[row, c]) for c in range(c0, c1)]
        if any(v != 0 for v in row_values):
            continue
        for c in range(c0, c1):
            grid[row, c] = base
        for c in pos:
            grid[row, c] = 4
        return


def generate_input() -> np.ndarray:
    H = random.randint(12, 18)
    W = random.randint(14, 20)
    if H <= 2:
        H = 12
    if W <= 4:
        W = 14
    grid = np.zeros((H, W), dtype=np.int64)
    grid[0, :] = 5
    n = random.randint(3, 5)
    rows = random.sample(range(2, H - 1), n)
    colors = [1, 2, 3]
    for r in rows:
        _rand_segment(grid, r, colors)
    return grid

