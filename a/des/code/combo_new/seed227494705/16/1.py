from collections import deque
import random

import numpy as np

BLUE = 1


def _most_common(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _same_color_components(grid, colors=None):
    colors = None if colors is None else set(colors)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or (colors is not None and color not in colors):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return comps


def _rect_perimeter_cells(r0, c0, r1, c1):
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_frame(grid, r0, c0, h, w, color):
    r1, c1 = r0 + h - 1, c0 + w - 1
    for r, c in _rect_perimeter_cells(r0, c0, r1, c1):
        grid[r, c] = color


def _read_legend(grid, bg, border):
    mapping = {}
    h, w = grid.shape
    for r in range(h):
        key = int(grid[r, 0])
        if key in (bg, border) or key in mapping:
            continue
        for size in (3, 2):
            if r + size <= h and 1 + size <= w:
                pat = grid[r : r + size, 1 : 1 + size]
                if np.all(pat != bg) and not np.any(pat == border):
                    mapping[key] = pat.copy()
                    break
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each closed rectangle interior by tiling the pattern selected by its marker color.

    Input structure: A dominant background contains a corner legend and several blue
    rectangular frames.  Each legend entry has a singleton key color beside a small
    non-background pattern, and each frame interior has one singleton marker color.
    Rule: Read the legend as marker color to pattern.  For every blue rectangular
    frame, find the marker color inside it and tile the matched pattern over the
    whole frame interior from that interior's upper-left corner, preserving the
    frame and the legend.
    Key insight: The interior marker is an index into the legend, not the fill
    color; the visible value copied into the rectangle is the whole small texture.
    """
    out = grid.copy()
    bg = _most_common(grid)
    non_bg = [int(v) for v in np.unique(grid) if int(v) != bg]
    border = max(non_bg, key=lambda c: int(np.sum(grid == c))) if non_bg else BLUE
    legend = _read_legend(grid, bg, border)

    for comp in _same_color_components(grid, colors=[border]):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perim = _rect_perimeter_cells(r0, c0, r1, c1)
        if sum(1 for r, c in perim if int(grid[r, c]) == border) < len(perim) - 1:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        markers = [int(v) for v in np.unique(interior) if int(v) not in (bg, border)]
        marker = next((m for m in markers if m in legend), None)
        if marker is None:
            continue
        pat = legend[marker]
        ph, pw = pat.shape
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                out[rr, cc] = pat[(rr - r0 - 1) % ph, (cc - c0 - 1) % pw]
    return out


def generate_input() -> np.ndarray:
    size = random.choice([2, 3])
    n = random.randint(2, 4)
    h = random.randint(max(18, n * (size + 1) + 12), 30)
    w = random.randint(22, 30)
    grid = np.zeros((h, w), dtype=int)
    keys = random.sample([2, 3, 4, 6, 8], n)
    pattern_colors = [5, 7, 9, 3, 4]

    for i, key in enumerate(keys):
        r = i * (size + 1)
        grid[r, 0] = key
        a = pattern_colors[i % len(pattern_colors)]
        b = pattern_colors[(i + 2) % len(pattern_colors)]
        pat = np.fromfunction(lambda rr, cc: np.where((rr + cc + i) % 2 == 0, a, b), (size, size), dtype=int)
        if random.random() < 0.35:
            pat[:, :] = a
            pat[:, ::2] = b
        grid[r : r + size, 1 : 1 + size] = pat

    used = np.zeros_like(grid, dtype=bool)
    used[: n * (size + 1) + 1, : size + 3] = True
    for key in keys:
        for _ in range(300):
            rh = random.randint(5, 9)
            rw = random.randint(5, 10)
            r0 = random.randint(size * n + 2, h - rh - 1)
            c0 = random.randint(4, w - rw - 1)
            if not used[max(0, r0 - 1) : r0 + rh + 1, max(0, c0 - 1) : c0 + rw + 1].any():
                _draw_frame(grid, r0, c0, rh, rw, BLUE)
                mr = random.randint(r0 + 1, r0 + rh - 2)
                mc = random.randint(c0 + 1, c0 + rw - 2)
                grid[mr, mc] = key
                used[max(0, r0 - 1) : r0 + rh + 1, max(0, c0 - 1) : c0 + rw + 1] = True
                break
    return grid
