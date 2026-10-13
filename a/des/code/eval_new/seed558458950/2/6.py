"""Rearrange left rays through a gray lens into right-side rays using the lens path mappings.
Input structure:
- Background is black.
- A vertical gray rectangular strip (width 3-5) sits near the middle of the grid and forms the lens body.
- On the left of the lens, several horizontal single-pixel-high colored rays (different colors, one row each) connect the left edge to the lens left boundary.
- Inside the lens, each color draws a 1-cell-wide staircase path from some row on the lens left edge to some row on the lens right edge, with each step moving right and up/down/straight by at most one row.
- The lens area includes those colored paths overlaid on gray background.

Rule:
- Identify the gray lens region and for each non-background, non-gray color trace its endpoint rows: the row on the lens left edge is the input row, and the row on the lens right edge is the output row.
- Keep the lens (including its internal paths) unchanged.
- For each input row segment, clear that input ray and draw a same-color segment on the right side of the lens with identical length, ending row determined by the lens path endpoint row.
- The new segment starts at the lens right boundary + 1 and extends rightward.

Key insight:
- The visible staircase routes inside the lens directly encode a row permutation; the transformation is a pure rearrangement of input rays according to those encoded endpoint mappings.
"""

import random
from typing import List, Tuple

import numpy as np


def _connected_components(mask: np.ndarray) -> List[Tuple[List[Tuple[int, int]], int, int, int, int]]:
    """Return components as list of (cells,row_min,row_max,col_min,col_max)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[Tuple[List[Tuple[int, int]], int, int, int, int]] = []
    for sr, sc in zip(*np.where(mask)):
        if visited[sr, sc]:
            continue
        stack = [(sr, sc)]
        visited[sr, sc] = True
        cells: List[Tuple[int, int]] = []
        rmin = rmax = sr
        cmin = cmax = sc
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            rmin = min(rmin, r)
            rmax = max(rmax, r)
            cmin = min(cmin, c)
            cmax = max(cmax, c)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append((cells, rmin, rmax, cmin, cmax))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    # ARC tasks use ndarray integers.
    g = np.array(grid, copy=True)
    if g.ndim != 2:
        return g

    h, w = g.shape
    if h == 0 or w == 0:
        return g

    bg = int(np.bincount(g.ravel()).argmax())

    # 1) Detect gray lens rectangle component.
    lens_color = None
    lens_bbox = None
    best_score = -1.0
    colors = [int(c) for c in np.unique(g)]
    colors = [c for c in colors if c != bg]

    for color in colors:
        mask = g == color
        for cells, rmin, rmax, cmin, cmax in _connected_components(mask):
            comp_h = rmax - rmin + 1
            comp_w = cmax - cmin + 1
            area = len(cells)
            if not (3 <= comp_w <= 5):
                continue
            if comp_h < h - 4:
                continue
            rect_area = comp_h * comp_w
            fill_ratio = area / float(rect_area)
            if fill_ratio < 0.58:
                continue
            score = fill_ratio * area
            if score > best_score:
                lens_color = color
                lens_bbox = (rmin, rmax, cmin, cmax)
                best_score = score

    if lens_color is None or lens_bbox is None:
        return g

    lr0, lr1, lc0, lc1 = lens_bbox
    if not (0 <= lc0 < lc1 < w):
        return g

    # 2) Parse per-color mappings from left row -> right row via lens endpoints.
    mappings = []  # (color, input_row, output_row, segment_len)
    for color in colors:
        if color == lens_color:
            continue
        cm = (g == color)

        left_mask = cm[:, :lc0]
        if not np.any(left_mask):
            continue
        left_rows = np.where(np.any(left_mask, axis=1))[0]
        if left_rows.size != 1:
            continue
        input_row = int(left_rows[0])

        start_rows = np.where(cm[:, lc0])[0]
        end_rows = np.where(cm[:, lc1])[0]
        if start_rows.size != 1 or end_rows.size != 1:
            continue

        left_row_from_lens = int(start_rows[0])
        output_row = int(end_rows[0])
        if left_row_from_lens != input_row:
            continue

        left_row_cols = np.where(cm[input_row, :lc0])[0]
        if left_row_cols.size == 0:
            continue
        seg_min = int(left_row_cols[0])
        seg_max = int(left_row_cols[-1])
        if seg_min != 0:
            continue
        segment_len = seg_max - seg_min + 1
        if segment_len <= 0:
            continue
        if segment_len + lc1 + 1 > w:
            continue

        mappings.append((color, input_row, output_row, segment_len))

    if len(mappings) == 0:
        return g

    out = np.array(g, copy=True)

    # 3) Rearrange rays to the right side by the decoded mapping.
    for color, input_row, output_row, length in mappings:
        left_mask = out[input_row, :lc0] == color
        out[input_row, :lc0][left_mask] = bg

        end_col = lc1 + 1 + length
        if end_col <= w:
            out[output_row, lc1 + 1 : end_col] = color

    return out


def _draw_path(grid: np.ndarray, start_row: int, end_row: int, col_l: int, col_r: int, row_lo: int, row_hi: int, color: int, used: np.ndarray) -> bool:
    """Draw a one-cell-wide staircase from col_l to col_r with random +/-1/0 steps."""
    h, w = grid.shape
    cur_r = start_row
    cur_c = col_l
    path_ok = True

    if not (row_lo <= cur_r <= row_hi):
        return False

    if used[cur_r, cur_c]:
        return False
    grid[cur_r, cur_c] = color
    used[cur_r, cur_c] = True

    rng = random.Random()
    remaining_cols = col_r - col_l
    for _ in range(remaining_cols):
        options = [-1, 0, 1]
        rng.shuffle(options)
        moved = False
        for dr in options:
            nr = cur_r + dr
            nc = cur_c + 1
            if not (row_lo <= nr <= row_hi):
                continue
            if used[nr, nc]:
                continue
            if not (col_l <= nc <= col_r):
                continue
            cur_r, cur_c = nr, nc
            grid[cur_r, cur_c] = color
            used[cur_r, cur_c] = True
            moved = True
            break
        if not moved:
            path_ok = False
            break

    return path_ok and cur_r == end_row and cur_c == col_r


def generate_input() -> np.ndarray:
    """Generate random ARC-consistent inputs for the lens-rearrange puzzle."""
    while True:
        h = random.randint(14, 24)

        top_margin = random.randint(0, 2)
        bottom_margin = random.randint(0, 2)
        lens_top = top_margin
        lens_bottom = h - 1 - bottom_margin
        if lens_top >= lens_bottom:
            continue
        lens_h = lens_bottom - lens_top + 1

        lens_w = random.randint(3, 5)
        # keep enough room for left and right segments, output length equals lens_left.
        lens_left = random.randint(4, 12)
        min_w = lens_left + lens_w + lens_left + 1
        if min_w > 30:
            lens_left = min(lens_left, 11)
            min_w = lens_left + lens_w + lens_left + 1
            if min_w > 30:
                continue

        w = random.randint(min_w, 30)

        # background black, lens gray, rays colorful
        bg = 0
        lens_color = 5
        palette = [1, 2, 3, 4, 6, 7, 8, 9]
        if lens_color in palette:
            palette.remove(lens_color)

        lens_right = lens_left + lens_w - 1
        if lens_right >= w - 1:
            continue

        # ensure max possible output length fits
        max_out_len = w - (lens_right + 1)
        if max_out_len < lens_left:
            continue

        grid = np.full((h, w), bg, dtype=np.int32)
        grid[lens_top : lens_bottom + 1, lens_left : lens_right + 1] = lens_color

        available_rows = [r for r in range(lens_top, lens_bottom + 1)]
        n = random.randint(2, min(6, len(available_rows)))
        start_rows = sorted(random.sample(available_rows, n))

        # produce a nontrivial mapping permutation for output rows
        out_rows = start_rows.copy()
        if random.random() < 0.5:
            random.shuffle(out_rows)
        else:
            out_rows = start_rows[::-1]
        if out_rows == start_rows:
            if len(out_rows) > 1:
                out_rows[-1], out_rows[0] = out_rows[0], out_rows[-1]

        random.shuffle(palette)
        colors = palette[:n]

        # draw left rays first
        for row, color in zip(start_rows, colors):
            grid[row, :lens_left] = color

        # draw lens internal paths
        used = np.zeros_like(grid, dtype=bool)
        for row, dest_row, color in zip(start_rows, out_rows, colors):
            # keep each step moving within lens band and avoid overlaps.
            ok = _draw_path(grid, row, dest_row, lens_left, lens_right, lens_top, lens_bottom, color, used)
            if not ok:
                # restart to keep examples valid and deterministicly clean
                break
        else:
            # guarantee that all mapping rows are represented by path endpoints on lens edges
            good = True
            for row, dest_row, color in zip(start_rows, out_rows, colors):
                if grid[row, lens_left] != color or grid[dest_row, lens_right] != color:
                    good = False
                    break
            if good:
                return grid

        # retry generation on any failure
        continue
