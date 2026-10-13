"""Unfold a bent path into ordered horizontal bars inside the right-side frame.
Input structure:
- A black background with one rectangular one-cell-thick grey frame on the right side.
- A single-pixel-width path of one base color connects two grid edges.
- Each 90-degree turning pixel on the path is replaced by a non-background marker color.
Rule:
- Trace the non-background path from one endpoint to the other and split it at marker pixels.
- Compute each segment length as the number of path pixels between consecutive split points, including boundary markers.
- Segment color is the boundary marker color on the segment start side; only the first and last segments have one marker boundary and inherit that marker color.
- Draw the segments from top to bottom as left-aligned horizontal lines in the frame interior.
Key insight:
- Marker pixels are semantic segment boundaries, and each segment inherits color from the start-side marker, so path geometry is compressed into ordered 1D bars.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighs(grid: np.ndarray, r: int, c: int) -> List[Tuple[int, int]]:
    h, w = grid.shape
    out: List[Tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _find_background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_perimeter_rect(mask: np.ndarray, c: int, r1: int, c1: int, r2: int, c2: int) -> bool:
    if r1 == r2 or c1 == c2:
        return False
    h = r2 - r1 + 1
    w = c2 - c1 + 1
    expected = 2 * (h + w) - 4
    if np.sum(mask) != expected:
        return False

    # Top and bottom rows.
    for cc in range(c1, c2 + 1):
        if not mask[r1, cc] or not mask[r2, cc]:
            return False
    # Left and right columns.
    for rr in range(r1, r2 + 1):
        if not mask[rr, c1] or not mask[rr, c2]:
            return False

    # Ensure no interior pixels.
    interior = mask[r1 + 1 : r2, c1 + 1 : c2]
    if interior.size and np.any(interior):
        return False

    return True


def _find_frame(grid: np.ndarray, bg: int) -> Optional[Tuple[int, int, int, int, int]]:
    # Return (r1, r2, c1, c2, color).
    h, w = grid.shape
    best: Optional[Tuple[int, int, int, int, int]] = None

    for color in np.unique(grid):
        c = int(color)
        if c == bg:
            continue
        mask = grid == c
        if not np.any(mask):
            continue

        rr, cc = np.where(mask)
        r1, r2 = int(rr.min()), int(rr.max())
        c1, c2 = int(cc.min()), int(cc.max())
        if not _is_perimeter_rect(mask, c, r1, c1, r2, c2):
            continue

        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        interior_h = r2 - r1 - 1
        interior_w = c2 - c1 - 1
        if interior_h <= 0 or interior_w <= 0:
            continue

        if np.any(grid[r1 + 1 : r2, c1 + 1 : c2] != bg):
            continue

        score = (c2, area)
        if best is None or score > (best[1], best[0]):
            best = (r1, r2, c1, c2, c)

    return best


def _trace_path(grid: np.ndarray, path_mask: np.ndarray) -> List[Tuple[int, int]]:
    coords = np.argwhere(path_mask)
    if coords.size == 0:
        return []

    coord_set = {(int(r), int(c)) for r, c in coords}

    endpoints: List[Tuple[int, int]] = []
    for r, c in coords:
        r_i, c_i = int(r), int(c)
        deg = 0
        for nr, nc in _neighs(path_mask, r_i, c_i):
            if path_mask[nr, nc]:
                deg += 1
        if deg <= 1:
            endpoints.append((r_i, c_i))

    if endpoints:
        start = sorted(endpoints)[0]
    else:
        start = sorted(coord_set)[0]

    path: List[Tuple[int, int]] = [start]
    prev: Optional[Tuple[int, int]] = None
    cur = start
    visited = {start}

    while True:
        nxt: Optional[Tuple[int, int]] = None
        for nr, nc in _neighs(path_mask, cur[0], cur[1]):
            if not path_mask[nr, nc]:
                continue
            if prev is not None and (nr, nc) == prev:
                continue
            if (nr, nc) in visited:
                continue
            nxt = (nr, nc)
            break
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    return path


def _collect_segments(grid: np.ndarray, bg: int, path: List[Tuple[int, int]], path_color: int) -> List[Tuple[int, int]]:
    if not path:
        return []

    marker_indices: List[int] = []
    marker_colors: Dict[int, int] = {}
    for i, (r, c) in enumerate(path):
        v = int(grid[r, c])
        if v != bg and v != path_color:
            marker_indices.append(i)
            if i not in marker_colors:
                marker_colors[i] = v

    boundaries = [0]
    boundaries.extend(marker_indices)
    if boundaries[-1] != len(path) - 1:
        boundaries.append(len(path) - 1)
    if len(boundaries) < 2:
        boundaries = [0, len(path) - 1]

    segments: List[Tuple[int, int]] = []
    for idx in range(len(boundaries) - 1):
        start_i = boundaries[idx]
        end_i = boundaries[idx + 1]
        seg_len = end_i - start_i + 1

        if len(boundaries) == 2:
            seg_color = path_color
        elif idx == 0:
            seg_color = int(grid[path[boundaries[1]][0], path[boundaries[1]][1]])
        elif idx == len(boundaries) - 2:
            seg_color = int(grid[path[boundaries[-2]][0], path[boundaries[-2]][1]])
        else:
            seg_color = int(grid[path[boundaries[idx]][0], path[boundaries[idx]][1]])

        segments.append((seg_len, seg_color))

    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Unfold a bent path into ordered horizontal bars inside the right-side frame.
    Input structure:
    - A black background with one rectangular one-cell-thick grey frame on the right side.
    - A single-pixel-width path of one base color connects two grid edges.
    - Each 90-degree turning pixel on the path is replaced by a non-background marker color.
    Rule:
    - Trace the non-background path from one endpoint to the other and split it at marker pixels.
    - Compute each segment length as the number of path pixels between consecutive split points, including boundary markers.
    - Segment color is the boundary marker color on the segment start side; only the first and last segments have one marker boundary and inherit that marker color.
    - Draw the segments from top to bottom as left-aligned horizontal lines in the frame interior.
    Key insight:
    - Marker pixels are semantic segment boundaries, and each segment inherits color from the start-side marker, so path geometry is compressed into ordered 1D bars.
    """
    bg = _find_background_color(grid)

    frame = _find_frame(grid, bg)
    if frame is None:
        return grid.copy()

    r1, r2, c1, c2, frame_color = frame
    frame_h = r2 - r1 + 1
    frame_w = c2 - c1 + 1
    interior_h = frame_h - 2
    interior_w = frame_w - 2

    frame_mask = grid != bg
    if frame_color is not None:
        frame_mask &= grid != frame_color

    path_mask = frame_mask
    if not np.any(path_mask):
        return grid.copy()

    path_colors, counts = np.unique(grid[path_mask], return_counts=True)
    path_color = int(path_colors[np.argmax(counts)])
    path = _trace_path(grid, path_mask)
    if len(path) < 2:
        return grid.copy()

    segments = _collect_segments(grid, bg, path, path_color)
    if not segments:
        return grid.copy()

    out = np.full((frame_h, frame_w), bg, dtype=grid.dtype)
    out[r1 - 0, c1 : c2 + 1] = frame_color
    out[r2 - 0, c1 : c2 + 1] = frame_color
    out[r1 : r2 + 1, c1] = frame_color
    out[r1 : r2 + 1, c2] = frame_color

    for i, (seg_len, seg_color) in enumerate(segments):
        if i >= interior_h:
            break
        L = min(seg_len, interior_w)
        if L <= 0:
            continue
        out[r1 + 1 + i, c1 + 1 : c1 + 1 + L] = seg_color

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(5, 31))
    w = int(rng.integers(5, 31))
    grid = np.zeros((h, w), dtype=np.int32)
    for color in range(1, int(rng.integers(2, 6))):
        for _ in range(int(rng.integers(1, 4))):
            grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = color
    return grid
