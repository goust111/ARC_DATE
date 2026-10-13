from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_COLORS = {BLUE, GREEN}
BOUNDARY_COLORS = {BLUE, GREEN, RED}
REGION_COLORS = (YELLOW, GREY, PINK, ORANGE, TEAL, MAROON)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _simulate_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    starts = np.argwhere(g == RED)
    assert len(starts) == 1, "expected exactly one red start pixel"
    start_r, start_c = map(int, starts[0])
    assert start_c == 0, "start pixel must be on the left edge"

    out = g.copy()
    pending: list[tuple[int, int, str]] = [(start_r, start_c, "h")]
    seen_segments: set[tuple[int, int, str]] = set()
    opened_markers: set[tuple[int, int]] = set()
    h, w = g.shape

    while pending:
        r, c, axis = pending.pop()
        key = (r, c, axis)
        if key in seen_segments:
            continue
        seen_segments.add(key)
        out[r, c] = RED

        if axis == "h":
            directions = ((0, -1), (0, 1))
            next_axis = "v"
        else:
            directions = ((-1, 0), (1, 0))
            next_axis = "h"

        new_segments: list[tuple[int, int, str]] = []
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            prev_r, prev_c = r, c
            while 0 <= rr < h and 0 <= cc < w:
                cell = int(g[rr, cc])
                if cell in MARKER_COLORS:
                    marker = (rr, cc)
                    if marker not in opened_markers:
                        opened_markers.add(marker)
                        new_segments.append((prev_r, prev_c, next_axis))
                    break
                assert cell in {BLACK, RED}, "non-marker payload must stay off every ray path"
                out[rr, cc] = RED
                prev_r, prev_c = rr, cc
                rr += dr
                cc += dc

        # With a LIFO stack, scanning left/up first and then extending preserves
        # the required right/down priority when both directions open together.
        pending.extend(new_segments)

    return out


def _full_boundary_rows(boundary_mask: np.ndarray) -> list[int]:
    return [r for r in range(boundary_mask.shape[0]) if bool(np.all(boundary_mask[r, :]))]


def _full_boundary_cols(boundary_mask: np.ndarray) -> list[int]:
    return [c for c in range(boundary_mask.shape[1]) if bool(np.all(boundary_mask[:, c]))]


def _intervals(full_lines: list[int], size: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for line in sorted(full_lines):
        if start <= line - 1:
            intervals.append((start, line - 1))
        start = line + 1
    if start <= size - 1:
        intervals.append((start, size - 1))
    return intervals


def _extract_layout(out: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    boundary_mask = np.isin(out, list(BOUNDARY_COLORS))
    row_intervals = _intervals(_full_boundary_rows(boundary_mask), out.shape[0])
    col_intervals = _intervals(_full_boundary_cols(boundary_mask), out.shape[1])
    assert row_intervals and col_intervals, "ray network must create at least one region"

    compressed = np.zeros((len(row_intervals), len(col_intervals)), dtype=int)
    for i, (r0, r1) in enumerate(row_intervals):
        for j, (c0, c1) in enumerate(col_intervals):
            region = out[r0 : r1 + 1, c0 : c1 + 1]
            colors = sorted(
                int(v)
                for v in np.unique(region)
                if int(v) not in BOUNDARY_COLORS and int(v) != BLACK
            )
            assert len(colors) == 1, "each compressed region must contain exactly one payload color"
            compressed[i, j] = colors[0]
    return row_intervals, col_intervals, compressed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the recursive red ray network and compress each boundary-defined rectangle to one color cell.

    Input structure:
    - A black grid contains one red start pixel on the left edge, green/blue marker pixels, and payload pixels of other colors.
    - The payload pixels are scattered inside rectangular regions and never lie on any ray path.
    - After the marker-triggered ray expansion, the union of red rays and marker pixels forms complete partition rows and columns.

    Rule:
    - Fire a horizontal red ray rightward from the red start pixel.
    - Whenever a ray first meets a marker, stop before the marker and emit the perpendicular ray from that previous cell; later hits on the same marker only stop.
    - Process openings depth-first with right/down priority, preserving all marker pixels.
    - Treat the final red-ray network together with marker pixels as the partition boundary, then compress every rectangular region between full boundary rows and columns to one output pixel carrying that region's payload color.

    Key insight:
    - The markers are not decorative blockers: they recursively build the boundary network, and that dynamically created boundary geometry is exactly what defines the compression grid.
    """
    out = _simulate_rays(np.asarray(grid, dtype=int))
    _row_intervals, _col_intervals, compressed = _extract_layout(out)
    return compressed


def _region_cells(r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _scatter_region(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> int:
    r0, r1, c0, c1 = rect
    cells = _region_cells(r0, r1, c0, c1)
    count = random.randint(2, min(4, len(cells)))
    for r, c in random.sample(cells, count):
        grid[r, c] = color
    return count


def _build_candidate() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(11, 18)
    w = random.randint(11, 18)
    split_c = random.randint(4, w - 5)
    anchor_top = bool(random.getrandbits(1))

    if anchor_top:
        start_r = random.randint(1, h - 5)
        edge_marker_r = h - 1
        row_intervals = [(0, start_r - 1), (start_r + 1, h - 3), (h - 1, h - 1)]
    else:
        start_r = random.randint(4, h - 2)
        edge_marker_r = 0
        row_intervals = [(0, 0), (2, start_r - 1), (start_r + 1, h - 1)]

    col_intervals = [(0, split_c - 1), (split_c + 1, w - 1)]

    grid = np.zeros((h, w), dtype=int)
    grid[start_r, 0] = RED

    for c in range(split_c + 1, w):
        grid[start_r, c] = random.choice((BLUE, GREEN))
    grid[edge_marker_r, split_c] = random.choice((BLUE, GREEN))

    colors = list(REGION_COLORS)
    random.shuffle(colors)
    expected = np.zeros((3, 2), dtype=int)
    color_idx = 0
    for i, (r0, r1) in enumerate(row_intervals):
        for j, (c0, c1) in enumerate(col_intervals):
            color = colors[color_idx]
            color_idx += 1
            expected[i, j] = color
            _scatter_region(grid, (r0, r1, c0, c1), color)

    return grid, expected


def _validate_candidate(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        out = _simulate_rays(grid)
        row_intervals, col_intervals, compressed = _extract_layout(out)
    except AssertionError:
        return False

    if compressed.shape != (3, 2):
        return False
    if not np.array_equal(compressed, expected):
        return False

    boundary_mask = np.isin(out, list(BOUNDARY_COLORS))
    if len(_full_boundary_rows(boundary_mask)) != 2:
        return False
    if len(_full_boundary_cols(boundary_mask)) != 1:
        return False
    if len(row_intervals) != 3 or len(col_intervals) != 2:
        return False

    seen_colors: set[int] = set()
    for i, (r0, r1) in enumerate(row_intervals):
        for j, (c0, c1) in enumerate(col_intervals):
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            payload = [int(v) for v in np.unique(region) if int(v) not in BOUNDARY_COLORS and int(v) != BLACK]
            if len(payload) != 1:
                return False
            color = payload[0]
            if color in seen_colors:
                return False
            seen_colors.add(color)
            count = int(np.count_nonzero(region == color))
            if not (2 <= count <= 4):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, expected = _build_candidate()
        if _validate_candidate(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid recursive ray compression puzzle")
