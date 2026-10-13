from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRAY = 5
RED = 2
ORANGE = 7
STRIPE_COLORS = (1, 3, 4, 6, 8, 9)
CELL = 3
PITCH = CELL + 1
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored strip outward in its arrow direction by the number of orange markers it contains.

    Input structure:
    - The background is black, and gray one-pixel grid lines divide the canvas into equal rectangular cells.
    - Some cells are filled with one non-black, non-gray strip color; same-colored cells form a one-cell-thick horizontal or vertical strip.
    - One strip-end has a red arrow drawn on the nearby grid lines, indicating the outward push direction.
    - Some strip cells contain one small orange center marker pixel.

    Rule:
    - Count how many cells of each strip contain an orange marker; call that number N.
    - Keep the original strip in place, but remove its orange markers by turning them back into the strip color.
    - Extend the strip color from the arrow-side end into the next N black cells along that direction.
    - Keep the gray grid and the red arrow pixels unchanged.

    Key insight:
    - The arrow and the orange markers encode two different pieces of information:
      the arrow chooses the direction, while the marker count chooses the push depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_lines, col_lines = _grid_lines(g)
    cell_colors, cell_markers = _cell_contents(g, row_lines, col_lines)
    strips = _find_strips(cell_colors)

    out = g.copy()
    for strip in strips:
        color = int(strip["color"])
        cells = strip["cells"]
        depth = sum(
            int(cell_markers[int(r), int(c)])
            for r, c in cells
        )
        assert depth >= 1, "every strip must contain at least one marker"

        for r, c in cells:
            _paint_cell(out, int(r), int(c), color, row_lines, col_lines)

        dr, dc = _arrow_direction(g, strip, row_lines, col_lines)
        if dr != 0:
            edge_r = min(int(r) for r, _c in cells) if dr < 0 else max(int(r) for r, _c in cells)
            c = int(cells[0][1])
            for step in range(1, depth + 1):
                rr = int(edge_r + dr * step)
                assert 0 <= rr < cell_colors.shape[0], "vertical extension left the cell grid"
                assert int(cell_colors[rr, c]) == BG, "vertical extension must target black cells"
                _paint_cell(out, rr, c, color, row_lines, col_lines)
        else:
            r = int(cells[0][0])
            edge_c = min(int(c) for _r, c in cells) if dc < 0 else max(int(c) for _r, c in cells)
            for step in range(1, depth + 1):
                cc = int(edge_c + dc * step)
                assert 0 <= cc < cell_colors.shape[1], "horizontal extension left the cell grid"
                assert int(cell_colors[r, cc]) == BG, "horizontal extension must target black cells"
                _paint_cell(out, r, cc, color, row_lines, col_lines)

    return out


def _grid_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = g.shape
    row_lines = [int(r) for r in range(h) if int(np.sum(g[r, :] == GRAY)) >= max(3, w // 2)]
    col_lines = [int(c) for c in range(w) if int(np.sum(g[:, c] == GRAY)) >= max(3, h // 2)]
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "could not find the grid lines"

    row_gaps = [b - a for a, b in zip(row_lines, row_lines[1:], strict=False)]
    col_gaps = [b - a for a, b in zip(col_lines, col_lines[1:], strict=False)]
    assert len(set(row_gaps)) == 1 and len(set(col_gaps)) == 1, "grid lines must be evenly spaced"
    assert row_gaps[0] == col_gaps[0] == PITCH, "expected square cells with the configured pitch"
    return row_lines, col_lines


def _cell_contents(
    g: np.ndarray, row_lines: list[int], col_lines: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    colors = np.full((n_rows, n_cols), BG, dtype=int)
    markers = np.zeros((n_rows, n_cols), dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            r0, r1 = row_lines[r], row_lines[r + 1]
            c0, c1 = col_lines[c], col_lines[c + 1]
            patch = g[r0 + 1 : r1, c0 + 1 : c1]
            mid_r = (r0 + r1) // 2
            mid_c = (c0 + c1) // 2
            vals = {
                int(v)
                for v in np.unique(patch)
                if int(v) not in (BG, ORANGE)
            }
            assert len(vals) <= 1, "a cell may contain at most one base color"
            if vals:
                colors[r, c] = vals.pop()
            markers[r, c] = int(g[mid_r, mid_c]) == ORANGE
            if markers[r, c]:
                assert int(colors[r, c]) != BG, "markers must sit inside strip cells"

    return colors, markers


def _find_strips(cell_colors: np.ndarray) -> list[dict]:
    h, w = cell_colors.shape
    seen = np.zeros((h, w), dtype=bool)
    strips: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(cell_colors[r, c])
            if color == BG or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(cell_colors[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            rows = {int(rr) for rr, _cc in comp}
            cols = {int(cc) for _rr, cc in comp}
            assert len(rows) == 1 or len(cols) == 1, "each strip must be axis-aligned and one cell thick"
            if len(rows) == 1:
                comp.sort(key=lambda x: x[1])
                assert all(comp[i + 1][1] == comp[i][1] + 1 for i in range(len(comp) - 1))
                orient = "h"
            else:
                comp.sort(key=lambda x: x[0])
                assert all(comp[i + 1][0] == comp[i][0] + 1 for i in range(len(comp) - 1))
                orient = "v"
            strips.append({"color": color, "cells": comp, "orient": orient})

    assert strips, "expected at least one strip"
    return strips


def _arrow_direction(
    g: np.ndarray,
    strip: dict,
    row_lines: list[int],
    col_lines: list[int],
) -> tuple[int, int]:
    cells = strip["cells"]
    if strip["orient"] == "h":
        r = int(cells[0][0])
        c0 = min(int(c) for _r, c in cells)
        c1 = max(int(c) for _r, c in cells)
        top = row_lines[r]
        bottom = row_lines[r + 1]
        left_lo = max(0, col_lines[max(0, c0 - 2)])
        left_hi = col_lines[c0] + 1
        right_lo = col_lines[c1 + 1]
        right_hi = min(g.shape[1], col_lines[min(len(col_lines) - 1, c1 + 3)] + 1)
        left_count = int(np.sum(g[top : bottom + 1, left_lo:left_hi] == RED))
        right_count = int(np.sum(g[top : bottom + 1, right_lo:right_hi] == RED))
        assert (left_count > 0) ^ (right_count > 0), "horizontal strip must have exactly one arrow side"
        return (0, -1) if left_count > 0 else (0, 1)

    c = int(cells[0][1])
    r0 = min(int(r) for r, _c in cells)
    r1 = max(int(r) for r, _c in cells)
    left = col_lines[c]
    right = col_lines[c + 1]
    up_lo = max(0, row_lines[max(0, r0 - 2)])
    up_hi = row_lines[r0] + 1
    down_lo = row_lines[r1 + 1]
    down_hi = min(g.shape[0], row_lines[min(len(row_lines) - 1, r1 + 3)] + 1)
    up_count = int(np.sum(g[up_lo:up_hi, left : right + 1] == RED))
    down_count = int(np.sum(g[down_lo:down_hi, left : right + 1] == RED))
    assert (up_count > 0) ^ (down_count > 0), "vertical strip must have exactly one arrow side"
    return (-1, 0) if up_count > 0 else (1, 0)


def _paint_cell(
    grid: np.ndarray,
    cell_r: int,
    cell_c: int,
    color: int,
    row_lines: list[int],
    col_lines: list[int],
) -> None:
    r0, r1 = row_lines[cell_r], row_lines[cell_r + 1]
    c0, c1 = col_lines[cell_c], col_lines[cell_c + 1]
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(color)


def _cell_bbox(cell_r: int, cell_c: int) -> tuple[int, int, int, int]:
    top = cell_r * PITCH
    left = cell_c * PITCH
    return top, left, top + PITCH, left + PITCH


def _center_between(a: int, b: int) -> int:
    return (int(a) + int(b)) // 2


def _draw_left_arrow(grid: np.ndarray, row: int, start_col: int) -> None:
    top, _left, bottom, _right = _cell_bbox(row, start_col - 1)
    outer = (start_col - 2) * PITCH
    middle = (start_col - 1) * PITCH
    stripe = start_col * PITCH
    mid_r = _center_between(top, bottom)
    grid[top, middle + 1 : stripe + 1] = RED
    grid[bottom, middle + 1 : stripe + 1] = RED
    grid[top + 1 : bottom, middle] = RED
    grid[mid_r, outer] = RED


def _draw_right_arrow(grid: np.ndarray, row: int, end_col: int) -> None:
    top, _left, bottom, _right = _cell_bbox(row, end_col + 1)
    stripe = (end_col + 1) * PITCH
    middle = (end_col + 2) * PITCH
    outer = (end_col + 3) * PITCH
    mid_r = _center_between(top, bottom)
    grid[top, stripe:middle] = RED
    grid[bottom, stripe:middle] = RED
    grid[top + 1 : bottom, middle] = RED
    grid[mid_r, outer] = RED


def _draw_up_arrow(grid: np.ndarray, start_row: int, col: int) -> None:
    _top, left, _bottom, right = _cell_bbox(start_row - 1, col)
    outer = (start_row - 2) * PITCH
    middle = (start_row - 1) * PITCH
    stripe = start_row * PITCH
    mid_c = _center_between(left, right)
    grid[middle, left + 1 : right] = RED
    grid[stripe - 1, left] = RED
    grid[stripe - 1, right] = RED
    grid[middle + 1 : stripe, left] = RED
    grid[middle + 1 : stripe, right] = RED
    grid[outer, mid_c] = RED


def _draw_down_arrow(grid: np.ndarray, end_row: int, col: int) -> None:
    _top, left, _bottom, right = _cell_bbox(end_row + 1, col)
    stripe = (end_row + 1) * PITCH
    middle = (end_row + 2) * PITCH
    outer = (end_row + 3) * PITCH
    mid_c = _center_between(left, right)
    grid[middle, left + 1 : right] = RED
    grid[stripe + 1, left] = RED
    grid[stripe + 1, right] = RED
    grid[stripe + 1 : middle, left] = RED
    grid[stripe + 1 : middle, right] = RED
    grid[outer, mid_c] = RED


def _draw_arrow(grid: np.ndarray, orient: str, direction: int, fixed: int, start: int, end: int) -> None:
    if orient == "h":
        if direction < 0:
            _draw_left_arrow(grid, fixed, start)
        else:
            _draw_right_arrow(grid, fixed, end)
    else:
        if direction < 0:
            _draw_up_arrow(grid, start, fixed)
        else:
            _draw_down_arrow(grid, end, fixed)


def _reserve_ok(
    used: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    pad: int = 1,
) -> bool:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(
    used: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    pad: int = 1,
) -> None:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_base_grid(cell_rows: int, cell_cols: int) -> np.ndarray:
    h = 1 + cell_rows * PITCH
    w = 1 + cell_cols * PITCH
    grid = np.full((h, w), BG, dtype=int)
    for r in range(0, h, PITCH):
        grid[r, :] = GRAY
    for c in range(0, w, PITCH):
        grid[:, c] = GRAY
    return grid


def _candidate_horizontal(cell_rows: int, cell_cols: int) -> tuple[int, int, int, int, int] | None:
    row = random.randrange(cell_rows)
    direction = random.choice((-1, 1))
    max_len = min(4, cell_cols - 2)
    if max_len < 2:
        return None
    length = random.randint(2, max_len)
    if direction < 0:
        min_start = 2
        max_start = cell_cols - length
        if min_start > max_start:
            return None
        start = random.randint(min_start, max_start)
        end = start + length - 1
        depth_max = min(length, start)
    else:
        min_start = 0
        max_start = cell_cols - length - 2
        if min_start > max_start:
            return None
        start = random.randint(min_start, max_start)
        end = start + length - 1
        depth_max = min(length, cell_cols - end - 1)
    if depth_max < 1:
        return None
    depth = random.randint(1, depth_max)
    return row, start, end, direction, depth


def _candidate_vertical(cell_rows: int, cell_cols: int) -> tuple[int, int, int, int, int] | None:
    col = random.randrange(cell_cols)
    direction = random.choice((-1, 1))
    max_len = min(4, cell_rows - 2)
    if max_len < 2:
        return None
    length = random.randint(2, max_len)
    if direction < 0:
        min_start = 2
        max_start = cell_rows - length
        if min_start > max_start:
            return None
        start = random.randint(min_start, max_start)
        end = start + length - 1
        depth_max = min(length, start)
    else:
        min_start = 0
        max_start = cell_rows - length - 2
        if min_start > max_start:
            return None
        start = random.randint(min_start, max_start)
        end = start + length - 1
        depth_max = min(length, cell_rows - end - 1)
    if depth_max < 1:
        return None
    depth = random.randint(1, depth_max)
    return col, start, end, direction, depth


def _place_strip(
    grid: np.ndarray,
    used: np.ndarray,
    color: int,
    orient: str,
) -> bool:
    cell_rows, cell_cols = used.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        if orient == "h":
            cand = _candidate_horizontal(cell_rows, cell_cols)
            if cand is None:
                return False
            row, start, end, direction, depth = cand
            ext = max(2, depth)
            if direction < 0:
                rr0, cc0, rr1, cc1 = row, start - ext, row, end
            else:
                rr0, cc0, rr1, cc1 = row, start, row, end + ext
            if not _reserve_ok(used, rr0, cc0, rr1, cc1, pad=1):
                continue

            cells = [(row, c) for c in range(start, end + 1)]
            markers = set(random.sample(cells, depth))
            for r, c in cells:
                _paint_cell(grid, r, c, color, list(range(0, grid.shape[0], PITCH)), list(range(0, grid.shape[1], PITCH)))
                if (r, c) in markers:
                    mr = r * PITCH + 2
                    mc = c * PITCH + 2
                    grid[mr, mc] = ORANGE
            _draw_arrow(grid, "h", direction, row, start, end)
            _reserve(used, rr0, cc0, rr1, cc1, pad=1)
            return True

        cand = _candidate_vertical(cell_rows, cell_cols)
        if cand is None:
            return False
        col, start, end, direction, depth = cand
        ext = max(2, depth)
        if direction < 0:
            rr0, cc0, rr1, cc1 = start - ext, col, end, col
        else:
            rr0, cc0, rr1, cc1 = start, col, end + ext, col
        if not _reserve_ok(used, rr0, cc0, rr1, cc1, pad=1):
            continue

        cells = [(r, col) for r in range(start, end + 1)]
        markers = set(random.sample(cells, depth))
        for r, c in cells:
            _paint_cell(grid, r, c, color, list(range(0, grid.shape[0], PITCH)), list(range(0, grid.shape[1], PITCH)))
            if (r, c) in markers:
                mr = r * PITCH + 2
                mc = c * PITCH + 2
                grid[mr, mc] = ORANGE
        _draw_arrow(grid, "v", direction, col, start, end)
        _reserve(used, rr0, cc0, rr1, cc1, pad=1)
        return True

    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        cell_rows = random.randint(5, 7)
        cell_cols = random.randint(5, 7)
        grid = _paint_base_grid(cell_rows, cell_cols)
        used = np.zeros((cell_rows, cell_cols), dtype=bool)

        max_stripes = min(4, len(STRIPE_COLORS), max(2, (cell_rows + cell_cols) // 3))
        target = random.randint(2, max_stripes)
        colors = list(STRIPE_COLORS)
        random.shuffle(colors)
        placed = 0
        orientations = [random.choice(("h", "v")) for _ in range(target)]
        if target >= 2 and len(set(orientations)) == 1:
            orientations[0] = "h"
            orientations[1] = "v"
            random.shuffle(orientations)

        for color, orient in zip(colors, orientations, strict=False):
            if _place_strip(grid, used, int(color), orient):
                placed += 1
            if placed == target:
                break

        if placed < 2:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise AssertionError("failed to generate a valid puzzle instance")
