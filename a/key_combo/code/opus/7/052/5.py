from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

CellBox = tuple[int, int, int, int]
Point = tuple[int, int]


def _divider_spans(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for divider in dividers:
        if start <= divider - 1:
            spans.append((start, divider - 1))
        start = divider + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _extract_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[dict[str, object]]]]:
    row_dividers = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_dividers = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_spans = _divider_spans(int(grid.shape[0]), row_dividers)
    col_spans = _divider_spans(int(grid.shape[1]), col_dividers)

    assert row_spans and col_spans, "expected at least one rectangular cell"

    cells: list[list[dict[str, object]]] = []
    for r0, r1 in row_spans:
        row_cells: list[dict[str, object]] = []
        for c0, c1 in col_spans:
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            coords = np.argwhere((region != BLACK) & (region != GREY))
            assert len(coords) <= 1, "each cell may contain at most one colored pixel"
            color = None
            point = None
            if len(coords) == 1:
                rr, cc = coords[0]
                point = (r0 + int(rr), c0 + int(cc))
                color = int(grid[point])
            row_cells.append({"box": (r0, r1, c0, c1), "color": color, "point": point})
        cells.append(row_cells)
    return row_spans, col_spans, cells


def _collect_domino_ops(cells: list[list[dict[str, object]]]) -> list[tuple[Point, Point, tuple[str, int, int, int]]]:
    ops: list[tuple[Point, Point, tuple[str, int, int, int]]] = []
    n_rows = len(cells)
    n_cols = len(cells[0])

    for r in range(n_rows):
        for c in range(n_cols):
            here = cells[r][c]
            box = here["box"]
            assert isinstance(box, tuple)
            _, r1, _, c1 = box
            color_here = here["color"]

            if c + 1 < n_cols:
                right = cells[r][c + 1]
                if {color_here, right["color"]} == {RED, GREEN}:
                    red_point = here["point"] if color_here == RED else right["point"]
                    green_point = here["point"] if color_here == GREEN else right["point"]
                    assert red_point is not None and green_point is not None
                    ops.append((red_point, green_point, ("v", c1 + 1, box[0], box[1])))

            if r + 1 < n_rows:
                down = cells[r + 1][c]
                if {color_here, down["color"]} == {RED, GREEN}:
                    red_point = here["point"] if color_here == RED else down["point"]
                    green_point = here["point"] if color_here == GREEN else down["point"]
                    assert red_point is not None and green_point is not None
                    ops.append((red_point, green_point, ("h", r1 + 1, box[2], box[3])))
    return ops


def _connect_aligned_pairs(grid: np.ndarray, color: int) -> None:
    points = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    row_pairs: list[tuple[int, int, int]] = []
    col_pairs: list[tuple[int, int, int]] = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a = points[i]
            b = points[j]
            if a[0] == b[0]:
                c0, c1 = sorted((a[1], b[1]))
                row_pairs.append((a[0], c0, c1))
            if a[1] == b[1]:
                r0, r1 = sorted((a[0], b[0]))
                col_pairs.append((a[1], r0, r1))

    for row, c0, c1 in sorted(set(row_pairs), key=lambda item: (item[0], item[1], item[2])):
        if np.all(grid[row, c0 + 1 : c1] == BLACK):
            grid[row, c0 + 1 : c1] = color

    for col, r0, r1 in sorted(set(col_pairs), key=lambda item: (item[0], item[1], item[2])):
        if np.all(grid[r0 + 1 : r1, col] == BLACK):
            grid[r0 + 1 : r1, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase red-green wall-crossing pairs, knock out their divider segments, and then connect aligned blue/teal survivors.

    Input structure:
    - A black background contains a rectangular lattice of one-cell-thick grey divider rows and columns.
    - The divider bars partition the grid into rectangular cells; each cell interior is otherwise black.
    - Some cells contain exactly one colored pixel: red, green, or blue.

    Rule:
    - Treat each pair of orthogonally adjacent cells that share a grey divider segment and contain one red pixel and one green pixel as a wall-crossing domino pair.
    - For every such pair, erase the red pixel, recolor the green pixel to teal, and delete the shared divider segment between those two cells only.
    - After all divider deletions, consider the remaining blue pixels and teal pixels separately. For each same-color pair on one row, then for each same-color pair on one column, fill the black pixels between them if and only if the whole path is black.
    - Keep all untouched divider pieces and all other pixels unchanged.

    Key insight:
    - The red-green pairs matter because they do not just recolor pixels; they physically open gaps in the divider grid, and those new gaps determine which later blue or teal alignments become connectable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, _, cells = _extract_cells(g)
    ops = _collect_domino_ops(cells)
    assert ops, "expected at least one red-green divider-crossing pair"

    out = g.copy()
    for red_point, green_point, wall in ops:
        out[red_point] = BLACK
        out[green_point] = TEAL
        axis, fixed, start, end = wall
        if axis == "v":
            out[start : end + 1, fixed] = BLACK
        else:
            out[fixed, start : end + 1] = BLACK

    _connect_aligned_pairs(out, BLUE)
    _connect_aligned_pairs(out, TEAL)
    return out


GEN_CELL_ROWS = 7
GEN_CELL_COLS = 7
GEN_MIN_CELL_SIZE = 3
GEN_MAX_EXTRA_SIZE_CELLS = 3
GEN_RETRY_LIMIT = 50


def _sample_axis_sizes(n_cells: int) -> list[int]:
    sizes = [GEN_MIN_CELL_SIZE] * n_cells
    extra_count = random.randint(0, GEN_MAX_EXTRA_SIZE_CELLS)
    for idx in random.sample(range(n_cells), extra_count):
        sizes[idx] += 1
    return sizes


def _build_divider_grid(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(col_widths) + len(col_widths) - 1
    grid = np.zeros((height, width), dtype=int)

    row_spans: list[tuple[int, int]] = []
    r = 0
    for idx, cell_h in enumerate(row_heights):
        row_spans.append((r, r + cell_h - 1))
        r += cell_h
        if idx < len(row_heights) - 1:
            grid[r, :] = GREY
            r += 1

    col_spans: list[tuple[int, int]] = []
    c = 0
    for idx, cell_w in enumerate(col_widths):
        col_spans.append((c, c + cell_w - 1))
        c += cell_w
        if idx < len(col_widths) - 1:
            grid[:, c] = GREY
            c += 1

    return grid, row_spans, col_spans


def _pick_offset(size: int, mode: str) -> int:
    if mode == "low":
        return 0
    if mode == "high":
        return size - 1
    if mode == "center":
        return size // 2
    raise ValueError(f"unknown placement mode: {mode}")


def _place_in_cell(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    cell_r: int,
    cell_c: int,
    color: int,
    row_offset: int | None = None,
    col_offset: int | None = None,
    row_mode: str = "center",
    col_mode: str = "center",
) -> Point:
    r0, r1 = row_spans[cell_r]
    c0, c1 = col_spans[cell_c]
    if row_offset is None:
        row_offset = _pick_offset(r1 - r0 + 1, row_mode)
    if col_offset is None:
        col_offset = _pick_offset(c1 - c0 + 1, col_mode)
    point = (r0 + row_offset, c0 + col_offset)
    assert grid[point] == BLACK
    grid[point] = color
    return point


def _attempt_generate() -> np.ndarray | None:
    row_heights = _sample_axis_sizes(GEN_CELL_ROWS)
    col_widths = _sample_axis_sizes(GEN_CELL_COLS)
    grid, row_spans, col_spans = _build_divider_grid(row_heights, col_widths)

    horizontal_row = 1
    horizontal_offset = random.randrange(row_heights[horizontal_row])
    horizontal_roles = [RED, GREEN, RED, GREEN, RED]
    horizontal_col_modes = ["high", "center", "center", "center", "low"]
    horizontal_teals: list[Point] = []

    for idx, cell_c in enumerate(range(1, 6)):
        point = _place_in_cell(
            grid,
            row_spans,
            col_spans,
            horizontal_row,
            cell_c,
            horizontal_roles[idx],
            row_offset=horizontal_offset,
            col_mode=horizontal_col_modes[idx],
        )
        if horizontal_roles[idx] == GREEN:
            horizontal_teals.append(point)

    blue_points = [
        _place_in_cell(
            grid,
            row_spans,
            col_spans,
            horizontal_row,
            0,
            BLUE,
            row_offset=horizontal_offset,
            col_mode="high",
        ),
        _place_in_cell(
            grid,
            row_spans,
            col_spans,
            horizontal_row,
            6,
            BLUE,
            row_offset=horizontal_offset,
            col_mode="low",
        ),
    ]

    vertical_col = random.choice([1, 2, 4, 5])
    vertical_offset = random.randrange(col_widths[vertical_col])
    vertical_roles = [RED, GREEN, RED, GREEN, RED]
    vertical_row_modes = ["high", "center", "center", "center", "low"]
    vertical_teals: list[Point] = []

    for idx, cell_r in enumerate(range(2, 7)):
        point = _place_in_cell(
            grid,
            row_spans,
            col_spans,
            cell_r,
            vertical_col,
            vertical_roles[idx],
            col_offset=vertical_offset,
            row_mode=vertical_row_modes[idx],
        )
        if vertical_roles[idx] == GREEN:
            vertical_teals.append(point)

    out = main(grid)

    h_left, h_right = sorted(horizontal_teals, key=lambda p: p[1])
    assert np.all(out[h_left[0], h_left[1] : h_right[1] + 1] == TEAL)

    v_top, v_bottom = sorted(vertical_teals, key=lambda p: p[0])
    assert np.all(out[v_top[0] : v_bottom[0] + 1, v_top[1]] == TEAL)

    removed_wall_cols = [col_spans[c][1] + 1 for c in range(1, 5)]
    blue_row = blue_points[0][0]
    blue_c0, blue_c1 = sorted((blue_points[0][1], blue_points[1][1]))
    assert any(blue_c0 < wall_col < blue_c1 for wall_col in removed_wall_cols)
    assert out[blue_row, blue_c0] == BLUE and out[blue_row, blue_c1] == BLUE

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
