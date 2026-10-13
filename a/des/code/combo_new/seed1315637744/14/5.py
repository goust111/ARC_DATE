from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GRID_COLOR = 5
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _grid_line_positions(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    r_lines = np.flatnonzero(row_counts == row_counts.max())
    c_lines = np.flatnonzero(col_counts == col_counts.max())
    assert len(r_lines) >= 2 and len(c_lines) >= 2, "grid must contain multiple horizontal and vertical separators"

    r_steps = np.diff(r_lines)
    c_steps = np.diff(c_lines)
    assert len(set(map(int, r_steps.tolist()))) == 1, "horizontal separators must be evenly spaced"
    assert len(set(map(int, c_steps.tolist()))) == 1, "vertical separators must be evenly spaced"

    cell_size = int(r_steps[0] - 1)
    assert cell_size >= 1, "cell interiors must be at least 1 pixel wide"
    assert int(c_steps[0] - 1) == cell_size, "cells must be square"

    return r_lines, c_lines, cell_size


def _cell_centers(lines: np.ndarray, cell_size: int) -> list[int]:
    return [int(line + 1 + cell_size // 2) for line in lines[:-1]]


def _read_row_markers(g: np.ndarray, grid_color: int, r_lines: np.ndarray, left_col: int, cell_size: int) -> list[int]:
    colors = []
    for i, r0 in enumerate(r_lines[:-1]):
        center_r = int(r0 + 1 + cell_size // 2)
        color = int(g[center_r, left_col]) if 0 <= left_col < g.shape[1] else BG
        if color in (BG, grid_color):
            band = g[r0 + 1 : r_lines[i + 1], : left_col + 1]
            pts = np.argwhere((band != BG) & (band != grid_color))
            assert len(pts) == 1, "each row band must have one row marker"
            color = int(band[tuple(pts[0])])
        colors.append(color)
    assert len(set(colors)) == len(colors), "row marker colors must be distinct"
    return colors


def _read_col_markers(g: np.ndarray, grid_color: int, c_lines: np.ndarray, top_row: int, cell_size: int) -> list[int]:
    colors = []
    for j, c0 in enumerate(c_lines[:-1]):
        center_c = int(c0 + 1 + cell_size // 2)
        color = int(g[top_row, center_c]) if 0 <= top_row < g.shape[0] else BG
        if color in (BG, grid_color):
            band = g[: top_row + 1, c0 + 1 : c_lines[j + 1]]
            pts = np.argwhere((band != BG) & (band != grid_color))
            assert len(pts) == 1, "each column band must have one column marker"
            color = int(band[tuple(pts[0])])
        colors.append(color)
    assert len(set(colors)) == len(colors), "column marker colors must be distinct"
    return colors


def _border_run(g: np.ndarray, grid_color: int, start: tuple[int, int], step: tuple[int, int]) -> list[int]:
    h, w = g.shape
    r, c = start
    dr, dc = step
    colors = []
    while 0 <= r < h and 0 <= c < w:
        color = int(g[r, c])
        if color in (BG, grid_color):
            break
        colors.append(color)
        r += dr
        c += dc
    return colors


def _read_legend(g: np.ndarray, grid_color: int) -> list[int]:
    h, w = g.shape
    corners = [
        ((0, 0), ((0, 1), (1, 0))),
        ((0, w - 1), ((0, -1), (1, 0))),
        ((h - 1, 0), ((0, 1), (-1, 0))),
        ((h - 1, w - 1), ((0, -1), (-1, 0))),
    ]

    matches = [(corner, steps) for corner, steps in corners if int(g[corner]) not in (BG, grid_color)]
    assert len(matches) == 1, "legend must start at exactly one outer corner"
    corner, steps = matches[0]

    runs = [_border_run(g, grid_color, corner, step) for step in steps]
    valid = [run for run in runs if len(run) >= 2]
    assert len(valid) == 1, "legend must be a single straight strip from its corner"
    legend = valid[0]
    assert len(set(legend)) == len(legend), "legend colors must be unique"
    return legend


def _paint_cells(
    g: np.ndarray,
    r_lines: np.ndarray,
    c_lines: np.ndarray,
    legend: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    out = g.copy()
    index = {int(color): i for i, color in enumerate(legend)}
    assert set(map(int, row_colors)).issubset(index), "row markers must come from the legend"
    assert set(map(int, col_colors)).issubset(index), "column markers must come from the legend"

    for i, row_color in enumerate(row_colors):
        a = int(index[int(row_color)])
        for j, col_color in enumerate(col_colors):
            b = int(index[int(col_color)])
            fill = int(legend[(a + b) % len(legend)])
            out[r_lines[i] + 1 : r_lines[i + 1], c_lines[j] + 1 : c_lines[j + 1]] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each chessboard cell using the cyclic sum of its row-marker and column-marker colors on a corner legend.

    Input structure:
    - The background is black.
    - A grey 1-pixel lattice divides a rectangular board into equal square cells.
    - One colored pixel sits to the left of each board row, and one colored pixel sits above each board column.
    - A short strip of distinct colors starts at exactly one outer corner and runs horizontally or vertically along the border.
    - All row-marker and column-marker colors appear somewhere on that corner strip.

    Rule:
    - Read the corner strip from the corner pixel outward to obtain the cyclic color order.
    - Map every row-marker color and column-marker color to its position on that strip.
    - For each board cell, add the two positions and wrap around by the strip length.
    - Fill the whole cell interior with the strip color at that wrapped position, while keeping the grey lattice, markers, and legend unchanged.

    Key insight:
    - The cell color is not determined by a row cue or a column cue alone; it is the wrapped sum of both cues measured on the visible legend ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    non_bg_colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert non_bg_colors, "grid must contain non-background colors"
    grid_color = max(non_bg_colors, key=lambda color: int(np.count_nonzero(g == int(color))))

    r_lines, c_lines, cell_size = _grid_line_positions(g == grid_color)
    top_row = int(r_lines[0] - 1)
    left_col = int(c_lines[0] - 1)
    assert top_row >= 0 and left_col >= 0, "markers must sit immediately above and left of the board"

    row_colors = _read_row_markers(g, grid_color, r_lines, left_col, cell_size)
    col_colors = _read_col_markers(g, grid_color, c_lines, top_row, cell_size)
    legend = _read_legend(g, grid_color)

    return _paint_cells(g, r_lines, c_lines, legend, row_colors, col_colors)


def _board_span(count: int, cell_size: int) -> int:
    return int(count * cell_size + count + 1)


def _sample_margins(board_span: int) -> tuple[int, int]:
    budget = 30 - int(board_span)
    assert budget >= 4
    first = random.randint(2, min(3, budget - 2))
    second = random.randint(2, min(3, budget - first))
    return first, second


def _line_positions(start: int, count: int, cell_size: int) -> list[int]:
    return [int(start + i * (cell_size + 1)) for i in range(count + 1)]


def _place_legend(g: np.ndarray, legend: Iterable[int], corner: str, orientation: str) -> None:
    legend = list(map(int, legend))
    h, w = g.shape
    if corner == "tl":
        coords = [(0, i) for i in range(len(legend))] if orientation == "h" else [(i, 0) for i in range(len(legend))]
    elif corner == "tr":
        coords = [(0, w - 1 - i) for i in range(len(legend))] if orientation == "h" else [(i, w - 1) for i in range(len(legend))]
    elif corner == "bl":
        coords = [(h - 1, i) for i in range(len(legend))] if orientation == "h" else [(h - 1 - i, 0) for i in range(len(legend))]
    else:
        coords = [(h - 1, w - 1 - i) for i in range(len(legend))] if orientation == "h" else [(h - 1 - i, w - 1) for i in range(len(legend))]

    for (r, c), color in zip(coords, legend):
        g[int(r), int(c)] = int(color)


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(3, 6)
    legend = random.sample(LEGEND_COLORS, legend_len)

    row_count = random.randint(2, legend_len)
    col_count = random.randint(2, legend_len)
    cell_size = random.randint(1, 3)

    board_h = _board_span(row_count, cell_size)
    board_w = _board_span(col_count, cell_size)
    top_margin, bottom_margin = _sample_margins(board_h)
    left_margin, right_margin = _sample_margins(board_w)

    h = board_h + top_margin + bottom_margin
    w = board_w + left_margin + right_margin
    if max(h, w) > 30:
        return None

    g = np.full((h, w), BG, dtype=int)
    top = int(top_margin)
    left = int(left_margin)
    r_lines = _line_positions(top, row_count, cell_size)
    c_lines = _line_positions(left, col_count, cell_size)

    for r in r_lines:
        g[int(r), left : left + board_w] = GRID_COLOR
    for c in c_lines:
        g[top : top + board_h, int(c)] = GRID_COLOR

    row_ids = random.sample(range(legend_len), row_count)
    col_ids = random.sample(range(legend_len), col_count)
    row_colors = [int(legend[i]) for i in row_ids]
    col_colors = [int(legend[i]) for i in col_ids]

    for r, color in zip(_cell_centers(np.array(r_lines), cell_size), row_colors):
        g[int(r), left - 1] = int(color)
    for c, color in zip(_cell_centers(np.array(c_lines), cell_size), col_colors):
        g[top - 1, int(c)] = int(color)

    _place_legend(g, legend, random.choice(["tl", "tr", "bl", "br"]), random.choice(["h", "v"]))

    expected = _paint_cells(g, np.array(r_lines), np.array(c_lines), legend, row_colors, col_colors)
    try:
        if np.array_equal(main(g), expected):
            return g
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
