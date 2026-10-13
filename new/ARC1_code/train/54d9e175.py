from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg
import random


PAIRED_FILL = {
    Color.BLUE: Color.PINK,
    Color.RED: Color.ORANGE,
    Color.GREEN: Color.TEAL,
    Color.YELLOW: Color.MAROON,
}


def _non_separator_intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    prev = -1
    for idx in separators:
        assert 0 <= idx < length
        assert idx > prev
        if start < idx:
            intervals.append((start, idx))
        start = idx + 1
        prev = idx
    if start < length:
        intervals.append((start, length))
    assert intervals
    return intervals


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey-separated 3x3 window with the paired color of its center marker.

    Input structure
    - The background is the dominant color, and grey full rows/columns form separator lines.
    - The non-separator regions are 3x3 windows.
    - Each 3x3 window contains exactly one foreground marker at its center.

    Rule
    - Keep every grey separator line unchanged.
    - Replace each 3x3 window by a solid block using the marker's paired fill color:
      blue->pink, red->orange, green->teal, yellow->maroon.

    Key insight
    - The grey grid is only a partition; the center marker inside each cell chooses the
      solid color that fills that cell.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep = Color.GREY
    assert bg != sep

    height, width = g.shape
    sep_rows = [r for r in range(height) if bool(np.all(g[r, :] == sep))]
    sep_cols = [c for c in range(width) if bool(np.all(g[:, c] == sep))]
    assert sep_rows or sep_cols

    row_blocks = _non_separator_intervals(height, sep_rows)
    col_blocks = _non_separator_intervals(width, sep_cols)

    out = g.copy()
    for r0, r1 in row_blocks:
        for c0, c1 in col_blocks:
            window = g[r0:r1, c0:c1]
            assert window.shape == (3, 3)
            assert not bool(np.any(window == sep))

            marker_coords = np.argwhere(window != bg)
            assert len(marker_coords) == 1
            marker_r, marker_c = map(int, marker_coords[0])
            assert (marker_r, marker_c) == (1, 1)

            marker_color = int(window[marker_r, marker_c])
            assert marker_color in PAIRED_FILL
            out[r0:r1, c0:c1] = PAIRED_FILL[marker_color]

    return out

# --------------- generator (o5x) ---------------
# Geometry / layout: 3x3 cells separated by one grey line.
_GEN_CELL_SIZE, _GEN_STRIDE, _GEN_COL_CELLS = 3, 4, 3
_GEN_ROW_CELL_OPTIONS = [1, 2]
# Colors and validation.
_GEN_BG_COLOR, _GEN_SEPARATOR_COLOR = 0, 5
_GEN_MARKER_MIN, _GEN_MARKER_MAX = 1, 4
_GEN_MARKER_TO_FILL_54D9E175 = {1: 6, 2: 7, 3: 8, 4: 9}
_GEN_RETRY_LIMIT = 300

def _gen_make_base_grid_54d9e175(row_cells: int) -> np.ndarray:
    height = _GEN_STRIDE * row_cells - 1
    width = _GEN_STRIDE * _GEN_COL_CELLS - 1
    grid = np.full((height, width), _GEN_BG_COLOR, dtype=int)
    grid[_GEN_CELL_SIZE::_GEN_STRIDE, :] = _GEN_SEPARATOR_COLOR
    grid[:, _GEN_CELL_SIZE::_GEN_STRIDE] = _GEN_SEPARATOR_COLOR
    return grid

def _gen_place_markers_54d9e175(grid: np.ndarray, row_cells: int) -> None:
    for rr in range(row_cells):
        for cc in range(_GEN_COL_CELLS):
            grid[_GEN_STRIDE * rr + 1, _GEN_STRIDE * cc + 1] = random.randint(_GEN_MARKER_MIN, _GEN_MARKER_MAX)

def _gen_expected_output_54d9e175(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for r0 in range(0, inp.shape[0], _GEN_STRIDE):
        for c0 in range(0, inp.shape[1], _GEN_STRIDE):
            marker = int(inp[r0 + 1, c0 + 1])
            out[r0:r0 + _GEN_CELL_SIZE, c0:c0 + _GEN_CELL_SIZE] = _GEN_MARKER_TO_FILL_54D9E175[marker]
    return out

def _gen_attempt() -> np.ndarray | None:
    row_cells = random.choice(_GEN_ROW_CELL_OPTIONS)
    grid = _gen_make_base_grid_54d9e175(row_cells)
    _gen_place_markers_54d9e175(grid, row_cells)
    return grid

def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output_54d9e175(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError('failed to generate valid input for 54d9e175')
