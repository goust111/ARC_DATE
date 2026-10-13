from __future__ import annotations

import random

import numpy as np


BG = 0
MIN_SIZE = 8
MAX_SIZE = 20
MIN_GAP = 3
MAX_MARKERS_PER_AXIS = 4
MAX_GENERATION_ATTEMPTS = 300
MAX_POSITION_ATTEMPTS = 200

RowMarker = tuple[int, int]
ColMarker = tuple[int, int]
Coord = tuple[int, int]


def _plan_cell(planned: dict[Coord, int], cell: Coord, color: int) -> None:
    previous = planned.get(cell)
    assert previous is None or previous == color, "crosses must not request conflicting colors"
    planned[cell] = color


def _render_output(
    shape: tuple[int, int],
    bg: int,
    row_markers: list[RowMarker],
    col_markers: list[ColMarker],
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), bg, dtype=int)
    planned: dict[Coord, int] = {}

    for row, color in row_markers:
        assert 1 <= row <= height - 2, "row markers must leave room for a full cross"
        _plan_cell(planned, (row, 0), color)
    for col, color in col_markers:
        assert 1 <= col <= width - 2, "column markers must leave room for a full cross"
        _plan_cell(planned, (0, col), color)

    for row, row_color in row_markers:
        for col, col_color in col_markers:
            _plan_cell(planned, (row, col), row_color)
            _plan_cell(planned, (row, col - 1), row_color)
            _plan_cell(planned, (row, col + 1), row_color)
            _plan_cell(planned, (row - 1, col), col_color)
            _plan_cell(planned, (row + 1, col), col_color)

    for (row, col), color in planned.items():
        out[row, col] = color
    return out


def _extract_markers(grid: np.ndarray) -> tuple[int, list[RowMarker], list[ColMarker]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert 3 <= height <= 30 and 3 <= width <= 30, "grid must fit ARC limits"

    bg = int(g[0, 0])
    assert np.all(g[1:, 1:] == bg), "expected only top-row and left-column markers in the input"

    row_markers = [(int(row), int(g[row, 0])) for row in range(1, height) if int(g[row, 0]) != bg]
    col_markers = [(int(col), int(g[0, col])) for col in range(1, width) if int(g[0, col]) != bg]
    assert row_markers and col_markers, "expected at least one row marker and one column marker"

    return bg, row_markers, col_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw colored crosses at the intersections between left-edge row markers and top-edge column markers.

    Input structure:
    - The background is black, and the top-left corner stays background.
    - The top row contains isolated colored marker pixels that designate columns.
    - The left column contains isolated colored marker pixels that designate rows.
    - All interior cells start as background.

    Rule:
    - For every row-marker and column-marker pair, use their intersection as the center of a 3x3 cross.
    - Paint the center and the left/right arm cells with the row-marker color.
    - Paint the up/down arm cells with the column-marker color.
    - Keep the original edge markers unchanged.

    Key insight:
    - The output is the Cartesian product of marked rows and marked columns, with row markers controlling the horizontal paint and column markers controlling the vertical paint.
    """
    bg, row_markers, col_markers = _extract_markers(grid)
    return _render_output(np.asarray(grid).shape, bg, row_markers, col_markers)


def _max_spaced_count(size: int) -> int:
    if size < 3:
        return 0
    return 1 + (size - 3) // MIN_GAP


def _sample_spaced_positions(size: int, count: int, forced: tuple[int, ...]) -> list[int] | None:
    forced_positions = sorted(set(int(pos) for pos in forced))
    if len(forced_positions) != len(forced):
        return None
    if any(pos < 1 or pos > size - 2 for pos in forced_positions):
        return None
    for left, right in zip(forced_positions, forced_positions[1:]):
        if right - left < MIN_GAP:
            return None

    candidates = [pos for pos in range(1, size - 1) if pos not in forced_positions]
    for _ in range(MAX_POSITION_ATTEMPTS):
        picked = forced_positions[:]
        pool = candidates[:]
        random.shuffle(pool)
        for pos in pool:
            if all(abs(pos - other) >= MIN_GAP for other in picked):
                picked.append(pos)
                if len(picked) == count:
                    return sorted(picked)
        if len(picked) == count:
            return sorted(picked)
    return None


def _sample_axis_markers(size: int) -> list[int] | None:
    max_count = min(MAX_MARKERS_PER_AXIS, _max_spaced_count(size))
    if max_count <= 0:
        return None

    forced_options: list[tuple[int, ...]] = [tuple()]
    if size >= 4:
        forced_options.append((1,))
        forced_options.append((size - 2,))
    if size >= 7 and size - 3 >= MIN_GAP:
        forced_options.append((1, size - 2))

    forced = random.choice(forced_options)
    count = random.randint(max(1, len(forced)), max_count)
    return _sample_spaced_positions(size, count, forced)


def _build_input(
    shape: tuple[int, int],
    row_positions: list[int],
    col_positions: list[int],
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    height, width = shape
    grid = np.full((height, width), BG, dtype=int)
    for row, color in zip(row_positions, row_colors):
        grid[row, 0] = color
    for col, color in zip(col_positions, col_colors):
        grid[0, col] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)

    row_positions = _sample_axis_markers(height)
    col_positions = _sample_axis_markers(width)
    if row_positions is None or col_positions is None:
        return None

    total_markers = len(row_positions) + len(col_positions)
    if total_markers > 9:
        return None

    colors = random.sample(range(1, 10), total_markers)
    row_colors = colors[: len(row_positions)]
    col_colors = colors[len(row_positions) :]

    inp = _build_input((height, width), row_positions, col_positions, row_colors, col_colors)
    row_markers = list(zip(row_positions, row_colors))
    col_markers = list(zip(col_positions, col_colors))
    expected = _render_output(inp.shape, BG, row_markers, col_markers)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-intersection ARC input")
