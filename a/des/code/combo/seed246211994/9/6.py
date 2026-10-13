from __future__ import annotations

import random

import numpy as np


OUTPUT_SHAPE = (5, 3)
GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
SEGMENT_COUNT_RANGE = (1, 5)
SEGMENT_LENGTH_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 300
PADDING = 1
ORIENTATIONS = (
    ("h_right", 0, 1),
    ("h_left", 0, -1),
    ("v_down", 1, 0),
    ("diag_dr", 1, 1),
    ("diag_dl", 1, -1),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered colored line segments by their topmost row and compress them into stripes.

    Input structure
    - The background is the dominant color.
    - Up to five one-pixel-wide colored line segments are scattered on the grid.
    - Each foreground color has a unique topmost occupied row.

    Rule
    - For each foreground color, find its highest occupied row.
    - Sort foreground colors from top to bottom by that row.
    - Create a 5x3 output grid filled with the background color.
    - Starting from the bottom output row, paint one full 1x3 stripe per color in sorted order.
    - Leave any unused top rows as background.

    Key insight
    - Segment length, direction, and exact placement do not matter; only the vertical order of
      each color's highest pixel is preserved in the output stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    fg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert 1 <= len(fg_colors) <= OUTPUT_SHAPE[0]

    color_rows: list[tuple[int, int, int]] = []
    for color in fg_colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        top_row = int(coords[:, 0].min())
        top_cols = coords[coords[:, 0] == top_row, 1]
        left_col = int(top_cols.min())
        color_rows.append((top_row, left_col, color))

    top_rows = [row for row, _col, _color in color_rows]
    assert len(set(top_rows)) == len(top_rows), "foreground colors must have distinct top rows"
    color_rows.sort()

    out = np.full(OUTPUT_SHAPE, bg, dtype=int)
    for idx, (_row, _col, color) in enumerate(color_rows):
        out[OUTPUT_SHAPE[0] - 1 - idx, :] = color
    return out


def _segment_cells(top_row: int, start_col: int, dr: int, dc: int, length: int) -> list[tuple[int, int]]:
    return [(top_row + step * dr, start_col + step * dc) for step in range(length)]


def _start_columns(width: int, length: int, dc: int) -> range:
    if dc == 1:
        return range(0, width - length + 1)
    if dc == -1:
        return range(length - 1, width)
    return range(0, width)


def _max_length(height: int, width: int, top_row: int, dr: int, dc: int) -> int:
    max_len = SEGMENT_LENGTH_RANGE[1]
    if dr != 0:
        max_len = min(max_len, height - top_row)
    if dc != 0:
        max_len = min(max_len, width)
    return max_len


def _cells_are_clear(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not blocked[r, c] for r, c in cells)


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - PADDING), min(height, r + PADDING + 1)):
            for cc in range(max(0, c - PADDING), min(width, c + PADDING + 1)):
                blocked[rr, cc] = True


def _place_segment(
    grid: np.ndarray,
    blocked: np.ndarray,
    top_row: int,
    color: int,
) -> bool:
    height, width = grid.shape
    orientations = list(ORIENTATIONS)
    random.shuffle(orientations)

    for _name, dr, dc in orientations:
        max_len = _max_length(height, width, top_row, dr, dc)
        if max_len < SEGMENT_LENGTH_RANGE[0]:
            continue

        lengths = list(range(SEGMENT_LENGTH_RANGE[0], max_len + 1))
        random.shuffle(lengths)
        for length in lengths:
            starts = list(_start_columns(width, length, dc))
            random.shuffle(starts)
            for start_col in starts:
                cells = _segment_cells(top_row, start_col, dr, dc, length)
                if not _cells_are_clear(blocked, cells):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                _mark_blocked(blocked, cells)
                return True
    return False


def _expected_output(bg: int, ordered_colors: list[int]) -> np.ndarray:
    out = np.full(OUTPUT_SHAPE, bg, dtype=int)
    for idx, color in enumerate(ordered_colors):
        out[OUTPUT_SHAPE[0] - 1 - idx, :] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_segments = random.randint(*SEGMENT_COUNT_RANGE)
    bg = random.randrange(10)
    fg_colors = random.sample([c for c in range(10) if c != bg], n_segments)

    # Reserve unique top rows so the ordering signal is unambiguous.
    top_rows = random.sample(range(height - 1), n_segments)
    random.shuffle(top_rows)

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    for top_row, color in zip(top_rows, fg_colors):
        if not _place_segment(grid, blocked, top_row, color):
            return None

    ordered_colors = [color for _row, color in sorted(zip(top_rows, fg_colors))]
    expected = _expected_output(bg, ordered_colors)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
