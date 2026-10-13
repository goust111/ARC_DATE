from __future__ import annotations

import numpy as np

from arc_utils import colors_present, draw_line, get_bg
import random


def _is_consecutive(values: np.ndarray) -> bool:
    ordered = np.sort(values)
    return bool(ordered.size <= 1 or np.all(np.diff(ordered) == 1))


def _is_line_segment(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.unique(rows).size == 1:
        return _is_consecutive(cols)
    if np.unique(cols).size == 1:
        return _is_consecutive(rows)
    if np.unique(rows - cols).size == 1:
        return np.unique(rows).size == coords.shape[0] and _is_consecutive(rows)
    if np.unique(rows + cols).size == 1:
        return np.unique(rows).size == coords.shape[0] and _is_consecutive(rows)
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize the foreground line objects as ordered horizontal color stripes.

    Input structure:
    - A dominant background fills a rectangular grid.
    - Each non-background color forms one contiguous one-cell-thick line segment.
    - The line segments have distinct topmost rows, and there are at most five of
      them.

    Rule:
    - Sort the foreground colors by the topmost row occupied by their line
      segment.
    - Create a 5x3 output filled with the background color.
    - Place one solid 1x3 row for each foreground color, preserving that
      top-to-bottom order and padding any unused rows above with background.

    Key insight: The exact line length and slope do not matter; only the
    vertical order of the colored line segments is encoded in the output stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    ordered: list[tuple[int, int]] = []
    for color in colors_present(g, exclude=[bg]):
        coords = np.argwhere(g == color)
        assert coords.size > 0, "foreground color must occupy at least one pixel"
        assert _is_line_segment(coords), "each foreground color must be one line segment"
        ordered.append((int(coords[:, 0].min()), int(color)))

    assert 0 < len(ordered) <= 5, "expected one to five foreground line segments"
    top_rows = [top for top, _ in ordered]
    assert len(set(top_rows)) == len(top_rows), "line segments must have distinct top rows"

    ordered.sort()
    out = np.full((5, 3), bg, dtype=int)
    start_row = out.shape[0] - len(ordered)
    for out_r, (_, color) in enumerate(ordered, start=start_row):
        out[out_r, :] = color
    return out

# --------------- generator (att2) ---------------

# Canvas, palette, and output shape.
BG, GRID_SIZE = 7, 16
OUTPUT_SHAPE = (5, 3)
FG_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# Segment counts, lengths, and allowed row extents.
LINE_COUNT_BOUNDS = (3, 5)
LINE_LENGTH_BOUNDS = (3, 15)
TOP_ROW_BOUNDS = (2, 15)
BOTTOM_BAND_MIN, BOTTOM_DIAGONAL_TOP_MAX = 12, 13
BOTTOM_ROW = GRID_SIZE - 1

# Rejection sampling limit.
RETRY_LIMIT = 300


def _coords_for_line(top: int, start_col: int, length: int, dc: int) -> set[tuple[int, int]]:
    if dc == 0:
        return {(top, start_col + i) for i in range(length)}
    return {(top + i, start_col + i * dc) for i in range(length)}


def _valid_starts(top: int, length: int, dc: int, occupied: set[tuple[int, int]]) -> list[int]:
    starts = range(0, GRID_SIZE - length + 1) if dc >= 0 else range(length - 1, GRID_SIZE)
    return [
        start_col
        for start_col in starts
        if not (_coords_for_line(top, start_col, length, dc) & occupied)
    ]


def _line_options_for_top(top: int, must_reach_bottom: bool) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    if BOTTOM_BAND_MIN <= top <= BOTTOM_ROW and not must_reach_bottom:
        for length in range(LINE_LENGTH_BOUNDS[0], LINE_LENGTH_BOUNDS[1] + 1):
            options.append((0, length))

    if top <= BOTTOM_DIAGONAL_TOP_MAX:
        if must_reach_bottom:
            lengths = [GRID_SIZE - top]
        else:
            lo = max(LINE_LENGTH_BOUNDS[0], BOTTOM_BAND_MIN - top + 1)
            hi = GRID_SIZE - top
            lengths = list(range(lo, hi + 1))
        for length in lengths:
            options.append((1, length))
            options.append((-1, length))

    if top == BOTTOM_ROW and must_reach_bottom:
        for length in range(LINE_LENGTH_BOUNDS[0], LINE_LENGTH_BOUNDS[1] + 1):
            options.append((0, length))
    return options


def _build_expected(colors: list[int]) -> np.ndarray:
    expected = np.full(OUTPUT_SHAPE, BG, dtype=int)
    for i, color in enumerate(colors):
        expected[OUTPUT_SHAPE[0] - len(colors) + i, :] = color
    return expected


def _sample_top_rows_and_bottom(n_lines: int) -> tuple[list[int], int] | None:
    top_rows = sorted(random.sample(range(TOP_ROW_BOUNDS[0], TOP_ROW_BOUNDS[1] + 1), n_lines))
    bottom_capable = [i for i, top in enumerate(top_rows) if top <= BOTTOM_DIAGONAL_TOP_MAX or top == BOTTOM_ROW]
    if not bottom_capable:
        return None
    return top_rows, random.choice(bottom_capable)


def _place_segment(
    grid: np.ndarray,
    color: int,
    top: int,
    must_reach_bottom: bool,
    occupied: set[tuple[int, int]],
) -> bool:
    options = _line_options_for_top(top, must_reach_bottom)
    random.shuffle(options)
    for dc, length in options:
        starts = _valid_starts(top, length, dc, occupied)
        if not starts:
            continue
        start_col = random.choice(starts)
        direction = (0, 1) if dc == 0 else (1, dc)
        draw_line(grid, top, start_col, length=length, direction=direction, color=color)
        occupied.update(_coords_for_line(top, start_col, length, dc))
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_lines = random.randint(*LINE_COUNT_BOUNDS)
    colors = random.sample(FG_COLORS, n_lines)
    expected = _build_expected(colors)

    design = _sample_top_rows_and_bottom(n_lines)
    if design is None:
        return None
    top_rows, bottom_idx = design

    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for i, (color, top) in enumerate(zip(colors, top_rows)):
        if not _place_segment(grid, color, top, i == bottom_idx, occupied):
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 668eec9a input")
