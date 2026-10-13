from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TEAL = 8

MARKER_COUNT_RANGE = (3, 5)
WIDTH_OPTIONS = (3, 5, 7, 9, 11)
GRID_HEIGHT_RANGE = (11, 19)
GRID_WIDTH_RANGE = (15, 24)
MARKER_GAP_RANGE = (1, 3)
NON_MARKER_RED_PROB_RANGE = (0.05, 0.22)
MARKER_OUTSIDE_RED_PROB_RANGE = (0.0, 0.18)
MAX_GENERATION_ATTEMPTS = 400


def _random_binary_row(width: int, red_prob: float) -> np.ndarray:
    return (np.random.random(width) < red_prob).astype(int) * RED


def _marked_rows(grid: np.ndarray) -> list[tuple[int, np.ndarray, int]]:
    rows: list[tuple[int, np.ndarray, int]] = []
    for r in range(grid.shape[0]):
        blue_cols = np.flatnonzero(grid[r] == BLUE)
        teal_cols = np.flatnonzero(grid[r] == TEAL)
        if blue_cols.size == 0 and teal_cols.size == 0:
            continue

        assert blue_cols.size == 1 and teal_cols.size == 1, "each marked row must contain one blue and one teal marker"
        blue_col = int(blue_cols[0])
        teal_col = int(teal_cols[0])
        assert blue_col < teal_col, "blue marker must be left of teal marker"

        segment = grid[r, blue_col + 1 : teal_col]
        assert segment.size > 0, "markers must bracket at least one cell"
        assert np.all((segment == BLACK) | (segment == RED)), "bracketed segments may only contain black and red"
        rows.append((int(np.count_nonzero(segment == RED)), segment.copy(), r))

    assert rows, "expected at least one marked row"
    counts = [count for count, _segment, _r in rows]
    assert len(set(counts)) == len(counts), "red counts must be distinct"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract bracketed red/black row segments, sort them by red count, and center-stack them into a widening wedge.

    Input structure:
    - The grid has a black background with a few marked rows containing exactly one blue pixel on the left and one teal pixel on the right.
    - Each marked row brackets a horizontal segment made only of black and red cells.
    - Different marked rows may appear at different input rows and use different odd segment widths.

    Rule:
    - Find every row that contains one blue marker and one teal marker, then extract the cells strictly between them.
    - Count the red pixels inside each extracted segment and sort the segments from smallest count to largest count.
    - Create a black output whose height is the number of segments and whose width is the widest segment, then place each sorted segment centered on its output row.

    Key insight:
    - The markers only delimit reusable row segments; once extracted, sorting by each segment's red mass reorders narrow-to-wide rows so the centered stack becomes a natural wedge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    ordered = sorted(_marked_rows(g), key=lambda item: item[0])
    max_width = max(segment.size for _count, segment, _r in ordered)
    out = np.full((len(ordered), max_width), BLACK, dtype=int)

    for out_r, (_count, segment, _src_r) in enumerate(ordered):
        pad = max_width - int(segment.size)
        left_pad = pad // 2
        out[out_r, left_pad : left_pad + segment.size] = segment

    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _sample_marker_rows(height: int, count: int) -> list[int]:
    assert height >= count
    while True:
        rows = sorted(random.sample(range(height), count))
        gaps = [b - a for a, b in zip(rows, rows[1:])]
        if all(MARKER_GAP_RANGE[0] <= gap <= MARKER_GAP_RANGE[1] for gap in gaps):
            return rows


def _sample_red_counts(widths: list[int]) -> list[int]:
    counts = [0] * len(widths)
    next_count = None
    for i in range(len(widths) - 1, -1, -1):
        lo = i + 1
        hi = widths[i] - 1
        if next_count is not None:
            hi = min(hi, next_count - 1)
        if lo > hi:
            raise AssertionError("unable to assign strictly increasing red counts")
        counts[i] = _randint(lo, hi)
        next_count = counts[i]
    return counts


def _build_segment(width: int, red_count: int) -> np.ndarray:
    segment = np.full(width, BLACK, dtype=int)
    red_cols = np.random.choice(width, size=red_count, replace=False)
    segment[red_cols] = RED
    return segment


def _expected_output(segments: list[np.ndarray]) -> np.ndarray:
    max_width = max(segment.size for segment in segments)
    out = np.full((len(segments), max_width), BLACK, dtype=int)
    for r, segment in enumerate(segments):
        left_pad = (max_width - segment.size) // 2
        out[r, left_pad : left_pad + segment.size] = segment
    return out


def _gen_attempt() -> np.ndarray | None:
    marker_count = _randint(*MARKER_COUNT_RANGE)
    widths = sorted(random.sample(WIDTH_OPTIONS, marker_count))
    red_counts = _sample_red_counts(widths)
    ordered_segments = [_build_segment(width, red_count) for width, red_count in zip(widths, red_counts)]
    expected = _expected_output(ordered_segments)

    max_width = widths[-1]
    min_grid_width = max(GRID_WIDTH_RANGE[0], max_width + 4)
    max_grid_width = GRID_WIDTH_RANGE[1]
    if min_grid_width > max_grid_width:
        return None

    grid_height = _randint(max(marker_count + 4, GRID_HEIGHT_RANGE[0]), GRID_HEIGHT_RANGE[1])
    grid_width = _randint(min_grid_width, max_grid_width)
    grid = np.full((grid_height, grid_width), BLACK, dtype=int)

    base_red_prob = random.uniform(*NON_MARKER_RED_PROB_RANGE)
    for r in range(grid_height):
        grid[r] = _random_binary_row(grid_width, base_red_prob)

    marker_rows = _sample_marker_rows(grid_height, marker_count)
    segment_order = list(range(marker_count))
    random.shuffle(segment_order)

    max_blue_col = min(7, grid_width - max_width - 2)
    if max_blue_col < 0:
        return None
    blue_choices = list(range(max_blue_col + 1))

    blue_cols = [int(random.choice(blue_choices)) for _ in range(marker_count)]
    if len(set(blue_cols)) == 1 and len(blue_choices) > 1:
        idx = random.randrange(marker_count)
        alternatives = [value for value in blue_choices if value != blue_cols[idx]]
        blue_cols[idx] = int(random.choice(alternatives))

    for row_index, blue_col, segment_idx in zip(marker_rows, blue_cols, segment_order):
        segment = ordered_segments[segment_idx]
        teal_col = blue_col + 1 + segment.size
        if teal_col >= grid_width:
            return None
        row = _random_binary_row(grid_width, random.uniform(*MARKER_OUTSIDE_RED_PROB_RANGE))
        row[blue_col] = BLUE
        row[blue_col + 1 : teal_col] = segment
        row[teal_col] = TEAL
        grid[row_index] = row

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if solved.shape[1] != max_width:
        return None
    if not np.all(np.diff([np.count_nonzero(seg == RED) for seg in ordered_segments]) > 0):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
