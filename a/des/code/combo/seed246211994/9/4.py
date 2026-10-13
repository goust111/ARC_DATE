from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _base_and_marker(values: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, values))
    max_count = max(counts.values())
    bases = [color for color, count in counts.items() if count == max_count]
    assert len(bases) == 1, "expected a unique base color"
    base = int(bases[0])
    markers = [color for color, count in counts.items() if color != base and count == 2]
    assert len(markers) == 1, "expected exactly one pair marker color"
    return base, int(markers[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Build the woven rectangle determined by the marker pairs on two crossed bars."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fg = g != bg

    h_rows = np.flatnonzero(fg.sum(axis=1) > 1)
    v_cols = np.flatnonzero(fg.sum(axis=0) > 1)
    assert h_rows.size == 1, "expected exactly one horizontal bar row"
    assert v_cols.size == 1, "expected exactly one vertical bar column"
    h_row = int(h_rows[0])
    v_col = int(v_cols[0])

    h_cols = np.flatnonzero(g[h_row] != bg)
    v_rows = np.flatnonzero(g[:, v_col] != bg)
    assert h_cols.size >= 3
    assert v_rows.size >= 3
    assert np.array_equal(h_cols, np.arange(h_cols[0], h_cols[-1] + 1))
    assert np.array_equal(v_rows, np.arange(v_rows[0], v_rows[-1] + 1))

    h_base, h_marker = _base_and_marker(g[h_row, h_cols])
    v_base, v_marker = _base_and_marker(g[v_rows, v_col])

    h_marker_cols = h_cols[g[h_row, h_cols] == h_marker]
    v_marker_rows = v_rows[g[v_rows, v_col] == v_marker]
    assert h_marker_cols.size == 2
    assert v_marker_rows.size == 2
    left_col = int(h_marker_cols.min())
    right_col = int(h_marker_cols.max())

    if np.all(v_marker_rows > h_row):
        direction = 1
        near_row = int(v_marker_rows.min())
        far_row = int(v_marker_rows.max())
    elif np.all(v_marker_rows < h_row):
        direction = -1
        near_row = int(v_marker_rows.max())
        far_row = int(v_marker_rows.min())
    else:
        raise AssertionError("vertical markers must lie on one side of the horizontal bar")

    cross_color = int(g[h_row, v_col])
    assert cross_color in (h_base, v_base), "crossing color must be one of the base colors"

    near_step = abs(near_row - h_row)
    far_step = abs(far_row - h_row)
    out = np.full_like(g, bg)

    for step in range(far_step + 1):
        r = h_row + direction * step
        even_step = step % 2 == 0
        for offset, c in enumerate(range(left_col, right_col + 1)):
            even_offset = offset % 2 == 0
            if even_step and even_offset:
                out[r, c] = h_marker if step < near_step else v_marker
            elif even_step:
                out[r, c] = cross_color if step == near_step else v_base
            else:
                out[r, c] = h_base if even_offset else cross_color

    return out


BG_COLOR = 0
GRID_ROWS_RANGE = (10, 24)
GRID_COLS_RANGE = (10, 18)
MARKER_SPAN_RANGE = (2, 7)
FAR_STEP_RANGE = (4, 15)
NON_MARKER_SIDE_PAD = 2
H_BAR_MAX_TRIM = 2
V_BAR_MAX_TRIM = 4
GENERATION_ATTEMPTS = 300


def _sample_interval(limit: int, min_len: int, required: list[int]) -> tuple[int, int] | None:
    candidates = [
        (start, end)
        for start in range(limit)
        for end in range(start + min_len - 1, limit)
        if all(start <= value <= end for value in required)
    ]
    return random.choice(candidates) if candidates else None


def _sample_layout() -> tuple[int, ...] | None:
    height = random.randint(*GRID_ROWS_RANGE)
    width = random.randint(*GRID_COLS_RANGE)

    max_far = min(FAR_STEP_RANGE[1], height - (NON_MARKER_SIDE_PAD + 1))
    if max_far < FAR_STEP_RANGE[0]:
        return None

    far_step = random.randint(FAR_STEP_RANGE[0], max_far)
    near_step = random.randint(1, far_step - 1)

    directions = []
    if height - far_step - 1 >= NON_MARKER_SIDE_PAD:
        directions.append(1)
    if far_step <= height - (NON_MARKER_SIDE_PAD + 1):
        directions.append(-1)
    if not directions:
        return None
    direction = random.choice(directions)

    if direction == 1:
        h_row = random.randint(NON_MARKER_SIDE_PAD, height - far_step - 1)
    else:
        h_row = random.randint(far_step, height - (NON_MARKER_SIDE_PAD + 1))

    max_span = min(MARKER_SPAN_RANGE[1], width - 2)
    if max_span < MARKER_SPAN_RANGE[0]:
        return None
    marker_span = random.randint(MARKER_SPAN_RANGE[0], max_span)
    left_col = random.randint(0, width - marker_span - 1)
    right_col = left_col + marker_span

    v_choices = [col for col in range(width) if col not in (left_col, right_col)]
    if not v_choices:
        return None
    v_col = random.choice(v_choices)

    near_row = h_row + direction * near_step
    far_row = h_row + direction * far_step
    opposite_side_row = h_row - direction * NON_MARKER_SIDE_PAD

    h_interval = _sample_interval(width, max(6, width - H_BAR_MAX_TRIM), [left_col, right_col, v_col])
    v_interval = _sample_interval(
        height,
        max(6, height - V_BAR_MAX_TRIM),
        [h_row, near_row, far_row, opposite_side_row],
    )
    if h_interval is None or v_interval is None:
        return None

    return (
        height,
        width,
        h_row,
        v_col,
        left_col,
        right_col,
        near_step,
        far_step,
        direction,
        h_interval[0],
        h_interval[1],
        v_interval[0],
        v_interval[1],
    )


def _render_input(layout: tuple[int, ...], colors: tuple[int, int, int, int, int]) -> np.ndarray:
    (
        height,
        width,
        h_row,
        v_col,
        left_col,
        right_col,
        near_step,
        far_step,
        direction,
        h_start,
        h_end,
        v_start,
        v_end,
    ) = layout
    h_base, h_marker, v_base, v_marker, cross_color = colors
    near_row = h_row + direction * near_step
    far_row = h_row + direction * far_step

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[h_row, h_start : h_end + 1] = h_base
    grid[v_start : v_end + 1, v_col] = v_base
    grid[h_row, left_col] = h_marker
    grid[h_row, right_col] = h_marker
    grid[near_row, v_col] = v_marker
    grid[far_row, v_col] = v_marker
    grid[h_row, v_col] = cross_color
    return grid


def _render_output(layout: tuple[int, ...], colors: tuple[int, int, int, int, int]) -> np.ndarray:
    (
        height,
        width,
        h_row,
        _v_col,
        left_col,
        right_col,
        near_step,
        far_step,
        direction,
        _h_start,
        _h_end,
        _v_start,
        _v_end,
    ) = layout
    h_base, h_marker, v_base, v_marker, cross_color = colors

    out = np.full((height, width), BG_COLOR, dtype=int)
    for step in range(far_step + 1):
        r = h_row + direction * step
        even_step = step % 2 == 0
        for offset, c in enumerate(range(left_col, right_col + 1)):
            even_offset = offset % 2 == 0
            if even_step and even_offset:
                out[r, c] = h_marker if step < near_step else v_marker
            elif even_step:
                out[r, c] = cross_color if step == near_step else v_base
            else:
                out[r, c] = h_base if even_offset else cross_color
    return out


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    h_base, h_marker, v_base, v_marker = random.sample(range(1, 10), 4)
    cross_color = random.choice((h_base, v_base))
    colors = (h_base, h_marker, v_base, v_marker, cross_color)
    grid = _render_input(layout, colors)
    expected = _render_output(layout, colors)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
