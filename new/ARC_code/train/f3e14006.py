from collections import Counter

import numpy as np

from arc_utils import get_bg
import random


def _base_and_pair_marker(values: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, values))
    max_count = max(counts.values())
    bases = [color for color, n in counts.items() if n == max_count]
    assert len(bases) == 1
    base = bases[0]
    markers = [color for color, n in counts.items() if color != base and n == 2]
    assert len(markers) == 1
    return int(base), int(markers[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Build the woven rectangle determined by the two marked cells on each crossed bar.

    Input structure
    A black-background grid contains one long horizontal bar and one long vertical bar.
    Each bar has a dominant base color and exactly two same-colored marker pixels.

    Rule
    The horizontal marker pair gives the output column span.  From the horizontal bar
    row, walk toward the farther vertical marker; rows and columns alternate every
    other cell.  Even step/even offset cells use the horizontal marker color until
    the nearer vertical marker is reached, then use the vertical marker color.
    Odd row steps use the horizontal base on marker columns and the original
    crossing color on spacer columns.  Spacer columns on even row steps use the
    vertical base, except at the switch row where they use the crossing color.

    Key insight
    The original cross cell records which base color wins at base/base lattice
    crossings; the nearer vertical marker is the color-switch boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg = g != bg

    h_rows = np.flatnonzero(fg.sum(axis=1) > 1)
    v_cols = np.flatnonzero(fg.sum(axis=0) > 1)
    assert len(h_rows) == 1
    assert len(v_cols) == 1
    h_row = int(h_rows[0])
    v_col = int(v_cols[0])

    h_cols = np.flatnonzero(g[h_row] != bg)
    v_rows = np.flatnonzero(g[:, v_col] != bg)
    assert h_cols.size >= 2
    assert v_rows.size >= 2
    assert np.array_equal(h_cols, np.arange(h_cols[0], h_cols[-1] + 1))
    assert np.array_equal(v_rows, np.arange(v_rows[0], v_rows[-1] + 1))

    h_base, h_marker = _base_and_pair_marker(g[h_row, h_cols])
    v_base, v_marker = _base_and_pair_marker(g[v_rows, v_col])

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
    assert cross_color in (h_base, v_base)

    out = np.full_like(g, bg)
    near_step = abs(near_row - h_row)
    far_step = abs(far_row - h_row)

    for step in range(far_step + 1):
        r = h_row + direction * step
        for offset, c in enumerate(range(left_col, right_col + 1)):
            if offset % 2 == 0:
                if step % 2 == 0:
                    out[r, c] = h_marker if step < near_step else v_marker
                else:
                    out[r, c] = h_base
            elif step % 2 == 0:
                out[r, c] = cross_color if step == near_step else v_base
            else:
                out[r, c] = cross_color

    return out

# --------------- generator (o5x) ---------------

# Bounds and placement constraints.
BG_COLOR, GRID_ROWS_RANGE, GRID_COLS_RANGE = 0, (10, 25), (10, 15)
H_SPAN_RANGE, FAR_STEP_RANGE = (2, 6), (4, 15)
NON_MARKER_SIDE_PAD, H_BAR_MAX_TRIM, V_BAR_MAX_TRIM = 2, 2, 4
# Palette and retry policy.
COLOR_RANGE, COLOR_COUNT, GENERATION_RETRIES = (1, 10), 4, 300


def _gen_interval(limit: int, min_len: int, required: list[int]) -> tuple[int, int] | None:
    """Sample a contiguous interval that covers every required coordinate."""
    candidates = [
        (start, end)
        for start in range(limit)
        for end in range(start + min_len - 1, limit)
        if all(start <= x <= end for x in required)
    ]
    return random.choice(candidates) if candidates else None


def _sample_layout() -> tuple | None:
    """Sample grid geometry, marker positions, and bar intervals."""
    H = random.randint(*GRID_ROWS_RANGE)
    W = random.randint(*GRID_COLS_RANGE)

    max_far = min(FAR_STEP_RANGE[1], H - (NON_MARKER_SIDE_PAD + 1))
    if max_far < FAR_STEP_RANGE[0]:
        return None
    far_step = random.randint(FAR_STEP_RANGE[0], max_far)
    near_step = random.randint(1, far_step - 1)

    directions = []
    if H - far_step - 1 >= NON_MARKER_SIDE_PAD:
        directions.append(1)
    if far_step <= H - (NON_MARKER_SIDE_PAD + 1):
        directions.append(-1)
    if not directions:
        return None
    direction = random.choice(directions)
    if direction == 1:
        h_row = random.randint(NON_MARKER_SIDE_PAD, H - far_step - 1)
    else:
        h_row = random.randint(far_step, H - (NON_MARKER_SIDE_PAD + 1))

    h_span = random.randint(H_SPAN_RANGE[0], min(H_SPAN_RANGE[1], W - 1))
    left_col = random.randint(0, W - h_span - 1)
    right_col = left_col + h_span

    v_choices = [c for c in range(W) if c not in (left_col, right_col)]
    if not v_choices:
        return None
    v_col = random.choice(v_choices)

    near_row = h_row + direction * near_step
    far_row = h_row + direction * far_step
    non_marker_side_row = h_row - direction * NON_MARKER_SIDE_PAD

    h_interval = _gen_interval(W, W - H_BAR_MAX_TRIM, [left_col, right_col, v_col])
    v_interval = _gen_interval(H, H - V_BAR_MAX_TRIM, [h_row, near_row, far_row, non_marker_side_row])
    if h_interval is None or v_interval is None:
        return None

    return (H, W, h_row, v_col, left_col, right_col, near_step, far_step, direction, h_interval, v_interval)


def _render_grids(layout: tuple, colors: tuple[int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Render the sampled input grid and its expected woven output."""
    H, W, h_row, v_col, left_col, right_col, near_step, far_step, direction, h_interval, v_interval = layout
    h_base, h_marker, v_base, v_marker, cross_color = colors
    h_start, h_end = h_interval
    v_start, v_end = v_interval
    near_row = h_row + direction * near_step
    far_row = h_row + direction * far_step

    grid = np.full((H, W), BG_COLOR, dtype=int)
    grid[h_row, h_start : h_end + 1] = h_base
    grid[v_start : v_end + 1, v_col] = v_base
    grid[h_row, left_col] = h_marker
    grid[h_row, right_col] = h_marker
    grid[near_row, v_col] = v_marker
    grid[far_row, v_col] = v_marker
    grid[h_row, v_col] = cross_color

    out = np.full((H, W), BG_COLOR, dtype=int)
    for step in range(far_step + 1):
        r = h_row + direction * step
        even_step = step % 2 == 0
        for offset, c in enumerate(range(left_col, right_col + 1)):
            if even_step and offset % 2 == 0:
                out[r, c] = h_marker if step < near_step else v_marker
            elif even_step:
                out[r, c] = cross_color if step == near_step else v_base
            else:
                out[r, c] = h_base if offset % 2 == 0 else cross_color
    return grid, out


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    colors = random.sample(range(*COLOR_RANGE), COLOR_COUNT)
    h_base, _, v_base, _ = colors
    grid, expected = _render_grids(layout, (*colors, random.choice((h_base, v_base))))
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f3e14006 input")
