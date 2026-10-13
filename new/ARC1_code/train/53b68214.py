from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _shift_row(row: np.ndarray, dc: int, bg: int) -> np.ndarray:
    out = np.full(row.shape, int(bg), dtype=int)
    cols = np.flatnonzero(row != int(bg))
    dst_cols = cols + int(dc)
    keep = (0 <= dst_cols) & (dst_cols < row.size)
    out[dst_cols[keep]] = row[cols[keep]]
    return out


def _find_period_and_shift(g: np.ndarray, bg: int) -> tuple[int, int]:
    height, width = map(int, g.shape)
    for period in range(1, height):
        shifts = []
        for dc in range(-(width - 1), width):
            if all(
                np.array_equal(_shift_row(g[row], dc, bg), g[row + period])
                for row in range(height - period)
            ):
                shifts.append(int(dc))
        if shifts:
            assert len(shifts) == 1, "expected a unique horizontal shift for the minimal row period"
            return int(period), shifts[0]
    raise AssertionError("expected a translated periodic row pattern")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the top-aligned foreground row pattern downward to a ten-row grid.

    Input structure:
    - The input is a width-10 grid with height below 10.
    - Background is the dominant color, and the foreground uses one color.
    - Rows form a translated periodic sequence: after a fixed vertical period,
      the row mask reappears shifted horizontally by a fixed offset.

    Rule:
    - Copy the input rows into the top of a 10-row background canvas.
    - Infer the minimal vertical period and its unique horizontal shift from the
      observed rows.
    - Continue each missing row from the row one period above it, applying the
      same horizontal shift and clipping cells that move outside the grid.

    Key insight:
    - The partial drawing is the top of a periodic row sequence, so the missing
      bottom rows are determined by the matching earlier row in the period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 1 < height < 10 and width == 10, "expected a partial 10-column grid"

    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 1, "expected exactly one foreground color"

    period, dc = _find_period_and_shift(g, bg)

    out = np.full((10, width), int(bg), dtype=int)
    out[:height, :] = g
    for row in range(height, 10):
        out[row] = _shift_row(out[row - period], dc, bg)
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_BG = 0
_GEN_WIDTH = 10
_GEN_OUT_HEIGHT = 10
_GEN_COLOR_MIN, _GEN_COLOR_MAX = 1, 9

# Period, horizontal drift, and visible input height.
_GEN_PERIOD_MIN, _GEN_PERIOD_MAX = 1, 3
_GEN_SHIFT_MIN, _GEN_SHIFT_MAX = 0, 2
_GEN_INPUT_HEIGHT_MIN, _GEN_INPUT_HEIGHT_MAX = 5, 8

# Sparse base-row foreground counts.
_GEN_PERIOD1_COUNT_MIN, _GEN_PERIOD1_COUNT_MAX = 1, 4
_GEN_PERIOD2_TOTAL_MIN, _GEN_PERIOD2_TOTAL_MAX = 3, 5
_GEN_PERIOD3_FIRST_COUNT = 1
_GEN_PERIOD3_THIRD_CHOICES = (2, 3)
_GEN_BASE_TOTAL_MAX = 5

# Column placement and retry behavior.
_GEN_ANCHOR_SHIFT_CAP = 2
_GEN_CONTIGUOUS_RUN_PROB = 0.45
_GEN_MAX_EXTRA_SPAN = 3
_GEN_MAX_ATTEMPTS = 300


def _gen_shift_row(row: np.ndarray, dc: int) -> np.ndarray:
    out = np.full(row.shape, _GEN_BG, dtype=int)
    cols = np.flatnonzero(row != _GEN_BG)
    dst_cols = cols + int(dc)
    keep = (0 <= dst_cols) & (dst_cols < row.size)
    out[dst_cols[keep]] = row[cols[keep]]
    return out


def _gen_base_counts(period: int) -> list[int]:
    if period == 1:
        return [random.randint(_GEN_PERIOD1_COUNT_MIN, _GEN_PERIOD1_COUNT_MAX)]

    if period == 2:
        total = random.randint(_GEN_PERIOD2_TOTAL_MIN, _GEN_PERIOD2_TOTAL_MAX)
        first = random.randint(1, total - 1)
        if first * 2 == total:
            first = 1
        if random.choice([False, True]):
            first = total - first
        return [first, total - first]

    third = random.choice(_GEN_PERIOD3_THIRD_CHOICES)
    second_max = _GEN_BASE_TOTAL_MAX - _GEN_PERIOD3_FIRST_COUNT - third
    second = random.randint(1, second_max)
    return [_GEN_PERIOD3_FIRST_COUNT, second, third]


def _gen_column_pool(dc: int, period: int, row_index: int) -> tuple[int, int, int | None]:
    """Return inclusive column bounds plus an optional required anchor column."""
    if dc == 0:
        return 0, _GEN_WIDTH - 1, None

    max_shifts = (_GEN_OUT_HEIGHT - 1 - row_index) // period
    anchor_shifts = min(_GEN_ANCHOR_SHIFT_CAP, max(1, max_shifts))
    anchor_max = max(0, _GEN_WIDTH - 1 - dc * anchor_shifts)
    required = random.randint(0, anchor_max)
    pool_max = min(_GEN_WIDTH - 1, max(anchor_max, _GEN_WIDTH - 1 - dc))
    return 0, pool_max, required


def _gen_window_start(pool_min: int, pool_max: int, size: int, required: int | None) -> int:
    if required is None:
        return random.randint(pool_min, pool_max - size + 1)

    lo = max(pool_min, required - size + 1)
    hi = min(required, pool_max - size + 1)
    return random.randint(lo, hi)


def _gen_pick_columns(count: int, dc: int, period: int, row_index: int) -> list[int]:
    pool_min, pool_max, required = _gen_column_pool(dc, period, row_index)

    if count == 1:
        if required is not None:
            return [required]
        return [random.randint(pool_min, pool_max)]

    if random.random() < _GEN_CONTIGUOUS_RUN_PROB:
        start = _gen_window_start(pool_min, pool_max, count, required)
        return list(range(start, start + count))

    pool_size = pool_max - pool_min + 1
    span = count + random.randint(0, min(_GEN_MAX_EXTRA_SPAN, pool_size - count))
    start = _gen_window_start(pool_min, pool_max, span, required)

    window = list(range(start, start + span))
    cols = set() if required is None else {required}
    cols.update(random.sample([col for col in window if col not in cols], count - len(cols)))
    return sorted(cols)


def _gen_make_base_rows(period: int, dc: int, color: int) -> np.ndarray:
    base = np.full((period, _GEN_WIDTH), _GEN_BG, dtype=int)
    for row_index, count in enumerate(_gen_base_counts(period)):
        cols = _gen_pick_columns(count, dc, period, row_index)
        base[row_index, cols] = color
    return base


def _gen_build_output(base: np.ndarray, dc: int) -> np.ndarray:
    period = int(base.shape[0])
    out = np.full((_GEN_OUT_HEIGHT, _GEN_WIDTH), _GEN_BG, dtype=int)
    out[:period] = base
    for row in range(period, _GEN_OUT_HEIGHT):
        out[row] = _gen_shift_row(out[row - period], dc)
    return out


def _gen_attempt() -> np.ndarray | None:
    color = random.randint(_GEN_COLOR_MIN, _GEN_COLOR_MAX)
    period = random.randint(_GEN_PERIOD_MIN, _GEN_PERIOD_MAX)
    dc = random.randint(_GEN_SHIFT_MIN, _GEN_SHIFT_MAX)

    base = _gen_make_base_rows(period, dc, color)
    out = _gen_build_output(base, dc)

    min_height = max(2 * period + 1, _GEN_INPUT_HEIGHT_MIN)
    input_height = random.randint(min_height, _GEN_INPUT_HEIGHT_MAX)
    inp = out[:input_height].copy()

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
