from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
MASK = 5
FG_CHOICES = [1, 2, 3, 6, 7, 8, 9]
GEN_RETRIES = 500


def _shift_row(row: np.ndarray, dx: int, bg: int = BG) -> np.ndarray:
    out = np.full(row.shape, int(bg), dtype=int)
    cols = np.flatnonzero(row != int(bg))
    dst = cols + int(dx)
    keep = (0 <= dst) & (dst < row.size)
    out[dst[keep]] = row[cols[keep]]
    return out


def _candidate_period_shifts(rows: np.ndarray, periods: tuple[int, ...] = (2, 3)) -> list[tuple[int, int]]:
    rows = np.asarray(rows, dtype=int)
    height, width = map(int, rows.shape)
    candidates: list[tuple[int, int]] = []
    for period in periods:
        if period >= height:
            continue
        for dx in range(-(width - 1), width):
            if all(np.array_equal(_shift_row(rows[r], dx), rows[r + period]) for r in range(height - period)):
                candidates.append((int(period), int(dx)))
    return candidates


def _find_period_shift(rows: np.ndarray) -> tuple[int, int]:
    by_period: dict[int, set[int]] = {}
    for period, dx in _candidate_period_shifts(rows):
        by_period.setdefault(int(period), set()).add(int(dx))

    for period in (2, 3):
        if period not in by_period:
            continue
        dxs = sorted(by_period[period])
        assert len(dxs) == 1, "expected a unique horizontal shift for the minimal valid period"
        return int(period), int(dxs[0])

    raise AssertionError("expected a unique translated periodic pattern on the trusted side")


def _layout(width: int, axis: int, source_side: str) -> tuple[list[int], list[int], list[int]]:
    if source_side == "left":
        source_cols = list(range(axis))
        pair_count = min(len(source_cols), width - axis - 1)
        paired_source = list(range(axis - 1, axis - pair_count - 1, -1))
        target_cols = list(range(axis + 1, axis + pair_count + 1))
    else:
        source_cols = list(range(axis + 1, width))
        pair_count = min(len(source_cols), axis)
        paired_source = list(range(axis + 1, axis + pair_count + 1))
        target_cols = list(range(axis - 1, axis - pair_count - 1, -1))
    return source_cols, paired_source, target_cols


def _find_axis_and_source(grid: np.ndarray) -> tuple[int, str]:
    height, _width = map(int, grid.shape)
    axis_cols = np.flatnonzero(np.all(grid == YELLOW, axis=0))
    assert len(axis_cols) == 1, "expected exactly one full-height yellow axis"
    axis = int(axis_cols[0])

    extras = [(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW) if int(c) != axis]
    assert len(extras) == 1, "expected exactly one off-axis yellow control pixel"
    _r, control_c = extras[0]
    source_side = "left" if control_c < axis else "right"

    source_cols, _paired_source, target_cols = _layout(grid.shape[1], axis, source_side)
    assert source_cols and target_cols and len(source_cols) >= len(target_cols)
    assert height >= 1
    return axis, source_side


def _repair_top_rows(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, str]:
    g = np.asarray(grid, dtype=int)
    axis, source_side = _find_axis_and_source(g)
    source_cols, paired_source, target_cols = _layout(g.shape[1], axis, source_side)

    repaired = g.copy()
    repaired[repaired == YELLOW] = BG
    for src_c, dst_c in zip(paired_source, target_cols):
        repaired[:, dst_c] = repaired[:, src_c]

    source_rows = repaired[:, source_cols]
    return repaired, source_rows, axis, source_side


def _render_from_source_rows(source_rows: np.ndarray, width: int, axis: int, source_side: str) -> np.ndarray:
    source_rows = np.asarray(source_rows, dtype=int)
    source_cols, paired_source, target_cols = _layout(width, axis, source_side)

    out = np.full((source_rows.shape[0], width), BG, dtype=int)
    out[:, source_cols] = source_rows

    src_pos = {col: i for i, col in enumerate(source_cols)}
    for src_c, dst_c in zip(paired_source, target_cols):
        out[:, dst_c] = source_rows[:, src_pos[src_c]]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the damaged mirrored side, then continue the trusted half-row drift down to a 10x10 canvas.

    Input structure:
    - The input is a width-10 grid of height 4-6 on a black background.
    - A full-height yellow vertical axis and one off-axis yellow control pixel
      mark a left or right trusted source side.
    - One non-black foreground color forms a row pattern on the trusted side.
    - The opposite side is the mirror target; some of its foreground pixels are
      replaced by gray mask cells.

    Rule:
    - Use the yellow control pixel to choose the trusted source side.
    - Remove the yellow markers to background, then mirror the source columns
      that have partners across the axis onto the target side, overwriting any
      gray masks and yielding repaired top rows.
    - On the repaired trusted-side half-rows, infer the minimal valid vertical
      period 2 or 3 and its unique horizontal shift.
    - Copy the repaired rows to the top of a 10x10 black canvas. For each
      missing lower row, shift the trusted half-row from one period above by the
      inferred offset, then mirror it back onto the target side with the same axis.

    Key insight:
    - The yellow marker identifies which half is reliable. Only after trusting
      that half and repairing its mirror can the row-drift rule be continued
      consistently to the missing lower rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width == 10 and 4 <= height <= 6, "expected a 4-6 by 10 input grid"

    repaired, source_rows, axis, source_side = _repair_top_rows(g)
    period, dx = _find_period_shift(source_rows)

    out_source = np.full((10, source_rows.shape[1]), BG, dtype=int)
    out_source[:height] = source_rows
    for row in range(height, 10):
        out_source[row] = _shift_row(out_source[row - period], dx)

    out = _render_from_source_rows(out_source, width, axis, source_side)
    out[:height] = repaired
    return out


def _gen_base_rows(period: int, dx: int, color: int) -> np.ndarray:
    width = 5
    base = np.full((period, width), BG, dtype=int)
    counts = [random.randint(1, 3) for _ in range(period)]

    survivor = random.randrange(period)
    for row in range(period):
        cols: set[int] = set()
        want = counts[row]

        if dx > 0:
            cols.add(random.randint(0, max(0, width - 1 - dx)))
            if row == survivor:
                cols.add(random.randint(0, max(0, width - 1 - dx * 2)))
        else:
            cols.add(random.randint(min(width - 1, -dx), width - 1))
            if row == survivor:
                cols.add(random.randint(min(width - 1, -dx * 2), width - 1))

        pool = list(range(width))
        random.shuffle(pool)
        for col in pool:
            cols.add(col)
            if len(cols) >= want:
                break

        for col in list(sorted(cols))[:want]:
            base[row, col] = int(color)

    return base


def _build_source_output(base: np.ndarray, dx: int) -> np.ndarray:
    period = int(base.shape[0])
    out = np.full((10, base.shape[1]), BG, dtype=int)
    out[:period] = base
    for row in range(period, 10):
        out[row] = _shift_row(out[row - period], dx)
    return out


def _gen_attempt() -> np.ndarray | None:
    period = random.choice([2, 3])
    dx = random.choice([-1, 1] if period == 2 else [-2, -1, 1, 2])
    height = random.randint(max(4, period + 2), 6)
    axis, source_side = random.choice([(4, "right"), (5, "left")])
    color = random.choice(FG_CHOICES)

    for _ in range(200):
        base = _gen_base_rows(period, dx, color)
        source_out = _build_source_output(base, dx)
        top_source = source_out[:height]

        try:
            actual_period, actual_dx = _find_period_shift(top_source)
        except Exception:
            continue
        if (actual_period, actual_dx) != (period, dx):
            continue

        if len({tuple(row.tolist()) for row in top_source}) < 2:
            continue
        if not np.any(source_out[height:] != BG):
            continue

        repaired_top = _render_from_source_rows(top_source, 10, axis, source_side)
        if _candidate_period_shifts(repaired_top) == [(period, dx)]:
            continue

        _source_cols, _paired_source, target_cols = _layout(10, axis, source_side)
        target_fg = [(r, c) for r in range(height) for c in target_cols if int(repaired_top[r, c]) == color]
        if not target_fg:
            continue

        control_col = axis - 2 if source_side == "left" else axis + 2
        control_rows = [r for r in range(height) if int(repaired_top[r, control_col]) == BG]
        if not control_rows:
            continue

        inp = repaired_top.copy()
        mask_count = random.randint(1, min(3, len(target_fg)))
        for r, c in random.sample(target_fg, mask_count):
            inp[r, c] = MASK
        inp[:, axis] = YELLOW
        inp[random.choice(control_rows), control_col] = YELLOW

        stripped = inp.copy()
        stripped[stripped == YELLOW] = BG
        if _candidate_period_shifts(stripped) == [(period, dx)]:
            continue

        expected = _render_from_source_rows(source_out, 10, axis, source_side)
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue

        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
