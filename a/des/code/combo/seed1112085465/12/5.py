from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
BAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_baseline(grid: np.ndarray) -> tuple[int, np.ndarray]:
    candidates: list[tuple[int, np.ndarray]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        grey_cols = np.flatnonzero(row == GREY)
        if len(grey_cols) < 2:
            continue
        if not np.all((row == BLACK) | (row == GREY)):
            continue
        diffs = np.diff(grey_cols)
        if len(set(map(int, diffs))) != 1:
            continue
        candidates.append((int(r), grey_cols.astype(int)))

    assert len(candidates) == 1, "expected exactly one grey baseline row"
    baseline_row, baseline_cols = candidates[0]
    assert int(np.count_nonzero(grid == GREY)) == len(baseline_cols), "grey may only appear on the baseline"
    return baseline_row, baseline_cols


def _nearest_baseline_col(center2: int, baseline_cols: np.ndarray) -> int | None:
    distances = np.abs(2 * baseline_cols - center2)
    best = int(np.min(distances))
    matches = baseline_cols[distances == best]
    if len(matches) != 1:
        return None
    return int(matches[0])


def _parse_bar_row(row: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    cols = np.flatnonzero(row != BLACK)
    if len(cols) == 0:
        return None

    c0 = int(cols[0])
    c1 = int(cols[-1])
    assert len(cols) == c1 - c0 + 1, "each fraction bar must be a single contiguous run"

    span = row[c0 : c1 + 1]
    assert np.all(span != BLACK)
    assert GREY not in set(map(int, span)), "grey is reserved for the baseline"

    changes = np.flatnonzero(span[1:] != span[:-1]) + 1
    assert len(changes) == 1, "each fraction bar must have exactly two color segments"

    split = int(changes[0])
    left_color = int(span[0])
    right_color = int(span[split])
    assert left_color != right_color
    assert np.all(span[:split] == left_color)
    assert np.all(span[split:] == right_color)

    left_len = split
    right_len = int(span.size - split)
    assert left_len > 0 and right_len > 0
    assert left_len != right_len, "shorter side must be unique"
    return c0, c1, left_color, right_color, left_len, right_len


def main(grid: np.ndarray) -> np.ndarray:
    """Turn two-color horizontal fraction bars into vertical columns showing the shorter segment."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(np.min(g)) >= 0
    assert BLACK in set(map(int, np.unique(g))), "black background must be present"

    baseline_row, baseline_cols = _find_baseline(g)

    bars: list[dict[str, int]] = []
    used_targets: set[int] = set()
    for r in range(g.shape[0]):
        if r == baseline_row:
            continue

        parsed = _parse_bar_row(g[r])
        if parsed is None:
            continue
        assert r < baseline_row, "all bars must lie above the baseline"

        c0, c1, left_color, right_color, left_len, right_len = parsed
        center2 = c0 + c1
        target_col = _nearest_baseline_col(center2, baseline_cols)
        assert target_col is not None, "bar midpoint must map to a unique nearest baseline pixel"
        assert target_col not in used_targets, "each baseline pixel may receive at most one output column"
        used_targets.add(target_col)

        if left_len < right_len:
            short_color = left_color
            short_len = left_len
        else:
            short_color = right_color
            short_len = right_len

        bars.append(
            {
                "row": int(r),
                "c0": c0,
                "c1": c1,
                "target_col": target_col,
                "short_color": short_color,
                "short_len": short_len,
            }
        )

    assert bars, "expected at least one fraction bar"

    out = g.copy()
    for bar in bars:
        out[bar["row"], bar["c0"] : bar["c1"] + 1] = BLACK

    for bar in bars:
        top = baseline_row - bar["short_len"]
        assert top >= 0
        segment = out[top:baseline_row, bar["target_col"]]
        assert np.all(segment == BLACK), "output columns may not overlap"
        out[top:baseline_row, bar["target_col"]] = bar["short_color"]

    return out


# --------------- generator ---------------

BAR_COUNT_RANGE = (2, 5)
STRIDE_RANGE = (3, 5)
HEIGHT_PADDING_RANGE = (3, 7)
GEN_RETRY_LIMIT = 300


def _sample_rows(height: int, count: int) -> list[int] | None:
    baseline_row = height - 1
    first_phase = random.randint(0, 1)
    phases = [first_phase, 1 - first_phase]
    for phase in phases:
        choices = list(range(phase, baseline_row - 1, 2))
        if len(choices) >= count:
            return sorted(random.sample(choices, count))
    for phase in (0, 1):
        choices = list(range(phase, baseline_row - 1, 2))
        if len(choices) >= count:
            return sorted(random.sample(choices, count))
    return None


def _choose_lengths(stride: int, height: int) -> tuple[int, int] | None:
    max_total = 2 * stride - 1
    max_short = min(5, stride - 1, height - 2)
    if max_short < 1:
        return None
    short_len = random.randint(1, max_short)
    long_hi = min(short_len + 4, max_total - short_len)
    if long_hi <= short_len:
        return None
    long_len = random.randint(short_len + 1, long_hi)
    return short_len, long_len


def _feasible_starts(target_col: int, total_len: int, width: int, baseline_cols: np.ndarray) -> list[int]:
    starts: list[int] = []
    for start in range(width - total_len + 1):
        center2 = 2 * start + total_len - 1
        if _nearest_baseline_col(center2, baseline_cols) == target_col:
            starts.append(start)
    return starts


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(*BAR_COUNT_RANGE)
    stride = random.randint(*STRIDE_RANGE)
    left_margin = random.randint(stride - 1, stride + 1)
    right_margin = random.randint(stride - 1, stride + 1)

    baseline_cols = np.array([left_margin + i * stride for i in range(bar_count)], dtype=int)
    width = int(baseline_cols[-1] + right_margin + 1)

    min_height = max(7, 2 * bar_count + 1)
    height = random.randint(min_height, min_height + random.randint(*HEIGHT_PADDING_RANGE))
    rows = _sample_rows(height, bar_count)
    if rows is None:
        return None

    baseline_row = height - 1
    inp = np.full((height, width), BLACK, dtype=int)
    out = np.full_like(inp, BLACK)
    inp[baseline_row, baseline_cols] = GREY
    out[baseline_row, baseline_cols] = GREY

    shuffled_targets = random.sample(list(map(int, baseline_cols)), bar_count)
    for row, target_col in zip(rows, shuffled_targets):
        lengths = _choose_lengths(stride, height)
        if lengths is None:
            return None
        short_len, long_len = lengths
        total_len = short_len + long_len

        starts = _feasible_starts(target_col, total_len, width, baseline_cols)
        if not starts:
            return None
        start = random.choice(starts)

        short_color, long_color = random.sample(BAR_COLORS, 2)
        short_on_left = bool(random.getrandbits(1))
        if short_on_left:
            left_len, right_len = short_len, long_len
            left_color, right_color = short_color, long_color
        else:
            left_len, right_len = long_len, short_len
            left_color, right_color = long_color, short_color

        inp[row, start : start + left_len] = left_color
        inp[row, start + left_len : start + left_len + right_len] = right_color
        out[baseline_row - short_len : baseline_row, target_col] = short_color

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
