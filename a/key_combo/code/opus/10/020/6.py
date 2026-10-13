from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MARKER_COLORS = {YELLOW, ORANGE}

MIN_HEIGHT = 6
MAX_HEIGHT = 9
MIN_WIDTH = 10
MAX_WIDTH = 16
GEN_ATTEMPTS = 400


def _extract_row_features(row: np.ndarray) -> tuple[int, int, int, int]:
    left_marker = int(row[0])
    right_marker = int(row[-1])
    assert left_marker in MARKER_COLORS and right_marker in MARKER_COLORS, "row markers must be yellow or orange"

    inner = np.asarray(row[1:-1], dtype=int)
    assert inner.size >= 3, "row interior must have room for blue, green, and red"
    assert set(map(int, np.unique(inner))) <= {BLUE, RED, GREEN}, "interior may only use blue, red, and green"

    green_cols = np.flatnonzero(inner == GREEN)
    assert green_cols.size > 0, "each row must contain one green run"
    start_inner = int(green_cols[0])
    end_inner = int(green_cols[-1])
    assert green_cols.size == end_inner - start_inner + 1, "green cells must form a single contiguous run"

    start = start_inner + 1
    end = end_inner + 1
    assert start >= 2, "input rows must keep at least one blue cell before the green run"
    assert end <= row.size - 3, "input rows must keep at least one red cell after the green run"
    assert np.all(inner[:start_inner] == BLUE), "cells before the green run must be blue"
    assert np.all(inner[end_inner + 1 :] == RED), "cells after the green run must be red"
    return left_marker, right_marker, start, end


def _build_aligned_inner(width: int, anchor_end: int, total_width: int) -> np.ndarray:
    inner_width = total_width - 2
    assert 1 <= anchor_end <= total_width - 2
    new_start = max(1, anchor_end - width + 1)
    new_end = new_start + width - 1
    assert 1 <= new_start <= new_end <= total_width - 2

    row = np.empty(inner_width, dtype=int)
    row[: new_start - 1] = BLUE
    row[new_start - 1 : new_end] = GREEN
    row[new_end:] = RED
    return row


def _solve_row(row: np.ndarray, anchor_end: int) -> np.ndarray:
    left_marker, right_marker, start, end = _extract_row_features(row)
    width = end - start + 1

    out = np.empty_like(row)
    out[0] = left_marker
    out[-1] = right_marker
    out[1:-1] = _build_aligned_inner(width, anchor_end, row.size)
    if left_marker != right_marker:
        out[1:-1] = out[1:-1][::-1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Right-align every row's green run to the shared leftmost right edge, then mirror only the rows whose edge markers disagree.

    Input structure:
    - Each row has a yellow/orange marker in column 0 and another in the last
      column.
    - Between the markers, every row is exactly a blue prefix, one contiguous
      green run, and a red suffix.
    - Different rows place the green run at different horizontal positions, and
      the marker pair may match or mismatch.

    Rule:
    - Find the minimum original right-edge column among all green runs.
    - Rebuild every row so its green run ends at that column, clamping the run
      start to column 1 if necessary; cells before the run become blue and cells
      after it become red.
    - Keep marker pixels fixed.
    - After the alignment, horizontally mirror columns 1..W-2 only for rows
      whose two marker colors differ.

    Key insight:
    - The markers do not locate the run; they are a post-alignment switch that
      decides whether the already realigned row interior stays as-is or flips.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 5 <= g.shape[0] <= 30 and 6 <= g.shape[1] <= 30, "grid shape must stay within ARC limits"

    row_features = [_extract_row_features(g[r]) for r in range(g.shape[0])]
    anchor_end = min(end for _lm, _rm, _start, end in row_features)

    out = np.empty_like(g)
    for r in range(g.shape[0]):
        out[r] = _solve_row(g[r], anchor_end)
    return out


def _make_input_row(width: int, shift: int, same_markers: bool, total_width: int, anchor_end: int) -> np.ndarray:
    left_marker = random.choice([YELLOW, ORANGE])
    if same_markers:
        right_marker = left_marker
    else:
        right_marker = ORANGE if left_marker == YELLOW else YELLOW

    end = anchor_end + shift
    start = end - width + 1
    assert 2 <= start <= end <= total_width - 3

    row = np.empty(total_width, dtype=int)
    row[0] = left_marker
    row[-1] = right_marker
    row[1:start] = BLUE
    row[start : end + 1] = GREEN
    row[end + 1 : -1] = RED
    return row


def _shift_bounds(width: int, total_width: int, anchor_end: int, require_shift: bool) -> tuple[int, int] | None:
    min_shift = max(0, width - anchor_end + 1)
    if require_shift:
        min_shift = max(min_shift, 1)
    max_shift = (total_width - 3) - anchor_end
    if min_shift > max_shift:
        return None
    return min_shift, max_shift


def _choose_distinct(values: list[int], avoid: set[int]) -> int:
    candidates = [v for v in values if v not in avoid]
    if candidates:
        return random.choice(candidates)
    return random.choice(values)


def _baseline_without_alignment(grid: np.ndarray) -> np.ndarray:
    baseline = np.asarray(grid, dtype=int).copy()
    for r in range(baseline.shape[0]):
        if int(baseline[r, 0]) != int(baseline[r, -1]):
            baseline[r, 1:-1] = baseline[r, 1:-1][::-1]
    return baseline


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if grid.shape != out.shape or np.array_equal(grid, out):
        return False

    features = [_extract_row_features(grid[r]) for r in range(grid.shape[0])]
    anchor_end = min(end for _lm, _rm, _start, end in features)
    widths = {end - start + 1 for _lm, _rm, start, end in features}
    if len(widths) < 3:
        return False

    has_same = any(lm == rm for lm, rm, _start, _end in features)
    has_diff = any(lm != rm for lm, rm, _start, _end in features)
    if not (has_same and has_diff):
        return False

    shifted_rows = [(lm, rm, start, end) for lm, rm, start, end in features if end > anchor_end]
    if not shifted_rows:
        return False

    has_same_shifted = any(lm == rm for lm, rm, _start, _end in shifted_rows)
    has_diff_shifted = any(lm != rm for lm, rm, _start, _end in shifted_rows)
    if not (has_same_shifted and has_diff_shifted):
        return False

    has_clamp_row = any((end - start + 1) > anchor_end for _lm, _rm, start, end in features)
    if not has_clamp_row:
        return False

    baseline = _baseline_without_alignment(grid)
    dependency_visible = any(
        int(grid[r, 0]) != int(grid[r, -1]) and not np.array_equal(out[r], baseline[r])
        for r in range(grid.shape[0])
    )
    if not dependency_visible:
        return False

    # Prevent the generated examples from collapsing into a tiny set of output rows.
    unique_output_rows = {tuple(map(int, row)) for row in out}
    return len(unique_output_rows) >= 3


def _generate_rows(total_height: int, total_width: int, anchor_end: int) -> list[np.ndarray] | None:
    nonclamp_widths = list(range(2, anchor_end))
    clamp_widths = list(range(anchor_end + 1, total_width - 3))
    if not nonclamp_widths or not clamp_widths:
        return None

    used_widths: set[int] = set()

    width_anchor = _choose_distinct(nonclamp_widths, used_widths)
    used_widths.add(width_anchor)

    width_diff_shift = _choose_distinct(nonclamp_widths, used_widths)
    used_widths.add(width_diff_shift)

    width_same_shift = _choose_distinct(nonclamp_widths, used_widths)
    used_widths.add(width_same_shift)

    width_clamp = _choose_distinct(clamp_widths, used_widths)
    used_widths.add(width_clamp)

    specs: list[tuple[int, bool, bool]] = [
        (width_anchor, False, True),
        (width_diff_shift, True, False),
        (width_same_shift, True, True),
        (width_clamp, True, False),
    ]

    while len(specs) < total_height:
        branch_same = len([1 for _w, _req, same in specs if same]) <= len(specs) // 2
        pool = nonclamp_widths + clamp_widths
        width = _choose_distinct(pool, set() if len(used_widths) >= 4 else used_widths)
        used_widths.add(width)
        specs.append((width, True, branch_same if random.random() < 0.6 else not branch_same))

    rows: list[np.ndarray] = []
    for width, require_shift, same_markers in specs:
        bounds = _shift_bounds(width, total_width, anchor_end, require_shift)
        if bounds is None:
            return None
        min_shift, max_shift = bounds
        shift = 0 if not require_shift else random.randint(min_shift, max_shift)
        rows.append(_make_input_row(width, shift, same_markers, total_width, anchor_end))

    random.shuffle(rows)
    return rows


def _gen_attempt() -> np.ndarray | None:
    total_height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    total_width = random.randint(MIN_WIDTH, MAX_WIDTH)
    anchor_end = random.randint(4, total_width - 5)

    rows = _generate_rows(total_height, total_width, anchor_end)
    if rows is None:
        return None

    grid = np.vstack(rows)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
