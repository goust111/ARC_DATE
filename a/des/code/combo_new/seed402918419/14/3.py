from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_H_RANGE = (6, 18)
GRID_W_RANGE = (10, 30)
ACTIVE_ROWS_RANGE = (2, 6)
SIGNAL_ONLY_ROWS_MAX = 2
ARROW_ONLY_ROWS_MAX = 1
MAX_SIGNALS_PER_ROW = 6
GENERATION_ATTEMPTS = 300
ROW_ATTEMPTS = 80


def _contiguous_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    if cols.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    end = int(cols[0])
    for col in map(int, cols[1:]):
        if col == end + 1:
            end = col
            continue
        runs.append((start, end))
        start = end = col
    runs.append((start, end))
    return runs


def _has_vertical_red(g: np.ndarray, row: int, col: int) -> bool:
    h = g.shape[0]
    return (row > 0 and int(g[row - 1, col]) == RED) or (row + 1 < h and int(g[row + 1, col]) == RED)


def _row_direction(g: np.ndarray, row: int, green_cols: np.ndarray) -> str | None:
    red_cols = np.flatnonzero(g[row] == RED)
    if red_cols.size == 0:
        return None

    green_min = int(green_cols[0]) if green_cols.size else None
    green_max = int(green_cols[-1]) if green_cols.size else None
    directions: list[str] = []

    for start, end in _contiguous_runs(red_cols):
        length = end - start + 1

        if length == 3:
            left_stem = _has_vertical_red(g, row, start)
            right_stem = _has_vertical_red(g, row, end)
            if left_stem ^ right_stem:
                directions.append("right" if left_stem else "left")
                continue

            if green_cols.size:
                if end < green_min:
                    directions.append("right")
                    continue
                if start > green_max:
                    directions.append("left")
                    continue

        if length == 2:
            left_stem = _has_vertical_red(g, row, start)
            right_stem = _has_vertical_red(g, row, end)
            if left_stem ^ right_stem:
                directions.append("right" if left_stem else "left")

    if not directions:
        return None
    if len(set(directions)) != 1:
        return None
    return directions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a blue bar on each marked row, using the row's green-pixel count as the length and the red arrow as the direction.

    Input structure:
    - The background is black.
    - Rows may contain scattered green signal pixels and at most one short red arrow marker.
    - The arrow is a small red motif aligned to the same row as the signals and indicates whether growth goes left or right.

    Rule:
    - For each row, count the number N of green signal pixels in that row.
    - Detect whether that row's red arrow points left or right.
    - If the arrow points right, start immediately to the right of the row's rightmost green pixel and paint N blue cells, clipped by the grid boundary.
    - If the arrow points left, start immediately to the left of the row's leftmost green pixel and paint N blue cells, clipped by the grid boundary.
    - Keep all original red arrow pixels and green signal pixels unchanged; rows without arrows stay unchanged.

    Key insight:
    - The green pixels encode only the required length, while the red marker encodes only the direction, so the output row combines information from two separate cues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = g.copy()

    for row in range(h):
        green_cols = np.flatnonzero(g[row] == GREEN)
        direction = _row_direction(g, row, green_cols)
        if direction is None or green_cols.size == 0:
            continue

        n_signals = int(green_cols.size)
        if direction == "right":
            start = int(green_cols[-1]) + 1
            stop = min(w, start + n_signals)
            out[row, start:stop] = BLUE
        else:
            stop = int(green_cols[0])
            start = max(0, stop - n_signals)
            out[row, start:stop] = BLUE

    out[g == RED] = RED
    out[g == GREEN] = GREEN
    return out


def _sample_positions(start: int, span: int, count: int, forced: int | None = None) -> list[int]:
    cells = list(range(start, start + span))
    if forced is None:
        return sorted(random.sample(cells, count))
    assert forced in cells
    if count == 1:
        return [forced]
    cells.remove(forced)
    picks = random.sample(cells, count - 1)
    picks.append(forced)
    return sorted(picks)


def _gen_active_row(width: int, direction: str, truncated: bool) -> tuple[list[int], list[int]] | None:
    max_n = min(MAX_SIGNALS_PER_ROW, width - 4)
    if max_n < 1:
        return None

    for _ in range(ROW_ATTEMPTS):
        n_signals = random.randint(1, max_n)
        gap = random.randint(1, min(3, width - 4))
        max_span = min(width - 4 - gap, n_signals + 4)
        if max_span < n_signals:
            continue
        span = random.randint(n_signals, max_span)

        if direction == "right":
            max_space = width - (3 + gap + span)
            if max_space < 0:
                continue
            lo = 0 if truncated else n_signals
            hi = min(max_space, n_signals - 1 if truncated else n_signals + 4)
            if lo > hi:
                continue
            right_space = random.randint(lo, hi)
            left_margin = width - (3 + gap + span + right_space)
            arrow_cols = [left_margin + i for i in range(3)]
            signal_start = left_margin + 3 + gap
            signals = _sample_positions(signal_start, span, n_signals, forced=signal_start + span - 1)
            return arrow_cols, signals

        max_space = width - (span + gap + 3)
        if max_space < 0:
            continue
        lo = 0 if truncated else n_signals
        hi = min(max_space, n_signals - 1 if truncated else n_signals + 4)
        if lo > hi:
            continue
        left_space = random.randint(lo, hi)
        signal_start = left_space
        arrow_start = signal_start + span + gap
        arrow_cols = [arrow_start + i for i in range(3)]
        signals = _sample_positions(signal_start, span, n_signals, forced=signal_start)
        return arrow_cols, signals

    return None


def _gen_signals_only_row(width: int) -> list[int] | None:
    max_n = min(MAX_SIGNALS_PER_ROW, width)
    if max_n < 1:
        return None

    for _ in range(ROW_ATTEMPTS):
        n_signals = random.randint(1, max_n)
        span = random.randint(n_signals, min(width, n_signals + 5))
        start = random.randint(0, width - span)
        return _sample_positions(start, span, n_signals)

    return None


def _gen_arrow_only_row(width: int) -> list[int] | None:
    if width < 3:
        return None
    start = random.randint(0, width - 3)
    return [start, start + 1, start + 2]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)

    all_rows = list(range(height))
    max_active = min(ACTIVE_ROWS_RANGE[1], height)
    active_count = random.randint(ACTIVE_ROWS_RANGE[0], max_active)
    active_rows = random.sample(all_rows, active_count)

    directions = ["left", "right"] * (active_count // 2)
    if active_count % 2:
        directions.append(random.choice(["left", "right"]))
    random.shuffle(directions)

    truncated_flags = [True] * (active_count // 2) + [False] * (active_count - active_count // 2)
    random.shuffle(truncated_flags)
    if active_count >= 2 and all(truncated_flags):
        truncated_flags[0] = False
    if active_count >= 2 and not any(truncated_flags):
        truncated_flags[0] = True

    changed_rows = 0
    for row, direction, truncated in zip(active_rows, directions, truncated_flags):
        result = _gen_active_row(width, direction, truncated)
        if result is None:
            return None
        arrow_cols, signal_cols = result
        grid[row, arrow_cols] = RED
        grid[row, signal_cols] = GREEN
        if direction == "right" and signal_cols[-1] < width - 1:
            changed_rows += 1
        if direction == "left" and signal_cols[0] > 0:
            changed_rows += 1

    spare_rows = [row for row in all_rows if row not in active_rows]
    random.shuffle(spare_rows)

    signal_only_count = min(len(spare_rows), random.randint(0, SIGNAL_ONLY_ROWS_MAX))
    for _ in range(signal_only_count):
        row = spare_rows.pop()
        signal_cols = _gen_signals_only_row(width)
        if signal_cols is None:
            return None
        grid[row, signal_cols] = GREEN

    arrow_only_count = min(len(spare_rows), random.randint(0, ARROW_ONLY_ROWS_MAX))
    for _ in range(arrow_only_count):
        row = spare_rows.pop()
        arrow_cols = _gen_arrow_only_row(width)
        if arrow_cols is None:
            return None
        grid[row, arrow_cols] = RED

    if changed_rows == 0:
        return None
    if _variant_score(grid) < 2:
        return None
    return grid


def _variant_score(grid: np.ndarray) -> int:
    score = 0
    for row in range(grid.shape[0]):
        green_cols = np.flatnonzero(grid[row] == GREEN)
        direction = _row_direction(grid, row, green_cols)
        if direction is None:
            continue
        score += 1
        if green_cols.size == 0:
            continue
        if direction == "right" and int(green_cols[-1]) == grid.shape[1] - 1:
            score += 1
        if direction == "left" and int(green_cols[0]) == 0:
            score += 1
    return score


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
