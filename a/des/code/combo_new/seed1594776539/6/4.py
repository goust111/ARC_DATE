from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAROON = 9
STRIPE_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

GRID_MIN = 8
GRID_MAX = 24
MAX_GAP_SPAN = 9
BROKEN_LINES_MAX = 5
GEN_ATTEMPTS = 300


def _is_gap_value(value: int) -> bool:
    return int(value) in (BLACK, MAROON)


def _line_runs(line: np.ndarray) -> list[tuple[int | str, int, int]]:
    runs: list[tuple[int | str, int, int]] = []
    start = 0
    current: int | str = "gap" if _is_gap_value(int(line[0])) else int(line[0])
    for idx in range(1, len(line)):
        value = "gap" if _is_gap_value(int(line[idx])) else int(line[idx])
        if value != current:
            runs.append((current, start, idx - 1))
            start = idx
            current = value
    runs.append((current, start, len(line) - 1))
    return runs


def _line_is_valid(line: np.ndarray) -> bool:
    runs = _line_runs(line)
    if len(runs) == 1:
        return runs[0][0] != "gap"
    if len(runs) != 3:
        return False
    first, middle, last = runs
    if first[0] == "gap" or middle[0] != "gap" or last[0] == "gap":
        return False
    maroons = int(np.count_nonzero(line[middle[1] : middle[2] + 1] == MAROON))
    return maroons == 1


def _detect_orientation(grid: np.ndarray) -> str:
    horizontal_ok = all(_line_is_valid(grid[r, :]) for r in range(grid.shape[0]))
    vertical_ok = all(_line_is_valid(grid[:, c]) for c in range(grid.shape[1]))
    if horizontal_ok and not vertical_ok:
        return "horizontal"
    if vertical_ok and not horizontal_ok:
        return "vertical"
    raise AssertionError("grid must encode a unique stripe orientation")


def _gap_spec_from_line(line: np.ndarray) -> tuple[int, int, int] | None:
    runs = _line_runs(line)
    if len(runs) == 1:
        return None
    assert len(runs) == 3 and runs[1][0] == "gap"
    start, end = runs[1][1], runs[1][2]
    maroons = np.flatnonzero(line[start : end + 1] == MAROON)
    assert len(maroons) == 1
    mid = start + int(maroons[0])
    return (start, end, mid)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair broken one-cell stripes and collect each maroon gap marker onto the output edge.

    Input structure:
    - The grid is fully covered by one-pixel-wide stripes that all share one orientation: horizontal rows or vertical columns.
    - Most stripes are complete single-color lines, while some contain one internal gap made of black cells with one maroon signal pixel at its center.
    - A broken stripe may show the same stripe color on both sides of the gap or two different colors.
    - There is no black background outside those gap segments.

    Rule:
    - Detect whether the stripes run horizontally or vertically.
    - For each broken stripe, find the gap segment and read the nearest non-gap colors on its two sides along the stripe direction.
    - If the two side colors match, fill the whole gap with that color; otherwise fill the left/top part through the maroon midpoint with the left/top color and the remaining right/bottom part with the other color.
    - Remove the maroon from every repaired gap, then place maroon pixels on the output's left edge for broken horizontal rows or top edge for broken vertical columns at the same row or column indices.

    Key insight:
    - The maroon pixel is both the gap midpoint, which fixes the split point for mixed-color repair, and an index token that migrates to the output border to mark which stripes were broken.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orientation = _detect_orientation(g)
    out = g.copy()

    if orientation == "horizontal":
        broken_rows: list[int] = []
        for r in range(g.shape[0]):
            line = g[r, :]
            gap = _gap_spec_from_line(line)
            if gap is None:
                continue
            start, end, mid = gap
            left_color = int(line[start - 1])
            right_color = int(line[end + 1])
            assert not _is_gap_value(left_color) and not _is_gap_value(right_color)
            if left_color == right_color:
                out[r, start : end + 1] = left_color
            else:
                out[r, start : mid + 1] = left_color
                out[r, mid + 1 : end + 1] = right_color
            broken_rows.append(r)
        for r in broken_rows:
            out[r, 0] = MAROON
        return out

    broken_cols: list[int] = []
    for c in range(g.shape[1]):
        line = g[:, c]
        gap = _gap_spec_from_line(line)
        if gap is None:
            continue
        start, end, mid = gap
        top_color = int(line[start - 1])
        bottom_color = int(line[end + 1])
        assert not _is_gap_value(top_color) and not _is_gap_value(bottom_color)
        if top_color == bottom_color:
            out[start : end + 1, c] = top_color
        else:
            out[start : mid + 1, c] = top_color
            out[mid + 1 : end + 1, c] = bottom_color
        broken_cols.append(c)
    for c in broken_cols:
        out[0, c] = MAROON
    return out


def _stripe_sequence(length: int, rng: random.Random) -> list[int]:
    seq = [rng.choice(STRIPE_COLORS)]
    for _ in range(1, length):
        choices = [color for color in STRIPE_COLORS if color != seq[-1]]
        seq.append(rng.choice(choices))
    return seq


def _odd_gap_length(limit: int, rng: random.Random) -> int:
    max_len = min(MAX_GAP_SPAN, limit)
    if max_len % 2 == 0:
        max_len -= 1
    choices = [size for size in range(3, max_len + 1, 2)]
    return rng.choice(choices)


def _branch_flags(count: int, rng: random.Random) -> list[bool]:
    flags = [True] * (count // 2) + [False] * (count - count // 2)
    rng.shuffle(flags)
    return flags


def _expected_output(
    grid: np.ndarray,
    orientation: str,
    gaps: list[tuple[int, int, int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    if orientation == "horizontal":
        for row, start, end, mid, left_color, right_color in gaps:
            if left_color == right_color:
                out[row, start : end + 1] = left_color
            else:
                out[row, start : mid + 1] = left_color
                out[row, mid + 1 : end + 1] = right_color
            out[row, 0] = MAROON
        return out

    for col, start, end, mid, top_color, bottom_color in gaps:
        if top_color == bottom_color:
            out[start : end + 1, col] = top_color
        else:
            out[start : mid + 1, col] = top_color
            out[mid + 1 : end + 1, col] = bottom_color
        out[0, col] = MAROON
    return out


def _gen_horizontal(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    row_colors = _stripe_sequence(h, rng)
    grid = np.repeat(np.array(row_colors, dtype=int)[:, None], w, axis=1)

    max_broken = min(BROKEN_LINES_MAX, h // 2)
    broken_count = rng.randint(2, max_broken)
    rows = sorted(rng.sample(range(h), broken_count))
    branch_flags = _branch_flags(broken_count, rng)

    gaps: list[tuple[int, int, int, int, int, int]] = []
    for row, same_color in zip(rows, branch_flags, strict=False):
        gap_len = _odd_gap_length(w - 2, rng)
        start = rng.randint(1, w - gap_len - 1)
        end = start + gap_len - 1
        mid = start + gap_len // 2
        left_color = row_colors[row]
        if same_color:
            right_color = left_color
        else:
            right_choices = [color for color in STRIPE_COLORS if color != left_color]
            right_color = rng.choice(right_choices)
        grid[row, :start] = left_color
        grid[row, start : end + 1] = BLACK
        grid[row, end + 1 :] = right_color
        grid[row, mid] = MAROON
        gaps.append((row, start, end, mid, left_color, right_color))

    expected = _expected_output(grid, "horizontal", gaps)
    return grid, expected


def _gen_vertical(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    col_colors = _stripe_sequence(w, rng)
    grid = np.repeat(np.array(col_colors, dtype=int)[None, :], h, axis=0)

    max_broken = min(BROKEN_LINES_MAX, w // 2)
    broken_count = rng.randint(2, max_broken)
    cols = sorted(rng.sample(range(w), broken_count))
    branch_flags = _branch_flags(broken_count, rng)

    gaps: list[tuple[int, int, int, int, int, int]] = []
    for col, same_color in zip(cols, branch_flags, strict=False):
        gap_len = _odd_gap_length(h - 2, rng)
        start = rng.randint(1, h - gap_len - 1)
        end = start + gap_len - 1
        mid = start + gap_len // 2
        top_color = col_colors[col]
        if same_color:
            bottom_color = top_color
        else:
            bottom_choices = [color for color in STRIPE_COLORS if color != top_color]
            bottom_color = rng.choice(bottom_choices)
        grid[:start, col] = top_color
        grid[start : end + 1, col] = BLACK
        grid[end + 1 :, col] = bottom_color
        grid[mid, col] = MAROON
        gaps.append((col, start, end, mid, top_color, bottom_color))

    expected = _expected_output(grid, "vertical", gaps)
    return grid, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(("horizontal", "vertical"))
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)

    if orientation == "horizontal":
        grid, expected = _gen_horizontal(rng, h, w)
    else:
        grid, expected = _gen_vertical(rng, h, w)

    if _detect_orientation(grid) != orientation:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-gap task")
