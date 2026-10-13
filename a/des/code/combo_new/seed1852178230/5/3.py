from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 4
MARKER = 7
PALETTE = (1, 2, 3, 5, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _primitive_cycle(values: list[int]) -> list[int]:
    n = len(values)
    assert n > 0
    for period in range(1, n + 1):
        if all(values[i] == values[i % period] for i in range(n)):
            return values[:period]
    raise AssertionError("ruler sequence must have a valid period")


def _repeated(cycle: list[int], length: int) -> list[int]:
    return [cycle[i % len(cycle)] for i in range(length)]


def _first_mismatch(values: list[int], cycle: list[int]) -> int | None:
    for i, value in enumerate(values):
        if value != cycle[i % len(cycle)]:
            return i
    return None


def _divider_indices(grid: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(grid == DIVIDER, axis=1))[0]
    cols = np.where(np.all(grid == DIVIDER, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full horizontal and one full vertical divider"
    return int(rows[0]), int(cols[0])


def _top_ruler(grid: np.ndarray, divider_col: int) -> list[int]:
    values = [int(grid[0, c]) for c in range(1, grid.shape[1]) if c != divider_col]
    assert values and all(v != DIVIDER for v in values)
    return _primitive_cycle(values)


def _left_ruler(grid: np.ndarray, divider_row: int) -> list[int]:
    values = [int(grid[r, 0]) for r in range(1, grid.shape[0]) if r != divider_row]
    assert values and all(v != DIVIDER for v in values)
    return _primitive_cycle(values)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the first place where each quadrant's pattern breaks its edge-ruler cycle.

    Input structure:
    - A black-background grid is split by one full yellow horizontal line and one full yellow vertical line.
    - The top border row (except the top-left corner and the yellow crossing column) shows a repeating color ruler.
    - The left border column (except the top-left corner and the yellow crossing row) shows another repeating color ruler.
    - The four interior quadrants contain color patterns to be checked against those rulers.

    Rule:
    - Locate the yellow cross that defines the four quadrants.
    - The upper-left and upper-right quadrants are controlled by the top-row ruler: scan each interior column top-to-bottom and compare it with that ruler's repeating cycle.
    - The lower-left and lower-right quadrants are controlled by the left-column ruler: scan each interior row left-to-right and compare it with that ruler's repeating cycle.
    - In every scanned row or column, recolor the first mismatching cell to orange; leave fully matching lines unchanged and preserve every other pixel.

    Key insight:
    - The yellow cross is only the partition; the real instruction comes from the colored edge rulers, and each mark identifies the earliest deviation from the expected repeating sequence inside one quadrant line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    divider_row, divider_col = _divider_indices(g)
    assert 0 < divider_row < h - 1 and 0 < divider_col < w - 1

    top_cycle = _top_ruler(g, divider_col)
    left_cycle = _left_ruler(g, divider_row)
    out = g.copy()

    top_rows = list(range(1, divider_row))
    bottom_rows = list(range(divider_row + 1, h))
    left_cols = list(range(1, divider_col))
    right_cols = list(range(divider_col + 1, w))
    assert top_rows and bottom_rows and left_cols and right_cols

    for cols in (left_cols, right_cols):
        for c in cols:
            line = [int(g[r, c]) for r in top_rows]
            mismatch = _first_mismatch(line, top_cycle)
            if mismatch is not None:
                out[top_rows[mismatch], c] = MARKER

    for rows in (bottom_rows,):
        for cols in (left_cols, right_cols):
            for r in rows:
                line = [int(g[r, c]) for c in cols]
                mismatch = _first_mismatch(line, left_cycle)
                if mismatch is not None:
                    out[r, cols[mismatch]] = MARKER

    return out


def _choose_cycle(max_len: int) -> list[int]:
    length = random.randint(2, min(4, max_len))
    return random.sample(PALETTE, length)


def _mismatch_flags(count: int) -> list[bool]:
    assert count >= 2
    bad = random.randint(1, count - 1)
    flags = [True] * bad + [False] * (count - bad)
    random.shuffle(flags)
    return flags


def _different_color(expected: int) -> int:
    choices = [color for color in PALETTE if color != expected]
    return random.choice(choices)


def _paint_top_quadrant(grid: np.ndarray, rows: list[int], cols: list[int], cycle: list[int]) -> None:
    flags = _mismatch_flags(len(cols))
    for col, has_mismatch in zip(cols, flags, strict=False):
        expected = _repeated(cycle, len(rows))
        values = expected[:]
        if has_mismatch:
            first_bad = random.randrange(len(rows))
            values[first_bad] = _different_color(expected[first_bad])
            if random.random() < 0.35:
                for i in range(first_bad + 1, len(rows)):
                    if random.random() < 0.25:
                        values[i] = _different_color(expected[i])
        for r, value in zip(rows, values, strict=False):
            grid[r, col] = value


def _paint_bottom_quadrant(grid: np.ndarray, rows: list[int], cols: list[int], cycle: list[int]) -> None:
    flags = _mismatch_flags(len(rows))
    for row, has_mismatch in zip(rows, flags, strict=False):
        expected = _repeated(cycle, len(cols))
        values = expected[:]
        if has_mismatch:
            first_bad = random.randrange(len(cols))
            values[first_bad] = _different_color(expected[first_bad])
            if random.random() < 0.35:
                for i in range(first_bad + 1, len(cols)):
                    if random.random() < 0.25:
                        values[i] = _different_color(expected[i])
        for c, value in zip(cols, values, strict=False):
            grid[row, c] = value


def _build_input() -> np.ndarray:
    h = random.randint(12, 28)
    w = random.randint(12, 28)
    divider_row = random.randint(4, h - 4)
    divider_col = random.randint(4, w - 4)

    top_visible = w - 2
    left_visible = h - 2
    top_cycle = _choose_cycle(min(4, top_visible))
    left_cycle = _choose_cycle(min(4, left_visible))

    grid = np.full((h, w), BG, dtype=int)
    grid[divider_row, :] = DIVIDER
    grid[:, divider_col] = DIVIDER

    top_positions = [c for c in range(1, w) if c != divider_col]
    left_positions = [r for r in range(1, h) if r != divider_row]
    for c, value in zip(top_positions, _repeated(top_cycle, len(top_positions)), strict=False):
        grid[0, c] = value
    for r, value in zip(left_positions, _repeated(left_cycle, len(left_positions)), strict=False):
        grid[r, 0] = value

    top_rows = list(range(1, divider_row))
    bottom_rows = list(range(divider_row + 1, h))
    left_cols = list(range(1, divider_col))
    right_cols = list(range(divider_col + 1, w))

    _paint_top_quadrant(grid, top_rows, left_cols, top_cycle)
    _paint_top_quadrant(grid, top_rows, right_cols, top_cycle)
    _paint_bottom_quadrant(grid, bottom_rows, left_cols, left_cycle)
    _paint_bottom_quadrant(grid, bottom_rows, right_cols, left_cycle)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        if out.shape != grid.shape or out.ndim != 2 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
