from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

PRIMARY_COLORS = (RED, BLUE, YELLOW)
INPUT_COLORS = {BLACK, GREY, *PRIMARY_COLORS}
EXPECTED_OUTPUT_COLORS = {RED, BLUE, YELLOW, GREEN, MAGENTA, ORANGE}

GRID_HEIGHT_RANGE = (9, 18)
GRID_WIDTH_RANGE = (9, 18)
MAX_GENERATION_ATTEMPTS = 300


def _mix_colors(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if a == b:
        return a

    pair = frozenset((a, b))
    if pair == frozenset((RED, BLUE)):
        return MAGENTA
    if pair == frozenset((RED, YELLOW)):
        return ORANGE
    if pair == frozenset((BLUE, YELLOW)):
        return GREEN
    raise AssertionError(f"unexpected primary pair: {a}, {b}")


def _read_ticks(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 6 and w >= 6, "grid must allow border, rulers, and interior canvas"
    assert set(map(int, np.unique(g))) <= INPUT_COLORS, "unexpected input colors"

    assert np.all(g[0, :] == GREY)
    assert np.all(g[h - 1, :] == GREY)
    assert np.all(g[:, 0] == GREY)
    assert np.all(g[:, w - 1] == GREY)
    assert int(g[1, 1]) == BLACK

    top_ruler = g[1, 2 : w - 1]
    left_ruler = g[2 : h - 1, 1]
    assert np.all(np.isin(top_ruler, (BLACK, *PRIMARY_COLORS)))
    assert np.all(np.isin(left_ruler, (BLACK, *PRIMARY_COLORS)))
    assert np.all(g[2 : h - 1, 2 : w - 1] == BLACK), "interior canvas must start black"

    row_ticks = [(r, int(g[r, 1])) for r in range(2, h - 1) if int(g[r, 1]) in PRIMARY_COLORS]
    col_ticks = [(c, int(g[1, c])) for c in range(2, w - 1) if int(g[1, c]) in PRIMARY_COLORS]

    assert row_ticks, "expected at least one row tick"
    assert col_ticks, "expected at least one column tick"
    return row_ticks, col_ticks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read colored row/column ticks from the inner border and paint their
    intersection points in the black canvas using primary-color mixing.
    """
    g = np.asarray(grid, dtype=int)
    row_ticks, col_ticks = _read_ticks(g)

    out = g.copy()
    for r, row_color in row_ticks:
        for c, col_color in col_ticks:
            out[r, c] = _mix_colors(row_color, col_color)
    return out


def _choose_tick_count(available: int) -> int:
    max_count = min(6, available - 1)
    max_count = max(3, max_count)
    capped = min(max_count, max(3, available // 2 + 1))
    return random.randint(3, capped)


def _sample_positions(available: list[int], count: int) -> list[int]:
    assert count <= len(available)
    return sorted(random.sample(available, count))


def _sample_primary_multiset(count: int) -> list[int]:
    colors = list(PRIMARY_COLORS)
    for _ in range(count - len(PRIMARY_COLORS)):
        colors.append(random.choice(PRIMARY_COLORS))
    random.shuffle(colors)
    return colors


def _build_input(
    height: int,
    width: int,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[height - 1, :] = GREY
    grid[:, 0] = GREY
    grid[:, width - 1] = GREY

    for r, color in zip(row_positions, row_colors, strict=True):
        grid[r, 1] = int(color)
    for c, color in zip(col_positions, col_colors, strict=True):
        grid[1, c] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    row_slots = list(range(2, height - 1))
    col_slots = list(range(2, width - 1))
    if len(row_slots) < 3 or len(col_slots) < 3:
        return None

    row_count = _choose_tick_count(len(row_slots))
    col_count = _choose_tick_count(len(col_slots))

    row_positions = _sample_positions(row_slots, row_count)
    col_positions = _sample_positions(col_slots, col_count)
    row_colors = _sample_primary_multiset(row_count)
    col_colors = _sample_primary_multiset(col_count)

    inp = _build_input(height, width, row_positions, row_colors, col_positions, col_colors)
    out = main(inp)

    observed = {
        int(out[r, c])
        for r in row_positions
        for c in col_positions
    }
    if observed != EXPECTED_OUTPUT_COLORS:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
