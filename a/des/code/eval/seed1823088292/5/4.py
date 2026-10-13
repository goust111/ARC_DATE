from __future__ import annotations

import numpy as np


BG = 0
MAROON = 9
TICK_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

GRID_SIZE_RANGE = (15, 25)
TICK_COUNT_RANGE = (2, 4)
MIN_TICK_GAP = 2
GEN_RETRIES = 300

_RNG = np.random.default_rng()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top row and left column as sparse row/column tick axes.

    For each interior cell:
    - row tick only or column tick only -> leave background
    - both ticks present and equal -> paint that tick color
    - both ticks present and different -> paint maroon

    The axes themselves are preserved unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    row_ticks = g[1:, 0]
    col_ticks = g[0, 1:]
    both_present = (row_ticks[:, None] != BG) & (col_ticks[None, :] != BG)
    markers = np.where(row_ticks[:, None] == col_ticks[None, :], row_ticks[:, None], MAROON)

    out[1:, 1:] = BG
    out[1:, 1:] = np.where(both_present, markers, BG)
    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _sample_sparse_indices(limit: int, count: int, min_gap: int) -> list[int]:
    choices = np.arange(1, limit + 1, dtype=int)
    for gap in range(min_gap, 0, -1):
        for _ in range(100):
            picked = np.sort(_RNG.choice(choices, size=count, replace=False))
            if np.all(np.diff(picked) >= gap):
                return [int(v) for v in picked]
    raise AssertionError("failed to sample sparse tick positions")


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)
    out[:, 0] = g[:, 0]
    out[0, :] = g[0, :]

    row_tick = {int(r): int(g[r, 0]) for r in range(1, h) if int(g[r, 0]) != BG}
    col_tick = {int(c): int(g[0, c]) for c in range(1, w) if int(g[0, c]) != BG}
    for r in range(1, h):
        rv = row_tick.get(r, BG)
        for c in range(1, w):
            cv = col_tick.get(c, BG)
            if rv == BG or cv == BG:
                continue
            out[r, c] = rv if rv == cv else MAROON
    return out


def _sample_axis_colors(row_count: int, col_count: int) -> tuple[list[int], list[int]]:
    overlap = _randint(1, min(row_count, col_count))
    shared = list(map(int, _RNG.choice(np.array(TICK_COLORS, dtype=int), size=overlap, replace=False)))
    remaining = [color for color in TICK_COLORS if color not in shared]

    row_only_count = row_count - overlap
    col_only_count = col_count - overlap
    row_only = list(map(int, _RNG.choice(np.array(remaining, dtype=int), size=row_only_count, replace=False)))
    remaining = [color for color in remaining if color not in row_only]
    col_only = list(map(int, _RNG.choice(np.array(remaining, dtype=int), size=col_only_count, replace=False)))

    row_colors = shared + row_only
    col_colors = shared + col_only
    _RNG.shuffle(row_colors)
    _RNG.shuffle(col_colors)
    return row_colors, col_colors


def _build_case() -> np.ndarray:
    h = _randint(*GRID_SIZE_RANGE)
    w = _randint(*GRID_SIZE_RANGE)

    row_count = _randint(*TICK_COUNT_RANGE)
    col_count = _randint(*TICK_COUNT_RANGE)
    row_positions = _sample_sparse_indices(h - 1, row_count, MIN_TICK_GAP)
    col_positions = _sample_sparse_indices(w - 1, col_count, MIN_TICK_GAP)
    row_colors, col_colors = _sample_axis_colors(row_count, col_count)

    grid = np.full((h, w), BG, dtype=int)
    for r, color in zip(row_positions, row_colors):
        grid[int(r), 0] = int(color)
    for c, color in zip(col_positions, col_colors):
        grid[0, int(c)] = int(color)

    row_tick = {int(r): int(grid[r, 0]) for r in row_positions}
    col_tick = {int(c): int(grid[0, c]) for c in col_positions}

    same_cells: list[tuple[int, int]] = []
    diff_cells: list[tuple[int, int]] = []
    for r, rv in row_tick.items():
        for c, cv in col_tick.items():
            if rv == cv:
                same_cells.append((int(r), int(c)))
            else:
                diff_cells.append((int(r), int(c)))

    assert same_cells, "generator must always include same-color intersections"
    assert len(diff_cells) >= 2, "generator must always include at least two mismatched intersections"

    example_count = _randint(1, len(diff_cells) - 1)
    example_idx = _RNG.choice(len(diff_cells), size=example_count, replace=False)
    for idx in np.atleast_1d(example_idx):
        r, c = diff_cells[int(idx)]
        grid[r, c] = MAROON

    expected = _oracle_output(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    new_same = sum(int(grid[r, c] == BG and expected[r, c] != BG and expected[r, c] != MAROON) for r, c in same_cells)
    hidden_diff = sum(int(grid[r, c] == BG and expected[r, c] == MAROON) for r, c in diff_cells)
    row_only = sum(int(grid[r, 0] != BG and grid[0, c] == BG) for r in range(1, h) for c in range(1, w))
    none_none = sum(int(grid[r, 0] == BG and grid[0, c] == BG) for r in range(1, h) for c in range(1, w))

    assert new_same >= 1
    assert hidden_diff >= 1
    assert row_only >= 1
    assert none_none >= 1
    assert int(grid[0, 0]) == BG
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _build_case()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input grid")
