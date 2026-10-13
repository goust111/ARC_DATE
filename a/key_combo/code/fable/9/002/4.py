from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
TEAL = 8

HEIGHT_RANGE = (9, 18)
BAR_COUNT_RANGE = (3, 6)
WIDTH_CAP = 24
GENERATION_ATTEMPTS = 300


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    """Return solid single-column red bars as dicts with r0/r1/c/height."""
    bars: list[dict[str, int]] = []
    red_mask = grid == RED
    for col in range(grid.shape[1]):
        rows = np.flatnonzero(red_mask[:, col])
        if rows.size == 0:
            continue
        assert np.all(np.diff(rows) == 1), "each occupied column must be one contiguous segment"
        r0 = int(rows[0])
        r1 = int(rows[-1])
        height = int(rows.size)
        assert np.all(grid[r0 : r1 + 1, col] == RED), "bars must be solid"
        bars.append({"r0": r0, "r1": r1, "c": int(col), "height": height})

    assert bars, "expected at least one red bar"
    occupied_cols = {bar["c"] for bar in bars}
    for col in range(grid.shape[1]):
        if col in occupied_cols:
            continue
        assert np.all(grid[:, col] == BLACK), "non-bar columns must be background only"

    # No two bars may 4-touch; with width-1 vertical bars this means no adjacent occupied columns.
    sorted_cols = sorted(occupied_cols)
    assert all(b - a >= 2 for a, b in zip(sorted_cols, sorted_cols[1:])), "bars must be separated"
    return bars


def _transform(grid: np.ndarray) -> np.ndarray:
    bars = _extract_bars(grid)
    assert len(bars) >= 2, "expected at least two bars"

    baselines = {bar["r1"] for bar in bars}
    assert len(baselines) == 1, "all bars must stand on the same baseline"

    heights = [bar["height"] for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be distinct"

    ranked = sorted(bars, key=lambda bar: bar["height"], reverse=True)
    out = grid.copy()
    for idx, bar in enumerate(ranked):
        ref_height = ranked[(idx + 1) % len(ranked)]["height"]
        recolor = min(bar["height"], ref_height)
        r1 = bar["r1"]
        c = bar["c"]
        out[r1 - recolor + 1 : r1 + 1, c] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the bottom part of each red bar using the next shorter bar's height.

    Input structure:
    - The background is black.
    - The foreground consists of separated solid vertical red bars, each one cell wide.
    - All bars rest on the same horizontal baseline, and their heights are all distinct.

    Rule:
    - Rank the bars by height from tallest to shortest.
    - For each bar, look at the next bar in that ranking, wrapping from the shortest back to the tallest.
    - Recolor that many cells from the current bar's bottom upward to teal, leaving the rest red.

    Key insight:
    - The split size is not derived from a bar itself or from its neighbors in position space.
      It is transferred from another bar chosen by height-rank order, with the shortest bar
      using the tallest as its reference and therefore turning fully teal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, RED}, "input must contain only black and red"
    assert np.count_nonzero(g == RED) > 0, "expected at least one red cell"
    return _transform(g)


def _sample_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    if len(candidates) < 2 * count - 1:
        return None
    for _ in range(80):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _sample_heights(height: int, count: int, *, aligned: bool) -> list[int]:
    values = random.sample(range(2, height), count)
    values.sort(reverse=True)
    if aligned:
        return values

    while True:
        shuffled = values[:]
        random.shuffle(shuffled)
        if shuffled != values:
            return shuffled


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(*BAR_COUNT_RANGE)
    grid_h = random.randint(max(HEIGHT_RANGE[0], bar_count + 3), HEIGHT_RANGE[1])
    min_width = max(2 * bar_count + 1, 9)
    grid_w = random.randint(min_width, WIDTH_CAP)

    cols = _sample_columns(grid_w, bar_count)
    if cols is None:
        return None

    # Cover both the deceptive aligned case and the general mismatched case.
    aligned = random.random() < 0.35
    heights = _sample_heights(grid_h, bar_count, aligned=aligned)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for col, h in zip(cols, heights):
        grid[grid_h - h :, col] = RED

    if not np.all(grid[0] == BLACK):
        return None

    left_to_right = heights[:]
    rank_order = sorted(heights, reverse=True)
    if aligned:
        assert left_to_right == rank_order
    else:
        assert left_to_right != rank_order

    try:
        solved = main(grid)
    except AssertionError:
        return None

    expected = _transform(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
