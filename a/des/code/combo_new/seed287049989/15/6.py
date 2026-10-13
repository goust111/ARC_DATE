from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
PATTERN_COLORS = (4, 5, 6, 7, 8, 9)
INDICATOR_COLORS = (RED, BLUE)
MAX_ATTEMPTS = 300


def _kept_mask(bar: np.ndarray) -> np.ndarray:
    keep_color = int(bar[0])
    return np.asarray(bar == keep_color, dtype=bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the main pattern by the row/column indicator colors chosen at the green corner.

    Input structure:
    - The background is black.
    - The first column below the corner is a row-indicator strip, with one red-or-blue marker for each pattern row.
    - The first row to the right of the corner is a column-indicator strip, with one red-or-blue marker for each pattern column.
    - The top-left corner is a single green selector pixel.
    - The remaining cells form the main multi-color pattern.

    Rule:
    - Read the row selection color from the pixel directly below the green corner.
    - Read the column selection color from the pixel directly right of the green corner.
    - Keep exactly those pattern rows whose left indicator matches the row selection color.
    - Keep exactly those pattern columns whose top indicator matches the column selection color.
    - Return the compact subgrid formed by the kept rows and kept columns, in their original order.

    Key insight:
    - Red and blue do not have a fixed meaning; the first row-strip cell and first column-strip cell reveal which indicator color means "keep" for rows and columns independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2
    assert int(g[0, 0]) == GREEN

    row_bar = g[1:, 0]
    col_bar = g[0, 1:]
    pattern = g[1:, 1:]

    keep_rows = _kept_mask(row_bar)
    keep_cols = _kept_mask(col_bar)
    return pattern[keep_rows][:, keep_cols]


def _sample_mask(length: int, rng: random.Random) -> np.ndarray:
    keep_count = rng.randint(1, length - 1)
    keep = np.zeros(length, dtype=bool)
    keep[0] = True
    if keep_count > 1:
        extra = rng.sample(range(1, length), keep_count - 1)
        keep[extra] = True
    return keep


def _bar_from_mask(mask: np.ndarray, keep_color: int) -> np.ndarray:
    other_color = BLUE if keep_color == RED else RED
    return np.where(mask, keep_color, other_color).astype(int)


def _patchwork_pattern(rows: int, cols: int, palette: list[int], rng: random.Random) -> np.ndarray:
    row_cuts = sorted(rng.sample(range(1, rows), rng.randint(1, min(3, rows - 1))))
    col_cuts = sorted(rng.sample(range(1, cols), rng.randint(1, min(3, cols - 1))))
    row_blocks = np.searchsorted(row_cuts, np.arange(rows), side="right")
    col_blocks = np.searchsorted(col_cuts, np.arange(cols), side="right")
    scale_r = rng.randint(1, 3)
    scale_c = rng.randint(1, 3)
    idx = (scale_r * row_blocks[:, None] + scale_c * col_blocks[None, :]) % len(palette)
    return np.take(np.asarray(palette, dtype=int), idx)


def _formula_pattern(rows: int, cols: int, palette: list[int], rng: random.Random) -> np.ndarray:
    rr, cc = np.indices((rows, cols))
    mode = rng.randrange(3)
    if mode == 0:
        row_span = rng.randint(1, max(1, rows // 4))
        col_span = rng.randint(1, max(1, cols // 4))
        idx = (rr // row_span + 2 * (cc // col_span) + rr + cc) % len(palette)
    elif mode == 1:
        idx = (2 * rr - cc + rr // max(1, rows // 5)) % len(palette)
    else:
        idx = (rr + 2 * cc + (rr - cc)) % len(palette)
    return np.take(np.asarray(palette, dtype=int), idx)


def _overlay_features(pattern: np.ndarray, palette: list[int], rng: random.Random) -> None:
    rows, cols = pattern.shape
    for _ in range(rng.randint(1, 3)):
        color = rng.choice(palette)
        if rng.random() < 0.5:
            r0 = rng.randint(0, rows - 1)
            c0 = rng.randint(0, cols - 1)
            r1 = rng.randint(r0, min(rows - 1, r0 + max(1, rows // 3)))
            c1 = rng.randint(c0, min(cols - 1, c0 + max(1, cols // 3)))
            pattern[r0 : r1 + 1, c0 : c1 + 1] = color
        elif rng.random() < 0.5:
            pattern[rng.randrange(rows), :] = color
        else:
            pattern[:, rng.randrange(cols)] = color


def _make_pattern(rows: int, cols: int, rng: random.Random) -> np.ndarray:
    palette = rng.sample(PATTERN_COLORS, rng.randint(3, 5))
    if rng.random() < 0.5:
        pattern = _patchwork_pattern(rows, cols, palette, rng)
    else:
        pattern = _formula_pattern(rows, cols, palette, rng)
    _overlay_features(pattern, palette, rng)

    unique = np.unique(pattern)
    if unique.size < 3:
        pattern[0, 0] = palette[0]
        pattern[rows // 2, cols // 2] = palette[1]
        pattern[-1, -1] = palette[2]
    return pattern.astype(int)


def _expected_output(pattern: np.ndarray, row_bar: np.ndarray, col_bar: np.ndarray) -> np.ndarray:
    keep_rows = row_bar == row_bar[0]
    keep_cols = col_bar == col_bar[0]
    return pattern[keep_rows][:, keep_cols]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    rows = rng.randint(4, 20)
    cols = rng.randint(4, 20)

    row_keep_color = rng.choice(INDICATOR_COLORS)
    col_keep_color = rng.choice(INDICATOR_COLORS)
    row_mask = _sample_mask(rows, rng)
    col_mask = _sample_mask(cols, rng)
    row_bar = _bar_from_mask(row_mask, row_keep_color)
    col_bar = _bar_from_mask(col_mask, col_keep_color)
    pattern = _make_pattern(rows, cols, rng)

    grid = np.full((rows + 1, cols + 1), BLACK, dtype=int)
    grid[0, 0] = GREEN
    grid[1:, 0] = row_bar
    grid[0, 1:] = col_bar
    grid[1:, 1:] = pattern

    expected = _expected_output(pattern, row_bar, col_bar)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if row_bar[0] == row_bar[-1] and np.all(row_bar == row_bar[0]):
        return None
    if col_bar[0] == col_bar[-1] and np.all(col_bar == col_bar[0]):
        return None
    if np.unique(pattern).size < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
