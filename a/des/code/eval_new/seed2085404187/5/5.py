from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
DATA = 8
PATTERN_COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 300


def _separator_column(grid: np.ndarray) -> tuple[int, int]:
    candidates = [
        c
        for c in range(grid.shape[1])
        if grid[0, c] != BG and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    c = int(candidates[0])
    assert 0 < c < grid.shape[1] - 1, "separator must leave non-empty template and data areas"
    return c, int(grid[0, c])


def _extract_patterns(left: np.ndarray, separator_color: int) -> list[np.ndarray]:
    split_rows = [r for r in range(left.shape[0]) if np.all(left[r] == separator_color)]
    patterns: list[np.ndarray] = []
    start = 0
    for stop in split_rows + [left.shape[0]]:
        assert stop > start, "template patterns must have positive height"
        pattern = left[start:stop].copy()
        assert not np.any(pattern == separator_color), "separator color cannot appear inside patterns"
        assert all(np.any(row != BG) for row in pattern), "every pattern row must contain visible pixels"
        patterns.append(pattern)
        start = stop + 1
    assert start == left.shape[0] + 1, "template rows must be fully partitioned by separators"
    for index, pattern in enumerate(patterns, start=1):
        assert pattern.shape[0] == index, "pattern height must equal its top-to-bottom index"
    return patterns


def _bar_height(column: np.ndarray, data_color: int) -> int:
    rows = np.flatnonzero(column == data_color)
    assert rows.size > 0, "every data column must contain a bar"
    top = int(rows[0])
    assert np.all(column[:top] == BG), "bars must be bottom-aligned"
    assert np.all(column[top:] == data_color), "bars must be contiguous"
    return int(column.shape[0] - top)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read bottom-aligned data bars as indices into a stacked template catalogue, then concatenate the selected patterns.

    Input structure:
    - The background is black.
    - A full-height grey column splits the grid into a left template area and a right data area.
    - The template area contains equal-width colored patterns stacked top-to-bottom, with grey separator rows between them.
      The k-th template from the top has height exactly k.
    - The data area is made of adjacent one-pixel columns, each containing a bottom-aligned azure bar.

    Rule:
    - Split the left area into templates using the grey separator rows.
    - For each data column, measure the azure bar height H and select the H-th template.
    - Concatenate the selected templates from left to right.
    - Bottom-align all pasted templates in the output and pad any space above them with background.

    Key insight:
    - The azure bar height is not a drawing instruction by itself; it is a lookup key that identifies which template row-block to copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_c, separator_color = _separator_column(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    assert left.size > 0 and right.size > 0

    patterns = _extract_patterns(left, separator_color)
    pattern_count = len(patterns)

    right_colors = set(map(int, np.unique(right))) - {BG}
    assert len(right_colors) == 1, "data area must use a single non-background color"
    data_color = next(iter(right_colors))
    assert data_color != separator_color, "data color must differ from separator color"

    heights = [_bar_height(right[:, c], data_color) for c in range(right.shape[1])]
    assert all(1 <= h <= pattern_count for h in heights), "bar heights must index existing templates"

    pattern_w = int(left.shape[1])
    out_h = max(heights)
    out_w = pattern_w * len(heights)
    out = np.full((out_h, out_w), BG, dtype=int)

    for i, height in enumerate(heights):
        pattern = patterns[height - 1]
        ph = int(pattern.shape[0])
        c0 = i * pattern_w
        out[out_h - ph :, c0 : c0 + pattern_w] = pattern

    return out


def _random_pattern(height: int, width: int, palette: list[int]) -> np.ndarray:
    for _ in range(200):
        pattern = np.full((height, width), BG, dtype=int)
        density = random.uniform(0.4, 0.8)
        for r in range(height):
            active = [c for c in range(width) if random.random() < density]
            if not active:
                active = [random.randrange(width)]
            for c in active:
                pattern[r, c] = random.choice(palette)
        if np.count_nonzero(pattern != BG) >= height:
            return pattern
    raise RuntimeError("failed to sample a visible template pattern")


def _build_output(patterns: list[np.ndarray], heights: list[int]) -> np.ndarray:
    pattern_w = int(patterns[0].shape[1])
    out_h = max(heights)
    out = np.full((out_h, pattern_w * len(heights)), BG, dtype=int)
    for i, height in enumerate(heights):
        pattern = patterns[height - 1]
        ph = int(pattern.shape[0])
        c0 = i * pattern_w
        out[out_h - ph :, c0 : c0 + pattern_w] = pattern
    return out


def _gen_attempt() -> np.ndarray | None:
    pattern_count = random.randint(3, 5)
    pattern_w = random.randint(3, 5)
    max_data_cols = min(8, 30 // pattern_w)
    if max_data_cols < pattern_count:
        return None

    data_cols = random.randint(pattern_count, max_data_cols)
    heights = list(range(1, pattern_count + 1))
    while len(heights) < data_cols:
        heights.append(random.randint(1, pattern_count))
    random.shuffle(heights)

    palette_size = random.randint(2, min(4, len(PATTERN_COLOR_POOL)))
    palette = random.sample(list(PATTERN_COLOR_POOL), palette_size)
    patterns = [_random_pattern(height, pattern_w, palette) for height in range(1, pattern_count + 1)]

    height = sum(range(1, pattern_count + 1)) + (pattern_count - 1)
    width = pattern_w + 1 + data_cols
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    row = 0
    for index, pattern in enumerate(patterns):
        ph = int(pattern.shape[0])
        grid[row : row + ph, :pattern_w] = pattern
        row += ph
        if index != len(patterns) - 1:
            grid[row, :pattern_w] = SEPARATOR
            row += 1
    grid[:, pattern_w] = SEPARATOR

    for c, bar_h in enumerate(heights):
        grid[height - bar_h :, pattern_w + 1 + c] = DATA

    expected = _build_output(patterns, heights)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
