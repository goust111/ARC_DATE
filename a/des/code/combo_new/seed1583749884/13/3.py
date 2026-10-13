from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BAR_COLOR = 5
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
BAR_HEIGHTS = [1, 2, 3, 4, 5]
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _sample_spaced_positions(width: int, count: int, min_sep: int) -> list[int] | None:
    """Sample sorted column positions with at least min_sep between neighbors."""
    compressed_width = width - (count - 1) * (min_sep - 1)
    if compressed_width < count:
        return None
    picks = sorted(random.sample(range(compressed_width), count))
    return [p + i * (min_sep - 1) for i, p in enumerate(picks)]


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] != bg).tolist()
        if len(cols) != 5:
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != 5:
            continue
        candidates.append((r, cols, colors))

    assert len(candidates) == 1, "expected one legend row with five distinct colors"
    return candidates[0]


def _parse_bars(grid: np.ndarray, bg: int, legend_row: int) -> list[dict[str, int]]:
    height, width = grid.shape
    assert legend_row < height - 1, "legend row must leave room for bars"

    bars: list[dict[str, int]] = []
    for c in range(width):
        lower = grid[legend_row + 1 :, c]
        nz = np.flatnonzero(lower != bg)
        if nz.size == 0:
            continue

        start = int(nz[0])
        assert np.array_equal(nz, np.arange(start, lower.size)), "bar cells must form one bottom-aligned segment"
        color = int(lower[start])
        assert np.all(lower[start:] == color), "every bar must be monochromatic"

        top = legend_row + 1 + start
        bars.append({"col": c, "top": top, "height": height - top, "color": color})

    assert len(bars) == 5, "expected five bottom-aligned bars"
    assert len({bar["color"] for bar in bars}) == 1, "all bars must share one placeholder color"

    heights = sorted(bar["height"] for bar in bars)
    assert heights == BAR_HEIGHTS, "bar heights must be the five distinct ranks 1..5"

    return bars


def _assert_exact_structure(
    grid: np.ndarray,
    bg: int,
    legend_row: int,
    legend_cols: list[int],
    bars: list[dict[str, int]],
) -> None:
    expected = np.zeros(grid.shape, dtype=bool)
    expected[legend_row, legend_cols] = True
    for bar in bars:
        expected[bar["top"] :, bar["col"]] = True
    assert np.array_equal(grid != bg, expected), "input must contain only the legend and five bars"


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder the grey bottom bars by height rank and repaint them with the legend colors.

    Input structure:
    - The background is one dominant color surrounding a single legend row and five thin bars.
    - One upper row contains exactly five distinct non-background pixels ordered from left to right.
    - Below the legend are five one-pixel-wide vertical bars of one shared placeholder color, all bottom-aligned and all with different heights.

    Rule:
    - Read the five legend pixels from left to right as the colors for ranks 1 through 5.
    - Rank the bars by height from shortest to tallest.
    - Clear the original bars, then redraw them on the same five bar columns from left to right so rank 1 is at the leftmost bar column and rank 5 is at the rightmost bar column.
    - Recolor each redrawn bar with the legend color for its rank while keeping the legend row unchanged.

    Key insight:
    - Height determines both the bar's destination column rank and the color it receives from the ordered legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    legend_row, legend_cols, legend_colors = _parse_legend(g, bg)
    bars = _parse_bars(g, bg, legend_row)
    _assert_exact_structure(g, bg, legend_row, legend_cols, bars)

    sorted_bar_cols = sorted(bar["col"] for bar in bars)
    sorted_heights = sorted(bar["height"] for bar in bars)

    out = np.full_like(g, bg)
    out[legend_row, legend_cols] = legend_colors
    for rank, (col, bar_height) in enumerate(zip(sorted_bar_cols, sorted_heights)):
        out[g.shape[0] - bar_height :, col] = legend_colors[rank]
    return out


def _oracle_output(
    shape: tuple[int, int],
    legend_row: int,
    legend_cols: list[int],
    legend_colors: list[int],
    bar_cols: list[int],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    out[legend_row, legend_cols] = legend_colors
    for rank, (col, bar_height) in enumerate(zip(sorted(bar_cols), BAR_HEIGHTS)):
        out[shape[0] - bar_height :, col] = legend_colors[rank]
    return out


def _gen_attempt() -> np.ndarray | None:
    bar_sep = random.randint(2, 4)
    legend_sep = random.randint(1, 3)
    min_width = max(5 + 4 * (bar_sep - 1), 5 + 4 * (legend_sep - 1))
    width = random.randint(min_width, 24)
    height = random.randint(6, 18)

    max_legend_row = min(4, height - 6)
    legend_row = random.randint(0, max_legend_row)
    legend_cols = _sample_spaced_positions(width, 5, legend_sep)
    bar_cols = _sample_spaced_positions(width, 5, bar_sep)
    if legend_cols is None or bar_cols is None:
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 5)
    bar_heights = BAR_HEIGHTS.copy()
    random.shuffle(bar_heights)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[legend_row, legend_cols] = legend_colors
    for col, bar_height in zip(bar_cols, bar_heights):
        grid[height - bar_height :, col] = BAR_COLOR

    expected = _oracle_output(grid.shape, legend_row, legend_cols, legend_colors, bar_cols)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
