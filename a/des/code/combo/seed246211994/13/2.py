from __future__ import annotations

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_bars(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    bars: list[tuple[int, int, int]] = []

    for col in range(width):
        rows = np.flatnonzero(grid[:, col] != bg)
        if rows.size == 0:
            continue

        top = int(rows[0])
        assert int(rows[-1]) == height - 1, "bars must be bottom-aligned"
        assert np.array_equal(rows, np.arange(top, height)), "bar cells must be contiguous"

        colors = np.unique(grid[top:, col])
        assert colors.size == 1, "each bar must be monochrome"
        bars.append((col, int(colors[0]), height - top))

    assert len(bars) >= 2, "expected at least two bars"
    return bars


def _render_bars(
    shape: tuple[int, int],
    bg: int,
    cols: np.ndarray,
    colors: np.ndarray,
    heights: np.ndarray,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    total_rows = shape[0]
    for col, color, bar_height in zip(cols, colors, heights):
        grid[total_rows - int(bar_height) :, int(col)] = int(color)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate bar colors right and bar heights left while keeping columns fixed."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    bars = _extract_bars(g, bg)

    out = np.full(g.shape, bg, dtype=int)
    count = len(bars)
    total_rows = g.shape[0]
    for idx, (col, _, _) in enumerate(bars):
        color = bars[(idx - 1) % count][1]
        bar_height = bars[(idx + 1) % count][2]
        assert 1 <= bar_height <= total_rows
        out[total_rows - bar_height :, col] = color

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

MIN_HEIGHT = 6
MAX_HEIGHT_INCLUSIVE = 16
MIN_BARS = 2
MAX_BARS = 6
WIDTH_EXTRA_MAX = 8
MAX_ATTEMPTS = 300
PALETTE = np.arange(10, dtype=int)


def _sample_columns(width: int, count: int) -> np.ndarray:
    # Choose columns with at least one background column between neighboring bars.
    base = np.sort(_RNG.choice(np.arange(width - count + 1), size=count, replace=False))
    return base + np.arange(count)


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    cols: np.ndarray,
    colors: np.ndarray,
    heights: np.ndarray,
) -> np.ndarray:
    return _render_bars(shape, bg, cols, np.roll(colors, 1), np.roll(heights, -1))


def _gen_attempt() -> np.ndarray | None:
    grid_height = int(_RNG.integers(MIN_HEIGHT, MAX_HEIGHT_INCLUSIVE + 1))
    max_bars = min(MAX_BARS, grid_height, len(PALETTE) - 1)
    if max_bars < MIN_BARS:
        return None

    bar_count = int(_RNG.integers(MIN_BARS, max_bars + 1))
    bg = int(_RNG.choice(PALETTE))

    fg_colors = PALETTE[PALETTE != bg]
    colors = _RNG.choice(fg_colors, size=bar_count, replace=False).astype(int)

    heights = _RNG.choice(np.arange(1, grid_height + 1), size=bar_count, replace=False).astype(int)
    _RNG.shuffle(heights)

    min_clear_width = 2 * bar_count - 1
    min_bg_width = (2 * int(np.sum(heights))) // grid_height + 1
    grid_width = int(
        _RNG.integers(max(min_clear_width, min_bg_width), max(min_clear_width, min_bg_width) + WIDTH_EXTRA_MAX + 1)
    )
    cols = _sample_columns(grid_width, bar_count)

    inp = _render_bars((grid_height, grid_width), bg, cols, colors, heights)
    expected = _expected_output((grid_height, grid_width), bg, cols, colors, heights)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
