from __future__ import annotations

import random

import numpy as np


def _background_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_bars(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = g.shape
    bars: list[dict[str, int]] = []
    for c in range(w):
        col = g[:, c]
        nz = np.flatnonzero(col != bg)
        if nz.size == 0:
            continue
        r0 = int(nz[0])
        r1 = int(nz[-1])
        color = int(col[r0])
        assert r1 == h - 1, "bars must be bottom-aligned"
        assert np.all(col[r0:] == color), "bar must be solid and monochrome"
        assert np.all(col[:r0] == bg), "space above a bar must be background"
        bars.append({"col": c, "height": h - r0, "color": color})

    assert bars, "expected at least one vertical bar"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Match legend-bar heights to grey bars and recolor only the matches."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    bars = _extract_bars(g, bg)

    color_counts: dict[int, int] = {}
    for bar in bars:
        color_counts[bar["color"]] = color_counts.get(bar["color"], 0) + 1

    grey_candidates = [color for color, count in color_counts.items() if count >= 2]
    assert grey_candidates, "expected repeated grey bar color"
    grey = max(grey_candidates, key=lambda color: color_counts[color])

    legend = [bar for bar in bars if bar["color"] != grey]
    targets = [bar for bar in bars if bar["color"] == grey]
    assert legend and targets

    legend_by_height: dict[int, int] = {}
    for bar in legend:
        assert bar["height"] not in legend_by_height, "legend heights must be unique"
        legend_by_height[bar["height"]] = bar["color"]

    out = np.full_like(g, bg)
    for bar in legend:
        out[g[:, bar["col"]] == bar["color"], bar["col"]] = bar["color"]

    h = g.shape[0]
    for bar in targets:
        color = legend_by_height.get(bar["height"], bg)
        if color != bg:
            out[h - bar["height"] :, bar["col"]] = color

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

BG = 0
GREY = 5
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

HEIGHT_RANGE = (8, 16)
WIDTH_RANGE = (12, 24)
LEGEND_COUNT_RANGE = (2, 5)
EXTRA_TARGET_COUNT_RANGE = (1, 4)
GROUP_GAP_RANGE = (2, 6)
OUTER_MARGIN_RANGE = (1, 3)
MAX_ATTEMPTS = 400


def _sample_distinct(values: list[int], count: int) -> list[int]:
    picks = _RNG.choice(values, size=count, replace=False)
    return [int(v) for v in picks]


def _paint_bar(grid: np.ndarray, col: int, height: int, color: int) -> None:
    grid[grid.shape[0] - height :, col] = int(color)


def _layout_positions(start: int, count: int) -> list[int]:
    return [start + 2 * i for i in range(count)]


def _expected_output(
    grid: np.ndarray,
    legend_cols: list[int],
    legend_heights: list[int],
    legend_colors: list[int],
    target_cols: list[int],
    target_heights: list[int],
) -> np.ndarray:
    out = np.full_like(grid, BG)
    for col, height, color in zip(legend_cols, legend_heights, legend_colors):
        _paint_bar(out, col, height, color)
    for col, height in zip(target_cols, target_heights):
        if height in legend_heights:
            color = legend_colors[legend_heights.index(height)]
            _paint_bar(out, col, height, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))

    legend_count = int(_RNG.integers(LEGEND_COUNT_RANGE[0], LEGEND_COUNT_RANGE[1] + 1))
    extra_count = int(_RNG.integers(EXTRA_TARGET_COUNT_RANGE[0], EXTRA_TARGET_COUNT_RANGE[1] + 1))
    target_count = legend_count + extra_count

    legend_span = 2 * legend_count - 1
    target_span = 2 * target_count - 1
    left_margin = int(_RNG.integers(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1))
    right_margin = int(_RNG.integers(OUTER_MARGIN_RANGE[0], OUTER_MARGIN_RANGE[1] + 1))
    group_gap = int(_RNG.integers(GROUP_GAP_RANGE[0], GROUP_GAP_RANGE[1] + 1))
    grid_w = left_margin + legend_span + group_gap + target_span + right_margin
    if not (WIDTH_RANGE[0] <= grid_w <= WIDTH_RANGE[1]):
        return None

    all_heights = list(range(1, grid_h))
    if len(all_heights) < legend_count + extra_count:
        return None

    legend_heights = _sample_distinct(all_heights, legend_count)
    remaining = [h for h in all_heights if h not in legend_heights]
    extra_heights = _sample_distinct(remaining, extra_count)
    legend_colors = random.sample(LEGEND_COLORS, legend_count)

    target_heights = legend_heights + extra_heights
    random.shuffle(target_heights)
    random.shuffle(legend_heights)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    legend_cols = _layout_positions(left_margin, legend_count)
    target_start = left_margin + legend_span + group_gap
    target_cols = _layout_positions(target_start, target_count)

    for col, height, color in zip(legend_cols, legend_heights, legend_colors):
        _paint_bar(grid, col, height, color)
    for col, height in zip(target_cols, target_heights):
        _paint_bar(grid, col, height, GREY)

    expected = _expected_output(
        grid, legend_cols, legend_heights, legend_colors, target_cols, target_heights
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
