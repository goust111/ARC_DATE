from __future__ import annotations

import numpy as np


BG = 0
_RNG = np.random.default_rng()

# Grid and bar sampling bounds.
GRID_HEIGHT_MIN = 4
GRID_HEIGHT_MAX = 12
GRID_WIDTH_MAX = 30
BAR_COUNT_MIN = 2
BAR_COUNT_MAX = 6
WIDTH_MAX = 9
MAX_GENERATION_ATTEMPTS = 300
COLOR_POOL = tuple(range(1, 10))


def _extract_bars(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    foreground = np.argwhere(g != BG)
    assert foreground.size > 0, "expected at least one foreground bar"

    rows = np.unique(foreground[:, 0])
    assert rows.size == 1, "all bars must have height 1 and share the same row"
    row = int(rows[0])

    line = g[row]
    cols = np.flatnonzero(line != BG)
    assert cols.size > 0
    left = int(cols[0])
    right = int(cols[-1])
    assert np.all(line[left : right + 1] != BG), "bars must be tightly packed with no gaps"

    bars: list[dict[str, int]] = []
    c = left
    while c <= right:
        color = int(line[c])
        start = c
        while c <= right and int(line[c]) == color:
            c += 1
        bars.append({"color": color, "width": c - start, "start": start})

    assert len(bars) >= 2, "expected multiple bars"
    assert len({bar["color"] for bar in bars}) == len(bars), "bar colors must be distinct"
    assert len({bar["width"] for bar in bars}) == len(bars), "bar widths must be distinct"

    return row, left, right, bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort bottom-aligned horizontal bars by width and restack them as a centered staircase.
    """
    g = np.asarray(grid, dtype=int)
    row, span_left, span_right, bars = _extract_bars(g)

    ordered = sorted(bars, key=lambda bar: bar["width"], reverse=True)
    base_width = int(ordered[0]["width"])
    span_width = int(span_right - span_left + 1)
    base_left = int(span_left + (span_width - base_width) // 2)
    assert row - (len(ordered) - 1) >= 0, "not enough rows to stack the bars"

    out = np.full_like(g, BG)
    for level, bar in enumerate(ordered):
        width = int(bar["width"])
        start = int(base_left + (base_width - width) // 2)
        out[row - level, start : start + width] = int(bar["color"])

    return out


def _sample_unique_widths(count: int) -> list[int]:
    max_width = int(_RNG.integers(max(count, 4), WIDTH_MAX + 1))
    if max_width < count:
        return []
    widths = _RNG.choice(np.arange(1, max_width + 1), size=count, replace=False)
    widths = [int(w) for w in widths]
    widths.sort()
    return widths


def _gen_attempt() -> np.ndarray | None:
    bar_count = int(_RNG.integers(BAR_COUNT_MIN, BAR_COUNT_MAX + 1))
    widths = _sample_unique_widths(bar_count)
    if len(widths) != bar_count:
        return None

    total_width = int(sum(widths))
    min_height = bar_count
    if min_height > GRID_HEIGHT_MAX or total_width > GRID_WIDTH_MAX - 2:
        return None

    grid_h = int(_RNG.integers(max(GRID_HEIGHT_MIN, min_height), GRID_HEIGHT_MAX + 1))
    extra_w = int(_RNG.integers(0, GRID_WIDTH_MAX - total_width + 1))
    if extra_w < 2:
        return None
    left_margin = int(_RNG.integers(1, extra_w))
    right_margin = extra_w - left_margin
    if right_margin < 1:
        return None
    grid_w = left_margin + total_width + right_margin

    colors = [int(c) for c in _RNG.choice(COLOR_POOL, size=bar_count, replace=False)]
    pairs = list(zip(widths, colors))
    _RNG.shuffle(pairs)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    c = left_margin
    for width, color in pairs:
        grid[-1, c : c + width] = color
        c += width

    row, span_left, span_right, bars = _extract_bars(grid)
    assert row == grid_h - 1
    assert span_left == left_margin
    assert span_right == left_margin + total_width - 1
    assert len(bars) == bar_count
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
