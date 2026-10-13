from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

LOWER_PALETTE = (BLUE, GREEN, YELLOW, GREY)
MARKER_TEMPLATES = {
    3: ((BLUE, 7, BLUE), (GREEN, 9, GREEN)),
    4: ((YELLOW, 7, 9, YELLOW), (GREY, 6, 8, GREY)),
}

MIN_BAR_COUNT = 3
MAX_BAR_COUNT = 4
MIN_BAR_HEIGHT = 4
MAX_BAR_HEIGHT = 10
GEN_ATTEMPTS = 400


def _as_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    return g


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    g = _as_grid(grid)
    height, width = g.shape
    bars: list[dict[str, int]] = []

    for col in range(width):
        row = 0
        seen_segments = 0
        while row < height:
            if g[row, col] == BLACK:
                row += 1
                continue

            start = row
            while row < height and g[row, col] != BLACK:
                row += 1
            end = row - 1
            seen_segments += 1

            segment = g[start : end + 1, col]
            if segment.size < MIN_BAR_HEIGHT:
                raise AssertionError("each bar must have height at least four")

            marker_offsets = np.flatnonzero((segment != RED) & (segment != BLACK))
            if marker_offsets.size != 1:
                raise AssertionError("each bar must contain exactly one non-red non-black marker pixel")

            if int(np.count_nonzero(segment == RED)) != int(segment.size) - 1:
                raise AssertionError("all non-marker cells in a bar must be red")

            marker_offset = int(marker_offsets[0])
            marker_row = start + marker_offset
            if not (start < marker_row < end):
                raise AssertionError("marker pixel must be strictly inside the bar")

            bars.append(
                {
                    "col": int(col),
                    "top": int(start),
                    "bottom": int(end),
                    "marker_row": int(marker_row),
                    "marker_color": int(segment[marker_offset]),
                }
            )

        if seen_segments > 1:
            raise AssertionError("an occupied column may contain only one maximal vertical bar")

    if not bars:
        raise AssertionError("expected at least one bar")
    if len(bars) > len(LOWER_PALETTE):
        raise AssertionError("at most four bars are supported by the palette")

    bars.sort(key=lambda bar: bar["col"])
    for left, right in zip(bars, bars[1:]):
        if right["col"] - left["col"] <= 1:
            raise AssertionError("bars must occupy distinct non-adjacent columns")

    return bars


def _render_output(grid: np.ndarray, bars: list[dict[str, int]]) -> np.ndarray:
    out = _as_grid(grid).copy()
    for rank, bar in enumerate(bars):
        col = bar["col"]
        top = bar["top"]
        bottom = bar["bottom"]
        marker_row = bar["marker_row"]
        out[top : marker_row + 1, col] = RED
        out[marker_row + 1 : bottom + 1, col] = LOWER_PALETTE[rank]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each marked red vertical bar at its marker row, then recolor the lower segment by column rank.

    Input structure:
    - The background is black.
    - The foreground consists of three or four width-1 vertical red bars in distinct non-adjacent columns.
    - Each bar is a single contiguous non-black segment with exactly one non-red non-black marker pixel strictly inside it.

    Rule:
    - Treat the marker pixel's row as that bar's cut position.
    - Recolor the whole upper segment, including the marker row, back to red.
    - Sort the bars from left to right and recolor each lower segment by rank: blue, green, yellow, then grey.

    Key insight:
    - Two independent clues meet on each bar: the local marker row decides where to split, while the global left-to-right column order decides the lower-segment color.
    """
    g = _as_grid(grid)
    bars = _extract_bars(g)
    if len(bars) not in (MIN_BAR_COUNT, MAX_BAR_COUNT):
        raise AssertionError("expected three or four bars")
    return _render_output(g, bars)


def _sample_columns(width: int, count: int) -> list[int]:
    candidates = list(range(1, width - 1))
    for _ in range(100):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    raise RuntimeError("failed to choose separated columns")


def _sample_heights(bar_count: int, grid_height: int) -> list[int]:
    max_height = min(MAX_BAR_HEIGHT, grid_height - 2)
    if max_height < MIN_BAR_HEIGHT:
        raise RuntimeError("grid too short for valid bars")

    shared_height = random.randint(max(MIN_BAR_HEIGHT, 5), max_height)
    heights = [shared_height, shared_height]

    different_choices = [h for h in range(MIN_BAR_HEIGHT, max_height + 1) if h != shared_height]
    if not different_choices:
        different_choices = [shared_height]
    heights.append(random.choice(different_choices))

    if bar_count == 4:
        heights.append(random.randint(MIN_BAR_HEIGHT, max_height))

    return heights


def _sample_marker_offsets(heights: list[int]) -> list[int]:
    shared_height = heights[0]
    first = random.randint(1, shared_height - 2)
    second_choices = [offset for offset in range(1, shared_height - 1) if offset != first]
    second = random.choice(second_choices)

    offsets = [first, second]
    third_limit = heights[2] - 2
    if first <= third_limit:
        offsets.append(first)
    else:
        offsets.append(random.randint(1, third_limit))

    if len(heights) == 4:
        offsets.append(random.randint(1, heights[3] - 2))

    return offsets


def _sample_tops(grid_height: int, heights: list[int], offsets: list[int]) -> list[int]:
    tops: list[int] = []
    for idx, height in enumerate(heights):
        top = random.randint(1, grid_height - height - 1)
        if idx == 1 and top == tops[0]:
            alternatives = [r for r in range(1, grid_height - height) if r != tops[0]]
            if alternatives:
                top = random.choice(alternatives)
        tops.append(top)

    marker_rows = [top + offset for top, offset in zip(tops, offsets)]
    if len(set(marker_rows)) == 1:
        adjust_idx = len(tops) - 1
        height = heights[adjust_idx]
        current = tops[adjust_idx]
        alternatives = [r for r in range(1, grid_height - height) if r != current]
        if alternatives:
            tops[adjust_idx] = random.choice(alternatives)

    return tops


def _marker_colors_for_layout(
    bar_count: int, cols: list[int], heights: list[int], offsets: list[int], tops: list[int]
) -> list[int]:
    templates = MARKER_TEMPLATES[bar_count]
    template_index = (sum(cols) + sum(heights) + sum(offsets) + sum(tops)) % len(templates)
    return list(templates[template_index])


def _build_candidate() -> np.ndarray:
    grid_height = random.randint(12, 16)
    min_width = 11
    grid_width = random.randint(min_width, 16)
    bar_count = random.choice((3, 4))

    cols = _sample_columns(grid_width, bar_count)
    heights = _sample_heights(bar_count, grid_height)
    offsets = _sample_marker_offsets(heights)
    tops = _sample_tops(grid_height, heights, offsets)
    marker_colors = _marker_colors_for_layout(bar_count, cols, heights, offsets, tops)

    grid = np.zeros((grid_height, grid_width), dtype=int)
    for col, top, height, offset, marker_color in zip(cols, tops, heights, offsets, marker_colors):
        bottom = top + height - 1
        grid[top : bottom + 1, col] = RED
        grid[top + offset, col] = marker_color

    return grid


def _covers_counterexamples(grid: np.ndarray) -> bool:
    bars = _extract_bars(grid)
    if len(bars) not in (MIN_BAR_COUNT, MAX_BAR_COUNT):
        return False

    heights = [bar["bottom"] - bar["top"] + 1 for bar in bars]
    offsets = [bar["marker_row"] - bar["top"] for bar in bars]
    marker_colors = [bar["marker_color"] for bar in bars]

    if any(height > MAX_BAR_HEIGHT for height in heights):
        return False
    if len(set(offsets)) < 2:
        return False

    same_height_different_cut = any(
        heights[i] == heights[j] and offsets[i] != offsets[j]
        for i in range(len(bars))
        for j in range(i + 1, len(bars))
    )
    if not same_height_different_cut:
        return False

    repeated_marker_different_rank = any(
        marker_colors[i] == marker_colors[j]
        for i in range(len(bars))
        for j in range(i + 1, len(bars))
    )
    if not repeated_marker_different_rank:
        return False

    if len(set(bar["top"] for bar in bars)) < 2:
        return False
    if not any(color not in LOWER_PALETTE for color in marker_colors):
        return False

    out = _render_output(grid, bars)
    for rank, bar in enumerate(bars):
        lower = out[bar["marker_row"] + 1 : bar["bottom"] + 1, bar["col"]]
        if lower.size == 0 or not np.all(lower == LOWER_PALETTE[rank]):
            return False
        upper = out[bar["top"] : bar["marker_row"] + 1, bar["col"]]
        if not np.all(upper == RED):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if _covers_counterexamples(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
