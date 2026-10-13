from __future__ import annotations

import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "background must be uniquely dominant"
    return int(values[int(np.argmax(counts))])


def _find_baseline(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected exactly one full-width non-background baseline"
    return candidates[0]


def _extract_slots(grid: np.ndarray, bg: int) -> tuple[int, int, list[dict[str, int]]]:
    height, width = map(int, grid.shape)
    baseline_row, baseline_color = _find_baseline(grid, bg)
    assert 0 < baseline_row < height, "baseline must leave room for bars above it"

    support = grid[baseline_row - 1]
    slots: list[dict[str, int]] = []
    occupied = np.zeros((baseline_row, width), dtype=bool)

    c = 0
    while c < width:
        if int(support[c]) == bg:
            c += 1
            continue

        color = int(support[c])
        left = c
        while c + 1 < width and int(support[c + 1]) == color:
            c += 1
        right = c

        top = baseline_row - 1
        while top >= 0 and bool(np.all(grid[top, left : right + 1] == color)):
            top -= 1
        top += 1

        bar = grid[top:baseline_row, left : right + 1]
        assert bar.size > 0
        assert bool(np.all(bar == color)), "each bar must be a solid monochrome rectangle"
        if top > 0:
            assert bool(np.all(grid[:top, left : right + 1] == bg)), "bars must end cleanly above"

        occupied[top:baseline_row, left : right + 1] = True
        slots.append(
            {
                "left": int(left),
                "right": int(right),
                "width": int(right - left + 1),
                "color": color,
                "height": int(baseline_row - top),
            }
        )
        c += 1

    assert len(slots) >= 2, "expected multiple bar slots"
    assert len({slot["color"] for slot in slots}) == len(slots), "bar colors must be unique"
    assert bool(np.all(grid[baseline_row + 1 :, :] == bg)), "only the baseline may appear below the bars"
    assert np.array_equal(grid[:baseline_row, :] != bg, occupied), "unexpected foreground above baseline"
    return baseline_row, baseline_color, slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the bar heights from shortest to tallest, then assign them left-to-right to the original color slots.

    Input structure:
    - A dominant background surrounds one full-width horizontal baseline.
    - Several solid monochrome vertical bars stand immediately above the baseline.
    - Each bar defines a left-to-right slot by its original position and color.

    Rule:
    - Order the bar heights from shortest to tallest.
    - Order the slots by original column position from left to right.
    - Put the k-th shortest height into the k-th slot, repainting it with that slot's original color.
    - Keep the baseline unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    baseline_row, _baseline_color, slots = _extract_slots(g, bg)

    heights = sorted(slot["height"] for slot in slots)
    ordered_slots = sorted(slots, key=lambda slot: slot["left"])

    out = g.copy()
    out[:baseline_row, :] = bg
    for slot, bar_height in zip(ordered_slots, heights):
        top = baseline_row - int(bar_height)
        out[top:baseline_row, slot["left"] : slot["right"] + 1] = slot["color"]

    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(10))
BAR_COUNT_RANGE = (3, 6)
BAR_WIDTH_RANGE = (1, 3)
BAR_HEIGHT_RANGE = (1, 8)
TOP_MARGIN_RANGE = (0, 3)
BOTTOM_PADDING_RANGE = (0, 2)
SIDE_MARGIN_RANGE = (0, 3)
GAP_RANGE = (1, 3)
GRID_HEIGHT_LIMITS = (6, 16)
GRID_WIDTH_LIMITS = (8, 26)
GENERATION_RETRIES = 300


def _build_grid(
    bg: int,
    baseline_color: int,
    baseline_row: int,
    total_height: int,
    total_width: int,
    bars: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full((total_height, total_width), bg, dtype=int)
    grid[baseline_row, :] = baseline_color
    for bar in bars:
        top = baseline_row - int(bar["height"])
        grid[top:baseline_row, bar["left"] : bar["right"] + 1] = int(bar["color"])
    return grid


def _expected_output(
    bg: int,
    baseline_color: int,
    baseline_row: int,
    total_height: int,
    total_width: int,
    bars: list[dict[str, int]],
) -> np.ndarray:
    ordered = sorted(bars, key=lambda bar: bar["left"])
    sorted_heights = sorted(int(bar["height"]) for bar in ordered)
    out_bars = []
    for bar, new_height in zip(ordered, sorted_heights):
        out_bars.append(
            {
                "left": int(bar["left"]),
                "right": int(bar["right"]),
                "color": int(bar["color"]),
                "height": int(new_height),
            }
        )
    return _build_grid(bg, baseline_color, baseline_row, total_height, total_width, out_bars)


def _choose_bar_width(bar_count: int) -> int:
    widths = [w for w in range(BAR_WIDTH_RANGE[0], BAR_WIDTH_RANGE[1] + 1) if bar_count * w <= GRID_WIDTH_LIMITS[1] - 2]
    return random.choice(widths)


def _sample_layout(bar_count: int, bar_width: int) -> tuple[int, int, list[int]]:
    gaps = [random.randint(*GAP_RANGE) for _ in range(bar_count - 1)]
    left_margin = random.randint(*SIDE_MARGIN_RANGE)
    right_margin = random.randint(*SIDE_MARGIN_RANGE)
    total_width = left_margin + right_margin + bar_count * bar_width + sum(gaps)
    return left_margin, total_width, gaps


def _sample_heights(bar_count: int) -> list[int] | None:
    max_height = BAR_HEIGHT_RANGE[1]
    if bar_count > max_height:
        return None
    heights = random.sample(range(BAR_HEIGHT_RANGE[0], max_height + 1), bar_count)
    if heights == sorted(heights):
        return None
    return heights


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(*BAR_COUNT_RANGE)
    bar_width = _choose_bar_width(bar_count)
    heights = _sample_heights(bar_count)
    if heights is None:
        return None

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_padding = random.randint(*BOTTOM_PADDING_RANGE)
    tallest = max(heights)
    baseline_row = tallest + top_margin
    total_height = baseline_row + 1 + bottom_padding
    if not (GRID_HEIGHT_LIMITS[0] <= total_height <= GRID_HEIGHT_LIMITS[1]):
        return None

    left_margin, total_width, gaps = _sample_layout(bar_count, bar_width)
    if not (GRID_WIDTH_LIMITS[0] <= total_width <= GRID_WIDTH_LIMITS[1]):
        return None

    palette = random.sample(COLOR_POOL, bar_count + 2)
    bg = int(palette[0])
    baseline_color = int(palette[1])
    bar_colors = [int(color) for color in palette[2:]]

    bars: list[dict[str, int]] = []
    col = left_margin
    for idx in range(bar_count):
        bars.append(
            {
                "left": int(col),
                "right": int(col + bar_width - 1),
                "color": int(bar_colors[idx]),
                "height": int(heights[idx]),
            }
        )
        col += bar_width
        if idx < len(gaps):
            col += gaps[idx]

    inp = _build_grid(bg, baseline_color, baseline_row, total_height, total_width, bars)
    out = _expected_output(bg, baseline_color, baseline_row, total_height, total_width, bars)

    try:
        if _dominant_color(inp) != bg:
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
