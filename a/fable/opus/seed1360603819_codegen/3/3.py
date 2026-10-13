from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_bars(grid: np.ndarray, bg: int) -> tuple[list[dict], dict, int]:
    height, width = grid.shape
    bars: list[dict] = []
    source_bar: dict | None = None

    for col in range(width):
        rows = np.flatnonzero(grid[:, col] != bg)
        if rows.size == 0:
            continue

        expected = np.arange(int(rows[0]), int(rows[-1]) + 1)
        assert np.array_equal(rows, expected), "each foreground column must be one contiguous bar"

        colors = {int(v) for v in np.unique(grid[rows, col])}
        assert colors <= {GREEN, RED}, "bars may only use green and red"

        top = int(rows[0])
        bottom = int(rows[-1])
        bar_height = int(rows.size)
        assert bar_height % 2 == 0, "bar heights must be even so the marker can occupy exactly one half"

        bar = {
            "col": int(col),
            "top": top,
            "bottom": bottom,
            "height": bar_height,
        }

        red_rows = rows[grid[rows, col] == RED]
        if red_rows.size:
            assert source_bar is None, "expected exactly one marked source bar"
            assert GREEN in colors and RED in colors, "the source bar must contain one green half and one red half"
            assert red_rows.size * 2 == bar_height, "the red marker must occupy exactly half the source bar"
            assert np.array_equal(red_rows, np.arange(int(red_rows[0]), int(red_rows[-1]) + 1))

            half = bar_height // 2
            top_half = np.arange(top, top + half)
            bottom_half = np.arange(top + half, bottom + 1)
            if np.array_equal(red_rows, top_half):
                bar["marked_half"] = "top"
            elif np.array_equal(red_rows, bottom_half):
                bar["marked_half"] = "bottom"
            else:
                raise AssertionError("the red marker must be exactly the upper or lower half")
            source_bar = bar
        else:
            assert colors == {GREEN}, "unmarked bars must be fully green"

        bars.append(bar)

    assert len(bars) >= 2, "expected at least two bars"
    assert source_bar is not None, "expected exactly one marked source bar"

    bars.sort(key=lambda bar: int(bar["height"]))
    heights = [int(bar["height"]) for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be unique"
    assert all(b - a == 2 for a, b in zip(heights, heights[1:])), "adjacent ranks must differ by two pixels"

    source_index = bars.index(source_bar)
    assert source_index + 1 < len(bars), "the source bar must have a unique next rank"
    return bars, source_bar, source_index


def _paint_bar_half(grid: np.ndarray, bar: dict, half_name: str, color: int) -> None:
    split = int(bar["top"]) + int(bar["height"]) // 2
    col = int(bar["col"])
    if half_name == "top":
        grid[int(bar["top"]) : split, col] = color
    elif half_name == "bottom":
        grid[split : int(bar["bottom"]) + 1, col] = color
    else:
        raise AssertionError(f"unknown half {half_name}")


def _draw_nested_boxes(grid: np.ndarray, start_col: int, rank: int, bg: int) -> None:
    height, width = grid.shape
    rings = rank - 1
    size = 2 * rank - 1
    available_width = width - start_col

    assert size <= height, "right-side figure must fit vertically"
    assert size <= available_width, "right-side figure must fit horizontally"
    assert bool(np.all(grid[:, start_col:] == bg)), "the right half must be blank before drawing"

    top = (height - size) // 2
    left = start_col + (available_width - size) // 2
    center_r = top + size // 2
    center_c = left + size // 2
    grid[center_r, center_c] = RED

    for layer in range(1, rings + 1):
        r0 = center_r - layer
        r1 = center_r + layer
        c0 = center_c - layer
        c1 = center_c + layer
        grid[r0, c0 : c1 + 1] = GREEN
        grid[r1, c0 : c1 + 1] = GREEN
        grid[r0 : r1 + 1, c0] = GREEN
        grid[r0 : r1 + 1, c1] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    bars, source_bar, source_index = _extract_bars(g, bg)
    dest_bar = bars[source_index + 1]
    dest_rank = source_index + 2

    out = g.copy()
    _paint_bar_half(out, source_bar, str(source_bar["marked_half"]), GREEN)
    assert np.all(out[:, int(dest_bar["col"])] != RED), "destination bar must start fully green"
    _paint_bar_half(out, dest_bar, str(source_bar["marked_half"]), RED)

    right_start = max(int(bar["col"]) for bar in bars) + 1
    _draw_nested_boxes(out, right_start, dest_rank, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(3, 7)
    min_height = random.choice((2, 4, 6, 8))
    sorted_heights = [min_height + 2 * i for i in range(bar_count)]
    max_height = sorted_heights[-1]
    frame_size = 2 * bar_count - 1

    left_margin = random.randint(0, 3)
    cols = [left_margin]
    for _ in range(bar_count - 1):
        cols.append(cols[-1] + 1 + random.randint(1, 3))
    last_col = cols[-1]

    canvas_height = max(max_height, frame_size) + random.randint(2, 7)
    bottom_row = random.randint(max_height - 1, canvas_height - 1)

    right_blank_width = max(
        frame_size + random.randint(2, 8),
        last_col + random.randint(2, 8),
    )
    canvas_width = last_col + 1 + right_blank_width

    grid = np.full((canvas_height, canvas_width), BG, dtype=int)

    permuted_heights = sorted_heights[:]
    random.shuffle(permuted_heights)
    bars_by_rank: list[dict] = [None] * bar_count  # type: ignore[assignment]
    for col, bar_height in zip(cols, permuted_heights):
        top = bottom_row - bar_height + 1
        if top < 0:
            return None
        grid[top : bottom_row + 1, col] = GREEN
        rank = sorted_heights.index(bar_height) + 1
        bars_by_rank[rank - 1] = {
            "col": col,
            "top": top,
            "height": bar_height,
        }

    source_rank = random.randint(1, bar_count - 1)
    source_bar = bars_by_rank[source_rank - 1]
    half_name = random.choice(("top", "bottom"))
    split = int(source_bar["top"]) + int(source_bar["height"]) // 2
    if half_name == "top":
        grid[int(source_bar["top"]) : split, int(source_bar["col"])] = RED
    else:
        grid[split : int(source_bar["top"]) + int(source_bar["height"]), int(source_bar["col"])] = RED

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
