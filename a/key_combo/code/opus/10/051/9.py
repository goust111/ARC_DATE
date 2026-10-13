from __future__ import annotations

import random
from typing import Literal

import numpy as np


BG = 0
RED = 2
GREEN = 3
ORANGE = 7

GEN_ATTEMPTS = 600
GRID_ROWS_RANGE = (17, 26)
GRID_COLS_RANGE = (16, 30)
BAR_COUNT_RANGE = (5, 7)
STEP_RANGE = (1, 3)


def _is_contiguous(indices: np.ndarray) -> bool:
    if len(indices) == 0:
        return False
    return bool(np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)))


def _parse_top_step(grid: np.ndarray) -> int:
    red_cols = np.flatnonzero(grid[0] == RED)
    assert 1 <= len(red_cols) <= 3, "expected a 1..3-pixel top-row red segment"
    assert _is_contiguous(red_cols), "top-row red pixels must form one horizontal segment"
    assert not np.any(grid[1:] == RED), "red is reserved for the top-row step marker"
    return int(len(red_cols))


def _parse_bars(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    bar_cols = [c for c in range(w) if np.any(np.isin(grid[:, c], (GREEN, ORANGE)))]
    assert len(bar_cols) >= 3, "expected at least three vertical bars"

    bars: list[dict] = []
    common_bottom = None
    marked_count = 0

    for c in bar_cols:
        rows = np.flatnonzero(np.isin(grid[:, c], (GREEN, ORANGE)))
        assert _is_contiguous(rows), "each bar must occupy consecutive rows"
        top = int(rows[0])
        bottom = int(rows[-1])
        height = int(len(rows))
        assert top > 0, "bars must stay below the top marker row"
        assert height % 2 == 0, "bar heights must be even"

        if common_bottom is None:
            common_bottom = bottom
        else:
            assert bottom == common_bottom, "all bars must be bottom-aligned"

        strip = grid[top : bottom + 1, c]
        assert np.all(np.isin(strip, (GREEN, ORANGE))), "bars may only use green/orange"

        orange_rows = np.flatnonzero(strip == ORANGE)
        mark_half: Literal["top", "bottom"] | None = None
        if len(orange_rows):
            marked_count += 1
            half = height // 2
            top_half = np.arange(0, half)
            bot_half = np.arange(half, height)
            assert len(orange_rows) == half, "the marker must occupy exactly half the bar"
            if np.array_equal(orange_rows, top_half):
                mark_half = "top"
            else:
                assert np.array_equal(orange_rows, bot_half), "marker must be top or bottom half"
                mark_half = "bottom"

        bars.append(
            {
                "col": int(c),
                "top": top,
                "bottom": bottom,
                "height": height,
                "mark_half": mark_half,
            }
        )

    assert marked_count == 1, "expected exactly one marked bar"
    heights = [bar["height"] for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be pairwise distinct"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Move the orange half-bar marker forward by k steps in the height-sorted vertical-bar cycle.

    Input structure:
    - The grid is mostly black background with 5-7 single-column green vertical bars of distinct even heights.
    - All bars share one bottom row; exactly one bar has either its top half or bottom half recolored orange.
    - The top row contains one short red horizontal segment whose length k encodes the cyclic shift amount.

    Rule:
    - Count the red cells on the top row to obtain k.
    - Sort the vertical bars by height from shortest to tallest and find the marked bar's rank in that order.
    - Erase the orange half from the source bar, then move that half-marker to the bar k positions later in the sorted cycle, wrapping around.
    - On the destination bar, recolor the same half type (top half or bottom half) orange and keep the red step marker unchanged.

    Key insight:
    - The destination is determined jointly by two independent cues: bar-height ordering gives the cycle, while the separate red segment gives the forward offset along that cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    step = _parse_top_step(g)
    bars = _parse_bars(g)
    ordered = sorted(bars, key=lambda bar: bar["height"])

    source_idx = next(i for i, bar in enumerate(ordered) if bar["mark_half"] is not None)
    source_bar = ordered[source_idx]
    target_bar = ordered[(source_idx + step) % len(ordered)]
    mark_half = source_bar["mark_half"]
    assert mark_half is not None

    out = g.copy()
    out[source_bar["top"] : source_bar["bottom"] + 1, source_bar["col"]] = GREEN
    out[target_bar["top"] : target_bar["bottom"] + 1, target_bar["col"]] = GREEN

    target_half = target_bar["height"] // 2
    if mark_half == "top":
        out[target_bar["top"] : target_bar["top"] + target_half, target_bar["col"]] = ORANGE
    else:
        out[target_bar["top"] + target_half : target_bar["bottom"] + 1, target_bar["col"]] = ORANGE
    return out


def _sample_nonadjacent_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _choose_source_rank(bar_count: int, step: int) -> tuple[int, bool]:
    want_wrap = random.choice((False, True))
    if want_wrap:
        return random.randint(bar_count - step, bar_count - 1), True
    return random.randint(0, bar_count - step - 1), False


def _render_expected(
    inp: np.ndarray,
    sorted_bars: list[dict],
    source_rank: int,
    target_rank: int,
    mark_half: Literal["top", "bottom"],
) -> np.ndarray:
    out = inp.copy()
    source_bar = sorted_bars[source_rank]
    target_bar = sorted_bars[target_rank]

    out[source_bar["top"] : source_bar["bottom"] + 1, source_bar["col"]] = GREEN
    out[target_bar["top"] : target_bar["bottom"] + 1, target_bar["col"]] = GREEN

    split = target_bar["height"] // 2
    if mark_half == "top":
        out[target_bar["top"] : target_bar["top"] + split, target_bar["col"]] = ORANGE
    else:
        out[target_bar["top"] + split : target_bar["bottom"] + 1, target_bar["col"]] = ORANGE
    return out


def _semantic_output_ok(
    inp: np.ndarray,
    out: np.ndarray,
    sorted_bars: list[dict],
    source_rank: int,
    target_rank: int,
    step: int,
    mark_half: Literal["top", "bottom"],
) -> bool:
    if not np.array_equal(inp[0], out[0]):
        return False
    if _parse_top_step(out) != step:
        return False

    parsed_out = _parse_bars(out)
    parsed_sorted = sorted(parsed_out, key=lambda bar: bar["height"])
    marked = [i for i, bar in enumerate(parsed_sorted) if bar["mark_half"] is not None]
    if marked != [target_rank]:
        return False
    if parsed_sorted[target_rank]["mark_half"] != mark_half:
        return False

    source_bar = sorted_bars[source_rank]
    if not np.all(out[source_bar["top"] : source_bar["bottom"] + 1, source_bar["col"]] == GREEN):
        return False

    expected = _render_expected(inp, sorted_bars, source_rank, target_rank, mark_half)
    if not np.array_equal(out, expected):
        return False

    delta = inp != out
    changed_cols = {int(c) for _, c in np.argwhere(delta)}
    allowed_cols = {sorted_bars[source_rank]["col"], sorted_bars[target_rank]["col"]}
    return changed_cols <= allowed_cols and bool(np.any(delta))


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    bar_count = random.randint(*BAR_COUNT_RANGE)
    step = random.randint(*STEP_RANGE)

    bar_cols = _sample_nonadjacent_columns(cols, bar_count)
    if bar_cols is None:
        return None

    baseline = random.randint(rows - 2, rows - 1)
    max_height = baseline - 1
    even_heights = list(range(2, max_height + 1, 2))
    if len(even_heights) < bar_count:
        return None

    chosen_heights = random.sample(even_heights, bar_count)
    random.shuffle(chosen_heights)
    bars = []
    for col, height in zip(bar_cols, chosen_heights):
        top = baseline - height + 1
        bars.append({"col": col, "top": top, "bottom": baseline, "height": height})

    sorted_bars = sorted(bars, key=lambda bar: bar["height"])
    source_rank, _wrap = _choose_source_rank(bar_count, step)
    target_rank = (source_rank + step) % bar_count
    mark_half: Literal["top", "bottom"] = random.choice(("top", "bottom"))

    inp = np.full((rows, cols), BG, dtype=int)
    line_start = random.randint(0, cols - step)
    inp[0, line_start : line_start + step] = RED

    for bar in bars:
        inp[bar["top"] : bar["bottom"] + 1, bar["col"]] = GREEN

    source_bar = sorted_bars[source_rank]
    split = source_bar["height"] // 2
    if mark_half == "top":
        inp[source_bar["top"] : source_bar["top"] + split, source_bar["col"]] = ORANGE
    else:
        inp[source_bar["top"] + split : source_bar["bottom"] + 1, source_bar["col"]] = ORANGE

    try:
        expected = _render_expected(inp, sorted_bars, source_rank, target_rank, mark_half)
        pred = main(inp)
        if not np.array_equal(pred, expected):
            return None
        if not _semantic_output_ok(inp, pred, sorted_bars, source_rank, target_rank, step, mark_half):
            return None
    except AssertionError:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
