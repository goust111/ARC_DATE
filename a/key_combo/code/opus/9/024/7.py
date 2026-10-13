from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg


BAR_COLOR = Color.GREY
HIGHEST_COLOR = Color.BLUE
LOWEST_COLOR = Color.RED
BG_COLOR = Color.BLACK

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (15, 22)
BAR_COUNT_RANGE = (5, 7)
BAR_HEIGHT_RANGE = tuple(range(2, 13))
MAX_ATTEMPTS = 2000

_BRANCH_CALL_COUNT = 0
_UNUSED_OUTPUT_COMBOS = {
    0: {(hi, lo) for hi in BAR_HEIGHT_RANGE for lo in BAR_HEIGHT_RANGE if hi > lo},
    1: {(hi, lo) for hi in BAR_HEIGHT_RANGE for lo in BAR_HEIGHT_RANGE if hi > lo},
}


def _extract_bars(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected one foreground bar color"
    fg = int(fg_colors[0])

    cols = np.flatnonzero(np.any(g == fg, axis=0))
    assert cols.size >= 2, "expected at least two bars"
    if cols.size > 1:
        assert int(np.min(np.diff(cols))) >= 2, "bars must be separated by at least one empty column"

    bars: list[dict[str, int]] = []
    for c in cols:
        rows = np.flatnonzero(g[:, c] == fg)
        assert rows.size > 0
        r0 = int(rows[0])
        r1 = int(rows[-1])
        height = int(rows.size)
        assert height == r1 - r0 + 1, "bars must be vertically solid"
        assert r1 == h - 1, "bars must be bottom-aligned"
        assert np.all(g[r0 : r1 + 1, c] == fg)
        assert np.count_nonzero(g[:, c] != bg) == height, "bar columns must contain only one segment"
        bars.append({"height": height, "r0": r0, "r1": r1, "c": int(c)})

    heights = [bar["height"] for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be unique"
    return bg, bars


def _extreme_bars(bars: list[dict[str, int]]) -> tuple[dict[str, int], dict[str, int]]:
    tallest = max(bars, key=lambda bar: bar["height"])
    shortest = min(bars, key=lambda bar: bar["height"])
    assert tallest["height"] > shortest["height"]
    return tallest, shortest


def _cropped_extremes(grid: np.ndarray, bg: int, tallest: dict[str, int], shortest: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    recolored = np.full(grid.shape, bg, dtype=int)
    recolored[tallest["r0"] : tallest["r1"] + 1, tallest["c"]] = HIGHEST_COLOR
    recolored[shortest["r0"] : shortest["r1"] + 1, shortest["c"]] = LOWEST_COLOR

    tallest_box = bbox(recolored == HIGHEST_COLOR)
    shortest_box = bbox(recolored == LOWEST_COLOR)
    assert tallest_box is not None and shortest_box is not None

    tallest_crop = crop_bbox(recolored, tallest_box)
    shortest_crop = crop_bbox(recolored, shortest_box)
    assert tallest_crop.shape[1] == 1 and shortest_crop.shape[1] == 1

    tallest_crop = np.flipud(tallest_crop).copy()
    return tallest_crop, shortest_crop


def _compose_output(tallest_crop: np.ndarray, shortest_crop: np.ndarray, *, tallest_left: bool, bg: int) -> np.ndarray:
    tall_h = int(tallest_crop.shape[0])
    short_h = int(shortest_crop.shape[0])

    if tallest_left:
        out = np.full((max(tall_h, short_h), 2), bg, dtype=int)
        out[:tall_h, 0:1] = tallest_crop
        out[:short_h, 1:2] = shortest_crop
        return out

    out = np.full((tall_h + short_h, 1), bg, dtype=int)
    out[:tall_h, 0:1] = tallest_crop
    out[tall_h : tall_h + short_h, 0:1] = shortest_crop
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the tallest and shortest vertical bars, crop them, and pack them using their original left-right relation.

    Input structure
    - The grid has a dominant black background and one foreground color forming
      several separated one-cell-wide vertical bars.
    - Every bar is vertically solid, bottom-aligned to the last row, and has a
      unique height.

    Rule
    - Rank the bars by height and keep only the unique tallest and shortest.
    - Recolor the tallest bar blue and the shortest bar red, then crop each kept
      bar to its own bounding box.
    - Flip the tallest crop vertically, leave the shortest crop unchanged, and
      use the original column order to choose the packing direction: if the
      tallest was left of the shortest, pack them side-by-side with top
      alignment; otherwise stack them top-to-bottom with left alignment.

    Key insight
    - Height ranking chooses which two bars survive, while the original spatial
      relation between those two extremes encodes whether the output is a
      horizontal or vertical concatenation.
    """
    g = np.asarray(grid, dtype=int)
    bg, bars = _extract_bars(g)
    tallest, shortest = _extreme_bars(bars)
    tallest_crop, shortest_crop = _cropped_extremes(g, bg, tallest, shortest)
    return _compose_output(
        tallest_crop,
        shortest_crop,
        tallest_left=bool(tallest["c"] < shortest["c"]),
        bg=bg,
    )


def _sample_columns(width: int, count: int) -> list[int]:
    assert width >= 2 * count + 1
    slack = width - count - 1
    base = sorted(random.sample(range(slack), count))
    return [1 + base_i + i for i, base_i in enumerate(base)]


def _choose_output_pair(branch: int) -> tuple[int, int]:
    combos = _UNUSED_OUTPUT_COMBOS[branch]
    if combos:
        pair = random.choice(tuple(combos))
        combos.remove(pair)
        return pair
    hi = random.choice(BAR_HEIGHT_RANGE[1:])
    lo = random.choice([h for h in BAR_HEIGHT_RANGE if h < hi])
    return hi, lo


def _valid_extreme_indices(count: int, branch: int) -> list[tuple[int, int]]:
    pairs = []
    for tallest_idx in range(1, count - 1):
        for shortest_idx in range(1, count - 1):
            if abs(tallest_idx - shortest_idx) <= 1:
                continue
            if branch == 0 and tallest_idx < shortest_idx:
                pairs.append((tallest_idx, shortest_idx))
            if branch == 1 and tallest_idx > shortest_idx:
                pairs.append((tallest_idx, shortest_idx))
    assert pairs, "expected at least one valid extreme-index pair"
    return pairs


def _build_candidate(branch: int) -> np.ndarray:
    h = random.randint(*GRID_H_RANGE)
    count = random.randint(*BAR_COUNT_RANGE)
    min_width = max(GRID_W_RANGE[0], 2 * count + 1)
    w = random.randint(min_width, GRID_W_RANGE[1])

    columns = _sample_columns(w, count)
    tallest_idx, shortest_idx = random.choice(_valid_extreme_indices(count, branch))
    tallest_height, shortest_height = _choose_output_pair(branch)

    remaining_heights = [v for v in BAR_HEIGHT_RANGE if v not in {tallest_height, shortest_height}]
    middle_heights = random.sample(remaining_heights, count - 2)

    heights = [None] * count
    heights[tallest_idx] = tallest_height
    heights[shortest_idx] = shortest_height
    fill_positions = [idx for idx in range(count) if heights[idx] is None]
    random.shuffle(middle_heights)
    for idx, height in zip(fill_positions, middle_heights):
        heights[idx] = height

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for c, height in zip(columns, heights):
        grid[h - height :, c] = BAR_COLOR
    return grid


def _expected_from_structure(grid: np.ndarray) -> np.ndarray:
    bg, bars = _extract_bars(grid)
    tallest, shortest = _extreme_bars(bars)

    bars_by_col = sorted(bars, key=lambda bar: bar["c"])
    tallest_rank = next(i for i, bar in enumerate(bars_by_col) if bar["c"] == tallest["c"])
    shortest_rank = next(i for i, bar in enumerate(bars_by_col) if bar["c"] == shortest["c"])
    assert 0 < tallest_rank < len(bars_by_col) - 1
    assert 0 < shortest_rank < len(bars_by_col) - 1
    assert abs(tallest_rank - shortest_rank) > 1

    tallest_crop = np.flipud(np.full((tallest["height"], 1), HIGHEST_COLOR, dtype=int))
    shortest_crop = np.full((shortest["height"], 1), LOWEST_COLOR, dtype=int)
    return _compose_output(
        tallest_crop,
        shortest_crop,
        tallest_left=bool(tallest["c"] < shortest["c"]),
        bg=bg,
    )


def _validate_candidate(grid: np.ndarray, branch: int) -> bool:
    try:
        bg, bars = _extract_bars(grid)
        assert bg == BG_COLOR
        assert BAR_COUNT_RANGE[0] <= len(bars) <= BAR_COUNT_RANGE[1]

        tallest, shortest = _extreme_bars(bars)
        assert tallest["height"] in BAR_HEIGHT_RANGE
        assert shortest["height"] in BAR_HEIGHT_RANGE

        bars_by_col = sorted(bars, key=lambda bar: bar["c"])
        tallest_rank = next(i for i, bar in enumerate(bars_by_col) if bar["c"] == tallest["c"])
        shortest_rank = next(i for i, bar in enumerate(bars_by_col) if bar["c"] == shortest["c"])
        assert 0 < tallest_rank < len(bars_by_col) - 1
        assert 0 < shortest_rank < len(bars_by_col) - 1
        assert abs(tallest_rank - shortest_rank) > 1
        assert (tallest["c"] < shortest["c"]) == (branch == 0)

        expected = _expected_from_structure(grid)
        actual = main(grid)
        assert np.array_equal(actual, expected)

        if branch == 0:
            assert actual.shape == (tallest["height"], 2)
            assert np.all(actual[:, 0] == HIGHEST_COLOR)
            assert np.all(actual[: shortest["height"], 1] == LOWEST_COLOR)
            assert np.all(actual[shortest["height"] :, 1] == bg)
        else:
            assert actual.shape == (tallest["height"] + shortest["height"], 1)
            assert np.all(actual[: tallest["height"], 0] == HIGHEST_COLOR)
            assert np.all(actual[tallest["height"] :, 0] == LOWEST_COLOR)

        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    global _BRANCH_CALL_COUNT

    branch = _BRANCH_CALL_COUNT % 2
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate(branch)
        if _validate_candidate(grid, branch):
            _BRANCH_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
