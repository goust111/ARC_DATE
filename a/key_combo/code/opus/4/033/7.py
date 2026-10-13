from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL)
MIN_HEIGHT = 3
MAX_HEIGHT = 10
MIN_BAR_COUNT = 4
MAX_BAR_COUNT = 5
MAX_GENERATION_ATTEMPTS = 500


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert max(h, w) <= 30, "ARC grids must fit within 30x30"

    comps = components_mask(g != Color.BLACK, connectivity=4)
    assert comps, "expected at least one non-black component"

    bars: list[dict[str, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = bbox_coords(comp)
        assert c0 == c1, "each object must be one column wide"
        assert r1 == h - 1, "each bar must touch the bottom edge"

        column = g[r0 : r1 + 1, c0]
        assert int(column[0]) in MARKER_COLORS, "bar top must be a valid marker color"
        if len(column) > 1:
            assert np.all(column[1:] == Color.GREY), "all non-top bar cells must be grey"

        bars.append({"col": int(c0), "height": int(r1 - r0 + 1), "marker": int(column[0])})

    bars.sort(key=lambda bar: bar["col"])

    marker_colors = [bar["marker"] for bar in bars]
    heights = [bar["height"] for bar in bars]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be unique"
    assert len(heights) == len(set(heights)), "bar heights must be unique"
    for bar in bars:
        assert MIN_HEIGHT <= bar["height"] <= min(MAX_HEIGHT, h)
    for left, right in zip(bars, bars[1:]):
        assert right["col"] - left["col"] >= 4, "bars need at least three empty columns between them"
    return bars


def _rank_map(bars: list[dict[str, int]], mode: str) -> dict[int, int]:
    if mode == "height_desc":
        ordered = sorted(bars, key=lambda bar: (-bar["height"], bar["col"]))
        return {bar["height"]: idx + 1 for idx, bar in enumerate(ordered)}
    if mode == "height_asc":
        ordered = sorted(bars, key=lambda bar: (bar["height"], bar["col"]))
        return {bar["height"]: idx + 1 for idx, bar in enumerate(ordered)}
    if mode == "left_to_right":
        return {bar["height"]: idx + 1 for idx, bar in enumerate(sorted(bars, key=lambda bar: bar["col"]))}
    if mode == "right_to_left":
        ordered = sorted(bars, key=lambda bar: -bar["col"])
        return {bar["height"]: idx + 1 for idx, bar in enumerate(ordered)}
    raise ValueError(f"unknown rank mode: {mode}")


def _render_output(
    shape: tuple[int, int],
    bars: list[dict[str, int]],
    *,
    rank_mode: str = "height_desc",
    process_order: str = "left_to_right",
    recolor: bool = True,
    erase_originals: bool = True,
    copies_mode: str = "rank",
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=int)

    if not erase_originals:
        for bar in bars:
            top = h - bar["height"]
            out[top + 1 : h, bar["col"]] = Color.GREY
            out[top, bar["col"]] = bar["marker"]

    ranks = _rank_map(bars, rank_mode)
    ordered_bars = sorted(bars, key=lambda bar: bar["col"], reverse=(process_order == "right_to_left"))

    for bar in ordered_bars:
        top = h - bar["height"]
        if copies_mode == "rank":
            copies = ranks[bar["height"]]
        elif copies_mode == "height":
            copies = bar["height"]
        else:
            raise ValueError(f"unknown copies mode: {copies_mode}")

        color = bar["marker"] if recolor else Color.GREY
        for step in range(1, copies + 1):
            col = bar["col"] + 2 * step
            if 0 <= col < w:
                out[top:h, col] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the original marked grey bars, then copy each bar to the right as many times as its height rank.

    Input structure:
    - The grid is black except for several bottom-aligned one-column bars.
    - Each bar is grey except for its top cell, which is a unique marker color from the non-grey palette.
    - Bar heights are distinct, bars are separated horizontally, and height includes the colored top cell.

    Rule:
    - Identify every one-column bar whose bottom touches the grid edge and whose top cell is the only non-grey cell in that column-object.
    - Rank the bars by height from tallest to shortest, assigning ranks 1, 2, ..., N.
    - For a bar of height h at column c with marker color C and rank r, paint copies of height h at columns c+2, c+4, ..., c+2r, clipped at the right edge, using color C for the whole copy.
    - Remove all original bars and markers, and apply the copies in left-to-right bar order so later overlaps overwrite earlier ones.

    Key insight:
    - The number of copies is not the physical height and not the left-to-right order; it is the bar's ordinal height rank, while the marker color independently sets the copy color.
    """
    bars = _extract_bars(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid).shape, bars)


def _is_monotonic(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _sample_height_order(count: int, max_height: int) -> list[int] | None:
    candidates = list(range(MIN_HEIGHT, max_height + 1))
    if len(candidates) < count:
        return None
    for _ in range(80):
        heights = random.sample(candidates, count)
        if not _is_monotonic(heights):
            return heights
    return None


def _sample_columns(width: int, bars: list[dict[str, int]], want_overlap: bool) -> list[int] | None:
    count = len(bars)
    max_col = width - 1 - 2 * count
    if max_col < 4 * (count - 1):
        return None

    for _ in range(300):
        cols = sorted(random.sample(range(max_col + 1), count))
        if any(b - a < 4 for a, b in zip(cols, cols[1:])):
            continue

        candidate_bars = [
            {"col": col, "height": bar["height"], "marker": bar["marker"]}
            for col, bar in zip(cols, bars)
        ]
        actual = _render_output((12, width), candidate_bars)
        reversed_order = _render_output((12, width), candidate_bars, process_order="right_to_left")
        has_order_sensitive_overlap = not np.array_equal(actual, reversed_order)
        if has_order_sensitive_overlap == want_overlap:
            return cols
    return None


def _validate_generated_instance(grid: np.ndarray, *, expect_overlap: bool) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bars = _extract_bars(g)
    assert MIN_BAR_COUNT <= len(bars) <= MAX_BAR_COUNT
    assert np.all(g[:-1, :] >= 0)

    heights = [bar["height"] for bar in bars]
    assert not _is_monotonic(heights), "height order must not collapse to a left-right shortcut"

    actual = main(g)
    expected = _render_output(g.shape, bars)
    assert np.array_equal(actual, expected)

    wrong_left_rank = _render_output(g.shape, bars, rank_mode="left_to_right")
    wrong_low_rank = _render_output(g.shape, bars, rank_mode="height_asc")
    wrong_height_copies = _render_output(g.shape, bars, copies_mode="height")
    wrong_color = _render_output(g.shape, bars, recolor=False)
    wrong_erase = _render_output(g.shape, bars, erase_originals=False)
    reversed_order = _render_output(g.shape, bars, process_order="right_to_left")

    assert not np.array_equal(actual, wrong_left_rank), "sample allows left-to-right rank shortcut"
    assert not np.array_equal(actual, wrong_low_rank), "sample allows shortest-first rank shortcut"
    assert not np.array_equal(actual, wrong_height_copies), "sample allows copies=height shortcut"
    assert not np.array_equal(actual, wrong_color), "sample must require recoloring"
    assert not np.array_equal(actual, wrong_erase), "sample must require erasing originals"

    has_overlap = not np.array_equal(actual, reversed_order)
    if expect_overlap:
        assert has_overlap, "overlap branch missing"

    assert np.any(actual != Color.BLACK), "output must contain visible copies"
    assert not np.any((actual == Color.GREY)), "output should not retain grey originals"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        count = random.choice((4, 5))
        height = random.randint(11, 18)
        width_min = 21 if count == 4 else 27
        width = random.randint(width_min, 30)

        heights = _sample_height_order(count, min(MAX_HEIGHT, height))
        if heights is None:
            continue

        markers = random.sample(MARKER_COLORS, count)
        bars = [{"col": -1, "height": h, "marker": marker} for h, marker in zip(heights, markers)]

        want_overlap = random.random() < 0.55
        cols = _sample_columns(width, bars, want_overlap)
        if cols is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        for bar, col in zip(bars, cols):
            top = height - bar["height"]
            grid[top + 1 : height, col] = Color.GREY
            grid[top, col] = bar["marker"]

        try:
            _validate_generated_instance(grid, expect_overlap=want_overlap)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
