from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components


RANK_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK]
INPUT_BAR_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

MIN_BARS = 4
MAX_BARS = 6
MIN_HEIGHT = 2
MAX_HEIGHT = 10
MIN_GAP = 2
MAX_GAP = 4
MIN_LEFT_RIGHT_MARGIN = 0
MAX_LEFT_RIGHT_MARGIN = 3
MIN_TOP_PADDING = 1
MAX_TOP_PADDING = 5
MIN_WIDTH = 12
MAX_SIZE = 30
MAX_GENERATION_ATTEMPTS = 600


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the inclusive Bresenham line between two (row, col) points."""
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    line: list[tuple[int, int]] = []
    while True:
        line.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return line


def _find_baseline_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == Color.GREY, axis=1))
    assert rows.size == 1, "expected exactly one full-width grey baseline"
    return int(rows[0])


def _extract_bars(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(np.max(g.shape)) <= MAX_SIZE, "grid must fit ARC size bounds"
    assert np.any(g == Color.GREY), "expected a grey baseline"

    baseline_r = _find_baseline_row(g)
    assert not np.any(g[baseline_r + 1 :, :] != Color.BLACK), "baseline must be the bottommost non-black row"

    work = g.copy()
    work[baseline_r, :] = Color.BLACK
    bars = find_components(work, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict")
    assert MIN_BARS <= len(bars) <= MAX_BARS, "expected 4-6 vertical bars"

    parsed: list[dict] = []
    seen_cols: set[int] = set()
    seen_heights: set[int] = set()
    for obj in bars:
        coords = np.asarray(obj["coords"], dtype=int)
        color = int(obj["color"])
        assert color != Color.BLACK and color != Color.GREY, "bars must use non-black non-grey colors"

        col_values = coords[:, 1]
        row_values = np.sort(coords[:, 0])
        assert np.all(col_values == col_values[0]), "bars must be width 1"
        col = int(col_values[0])
        height = int(coords.shape[0])
        top_r = baseline_r - height

        assert MIN_HEIGHT <= height <= MAX_HEIGHT, "bar height out of expected range"
        assert np.array_equal(row_values, np.arange(top_r, baseline_r)), "bar must be a solid vertical segment"
        assert col not in seen_cols, "each bar must occupy a unique column"
        assert height not in seen_heights, "bar heights must be unique"

        seen_cols.add(col)
        seen_heights.add(height)
        parsed.append(
            {
                "color": color,
                "col": col,
                "height": height,
                "top": (top_r, col),
                "coords": {tuple(map(int, rc)) for rc in coords},
            }
        )

    parsed.sort(key=lambda bar: bar["col"])
    gaps = [parsed[i + 1]["col"] - parsed[i]["col"] - 1 for i in range(len(parsed) - 1)]
    assert all(MIN_GAP <= gap <= MAX_GAP for gap in gaps), "bars must be separated by 2-4 empty columns"
    return baseline_r, parsed


def _render_rank_output(shape: tuple[int, int], baseline_r: int, ranked_bars: list[dict], connect_by_rank: bool) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[baseline_r, :] = Color.GREY

    if connect_by_rank:
        pairs = list(zip(ranked_bars, ranked_bars[1:]))
    else:
        spatial = sorted(ranked_bars, key=lambda bar: bar["col"])
        pairs = list(zip(spatial, spatial[1:]))

    for first, second in pairs:
        line_color = int(first["rank_color"]) if first["rank"] < second["rank"] else int(second["rank_color"])
        for r, c in _bresenham(first["top"], second["top"]):
            out[r, c] = line_color

    for bar in ranked_bars:
        top_r = baseline_r - int(bar["height"])
        out[top_r:baseline_r, int(bar["col"])] = int(bar["rank_color"])

    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    baseline_r, bars = _extract_bars(grid)
    ranked = sorted(bars, key=lambda bar: bar["height"])
    for rank, bar in enumerate(ranked, start=1):
        bar["rank"] = rank
        bar["rank_color"] = int(RANK_COLORS[rank - 1])

    out = _render_rank_output(grid.shape, baseline_r, ranked, connect_by_rank=True)
    non_black = bbox(out != Color.BLACK)
    assert non_black is not None, "output must contain non-black pixels"
    return crop_bbox(out, non_black)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank the bars by height, connect adjacent ranks by diagonal lines, and recolor by rank before cropping.

    Input structure:
    - The grid has a black background and one full-width grey baseline.
    - Above the baseline are 4-6 single-pixel-wide vertical bars with distinct colors.
    - Each bar rises straight up from the baseline, has a unique height, and bars are separated by 2-4 empty columns.

    Rule:
    - Rank the bars from shortest to tallest.
    - Recolor rank 1..6 to blue, red, green, yellow, grey, and pink respectively.
    - Draw a Bresenham line between the top pixels of ranks k and k+1, using the lower rank's recolor.
    - Keep the baseline grey, then crop to the minimal bounding box of all non-black output pixels.

    Key insight:
    The diagonal connections follow adjacency in height rank, not adjacency in left-to-right position, so the height order controls both the line topology and the final bar colors.
    """
    return _solve(np.asarray(grid, dtype=int))


def _make_input(cols: list[int], heights: list[int], colors: list[int], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    baseline_r = height - 1
    grid[baseline_r, :] = Color.GREY
    for col, bar_height, color in zip(cols, heights, colors):
        top_r = baseline_r - bar_height
        grid[top_r:baseline_r, col] = color
    return grid


def _bar_lookup(cols: list[int], heights: list[int]) -> list[dict]:
    return [
        {
            "col": int(col),
            "height": int(height),
            "top": (-int(height), int(col)),
        }
        for col, height in zip(cols, heights)
    ]


def _line_crosses_other_bar(cols: list[int], heights: list[int]) -> bool:
    bars = _bar_lookup(cols, heights)
    ranked = sorted(bars, key=lambda bar: bar["height"])
    occupied = {
        (r, bar["col"])
        for bar in bars
        for r in range(-bar["height"], 0)
    }
    for first, second in zip(ranked, ranked[1:]):
        for point in _bresenham(first["top"], second["top"])[1:-1]:
            if point in occupied:
                return True
    return False


def _lines_are_clear(cols: list[int], heights: list[int]) -> bool:
    bars = _bar_lookup(cols, heights)
    ranked = sorted(bars, key=lambda bar: bar["height"])
    segments = [set(_bresenham(first["top"], second["top"])) for first, second in zip(ranked, ranked[1:])]
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            overlap = segments[i] & segments[j]
            allowed: set[tuple[int, int]] = set()
            if j == i + 1:
                allowed.add(ranked[i + 1]["top"])
            if overlap - allowed:
                return False
    return True


def _rank_and_spatial_outputs_differ(grid: np.ndarray) -> bool:
    baseline_r, bars = _extract_bars(grid)
    ranked = sorted(bars, key=lambda bar: bar["height"])
    for rank, bar in enumerate(ranked, start=1):
        bar["rank"] = rank
        bar["rank_color"] = int(RANK_COLORS[rank - 1])

    correct = _render_rank_output(grid.shape, baseline_r, ranked, connect_by_rank=True)
    wrong = _render_rank_output(grid.shape, baseline_r, ranked, connect_by_rank=False)
    return not np.array_equal(correct, wrong)


def _is_non_monotone_by_x(heights: list[int]) -> bool:
    return heights != sorted(heights) and heights != sorted(heights, reverse=True)


def _sample_layout() -> tuple[list[int], list[int], int, int] | None:
    n_bars = random.randint(MIN_BARS, MAX_BARS)
    heights = random.sample(range(MIN_HEIGHT, MAX_HEIGHT + 1), n_bars)
    random.shuffle(heights)
    if not _is_non_monotone_by_x(heights):
        return None

    gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(n_bars - 1)]
    left_margin = random.randint(MIN_LEFT_RIGHT_MARGIN, MAX_LEFT_RIGHT_MARGIN)
    right_margin = random.randint(MIN_LEFT_RIGHT_MARGIN, MAX_LEFT_RIGHT_MARGIN)
    width = left_margin + n_bars + sum(gaps) + right_margin
    if not (MIN_WIDTH <= width <= MAX_SIZE):
        return None

    cols: list[int] = []
    col = left_margin
    for i in range(n_bars):
        cols.append(col)
        col += 1
        if i < n_bars - 1:
            col += gaps[i]

    top_padding = random.randint(MIN_TOP_PADDING, MAX_TOP_PADDING)
    height = top_padding + max(heights) + 1
    if height > MAX_SIZE:
        return None

    if _line_crosses_other_bar(cols, heights):
        return None
    if not _lines_are_clear(cols, heights):
        return None
    return cols, heights, height, width


def _validate_generated_input(grid: np.ndarray) -> None:
    baseline_r, bars = _extract_bars(grid)
    assert baseline_r == grid.shape[0] - 1, "generator keeps the baseline on the last row"
    assert _rank_and_spatial_outputs_differ(grid), "generator must rule out spatial-adjacency shortcut"
    heights = [int(bar["height"]) for bar in bars]
    assert len(set(heights)) == len(heights), "heights must be unique"

    sorted_by_x = sorted(bars, key=lambda bar: bar["col"])
    ranks_by_x = [sorted(heights).index(int(bar["height"])) for bar in sorted_by_x]
    assert any(abs(a - b) > 1 for a, b in zip(ranks_by_x, ranks_by_x[1:])), "need an adjacent-by-x counterexample"

    out = main(grid)
    assert isinstance(out, np.ndarray) and out.ndim == 2
    assert max(out.shape) <= MAX_SIZE


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = _sample_layout()
        if layout is None:
            continue
        cols, heights, height, width = layout
        colors = random.sample(INPUT_BAR_COLORS, len(cols))
        grid = _make_input(cols, heights, colors, height, width)
        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 7")
