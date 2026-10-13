from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _vertical_bars(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected one foreground bar color"

    fg = fg_colors[0]
    mask = g == fg
    cols = np.flatnonzero(mask.any(axis=0))
    assert cols.size >= 2, "expected at least two bars"

    bars = []
    for c in cols:
        rows = np.flatnonzero(mask[:, c])
        assert rows.size > 0
        r0, r1 = int(rows[0]), int(rows[-1])
        height = int(rows.size)
        assert height == r1 - r0 + 1, "each bar must be vertically solid"
        assert np.all(g[r0 : r1 + 1, c] == fg)
        bars.append((height, r0, r1, int(c)))
    return bars


def _unique_bar_by_height(bars: list[tuple[int, int, int, int]], *, tallest: bool) -> tuple[int, int, int, int]:
    heights = [bar[0] for bar in bars]
    target = max(heights) if tallest else min(heights)
    matches = [bar for bar in bars if bar[0] == target]
    assert len(matches) == 1, "target bar height must be unique"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only the tallest and shortest vertical bars, recoloring them blue and red.

    Input structure
    - The grid has a dominant background color and one foreground color forming
      separated one-cell-wide vertical bars.
    - The tallest and shortest bars have unique heights.

    Rule
    - Clear the grid to background.
    - Repaint the tallest input bar at its original location in blue.
    - Repaint the shortest input bar at its original location in red.

    Key insight
    - The grey bars behave like a height ranking: only the maximum and minimum
      ranks survive, and their original columns and vertical extents are preserved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg not in (Color.BLUE, Color.RED), "output marker colors must differ from background"

    bars = _vertical_bars(g, bg)
    tallest = _unique_bar_by_height(bars, tallest=True)
    shortest = _unique_bar_by_height(bars, tallest=False)

    out = np.full(g.shape, bg, dtype=int)
    for color, (_height, r0, r1, c) in ((Color.BLUE, tallest), (Color.RED, shortest)):
        out[r0 : r1 + 1, c] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_H = GRID_W = 9
BG_COLOR, BAR_COLOR = Color.BLACK, Color.GREY
# Bar layout and retry budget.
BAR_COUNTS, BAR_HEIGHTS = (4, 5), range(1, GRID_H + 1)
START_COLS, COLUMN_STEP = (0, 1), 2
MAX_ATTEMPTS = 300

def _gen_expected_output(inp: np.ndarray, columns: list[int], heights: list[int]) -> np.ndarray:
    out = np.full(inp.shape, BG_COLOR, dtype=int)
    tallest_i = max(range(len(heights)), key=heights.__getitem__)
    shortest_i = min(range(len(heights)), key=heights.__getitem__)
    for color, idx in ((Color.BLUE, tallest_i), (Color.RED, shortest_i)):
        height = heights[idx]
        out[GRID_H - height : GRID_H, columns[idx]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(*BAR_COUNTS)
    start_choices = [start for start in START_COLS if start + COLUMN_STEP * (bar_count - 1) < GRID_W]
    start_col = random.choice(start_choices)
    columns = [start_col + COLUMN_STEP * i for i in range(bar_count)]
    heights = random.sample(BAR_HEIGHTS, bar_count)

    inp = np.full((GRID_H, GRID_W), BG_COLOR, dtype=int)
    for col, height in zip(columns, heights):
        inp[GRID_H - height : GRID_H, col] = BAR_COLOR

    expected = _gen_expected_output(inp, columns, heights)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
