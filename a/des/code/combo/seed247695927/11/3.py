from __future__ import annotations

from collections import Counter
import random

import numpy as np


PINK = 6
MAX_ATTEMPTS = 300
HEIGHT_RANGE = (4, 10)
WIDTH_RANGE = (5, 14)
TOP_FILL_RATIO_RANGE = (0.25, 0.75)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the L-shaped clues into three rows.

    Input structure:
    - A dominant background color.
    - Row 0 is a two-color horizontal clue.
    - In rows below, column 0 contains one or more marker pixels forming the
      vertical clue; the first such marker row is the one preserved.

    Rule:
    - Output row 0 is the input top row.
    - Output row 1 is the first row below the top whose first cell is not background.
    - Output row 2 is background except that every non-background cell from the
      top row is copied as pink.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 3 and width >= 2, "grid too small for the clue layout"

    bg = _background_color(g)
    top = g[0].copy()
    top_colors = {int(v) for v in np.unique(top)}
    assert bg in top_colors and len(top_colors) == 2, "top row must be exactly two colors including background"
    assert np.any(top != bg), "top row needs at least one foreground cell"

    marker_rows = np.flatnonzero(g[1:, 0] != bg)
    assert marker_rows.size > 0, "expected at least one vertical marker below the top row"
    marker_row = int(marker_rows[0] + 1)

    out = np.full((3, width), bg, dtype=int)
    out[0] = top
    out[1] = g[marker_row]
    out[2, top != bg] = PINK
    return out


def _random_palette() -> tuple[int, int, int]:
    colors = [c for c in range(10) if c != PINK]
    bg, top_fg, marker = random.sample(colors, 3)
    return bg, top_fg, marker


def _sample_top_row(width: int, bg: int, top_fg: int) -> np.ndarray:
    top = np.full(width, bg, dtype=int)
    fill_ratio = random.uniform(*TOP_FILL_RATIO_RANGE)
    count = max(1, min(width - 1, int(round(width * fill_ratio))))
    cols = sorted(random.sample(range(width), count))
    top[cols] = top_fg
    return top


def _sample_marker_rows(height: int) -> list[int]:
    count = random.randint(1, height - 1)
    return sorted(random.sample(range(1, height), count))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    bg, top_fg, marker = _random_palette()

    grid = np.full((height, width), bg, dtype=int)
    grid[0] = _sample_top_row(width, bg, top_fg)

    marker_rows = _sample_marker_rows(height)
    for r in marker_rows:
        grid[r, 0] = marker

    # Add optional full-width repetitions of the first marker row shape to make
    # the vertical clue visually explicit without changing the rule.
    if width > 2 and len(marker_rows) > 1 and random.random() < 0.5:
        tail_width = random.randint(0, min(2, width - 1))
        if tail_width:
            extra_color = random.choice([marker, top_fg])
            for r in marker_rows:
                grid[r, 1 : 1 + tail_width] = extra_color

    expected = np.full((3, width), bg, dtype=int)
    expected[0] = grid[0]
    expected[1] = grid[marker_rows[0]]
    expected[2, grid[0] != bg] = PINK

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
