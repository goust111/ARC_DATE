from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift every foreground horizontal segment one cell left and replace the old background with grey.

    Input structure:
    - The grid has one dominant background color and exactly one foreground color.
    - The foreground pixels form one or more 4-connected horizontal one-cell-thick segments.

    Rule:
    - Create an output grid of the same shape filled with grey.
    - Move every foreground pixel one column left; pixels shifted past the left edge are clipped.
    - Paint all shifted foreground pixels with their original foreground color.

    Key insight:
    - The output is not an extension of the segments; it is a global one-cell west translation of the foreground mask over a newly grey background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.GREY, "background must be recolored to grey"
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])
    assert fg != Color.GREY, "foreground color must differ from grey fill"

    segments = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert segments, "expected at least one foreground segment"
    for segment in segments:
        assert int(segment["color"]) == fg, "all foreground segments must share one color"
        r0, c0, r1, c1 = segment["bbox"]
        assert r0 == r1, "foreground components must be one row high"
        assert int(segment["area"]) == int(c1 - c0 + 1), "foreground components must be solid horizontal segments"

    mask = g == fg
    shifted = np.zeros_like(mask, dtype=bool)
    shifted[:, :-1] = mask[:, 1:]

    out = np.full(g.shape, Color.GREY, dtype=int)
    out[shifted] = fg
    return out

# --------------- generator (o5x) ---------------
# Canvas and palette bounds.
GRID_MIN_SIZE, GRID_MAX_SIZE = 3, 10
INPUT_BACKGROUND = Color.BLACK
FOREGROUND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Segment count, row spacing, and length bounds.
MIN_SEGMENTS, MAX_SEGMENTS = 1, 4
MIN_SEGMENT_LEN, MAX_SEGMENT_LEN = 2, 6
ROW_SPACING_OFFSET = 1
MAX_FG_CELL_MARGIN, MAX_FG_CELL_DIVISOR = 1, 2

# Validation retry limit.
GENERATION_RETRIES = 300


def _gen_spaced_rows(n: int, k: int) -> list[int]:
    base_rows = sorted(random.sample(range(n - ROW_SPACING_OFFSET * (k - 1)), k))
    return [r + ROW_SPACING_OFFSET * i for i, r in enumerate(base_rows)]


def _gen_segment_lengths(n: int, k: int) -> list[int]:
    lengths: list[int] = []
    remaining_fg = (n * n - MAX_FG_CELL_MARGIN) // MAX_FG_CELL_DIVISOR
    for i in range(k):
        remaining_segments = k - i - 1
        max_len = min(MAX_SEGMENT_LEN, n, remaining_fg - MIN_SEGMENT_LEN * remaining_segments)
        length = random.randint(MIN_SEGMENT_LEN, max_len)
        lengths.append(length)
        remaining_fg -= length
    random.shuffle(lengths)
    return lengths


def _gen_attempt() -> np.ndarray:
    k = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    min_rows = k + ROW_SPACING_OFFSET * (k - 1)
    n = random.randint(max(GRID_MIN_SIZE, min_rows), GRID_MAX_SIZE)
    fg = random.choice(FOREGROUND_COLORS)

    rows = _gen_spaced_rows(n, k)
    lengths = _gen_segment_lengths(n, k)

    grid = np.full((n, n), INPUT_BACKGROUND, dtype=int)
    for r, length in zip(rows, lengths):
        c0 = random.randint(0, n - length)
        grid[r, c0 : c0 + length] = fg

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        try:
            main(inp)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate valid input for 32e9702f")
