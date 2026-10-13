from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the top-left horizontal bar into a wrapped sequence of increasing and decreasing foreground runs.

    Input structure:
    - The grid has a background color and exactly one foreground color.
    - All foreground pixels form one contiguous horizontal bar in the top row.
    - The bar starts at the left edge; its length is the control value n.

    Rule:
    - Build a one-dimensional sequence of foreground runs with lengths
      1, 2, ..., n, n-1, ..., 2, 1.
    - Separate adjacent runs by one background pixel.
    - Wrap the sequence row-major across the input width and pad the final row
      with background pixels.

    Key insight:
    The output preserves the input width; the bar length controls the run-length
    diamond, and the apparent 2D pattern is just that 1D sequence wrapped into rows.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    color = int(fg_colors[0])

    coords = np.argwhere(g == color)
    assert coords.size, "expected a foreground bar"
    assert np.all(coords[:, 0] == 0), "foreground bar must be in the top row"
    cols = np.sort(coords[:, 1])
    n = int(len(cols))
    assert np.array_equal(cols, np.arange(n)), "foreground bar must start at the left edge and be contiguous"

    run_lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    seq: list[int] = []
    for i, run_len in enumerate(run_lengths):
        seq.extend([color] * int(run_len))
        if i != len(run_lengths) - 1:
            seq.append(bg)

    width = int(g.shape[1])
    out_h = (len(seq) + width - 1) // width
    out = np.full((out_h, width), bg, dtype=int)
    out.ravel()[: len(seq)] = np.asarray(seq, dtype=int)
    return out

# --------------- generator (att2) ---------------

# Color palette.
BACKGROUND_COLOR = 0
FOREGROUND_COLOR_RANGE = (1, 9)
# Bar and canvas bounds.
BAR_LENGTH_RANGE = (3, 8)
WIDTH_RANGE = (5, 23)
INPUT_HEIGHT_RANGE = (7, 13)
BAR_ROW = 0
BAR_WIDTH_PADDING = 2
# Oracle retry limit.
GENERATION_RETRIES = 300


def _expected_output(width: int, color: int, n: int) -> np.ndarray:
    run_lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    seq: list[int] = []
    for i, run_len in enumerate(run_lengths):
        seq.extend([color] * run_len)
        if i != len(run_lengths) - 1:
            seq.append(BACKGROUND_COLOR)
    out_h = (len(seq) + width - 1) // width
    out = np.full((out_h, width), BACKGROUND_COLOR, dtype=int)
    out.ravel()[: len(seq)] = np.asarray(seq, dtype=int)
    return out


def _sample_grid() -> np.ndarray:
    color = random.randint(*FOREGROUND_COLOR_RANGE)
    n = random.randint(*BAR_LENGTH_RANGE)
    seq_len = n * n + 2 * n - 2
    width = random.randint(max(n + BAR_WIDTH_PADDING, WIDTH_RANGE[0]), WIDTH_RANGE[1])
    min_h = max(INPUT_HEIGHT_RANGE[0], (seq_len + width - 1) // width)
    h_in = random.randint(min_h, INPUT_HEIGHT_RANGE[1])
    grid = np.full((h_in, width), BACKGROUND_COLOR, dtype=int)
    grid[BAR_ROW, :n] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _sample_grid()
        color = int(grid[BAR_ROW, 0])
        n = int(np.count_nonzero(grid[BAR_ROW]))
        expected = _expected_output(grid.shape[1], color, n)
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
