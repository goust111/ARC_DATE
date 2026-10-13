from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _is_contiguous(cols: np.ndarray) -> bool:
    return cols.size > 0 and np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the central bottom bar color into the flanking columns as an upward stepped outline.

    Input structure
    - The grid has a background and exactly two foreground colors.
    - The bottom row is all foreground: one color forms a single interior contiguous bar, and the other color flanks it.
    - The interior bottom bar color is the color that gets extended; the flanking color remains unchanged.

    Rule
    - Keep the input unchanged, then add one bottom-bar-colored pixel in every bottom-row flanking column.
    - For each flanking column, compute its horizontal distance to the nearest end of the interior bar.
    - Place the new pixel in that same column, one row above the bottom for each distance step, with an additional one-row gap above the bottom row.

    Key insight
    - The bottom row is a ruler: distance from the central bar determines how high the matching color is lifted in each side column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    bottom = g[height - 1, :]
    assert np.all(bottom != bg), "expected a fully foreground bottom row"

    extension_candidates = []
    for color in colors:
        cols = np.flatnonzero(bottom == color)
        if _is_contiguous(cols) and 0 < int(cols[0]) and int(cols[-1]) < width - 1:
            extension_candidates.append(int(color))

    assert len(extension_candidates) == 1, "expected exactly one interior bottom bar color"
    extension_color = extension_candidates[0]
    flanking_color = [int(color) for color in colors if int(color) != extension_color]
    assert len(flanking_color) == 1, "expected one flanking color"
    flanking_color = flanking_color[0]

    bar_cols = np.flatnonzero(bottom == extension_color)
    assert _is_contiguous(bar_cols), "extension color must form one bottom bar"
    left = int(bar_cols[0])
    right = int(bar_cols[-1])

    out = g.copy()
    for col_raw in np.flatnonzero(bottom == flanking_color):
        col = int(col_raw)
        assert col < left or col > right, "flanking columns must lie outside the extension bar"
        distance = left - col if col < left else col - right
        row = height - distance - 2
        assert 0 <= row < height - 1, "computed extension pixel must be above the bottom row"
        assert int(out[row, col]) == bg, "extension target must be background"
        out[row, col] = extension_color

    return out

# --------------- generator (o5x) ---------------

# Centered-bottom geometry.
FLANK_PER_SIDE_MIN, FLANK_PER_SIDE_MAX = 1, 4
BAR_WIDTH_CHOICES = [1, 3, 5]

# Palette and generation safety.
FOREGROUND_COLORS = range(1, 10)
GENERATION_RETRIES = 300

def _gen_attempt() -> np.ndarray | None:
    flank_per_side = random.randint(FLANK_PER_SIDE_MIN, FLANK_PER_SIDE_MAX)
    bar_width = random.choice(BAR_WIDTH_CHOICES)
    side = bar_width + 2 * flank_per_side

    interior_color, flanking_color = random.sample(FOREGROUND_COLORS, 2)

    grid = np.zeros((side, side), dtype=int)
    bar_left = flank_per_side
    bar_right = bar_left + bar_width

    grid[-1, :bar_left] = flanking_color
    grid[-1, bar_left:bar_right] = interior_color
    grid[-1, bar_right:] = flanking_color
    grid[-2, bar_left:bar_right] = flanking_color

    expected = grid.copy()
    for col in range(bar_left):
        distance = bar_left - col
        expected[side - distance - 2, col] = interior_color
    for col in range(bar_right, side):
        distance = col - bar_right + 1
        expected[side - distance - 2, col] = interior_color

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b8cdaf2b input")
