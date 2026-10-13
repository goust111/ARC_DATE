from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg, is_contiguous
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey placeholder pixel with the visible control color in its row.

    Input structure
    - The grid has a dominant background, grey placeholder pixels, and colored row control pixels in one marker column.
    - Every row containing grey pixels contains exactly one non-background, non-grey control pixel; rows without grey contain no other foreground.

    Rule
    - Preserve background and control pixels.
    - For each row, replace all grey pixels with that row's control color.

    Key insight
    - The grey shape is not filled as a connected object; the row-aligned control marker supplies the replacement color independently for every grey pixel in that row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.size

    placeholder = int(Color.GREY)
    bg = int(get_bg(g))
    assert bg != placeholder
    assert np.any(g == placeholder), "expected grey placeholder pixels"

    out = g.copy()
    control_cols: set[int] = set()
    for r in range(g.shape[0]):
        row = g[r]
        placeholder_mask = row == placeholder
        control_cols_in_row = np.flatnonzero((row != bg) & (row != placeholder))

        if np.any(placeholder_mask):
            assert control_cols_in_row.size == 1, "each grey row must have one control pixel"
            control_col = int(control_cols_in_row[0])
            control_cols.add(control_col)
            out[r, placeholder_mask] = int(row[control_col])
        else:
            assert control_cols_in_row.size == 0, "control pixels must label rows with grey placeholders"

    assert len(control_cols) == 1, "all row control pixels must share one marker column"
    return out

# --------------- generator (o5x) ---------------

# Canvas, marker, and shape placement.
GRID_SHAPE = (10, 10)
BACKGROUND_COLOR, FILL_COLOR = int(Color.BLACK), int(Color.GREY)
MARKER_COL, SHAPE_COL_MIN, SHAPE_COL_LIMIT = 0, 3, 8

# Active-row and color-band sampling.
ACTIVE_HEIGHTS, SHORT_ACTIVE_TOPS = [7, 8], [1, 2]
FULL_ACTIVE_HEIGHT, FULL_ACTIVE_TOP = 8, 1
BAND_COUNTS = [2, 3, 4]
BAND_LEN_MIN, BAND_LEN_MAX = 1, 4
BAND_PALETTE = [int(c) for c in Color.NOT_BLACK if int(c) != FILL_COLOR]

# Grey-shape density, size, and growth.
SHAPE_WIDTHS = [3, 4, 5]
DENSITY_MIN, DENSITY_MAX = 0.4, 0.6
GREY_PIXELS_MIN, GREY_PIXELS_MAX = 12, 20
NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 300


def _gen_band_lengths(total_rows: int, band_count: int) -> list[int]:
    choices: list[list[int]] = []

    def rec(remaining: int, slots: int, prefix: list[int]) -> None:
        if slots == 0:
            if remaining == 0:
                choices.append(prefix.copy())
            return
        for length in range(BAND_LEN_MIN, BAND_LEN_MAX + 1):
            if length <= remaining <= length + BAND_LEN_MAX * (slots - 1):
                prefix.append(length)
                rec(remaining - length, slots - 1, prefix)
                prefix.pop()

    rec(total_rows, band_count, [])
    return random.choice(choices)


def _gen_connected_grey_mask(height: int, width: int, total_pixels: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    spine_c = random.randrange(width)
    mask[:, spine_c] = True

    left_row = random.randrange(height)
    right_row = random.randrange(height)
    mask[left_row, : spine_c + 1] = True
    mask[right_row, spine_c:] = True

    while int(mask.sum()) < total_pixels:
        frontier = set()
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            for dr, dc in NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return mask
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True

    return mask


def _gen_attempt() -> np.ndarray | None:
    grid = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)

    active_h = random.choice(ACTIVE_HEIGHTS)
    top = FULL_ACTIVE_TOP if active_h == FULL_ACTIVE_HEIGHT else random.choice(SHORT_ACTIVE_TOPS)
    active_rows = list(range(top, top + active_h))

    band_count = random.choice(BAND_COUNTS)
    band_lengths = _gen_band_lengths(active_h, band_count)
    band_colors = random.sample(BAND_PALETTE, band_count)

    shape_w = random.choice(SHAPE_WIDTHS)
    shape_c0 = random.randint(SHAPE_COL_MIN, SHAPE_COL_LIMIT - shape_w)
    area = active_h * shape_w
    min_base_pixels = active_h + shape_w - 1
    min_pixels = max(GREY_PIXELS_MIN, int(np.ceil(DENSITY_MIN * area)), min_base_pixels)
    max_pixels = min(GREY_PIXELS_MAX, int(np.floor(DENSITY_MAX * area)))
    if min_pixels > max_pixels:
        return None

    total_pixels = random.randint(min_pixels, max_pixels)
    mask = _gen_connected_grey_mask(active_h, shape_w, total_pixels)
    if int(mask.sum()) != total_pixels:
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None

    expected = grid.copy()
    start = 0
    for color, length in zip(band_colors, band_lengths):
        for local_r in range(start, start + length):
            r = active_rows[local_r]
            cols = shape_c0 + np.flatnonzero(mask[local_r])
            expected[r, MARKER_COL] = grid[r, MARKER_COL] = color
            grid[r, cols] = FILL_COLOR
            expected[r, cols] = color
        start += length

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c7d4e6ad input")
