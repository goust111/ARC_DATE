from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist() + [False]):
        if flag and start is None:
            start = int(idx)
        elif not flag and start is not None:
            runs.append((int(start), int(idx)))
            start = None
    return runs


def _crop_non_black(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != 0)
    assert coords.size > 0, "expected a non-black panel grid"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _panel_matrix_from_separator(crop: np.ndarray, sep_color: int) -> np.ndarray:
    height, width = map(int, crop.shape)
    sep_rows = np.all(crop == int(sep_color), axis=1)
    sep_cols = np.all(crop == int(sep_color), axis=0)
    assert bool(sep_rows.any()) and bool(sep_cols.any()), "separator must form full rows and columns"

    row_bands = _runs(~sep_rows)
    col_bands = _runs(~sep_cols)
    row_sep_bands = _runs(sep_rows)
    col_sep_bands = _runs(sep_cols)

    assert len(row_bands) >= 3, "need at least three panel rows to identify a unique outlier"
    assert len(col_bands) >= 1, "need at least one panel column"

    row_heights = {stop - start for start, stop in row_bands}
    col_widths = {stop - start for start, stop in col_bands}
    assert len(row_heights) == 1 and next(iter(row_heights)) > 0
    assert len(col_widths) == 1 and next(iter(col_widths)) > 0

    row_sep_sizes = {stop - start for start, stop in row_sep_bands}
    col_sep_sizes = {stop - start for start, stop in col_sep_bands}
    assert len(row_sep_sizes) == 1 and next(iter(row_sep_sizes)) > 0
    assert len(col_sep_sizes) == 1 and next(iter(col_sep_sizes)) > 0

    row_has_border = row_sep_bands[0][0] == 0
    col_has_border = col_sep_bands[0][0] == 0
    if row_has_border:
        assert len(row_sep_bands) == len(row_bands) + 1
        assert row_sep_bands[-1][1] == height
        assert row_bands[0][0] > 0 and row_bands[-1][1] < height
    else:
        assert len(row_sep_bands) == len(row_bands) - 1
        assert row_bands[0][0] == 0 and row_bands[-1][1] == height
    if col_has_border:
        assert len(col_sep_bands) == len(col_bands) + 1
        assert col_sep_bands[-1][1] == width
        assert col_bands[0][0] > 0 and col_bands[-1][1] < width
    else:
        assert len(col_sep_bands) == len(col_bands) - 1
        assert col_bands[0][0] == 0 and col_bands[-1][1] == width

    panel_colors = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    for r_idx, (r0, r1) in enumerate(row_bands):
        for c_idx, (c0, c1) in enumerate(col_bands):
            patch = crop[int(r0) : int(r1), int(c0) : int(c1)]
            values = np.unique(patch)
            assert values.size == 1, "each panel must be a solid color"
            color = int(values[0])
            assert color not in (0, int(sep_color)), "panel color cannot be black or separator grey"
            panel_colors[int(r_idx), int(c_idx)] = color

    return panel_colors


def _decode_panels(grid: np.ndarray) -> np.ndarray:
    crop = _crop_non_black(np.asarray(grid, dtype=int))
    candidates: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for color in sorted(int(c) for c in np.unique(crop) if int(c) != 0):
        try:
            panel_colors = _panel_matrix_from_separator(crop, color)
            _compress_outliers(panel_colors)
        except AssertionError:
            continue
        full_rows = int(np.count_nonzero(np.all(crop == color, axis=1)))
        full_cols = int(np.count_nonzero(np.all(crop == color, axis=0)))
        score = (panel_colors.shape[0] * panel_colors.shape[1], full_rows + full_cols, int(color))
        candidates.append((score, panel_colors))
    assert candidates, "could not identify the separator grid"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _compress_outliers(panel_colors: np.ndarray) -> np.ndarray:
    panel_colors = np.asarray(panel_colors, dtype=int)
    rows, cols = map(int, panel_colors.shape)
    assert rows >= 3, "two-row columns are ambiguous"

    out = np.zeros((rows, cols), dtype=int)
    for c in range(cols):
        counts = Counter(map(int, panel_colors[:, c]))
        assert len(counts) == 2, "each column must contain one odd color and one majority color"
        odd = [(color, count) for color, count in counts.items() if count == 1]
        majority = [(color, count) for color, count in counts.items() if count == rows - 1]
        assert len(odd) == 1 and len(majority) == 1, "column must have a unique outlier"
        odd_color = int(odd[0][0])
        odd_row = int(np.flatnonzero(panel_colors[:, c] == odd_color)[0])
        out[odd_row, c] = odd_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a panel grid separated by grey lines, find the one odd panel in each
    panel-column and compress those odd positions into a smaller output canvas.

    Input structure:
    - Background is black.
    - Grey separator lines divide the non-black region into equal rectangular panels.
    - Each panel is a solid color.
    - In each panel-column, exactly one panel has a different color from the others.

    Rule:
    - For each panel-column, locate the unique odd-colored panel.
    - Create an output grid with one cell per panel position.
    - Put the odd panel's color at the matching row and column in the output.
    - Leave all other output cells black.
    """
    panel_colors = _decode_panels(np.asarray(grid, dtype=int))
    return _compress_outliers(panel_colors)


BG = 0
SEP = 5
PANEL_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ROW_COUNT_RANGE = (3, 6)
COL_COUNT_RANGE = (2, 7)
PANEL_HEIGHT_RANGE = (2, 5)
PANEL_WIDTH_RANGE = (2, 5)
SEP_THICKNESSES = (1, 2)
MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400


def _panel_starts(count: int, panel_size: int, sep_thickness: int, outer_border: bool) -> list[int]:
    if outer_border:
        return [sep_thickness + i * (panel_size + sep_thickness) for i in range(count)]
    return [i * (panel_size + sep_thickness) for i in range(count)]


def _grid_extent(count: int, panel_size: int, sep_thickness: int, outer_border: bool) -> int:
    if outer_border:
        return count * panel_size + (count + 1) * sep_thickness
    return count * panel_size + (count - 1) * sep_thickness


def _sample_panel_colors(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    palette_size = random.randint(3, min(6, len(PANEL_PALETTE)))
    palette = random.sample(PANEL_PALETTE, palette_size)

    panel_colors = np.zeros((rows, cols), dtype=int)
    output = np.zeros((rows, cols), dtype=int)
    odd_rows: list[int] = []
    odd_colors: list[int] = []

    for c in range(cols):
        base_color = random.choice(palette)
        odd_color = random.choice([color for color in palette if color != base_color])
        odd_row = random.randrange(rows)

        panel_colors[:, c] = int(base_color)
        panel_colors[odd_row, c] = int(odd_color)
        output[odd_row, c] = int(odd_color)
        odd_rows.append(int(odd_row))
        odd_colors.append(int(odd_color))

    assert len(set(map(int, panel_colors.reshape(-1)))) >= 3
    assert len(set(odd_rows)) >= 2
    assert len(set(odd_colors)) >= 2
    return panel_colors, output


def _render_input(
    panel_colors: np.ndarray,
    panel_height: int,
    panel_width: int,
    sep_thickness: int,
    outer_border: bool,
    with_margin: bool,
) -> np.ndarray:
    rows, cols = map(int, panel_colors.shape)
    crop_height = _grid_extent(rows, panel_height, sep_thickness, outer_border)
    crop_width = _grid_extent(cols, panel_width, sep_thickness, outer_border)
    crop = np.full((crop_height, crop_width), SEP, dtype=int)

    row_starts = _panel_starts(rows, panel_height, sep_thickness, outer_border)
    col_starts = _panel_starts(cols, panel_width, sep_thickness, outer_border)
    for r in range(rows):
        for c in range(cols):
            r0 = row_starts[r]
            c0 = col_starts[c]
            crop[r0 : r0 + panel_height, c0 : c0 + panel_width] = int(panel_colors[r, c])

    if with_margin:
        top = random.randint(*MARGIN_RANGE)
        bottom = random.randint(*MARGIN_RANGE)
        left = random.randint(*MARGIN_RANGE)
        right = random.randint(*MARGIN_RANGE)
    else:
        top = bottom = left = right = 0

    grid = np.full((top + crop_height + bottom, left + crop_width + right), BG, dtype=int)
    grid[top : top + crop_height, left : left + crop_width] = crop
    return grid


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    panel_height = random.randint(*PANEL_HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)
    sep_thickness = random.choice(SEP_THICKNESSES)
    outer_border = random.choice([False, True])
    with_margin = random.choice([False, True])

    try:
        panel_colors, expected = _sample_panel_colors(rows, cols)
    except AssertionError:
        return None

    grid = _render_input(
        panel_colors,
        panel_height=panel_height,
        panel_width=panel_width,
        sep_thickness=sep_thickness,
        outer_border=outer_border,
        with_margin=with_margin,
    )

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
