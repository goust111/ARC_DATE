import random
from collections import Counter

import numpy as np


BG = 0
MIN_BANDS = 2
MAX_BANDS = 4
MIN_WIDTH = 6
MAX_WIDTH = 24
MIN_BAND_HEIGHT = 1
MAX_BAND_HEIGHT = 3
MAX_ATTEMPTS = 400
NON_BG_COLORS = tuple(range(1, 10))


def _find_sort_row(grid: np.ndarray) -> int:
    height, width = grid.shape
    candidates: list[int] = []
    for row in (0, height - 1):
        cols = np.flatnonzero(grid[row] != BG)
        if cols.size == 0 or cols.size >= width:
            continue
        if np.unique(grid[row, cols]).size != cols.size:
            continue
        if cols.size > 1 and np.any(np.diff(cols) < 2):
            continue
        candidates.append(int(row))
    assert len(candidates) == 1, "expected exactly one sparse ordering row on the top or bottom edge"
    return candidates[0]


def _extract_bands(grid: np.ndarray, sort_row: int) -> list[dict[str, int]]:
    height, width = grid.shape
    bands: list[dict[str, int]] = []
    row = 0
    while row < height:
        if row == sort_row or not np.any(grid[row] != BG):
            row += 1
            continue
        start = row
        while row < height and row != sort_row and np.any(grid[row] != BG):
            row += 1
        patch = np.array(grid[start:row, :], dtype=int)
        values = [int(v) for v in patch.reshape(-1) if int(v) != BG]
        counts = Counter(values)
        assert counts, "band must contain foreground pixels"

        body_color = int(max(counts.items(), key=lambda kv: kv[1])[0])
        marker_colors = [int(color) for color in counts if int(color) != body_color]
        assert len(marker_colors) == 1, "each band must contain exactly one marker color"
        marker_color = marker_colors[0]

        marker_pixels = np.argwhere(patch == marker_color)
        assert marker_pixels.shape[0] == 1, "each band must contain exactly one marker pixel"
        marker_r, marker_c = map(int, marker_pixels[0])
        assert marker_c == 0, "marker pixel must sit on the left edge of its band"

        restored = patch.copy()
        restored[marker_r, marker_c] = body_color
        assert np.all(restored == body_color), "band must become a solid stripe after restoring the marker"

        bands.append(
            {
                "height": int(patch.shape[0]),
                "body_color": body_color,
                "marker_color": marker_color,
            }
        )
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder horizontal color bands by the hidden marker sequence shown in the sparse ordering row.

    Input structure:
    - The background is black.
    - Several horizontal monochrome bands fill the grid width, and adjacent bands are separated by one black row.
    - Each band is 1-3 rows tall and contains one off-color marker pixel at its left edge instead of the band's body color.
    - An extra top row or bottom row is sparse: it contains each marker color exactly once from left to right, with black gaps between them.

    Rule:
    - Identify the sparse ordering row on the top or bottom edge and read its non-black colors from left to right.
    - Match each marker color to the band whose left-edge marker pixel has that color.
    - Rebuild the bands from top to bottom in the ordering-row sequence, preserving each band's height and the single black separator row between bands.
    - Replace every marker pixel with its band's body color, and remove the ordering row entirely.

    Key insight:
    - The only purpose of the tiny left-edge marker pixel is to link a full-width stripe to the sparse color legend, which defines vertical order rather than a recoloring rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sort_row = _find_sort_row(g)
    order = [int(g[sort_row, c]) for c in np.flatnonzero(g[sort_row] != BG)]
    bands = _extract_bands(g, sort_row)

    by_marker = {int(band["marker_color"]): band for band in bands}
    assert len(by_marker) == len(bands) == len(order), "ordering row and stripe markers must match one-to-one"
    assert len(set(order)) == len(order), "ordering row markers must be unique"

    out_height = sum(int(band["height"]) for band in bands) + max(0, len(bands) - 1)
    out = np.full((out_height, g.shape[1]), BG, dtype=int)

    row = 0
    for index, marker_color in enumerate(order):
        band = by_marker[int(marker_color)]
        band_height = int(band["height"])
        out[row : row + band_height, :] = int(band["body_color"])
        row += band_height
        if index < len(order) - 1:
            row += 1

    return out


def _sample_sparse_positions(width: int, count: int) -> list[int]:
    for _ in range(200):
        cols = sorted(random.sample(range(width), count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    raise RuntimeError("failed to place sparse ordering markers")


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    band_count = random.randint(MIN_BANDS, MAX_BANDS)
    band_heights = [random.randint(MIN_BAND_HEIGHT, MAX_BAND_HEIGHT) for _ in range(band_count)]
    width_min = max(MIN_WIDTH, 2 * band_count - 1)
    width = random.randint(width_min, MAX_WIDTH)

    input_height = sum(band_heights) + (band_count - 1) + 1
    assert input_height <= 30 and width <= 30

    palette = random.sample(list(NON_BG_COLORS), 2 * band_count)
    body_colors = palette[:band_count]
    marker_colors = palette[band_count:]
    random.shuffle(body_colors)
    random.shuffle(marker_colors)

    target_order = list(range(band_count))
    while target_order == list(range(band_count)):
        random.shuffle(target_order)

    sort_on_top = random.choice([False, True])
    sort_positions = _sample_sparse_positions(width, band_count)

    grid = np.full((input_height, width), BG, dtype=int)
    sort_row = 0 if sort_on_top else input_height - 1
    for col, band_index in zip(sort_positions, target_order):
        grid[sort_row, int(col)] = int(marker_colors[band_index])

    band_specs: list[dict[str, int]] = []
    row = 1 if sort_on_top else 0
    for band_index, band_height in enumerate(band_heights):
        body_color = int(body_colors[band_index])
        marker_color = int(marker_colors[band_index])

        grid[row : row + band_height, :] = body_color
        marker_row = row + random.randrange(band_height)
        grid[marker_row, 0] = marker_color

        band_specs.append(
            {
                "height": int(band_height),
                "body_color": body_color,
                "marker_color": marker_color,
            }
        )
        row += band_height
        if band_index < band_count - 1:
            row += 1

    output_height = sum(band_heights) + (band_count - 1)
    expected = np.full((output_height, width), BG, dtype=int)
    row = 0
    for out_index, band_index in enumerate(target_order):
        band = band_specs[band_index]
        band_height = int(band["height"])
        expected[row : row + band_height, :] = int(band["body_color"])
        row += band_height
        if out_index < band_count - 1:
            row += 1

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid, expected = _build_case()
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
