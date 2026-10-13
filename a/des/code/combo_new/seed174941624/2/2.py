from __future__ import annotations

import random

import numpy as np


ARC_COLORS = tuple(range(10))
GENERATION_RETRIES = 400
BAND_COUNT_RANGE = (1, 4)
PAIR_COUNT_RANGE = (2, 5)
MIN_REGION_THICKNESS = 2
GRID_MAX_SIDE = 24


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        start, end = sorted((c0, c1))
        return [(r0, c) for c in range(start, end + 1)]
    assert c0 == c1, "same-color endpoints must share a row or a column"
    start, end = sorted((r0, r1))
    return [(r, c0) for r in range(start, end + 1)]


def _find_dye_bands(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_bands = [(r, int(g[r, 0])) for r in range(h) if int(g[r, 0]) != bg and bool(np.all(g[r] == g[r, 0]))]
    col_bands = [(c, int(g[0, c])) for c in range(w) if int(g[0, c]) != bg and bool(np.all(g[:, c] == g[0, c]))]

    assert row_bands or col_bands, "expected at least one full-length dye band"
    assert not (row_bands and col_bands), "valid inputs use one global band orientation"

    raw = [("row", idx, color) for idx, color in row_bands] or [("col", idx, color) for idx, color in col_bands]
    indices = [int(idx) for _axis, idx, _color in raw]
    assert all(b - a > 1 for a, b in zip(indices, indices[1:])), "dye bands must be non-adjacent"

    colors = [int(color) for _axis, _idx, color in raw]
    assert len(colors) == len(set(colors)), "dye band colors must be distinct"

    return [{"axis": axis, "index": int(idx), "color": int(color)} for axis, idx, color in raw]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair and recolor the whole segment with the unique full-length dye band it crosses.

    Input structure:
    - The background is the most common color.
    - One or more one-cell-thick dye bands span the entire grid in one global orientation: either full rows, or the same pattern rotated so they are full columns.
    - Each dye band has a distinct non-background color, and adjacent rows or columns are never both dye bands.
    - Every other non-background color appears as exactly two singleton endpoints, and each pair is aligned on one row or one column so its connecting segment crosses exactly one dye band.

    Rule:
    - Detect the full-length dye bands and their shared orientation.
    - For each remaining color, connect its two endpoints with the straight one-cell-wide horizontal or vertical segment that joins them.
    - Determine the unique dye band intersected by that segment.
    - Recolor the entire segment, including both endpoints, to the dye band's color while leaving the dye bands themselves unchanged.

    Key insight:
    - The endpoint color only tells you which two pixels belong together; the visible band that the hidden segment passes through determines the segment's final output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    bg = _most_common_color(g)
    bands = _find_dye_bands(g, bg)
    band_axis = str(bands[0]["axis"])
    band_colors = {int(band["color"]) for band in bands}

    band_mask = np.zeros(g.shape, dtype=bool)
    if band_axis == "row":
        for band in bands:
            band_mask[int(band["index"]), :] = True
    else:
        for band in bands:
            band_mask[:, int(band["index"])] = True

    pair_colors = [int(color) for color in np.unique(g) if int(color) not in band_colors and int(color) != bg]
    assert pair_colors, "expected at least one endpoint pair"

    out = g.copy()
    for color in pair_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each non-band color must appear exactly twice"
        p0 = (int(coords[0, 0]), int(coords[0, 1]))
        p1 = (int(coords[1, 0]), int(coords[1, 1]))
        cells = _segment_cells(p0, p1)

        if band_axis == "row":
            assert p0[1] == p1[1], "with full-row dye bands, endpoint pairs must be vertical"
            lo, hi = sorted((p0[0], p1[0]))
            hits = [band for band in bands if lo <= int(band["index"]) <= hi]
        else:
            assert p0[0] == p1[0], "with full-column dye bands, endpoint pairs must be horizontal"
            lo, hi = sorted((p0[1], p1[1]))
            hits = [band for band in bands if lo <= int(band["index"]) <= hi]

        assert len(hits) == 1, "each segment must cross exactly one dye band"
        target_color = int(hits[0]["color"])

        for r, c in cells:
            if band_mask[r, c]:
                assert int(g[r, c]) == target_color, "crossed band cells must keep their band color"
            elif (r, c) == p0 or (r, c) == p1:
                assert int(g[r, c]) == color, "segment endpoints must match their pair color"
            else:
                assert int(g[r, c]) == bg, "segment interiors must be background before drawing"

        for r, c in cells:
            out[r, c] = target_color

    return out


def _distribute_extra(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _sample_band_rows(height: int, band_count: int) -> list[int]:
    base = MIN_REGION_THICKNESS * (band_count + 1) + band_count
    assert height >= base
    extras = _distribute_extra(height - base, band_count + 1)
    region_heights = [MIN_REGION_THICKNESS + extra for extra in extras]

    rows: list[int] = []
    pos = region_heights[0]
    for idx in range(band_count):
        rows.append(int(pos))
        if idx + 1 < len(region_heights):
            pos += 1 + region_heights[idx + 1]
    return rows


def _region_bounds(height: int, band_rows: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [row + 1 for row in band_rows]
    ends = [row - 1 for row in band_rows] + [height - 1]
    bounds = [(int(start), int(end)) for start, end in zip(starts, ends)]
    assert all(start <= end for start, end in bounds)
    return bounds


def _build_base_scene() -> tuple[np.ndarray, np.ndarray]:
    band_count = random.randint(*BAND_COUNT_RANGE)
    max_pair_count = min(PAIR_COUNT_RANGE[1], 9 - band_count)
    min_pair_count = max(PAIR_COUNT_RANGE[0], band_count)
    pair_count = random.randint(min_pair_count, max_pair_count)

    min_height = MIN_REGION_THICKNESS * (band_count + 1) + band_count
    height = random.randint(min_height, GRID_MAX_SIDE)
    width = random.randint(max(8, pair_count + 4), GRID_MAX_SIDE)

    bg = random.choice(ARC_COLORS)
    palette = [color for color in ARC_COLORS if color != bg]
    random.shuffle(palette)
    band_colors = [int(color) for color in palette[:band_count]]
    pair_colors = [int(color) for color in palette[band_count : band_count + pair_count]]

    band_rows = _sample_band_rows(height, band_count)
    regions = _region_bounds(height, band_rows)

    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()
    for row, color in zip(band_rows, band_colors):
        grid[row, :] = color
        expected[row, :] = color

    assignments = list(range(band_count))
    while len(assignments) < pair_count:
        assignments.append(random.randrange(band_count))
    random.shuffle(assignments)

    cols = random.sample(range(1, width - 1), pair_count)
    for color, band_idx, col in zip(pair_colors, assignments, cols):
        top_lo, top_hi = regions[band_idx]
        bottom_lo, bottom_hi = regions[band_idx + 1]
        top_row = random.randint(top_lo, top_hi)
        bottom_row = random.randint(bottom_lo, bottom_hi)

        grid[top_row, col] = color
        grid[bottom_row, col] = color
        expected[top_row : bottom_row + 1, col] = band_colors[band_idx]

    turns = random.randrange(4)
    return np.rot90(grid, turns).copy(), np.rot90(expected, turns).copy()


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid, expected = _build_base_scene()
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid dye-band recoloring input")
