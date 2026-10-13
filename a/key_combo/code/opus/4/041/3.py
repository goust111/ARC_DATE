from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

INDICATOR_COLORS = (RED, BLUE, GREEN)
MARKER_COUNT_RANGE = (4, 6)
BAND_WIDTH_RANGE = (3, 6)
HEIGHT_RANGE = (10, 16)
MAX_ATTEMPTS = 300


def _indicator_bands(grid: np.ndarray) -> tuple[list[int], list[int]]:
    starts = [int(c) for c in np.flatnonzero(grid[0] != BACKGROUND)]
    if len(starts) != 3:
        raise ValueError("expected exactly three first-row indicator pixels")
    colors = [int(grid[0, c]) for c in starts]
    return starts, colors


def _band_index(col: int, starts: list[int], width: int) -> int:
    for idx in range(len(starts) - 1, -1, -1):
        if int(col) >= starts[idx]:
            return idx
    raise ValueError("column is left of the first band")


def _halo_cells(row: int, col: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    cells: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = int(row) + dr
            cc = int(col) + dc
            if 0 <= rr < height and 0 <= cc < width:
                cells.append((rr, cc))
    return cells


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each gray marker into a 3x3 halo whose color and draw order are encoded by its column band.

    Input structure:
    - The grid has a black background and exactly three non-black indicator pixels on the first row.
    - Those indicator pixels mark the starts of three left-to-right column bands.
    - Below the first row, gray singleton marker pixels are scattered across the bands.

    Rule:
    - Determine each marker's band from its column, then read that band's indicator color as the halo color.
    - Process markers by band priority from left to right; within one band, process top-to-bottom then left-to-right.
    - For each marker, paint its 3x3 Chebyshev-radius-1 halo only onto background cells, then recolor the marker itself to the halo color.
    - Existing non-background pixels, including the first-row indicators and earlier halos, are never overwritten.

    Key insight:
    - A marker's column band simultaneously specifies both its halo color and its conflict-resolution priority, so overlap outcomes come from ordered background-only stamping rather than from any per-pixel color comparison.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    starts, band_colors = _indicator_bands(g)
    out = g.copy()
    height, width = g.shape

    markers: list[tuple[int, int, int, int]] = []
    for row, col in np.argwhere(g == GREY):
        row = int(row)
        col = int(col)
        if row == 0:
            continue
        band = _band_index(col, starts, width)
        markers.append((band, row, col, int(band_colors[band])))

    markers.sort(key=lambda item: (item[0], item[1], item[2]))

    for _band, row, col, color in markers:
        for rr, cc in _halo_cells(row, col, (height, width)):
            if out[rr, cc] == BACKGROUND:
                out[rr, cc] = color
        out[row, col] = color

    return out


def _band_columns(band: int, band_width: int) -> list[int]:
    start = int(band) * int(band_width)
    return list(range(start, start + int(band_width)))


def _overlap_pair_candidates(height: int, band_width: int, left_band: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    left_cols = _band_columns(left_band, band_width)
    right_cols = _band_columns(left_band + 1, band_width)
    rows = range(2, height - 1)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r1 in rows:
        for c1 in left_cols:
            for r2 in rows:
                for c2 in right_cols:
                    if _chebyshev((r1, c1), (r2, c2)) == 2:
                        pairs.append(((r1, c1), (r2, c2)))
    return pairs


def _extra_candidates(height: int, width: int, band_width: int, band: int, placed: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cols = _band_columns(band, band_width)
    candidates: list[tuple[int, int]] = []
    for row in range(2, height - 1):
        for col in cols:
            cell = (row, col)
            if cell in placed:
                continue
            if any(_chebyshev(cell, other) < 3 for other in placed):
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _has_required_overlap(markers: list[tuple[int, int]], band_width: int) -> bool:
    for i, a in enumerate(markers):
        band_a = int(a[1]) // int(band_width)
        for b in markers[i + 1 :]:
            band_b = int(b[1]) // int(band_width)
            if band_a != band_b and _chebyshev(a, b) == 2:
                return True
    return False


def _generate_attempt() -> np.ndarray | None:
    band_width = random.randint(*BAND_WIDTH_RANGE)
    height = random.randint(*HEIGHT_RANGE)
    width = 3 * band_width
    marker_count = random.randint(*MARKER_COUNT_RANGE)

    grid = np.zeros((height, width), dtype=int)
    indicator_cols = [0, band_width, 2 * band_width]
    for col, color in zip(indicator_cols, INDICATOR_COLORS):
        grid[0, col] = color

    left_band = random.choice((0, 1))
    overlap_candidates = _overlap_pair_candidates(height, band_width, left_band)
    if not overlap_candidates:
        return None
    first, second = random.choice(overlap_candidates)
    markers = [first, second]

    remaining_band = ({0, 1, 2} - {left_band, left_band + 1}).pop()
    remaining_choices = _extra_candidates(height, width, band_width, remaining_band, markers)
    if not remaining_choices:
        return None
    markers.append(remaining_choices.pop())

    while len(markers) < marker_count:
        band = random.choice((0, 1, 2))
        choices = _extra_candidates(height, width, band_width, band, markers)
        if not choices:
            return None
        markers.append(choices.pop())

    for row, col in markers:
        grid[row, col] = GREY

    if not _has_required_overlap(markers, band_width):
        return None
    if not np.array_equal(main(grid).shape, grid.shape):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
