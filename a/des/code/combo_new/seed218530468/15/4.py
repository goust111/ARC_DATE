from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
BAND_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_ARROW_ROW_GAP = 3
HEIGHT_RANGE = (8, 24)
MAX_WIDTH = 30
ARROW_COUNT_CAP = 6
BAND_COUNT_RANGE = (3, 7)
GEN_RETRY_LIMIT = 300


def _find_arrow_rows(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    assert w >= 3, "expected room for left-edge arrows and interior bands"

    rows: list[int] = []
    for r in range(1, h - 1):
        if int(grid[r, 1]) != GREY:
            continue
        if int(grid[r - 1, 0]) != GREY or int(grid[r + 1, 0]) != GREY:
            continue
        if int(grid[r, 0]) != BG or int(grid[r - 1, 1]) != BG or int(grid[r + 1, 1]) != BG:
            continue
        rows.append(int(r))

    assert rows, "expected at least one left-edge arrow"
    return rows


def _find_bands(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    bands: list[tuple[int, int]] = []
    seen_colors: set[int] = set()
    for c in range(2, w):
        col = grid[:, c]
        if np.any(col == GREY):
            continue
        colors = {int(v) for v in np.unique(col) if int(v) != BG}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        if int(np.count_nonzero(col == color)) < h - 1:
            continue
        bands.append((int(c), int(color)))
        seen_colors.add(int(color))

    assert bands, "expected vertical color bands"
    assert len(seen_colors) == len(bands), "band colors must be distinct"
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire rays from left-edge arrows, collect colors from vertical bands, and stamp the collected sequence onto the opposite edge.

    Input structure:
    - The background is black.
    - Several gray right-pointing arrows sit on the left edge; each arrow is a 3-pixel shape with one tip pixel in column 1 and two tail pixels in column 0.
    - The middle of the grid contains several one-cell-wide vertical color bands, separated by black columns.
    - Some bands contain a single black gap pixel at a specific row.

    Rule:
    - For each arrow, cast a horizontal ray to the right from the arrow tip row.
    - When the ray crosses a band column, collect that band's color if the band has a colored pixel on the ray row; if the band has its black gap on that row, skip it.
    - Remove the original gray arrow pixels.
    - At the right edge of that same row, draw one pixel per collected color from right to left, with the first collected color closest to the edge.

    Key insight:
    - Each band is a filterable color source: the vertical bands define the candidate colors, and each band's single-row gap determines which arrow rows fail to collect it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    arrow_rows = _find_arrow_rows(g)
    bands = _find_bands(g)

    out = g.copy()
    for r in arrow_rows:
        out[r - 1, 0] = BG
        out[r, 1] = BG
        out[r + 1, 0] = BG

    for r in arrow_rows:
        picked: list[int] = []
        for c, color in bands:
            if int(g[r, c]) == int(color):
                picked.append(int(color))

        for offset, color in enumerate(picked):
            out[r, w - 1 - offset] = int(color)

    return out


def _sample_spaced_rows(height: int, count: int) -> list[int] | None:
    rows = list(range(1, height - 1))
    chosen: list[int] = []
    for _ in range(count):
        candidates = [r for r in rows if all(abs(r - prev) >= MIN_ARROW_ROW_GAP for prev in chosen)]
        if not candidates:
            return None
        chosen.append(random.choice(candidates))
    return sorted(chosen)


def _sample_layout() -> tuple[int, int, list[int], list[int], dict[int, int]] | None:
    height = random.randint(*HEIGHT_RANGE)
    max_arrows = len(list(range(1, height - 1, MIN_ARROW_ROW_GAP)))
    if max_arrows < 2:
        return None

    arrow_count = random.randint(2, min(ARROW_COUNT_CAP, max_arrows))
    arrow_rows = _sample_spaced_rows(height, arrow_count)
    if arrow_rows is None:
        return None

    band_count = random.randint(*BAND_COUNT_RANGE)
    if band_count > len(BAND_PALETTE):
        return None

    left_pad = random.randint(1, 3)
    separators = [random.randint(1, 2) for _ in range(band_count - 1)]
    band_cols: list[int] = []
    c = 2 + left_pad
    for i in range(band_count):
        band_cols.append(int(c))
        if i < band_count - 1:
            c += 1 + separators[i]

    right_margin = band_count + random.randint(1, 4)
    width = band_cols[-1] + 1 + right_margin
    if width > MAX_WIDTH:
        return None

    max_gap_count = min(band_count - 1, arrow_count - 1)
    if max_gap_count < 1:
        return None
    gap_count = random.randint(1, max_gap_count)

    gapped_indices = random.sample(range(band_count), gap_count)
    gap_rows = random.sample(arrow_rows, gap_count)
    gap_by_band = {int(idx): int(row) for idx, row in zip(gapped_indices, gap_rows)}
    return height, width, arrow_rows, band_cols, gap_by_band


def _expected_from_construction(
    grid: np.ndarray,
    arrow_rows: list[int],
    band_colors: list[int],
    gap_by_band: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    _h, w = out.shape
    for r in arrow_rows:
        out[r - 1, 0] = BG
        out[r, 1] = BG
        out[r + 1, 0] = BG

    for r in arrow_rows:
        picked: list[int] = []
        for idx, color in enumerate(band_colors):
            if gap_by_band.get(idx) != r:
                picked.append(int(color))
        for offset, color in enumerate(picked):
            out[r, w - 1 - offset] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_layout()
    if sampled is None:
        return None
    height, width, arrow_rows, band_cols, gap_by_band = sampled

    band_count = len(band_cols)
    band_colors = random.sample(list(BAND_PALETTE), band_count)

    grid = np.full((height, width), BG, dtype=int)

    for r in arrow_rows:
        grid[r - 1, 0] = GREY
        grid[r, 1] = GREY
        grid[r + 1, 0] = GREY

    for idx, (c, color) in enumerate(zip(band_cols, band_colors)):
        grid[:, c] = int(color)
        gap_row = gap_by_band.get(idx)
        if gap_row is not None:
            grid[gap_row, c] = BG

    pickup_sequences = []
    for r in arrow_rows:
        picked = [int(color) for idx, color in enumerate(band_colors) if gap_by_band.get(idx) != r]
        if not picked:
            return None
        pickup_sequences.append(tuple(picked))

    if not any(len(seq) < band_count for seq in pickup_sequences):
        return None
    if not any(len(seq) == band_count for seq in pickup_sequences):
        return None
    if len(set(pickup_sequences)) < 2:
        return None

    expected = _expected_from_construction(grid, arrow_rows, band_colors, gap_by_band)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray band puzzle")
