from __future__ import annotations

import random
from collections import deque

import numpy as np


AZURE = 8
GREY = 5
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 9)

MIN_BANDS = 2
MAX_BANDS = 4
MIN_BAND_SIZE = 2
MAX_BAND_SIZE = 6
MIN_GRID_SIZE = 10
MAX_GREY_FILL_RATIO = 0.82
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _band_layout(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    dividers: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        bands.append((cursor, cursor + size - 1))
        cursor += size
        if idx + 1 < len(sizes):
            dividers.append(cursor)
            cursor += 1
    return bands, dividers, cursor


def _choose_band_sizes(count: int) -> list[int]:
    return [random.randint(MIN_BAND_SIZE, MAX_BAND_SIZE) for _ in range(count)]


def _choose_pair_positions(bands: list[tuple[int, int]]) -> tuple[int, int]:
    same_band = len(bands) == 1 or random.random() < 0.5
    wide_bands = [idx for idx, (start, end) in enumerate(bands) if end - start + 1 >= 2]

    if same_band and wide_bands:
        band_idx = random.choice(wide_bands)
        start, end = bands[band_idx]
        return tuple(sorted(random.sample(range(start, end + 1), 2)))

    idx_a, idx_b = random.sample(range(len(bands)), 2)
    a0, a1 = bands[idx_a]
    b0, b1 = bands[idx_b]
    p0 = random.randint(a0, a1)
    p1 = random.randint(b0, b1)
    return tuple(sorted((p0, p1)))


def _line_specs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != AZURE)
    horizontal: list[tuple[int, int]] = []
    vertical: list[tuple[int, int]] = []
    points: list[tuple[int, int]] = []

    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each signal color must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        if r0 == r1 and c0 != c1:
            horizontal.append((r0, color))
        elif c0 == c1 and r0 != r1:
            vertical.append((c0, color))
        else:
            raise AssertionError("each signal pair must align horizontally or vertically")
        points.extend(((r0, c0), (r1, c1)))

    rows = [r for r, _color in horizontal]
    cols = [c for c, _color in vertical]
    assert len(rows) == len(set(rows)), "divider rows must be unique"
    assert len(cols) == len(set(cols)), "divider columns must be unique"
    return horizontal, vertical, points


def _draw_dividers(grid: np.ndarray, horizontal: list[tuple[int, int]], vertical: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for col, color in vertical:
        out[:, col] = color
    for row, color in horizontal:
        out[row, :] = color
    return out


def _label_regions(divider_grid: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = divider_grid.shape
    labels = np.full((h, w), -1, dtype=int)
    next_label = 0

    for r in range(h):
        for c in range(w):
            if divider_grid[r, c] != AZURE or labels[r, c] != -1:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_label
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and divider_grid[nr, nc] == AZURE and labels[nr, nc] == -1:
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
            next_label += 1
    return labels, next_label


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    horizontal, vertical, points = _line_specs(g)
    assert horizontal or vertical, "expected at least one aligned signal pair"

    divider_grid = _draw_dividers(g, horizontal, vertical)
    labels, region_count = _label_regions(divider_grid)
    flagged: set[int] = set()

    for r, c in points:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                label = int(labels[nr, nc])
                if label >= 0:
                    flagged.add(label)

    out = divider_grid.copy()
    for label in flagged:
        out[labels == label] = GREY
    return out, len(flagged), region_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend aligned signal pairs into full dividers, then grey-fill the touched partitions.

    Input structure:
    - The background is Azure.
    - Several non-background colors appear as isolated signal pairs, with exactly two pixels per color.
    - Each same-color pair lies on one shared row or one shared column.

    Rule:
    - For every horizontal pair, paint its whole row in that color; for every vertical pair, paint its whole column in that color.
    - Treat those full rows and columns as dividers that partition the remaining Azure cells into 4-connected regions.
    - Any region that touches an original signal pixel across one of its four sides is filled with Grey.
    - Keep the colored divider lines and the original signal pixels unchanged.

    Key insight:
    - The paired pixels are not endpoints of short segments; they announce full-width or full-height separator lines, and the regions adjacent to those boundary markers are the ones that turn Grey.
    """
    out, _flagged, _regions = _solve_with_meta(grid)
    return out


def _gen_attempt() -> np.ndarray | None:
    row_band_count = random.randint(MIN_BANDS, MAX_BANDS)
    col_band_count = random.randint(MIN_BANDS, MAX_BANDS)
    row_bands, divider_rows, height = _band_layout(_choose_band_sizes(row_band_count))
    col_bands, divider_cols, width = _band_layout(_choose_band_sizes(col_band_count))

    if height < MIN_GRID_SIZE or width < MIN_GRID_SIZE:
        return None

    line_count = len(divider_rows) + len(divider_cols)
    if line_count > len(SIGNAL_COLORS):
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    colors = random.sample(SIGNAL_COLORS, line_count)
    color_iter = iter(colors)

    for row in divider_rows:
        c0, c1 = _choose_pair_positions(col_bands)
        color = next(color_iter)
        grid[row, c0] = color
        grid[row, c1] = color

    for col in divider_cols:
        r0, r1 = _choose_pair_positions(row_bands)
        color = next(color_iter)
        grid[r0, col] = color
        grid[r1, col] = color

    try:
        out, marked_regions, total_regions = _solve_with_meta(grid)
    except AssertionError:
        return None

    if marked_regions < 2 or marked_regions >= total_regions:
        return None

    grey_cells = int(np.count_nonzero(out == GREY))
    if grey_cells == 0:
        return None

    bg_cells = int(np.count_nonzero(grid == AZURE))
    if grey_cells / bg_cells > MAX_GREY_FILL_RATIO:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
