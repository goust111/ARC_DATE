from __future__ import annotations

import numpy as np


BG_COLOR = 0
LINE_COLOR = 1
MARKER_COLORS = (2, 3)
EVEN_FILL_COLOR = 4
ODD_FILL_COLOR = 7

CELL_SIZES = (2, 3)
LINE_COUNT_MIN = 5
LINE_COUNT_MAX = 7
MARGIN_MIN = 1
MARGIN_MAX = 3
INSIDE_MARKER_FRACTION_RANGE = (0.18, 0.45)
OUTSIDE_MARKER_FRACTION_RANGE = (0.05, 0.22)
OUTSIDE_MARKER_PROB = 0.8
GENERATION_ATTEMPTS = 300


def _is_regular_spacing(positions: list[int]) -> bool:
    if len(positions) < 2:
        return False
    gaps = np.diff(np.asarray(positions, dtype=int))
    return bool(np.all(gaps == gaps[0]))


def _find_lattice_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    height, width = grid.shape
    line_rows = [r for r in range(height) if int(np.count_nonzero(grid[r, :] == LINE_COLOR)) >= width - 2]
    line_cols = [c for c in range(width) if int(np.count_nonzero(grid[:, c] == LINE_COLOR)) >= height - 2]

    assert len(line_rows) >= 4 and len(line_cols) >= 4, "expected a visible lattice"
    assert _is_regular_spacing(line_rows) and _is_regular_spacing(line_cols), "lattice spacing must be regular"
    return line_rows, line_cols


def _find_crop_bbox(
    grid: np.ndarray, line_rows: list[int], line_cols: list[int]
) -> tuple[int, int, int, int]:
    holes = [(r, c) for r in line_rows for c in line_cols if int(grid[r, c]) == BG_COLOR]
    assert len(holes) == 4, "expected exactly four missing lattice intersections"

    hole_rows = sorted({r for r, _c in holes})
    hole_cols = sorted({c for _r, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2, "holes must be rectangle corners"

    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return int(r0), int(c0), int(r1), int(c1)


def _fill_crop(crop: np.ndarray, line_rows: list[int], line_cols: list[int]) -> np.ndarray:
    out = crop.copy()
    cell_row_count = len(line_rows) - 1
    cell_col_count = len(line_cols) - 1
    assert cell_row_count >= 1 and cell_col_count >= 1

    for cell_r in range(cell_row_count):
        for cell_c in range(cell_col_count):
            depth = min(
                cell_r,
                cell_c,
                cell_row_count - 1 - cell_r,
                cell_col_count - 1 - cell_c,
            )
            fill_color = EVEN_FILL_COLOR if depth % 2 == 0 else ODD_FILL_COLOR
            region = out[
                line_rows[cell_r] + 1 : line_rows[cell_r + 1],
                line_cols[cell_c] + 1 : line_cols[cell_c + 1],
            ]
            region[region == BG_COLOR] = fill_color

    return out


def _expected_output(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    top_idx: int,
    bottom_idx: int,
    left_idx: int,
    right_idx: int,
) -> np.ndarray:
    r0 = line_rows[top_idx]
    r1 = line_rows[bottom_idx]
    c0 = line_cols[left_idx]
    c1 = line_cols[right_idx]
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    crop_rows = [line_rows[i] - r0 for i in range(top_idx, bottom_idx + 1)]
    crop_cols = [line_cols[i] - c0 for i in range(left_idx, right_idx + 1)]
    return _fill_crop(crop, crop_rows, crop_cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the lattice rectangle marked by four missing intersections and fill its cells by alternating depth parity.

    Input structure:
    - A black background contains a one-pixel-thick blue rectangular lattice with regular row and column spacing.
    - Exactly four lattice intersections are black holes instead of blue pixels, and those four holes form the corners of one axis-aligned rectangle.
    - Some cell interiors contain one red or green marker pixel, while lattice lines remain blue.

    Rule:
    - Detect the regular blue lattice rows and columns and locate the four black missing intersections on that lattice.
    - Crop the grid to the inclusive rectangle whose corners are those four holes.
    - Within the cropped lattice, fill each cell interior by its layer depth from the crop boundary: even depth cells become yellow and odd depth cells become orange.
    - Only repaint black background pixels inside each cell; preserve existing red/green markers, blue lattice lines, and the four black corner holes.

    Key insight:
    - The same four missing intersections serve two roles at once: they choose the crop and also define the boundary from which the alternating cell-depth coloring is measured.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line_rows, line_cols = _find_lattice_lines(g)
    r0, c0, r1, c1 = _find_crop_bbox(g, line_rows, line_cols)

    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    crop_rows = [r - r0 for r in line_rows if r0 <= r <= r1]
    crop_cols = [c - c0 for c in line_cols if c0 <= c <= c1]
    assert crop_rows[0] == 0 and crop_rows[-1] == r1 - r0
    assert crop_cols[0] == 0 and crop_cols[-1] == c1 - c0
    return _fill_crop(crop, crop_rows, crop_cols)


def _sample_lattice_spec(
    rng: np.random.Generator,
) -> tuple[int, int, list[int], list[int]] | None:
    cell_size = int(rng.choice(CELL_SIZES))
    spacing = cell_size + 1
    h_lines = int(rng.integers(LINE_COUNT_MIN, LINE_COUNT_MAX + 1))
    w_lines = int(rng.integers(LINE_COUNT_MIN, LINE_COUNT_MAX + 1))
    top = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    bottom = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    left = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    right = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))

    height = top + (h_lines - 1) * spacing + 1 + bottom
    width = left + (w_lines - 1) * spacing + 1 + right
    if max(height, width) > 30:
        return None

    line_rows = [top + i * spacing for i in range(h_lines)]
    line_cols = [left + i * spacing for i in range(w_lines)]
    return height, width, line_rows, line_cols


def _sample_span(rng: np.random.Generator, line_count: int) -> tuple[int, int]:
    start = int(rng.integers(0, line_count - 3))
    stop = int(rng.integers(start + 3, line_count))
    return start, stop


def _choose_cells(
    rng: np.random.Generator,
    cells: list[tuple[int, int]],
    fraction_range: tuple[float, float],
    *,
    min_count: int = 0,
) -> list[tuple[int, int]]:
    if not cells:
        return []

    low = max(min_count, int(np.floor(len(cells) * fraction_range[0])))
    high = max(low, int(np.ceil(len(cells) * fraction_range[1])))
    high = min(high, len(cells))
    count = int(rng.integers(low, high + 1))
    if count == 0:
        return []
    picks = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in np.atleast_1d(picks)]


def _place_marker(
    grid: np.ndarray,
    rng: np.random.Generator,
    line_rows: list[int],
    line_cols: list[int],
    cell_r: int,
    cell_c: int,
) -> None:
    r0 = line_rows[cell_r] + 1
    r1 = line_rows[cell_r + 1]
    c0 = line_cols[cell_c] + 1
    c1 = line_cols[cell_c + 1]
    rr = int(rng.integers(r0, r1))
    cc = int(rng.integers(c0, c1))
    grid[rr, cc] = int(rng.choice(MARKER_COLORS))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    spec = _sample_lattice_spec(rng)
    if spec is None:
        return None
    height, width, line_rows, line_cols = spec

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[line_rows, :] = LINE_COLOR
    grid[:, line_cols] = LINE_COLOR

    top_idx, bottom_idx = _sample_span(rng, len(line_rows))
    left_idx, right_idx = _sample_span(rng, len(line_cols))

    hole_rows = [line_rows[top_idx], line_rows[bottom_idx]]
    hole_cols = [line_cols[left_idx], line_cols[right_idx]]
    for r in hole_rows:
        for c in hole_cols:
            grid[r, c] = BG_COLOR

    all_cells = [(r, c) for r in range(len(line_rows) - 1) for c in range(len(line_cols) - 1)]
    inside_cells = [
        (r, c)
        for r, c in all_cells
        if top_idx <= r < bottom_idx and left_idx <= c < right_idx
    ]
    inside_set = set(inside_cells)
    outside_cells = [cell for cell in all_cells if cell not in inside_set]

    inside_picks = _choose_cells(rng, inside_cells, INSIDE_MARKER_FRACTION_RANGE, min_count=1)
    for cell_r, cell_c in inside_picks:
        _place_marker(grid, rng, line_rows, line_cols, cell_r, cell_c)

    if outside_cells and float(rng.random()) < OUTSIDE_MARKER_PROB:
        outside_picks = _choose_cells(rng, outside_cells, OUTSIDE_MARKER_FRACTION_RANGE, min_count=1)
        for cell_r, cell_c in outside_picks:
            _place_marker(grid, rng, line_rows, line_cols, cell_r, cell_c)

    expected = _expected_output(grid, line_rows, line_cols, top_idx, bottom_idx, left_idx, right_idx)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
