from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg


def _wall_row_interval(wall_cols: np.ndarray, marker_col: int, width: int) -> tuple[int, int]:
    left = wall_cols[wall_cols < marker_col]
    right = wall_cols[wall_cols >= marker_col]
    assert left.size or right.size, "wall row must constrain the marker column"

    if left.size and right.size:
        start = int(left.max()) + 1
        stop = int(right.min()) - 1
    elif left.size:
        start = int(left.max()) + 1
        stop = int(width) - 1
    else:
        start = 0
        stop = int(right.min()) - 1

    assert 0 <= start <= stop < int(width), "wall row must leave a nonempty passage"
    return start, stop


def _assert_wall_connectivity(g: np.ndarray, wall_color: int, marker_col: int) -> None:
    wall_coords = {tuple(map(int, coord)) for coord in np.argwhere(g == wall_color)}
    remaining = set(wall_coords)
    component_count = 0
    while remaining:
        component_count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    assert component_count == 4, "wall must form four 8-connected arcs around the four openings"

    rows: dict[int, tuple[list[int], list[int]]] = {}
    for r in range(g.shape[0]):
        wall_cols = [int(c) for c in np.flatnonzero(g[r] == wall_color)]
        if not wall_cols:
            continue
        left = [c for c in wall_cols if c < marker_col]
        right = [c for c in wall_cols if c > marker_col]
        for side_cols in (left, right):
            if side_cols:
                assert side_cols == list(range(min(side_cols), max(side_cols) + 1)), "wall side blocks must be contiguous"
        rows[r] = (left, right)

    for r in sorted(rows):
        if r - 1 not in rows:
            continue
        prev_left, prev_right = rows[r - 1]
        left, right = rows[r]
        for prev_side, side in ((prev_left, left), (prev_right, right)):
            if prev_side and side:
                assert min(abs(a - b) for a in prev_side for b in side) <= 1, "adjacent wall rows must remain 8-connected"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the row-wise passage indicated by wall pixels with the singleton marker color.

    Input structure:
    - A dominant background contains one singleton marker pixel and one wall color.
    - Wall pixels lie on several rows and form left/right limits around the marker column.
    - Empty rows between the top and bottom wall rows are open horizontal passages.

    Rule:
    - On a wall row, paint background pixels between the nearest wall strictly
      left of the marker column and the nearest wall at or right of it; if only
      one side is present, paint from that wall to the opposite grid edge.
    - On empty rows between the top and bottom wall rows, paint the whole row.
    - On empty rows outside the wall-row band, reuse the nearest end wall-row interval.
    - Preserve wall pixels and the original marker pixel.

    Key insight:
    - The wall color defines scanline limits relative to the marker column; the
      marker color fills those implied horizontal spans rather than flood-filling
      through connected background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly a marker color and a wall color"

    marker_colors = [color for color in fg_colors if int(np.count_nonzero(g == color)) == 1]
    assert len(marker_colors) == 1, "expected a unique singleton marker color"
    marker_color = int(marker_colors[0])
    wall_colors = [color for color in fg_colors if int(color) != marker_color]
    assert len(wall_colors) == 1, "expected one wall color"
    wall_color = int(wall_colors[0])

    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape == (1, 2), "expected exactly one marker pixel"
    marker_row, marker_col = map(int, marker_coords[0])
    _assert_wall_connectivity(g, wall_color, marker_col)

    row_intervals: dict[int, tuple[int, int]] = {}
    for r in range(height):
        wall_cols = np.flatnonzero(g[r] == wall_color)
        if wall_cols.size:
            row_intervals[int(r)] = _wall_row_interval(wall_cols, marker_col, width)

    assert row_intervals, "expected at least one wall row"
    top_wall_row = min(row_intervals)
    bottom_wall_row = max(row_intervals)
    assert top_wall_row <= marker_row <= bottom_wall_row, "marker must lie within the wall-row band"

    out = g.copy()
    for r in range(height):
        if r in row_intervals:
            start, stop = row_intervals[r]
        elif top_wall_row < r < bottom_wall_row:
            start, stop = 0, int(width) - 1
        elif r < top_wall_row:
            start, stop = row_intervals[top_wall_row]
        else:
            start, stop = row_intervals[bottom_wall_row]

        span = out[r, start : stop + 1]
        span[span == bg] = marker_color

    return out

# --------------- generator (att2) ---------------

from arc_utils import components_mask

# Canvas and band choices.
HEIGHT_BAND_EMPTY_OPTIONS = ((11, 5, 1), (11, 6, 1), (11, 6, 2), (12, 6, 1), (12, 6, 2), (12, 7, 1), (12, 7, 2))
GRID_WIDTH_RANGE = (11, 12)
BAND_TOP_ROW = 2

# Palette and marker placement.
COLOR_IDS = range(10)
COLOR_COUNT = 3
MARKER_COL_RANGE = (3, 8)

# Wall shape and pixel-count controls.
WALL_PIXEL_RANGE = (8, 14)
BASE_WALL_OFFSET = 1
PEAK_WALL_OFFSET_MIN = 2
SIDE_BLOCK_MAX_LEN = 2
EXPECTED_WALL_COMPONENTS = 4
ONE_SIDED_ROW_PROB = 0.25
ONE_SIDED_OPTIONS = ("left", "right")

GENERATION_RETRIES = 300


def _wall_interval(wall_cols: list[int], marker_col: int, width: int) -> tuple[int, int]:
    left = [c for c in wall_cols if c < marker_col]
    right = [c for c in wall_cols if c > marker_col]
    if left and right:
        return max(left) + 1, min(right) - 1
    if left:
        return max(left) + 1, width - 1
    return 0, min(right) - 1


def _expected_output(grid, bg, marker, intervals, top, bottom):
    """Return the solver output implied by the generated wall intervals."""
    height, width = grid.shape
    out = grid.copy()
    for r in range(height):
        if r in intervals:
            start, stop = intervals[r]
        elif top < r < bottom:
            start, stop = 0, width - 1
        elif r < top:
            start, stop = intervals[top]
        else:
            start, stop = intervals[bottom]
        span = out[r, start : stop + 1]
        span[span == bg] = marker
    return out


def _center_empty_rows(top: int, band_len: int, empty_count: int) -> set[int]:
    bottom = top + band_len - 1
    if empty_count == 1:
        if band_len % 2:
            return {(top + bottom) // 2}
        return {random.choice([(top + bottom) // 2, (top + bottom) // 2 + 1])}
    if band_len % 2:
        mid = (top + bottom) // 2
        return set(random.choice([(mid - 1, mid), (mid, mid + 1)]))
    mid = (top + bottom) // 2
    return {mid, mid + 1}


def _peak_offsets(wall_rows: list[int], top: int, bottom: int, maximum: int) -> dict[int, int]:
    max_depth = max(min(r - top, bottom - r) for r in wall_rows)
    if max_depth == 0:
        return {r: BASE_WALL_OFFSET for r in wall_rows}
    peak = random.randint(PEAK_WALL_OFFSET_MIN, min(maximum, BASE_WALL_OFFSET + max_depth))
    offsets = {}
    for r in wall_rows:
        depth = min(r - top, bottom - r)
        offsets[r] = BASE_WALL_OFFSET + round((peak - BASE_WALL_OFFSET) * depth / max_depth)
    return offsets


def _wall_component_count(wall_cols_by_row: dict[int, list[int]]) -> int:
    max_row = max(wall_cols_by_row)
    max_col = max(c for cols in wall_cols_by_row.values() for c in cols)
    mask = np.zeros((max_row + 1, max_col + 1), dtype=bool)
    for r, cols in wall_cols_by_row.items():
        mask[r, cols] = True
    return len(components_mask(mask, connectivity=8))


def _cols_on_side(cols: list[int], marker_col: int, side: str) -> list[int]:
    if side == "left":
        return [c for c in cols if c < marker_col]
    return [c for c in cols if c > marker_col]


def _wall_has_intended_connectivity(wall_cols_by_row: dict[int, list[int]], marker_col: int) -> bool:
    if _wall_component_count(wall_cols_by_row) != EXPECTED_WALL_COMPONENTS:
        return False

    for r in sorted(wall_cols_by_row):
        if r - 1 not in wall_cols_by_row:
            continue
        prev_cols = wall_cols_by_row[r - 1]
        cols = wall_cols_by_row[r]
        for side in ONE_SIDED_OPTIONS:
            prev_side = _cols_on_side(prev_cols, marker_col, side)
            curr_side = _cols_on_side(cols, marker_col, side)
            if prev_side and curr_side and min(abs(a - b) for a in prev_side for b in curr_side) > 1:
                return False
    return True


def _sample_scene() -> tuple[int, int, int, int, int, int, int, int, int, list[int]]:
    height, band_len, empty_count = random.choice(HEIGHT_BAND_EMPTY_OPTIONS)
    width = random.randint(*GRID_WIDTH_RANGE)
    top = BAND_TOP_ROW
    bottom = top + band_len - 1

    bg, wall, marker = random.sample(COLOR_IDS, COLOR_COUNT)
    marker_col = random.randint(*MARKER_COL_RANGE)
    marker_row = random.randint(top, bottom)

    empty_rows = _center_empty_rows(top, band_len, empty_count)
    wall_rows = [r for r in range(top, bottom + 1) if r not in empty_rows]
    return height, width, top, bottom, bg, wall, marker, marker_row, marker_col, wall_rows


def _wall_side_specs(wall_rows, top, bottom, marker_col, width):
    """Choose side presence and wall boundaries for each non-empty band row."""
    side_mode = {r: "both" for r in wall_rows}
    inner_wall_rows = [r for r in wall_rows if r not in (top, bottom)]
    if inner_wall_rows and random.random() < ONE_SIDED_ROW_PROB:
        side_mode[random.choice(inner_wall_rows)] = random.choice(ONE_SIDED_OPTIONS)
    left_offsets = _peak_offsets(wall_rows, top, bottom, marker_col)
    right_offsets = _peak_offsets(wall_rows, top, bottom, width - marker_col - 1)

    sides: list[tuple[int, str, int, int]] = []
    for r in wall_rows:
        if side_mode[r] in ("both", "left"):
            boundary = marker_col - left_offsets[r]
            sides.append((r, "left", boundary, min(SIDE_BLOCK_MAX_LEN, boundary + 1)))
        if side_mode[r] in ("both", "right"):
            boundary = marker_col + right_offsets[r]
            sides.append((r, "right", boundary, min(SIDE_BLOCK_MAX_LEN, width - boundary)))
    return sides


def _choose_extra_side_pixels(sides: list[tuple[int, str, int, int]]) -> set[tuple[int, str, int]] | None:
    base_pixels = len(sides)
    extra_slots = [(r, side, boundary) for r, side, boundary, capacity in sides if capacity >= SIDE_BLOCK_MAX_LEN]
    min_total = max(WALL_PIXEL_RANGE[0], base_pixels)
    max_total = min(WALL_PIXEL_RANGE[1], base_pixels + len(extra_slots))
    if min_total > max_total:
        return None
    target_pixels = random.randint(min_total, max_total)
    extra_count = target_pixels - base_pixels
    return set(random.sample(extra_slots, extra_count))


def _design_wall_columns(wall_rows, top, bottom, marker_col, width):
    """Assign one or two wall pixels to each selected side boundary."""
    sides = _wall_side_specs(wall_rows, top, bottom, marker_col, width)
    extra_set = _choose_extra_side_pixels(sides)
    if extra_set is None:
        return None

    wall_cols_by_row: dict[int, list[int]] = {r: [] for r in wall_rows}
    for r, side, boundary, _capacity in sides:
        length = SIDE_BLOCK_MAX_LEN if (r, side, boundary) in extra_set else 1
        if side == "left":
            wall_cols_by_row[r].extend(range(boundary - length + 1, boundary + 1))
        else:
            wall_cols_by_row[r].extend(range(boundary, boundary + length))
    return wall_cols_by_row


def _render_grid(height, width, bg, wall, marker, marker_row, marker_col, wall_rows, wall_cols_by_row):
    """Paint wall pixels and return row intervals used by the oracle check."""
    grid = np.full((height, width), bg, dtype=int)
    intervals: dict[int, tuple[int, int]] = {}
    for r in wall_rows:
        cols = sorted(wall_cols_by_row[r])
        grid[r, cols] = wall
        intervals[r] = _wall_interval(cols, marker_col, width)
    grid[marker_row, marker_col] = marker
    return grid, intervals


def _passes_oracle_check(grid, bg, marker, intervals, top, bottom):
    expected = _expected_output(grid, bg, marker, intervals, top, bottom)
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    height, width, top, bottom, bg, wall, marker, marker_row, marker_col, wall_rows = _sample_scene()

    wall_cols_by_row = _design_wall_columns(wall_rows, top, bottom, marker_col, width)
    if wall_cols_by_row is None or not _wall_has_intended_connectivity(wall_cols_by_row, marker_col):
        return None

    grid, intervals = _render_grid(height, width, bg, wall, marker, marker_row, marker_col, wall_rows, wall_cols_by_row)
    if not _passes_oracle_check(grid, bg, marker, intervals, top, bottom):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 78e78cff input")
