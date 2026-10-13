from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OBSTACLE = 6
CORRIDOR = 4
MARKER_COLORS = (1, 2, 3, 7, 8, 9)

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (12, 20)
MARKER_COUNT_CHOICES = (3, 4)
OBSTACLE_COUNT_RANGE = (3, 5)
ROW_MARGIN = 2
COL_MARGIN = 1
MIN_MARKER_GAP = 4
GEN_ATTEMPTS = 400


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BACKGROUND, OBSTACLE)):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 1, "each marker color must appear exactly once"
        r, c = map(int, coords[0])
        markers.append((r, c, color))
    markers.sort(key=lambda item: item[0])
    assert 3 <= len(markers) <= 4, "expected 3-4 colored markers"
    assert len({r for r, _c, _color in markers}) == len(markers), "marker rows must be distinct"
    return markers


def _band_ranges(marker_rows: list[int], height: int) -> list[tuple[int, int]]:
    mids = [(marker_rows[i] + marker_rows[i + 1]) // 2 for i in range(len(marker_rows) - 1)]
    starts = [0] + [mid + 1 for mid in mids]
    ends = mids + [height - 1]
    return list(zip(starts, ends))


def _paint_row_preserving_obstacles(out: np.ndarray, row: int, color: int) -> None:
    mask = out[row] != OBSTACLE
    out[row, mask] = int(color)


def _paint_col_segment_preserving_obstacles(
    out: np.ndarray, col: int, r0: int, r1: int, color: int
) -> None:
    segment = out[r0 : r1 + 1, col]
    mask = segment != OBSTACLE
    segment[mask] = int(color)


def _paint_frames(grid: np.ndarray, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    height, width = out.shape
    bands = _band_ranges([r for r, _c, _color in markers], height)

    for (marker_r, _marker_c, color), (band_r0, band_r1) in zip(markers, bands):
        _paint_row_preserving_obstacles(out, marker_r, color)
        _paint_col_segment_preserving_obstacles(out, 0, band_r0, band_r1, color)
        _paint_col_segment_preserving_obstacles(out, width - 1, band_r0, band_r1, color)
        if band_r0 == 0:
            _paint_row_preserving_obstacles(out, 0, color)
        if band_r1 == height - 1:
            _paint_row_preserving_obstacles(out, height - 1, color)

    return out


def _paint_corridors(pre_corridor: np.ndarray, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = pre_corridor.copy()
    width = out.shape[1]
    marker_rows = [r for r, _c, _color in markers]

    for upper_r, lower_r in zip(marker_rows, marker_rows[1:]):
        if lower_r - upper_r <= 1:
            continue
        for c in range(width):
            segment = pre_corridor[upper_r + 1 : lower_r, c]
            if segment.size > 0 and np.all(segment == BACKGROUND):
                out[upper_r + 1 : lower_r, c] = CORRIDOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker row into a band-framed structure, then color the clear columns between adjacent frame walls.

    Input structure:
    - A black background contains 3-4 colored single-pixel markers, each on a distinct row.
    - Magenta single-pixel obstacles may appear anywhere except on marker rows.
    - Marker rows are interpreted from top to bottom, and midpoint rows between adjacent markers define horizontal bands.

    Rule:
    - Sort the markers by row. Use the floor midpoints between adjacent marker rows to partition the full grid into one band per marker.
    - For each band, paint the marker's whole row, plus the leftmost and rightmost columns across that band's row span, in the marker color; if the band touches the grid top or bottom, also paint that outer boundary row. Never overwrite magenta obstacles.
    - For each adjacent pair of markers, treat their painted marker rows as the two opposing horizontal walls. In every column where the cells strictly between those two rows are all black after frame drawing, recolor that full in-between segment yellow.
    - Keep the magenta obstacles unchanged, and do not repaint any non-background frame pixels during corridor coloring.

    Key insight:
    - The walkable corridor is not an input object: it appears only after the marker rows are expanded into framed bands, because those frames create the facing wall rows that define where a clear column can exist.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    markers = _extract_markers(g)
    pre_corridor = _paint_frames(g, markers)
    return _paint_corridors(pre_corridor, markers)


def _sample_marker_rows(height: int, count: int) -> list[int] | None:
    rows = list(range(ROW_MARGIN, height - ROW_MARGIN))
    if len(rows) < count:
        return None
    for _ in range(80):
        chosen = sorted(random.sample(rows, count))
        if min(b - a for a, b in zip(chosen, chosen[1:])) < MIN_MARKER_GAP:
            continue
        return chosen
    return None


def _pair_clear_blocked_columns(grid: np.ndarray, upper_r: int, lower_r: int) -> tuple[list[int], list[int]]:
    clear_cols: list[int] = []
    blocked_cols: list[int] = []
    for c in range(1, grid.shape[1] - 1):
        segment = grid[upper_r + 1 : lower_r, c]
        if segment.size == 0:
            continue
        if np.all(segment == BACKGROUND):
            clear_cols.append(c)
        if np.any(segment == OBSTACLE):
            blocked_cols.append(c)
    return clear_cols, blocked_cols


def _validate_generator_constraints(grid: np.ndarray) -> None:
    markers = _extract_markers(grid)
    marker_rows = [r for r, _c, _color in markers]
    obstacle_coords = np.argwhere(grid == OBSTACLE)
    assert OBSTACLE not in [int(grid[r, c]) for r in marker_rows for c in range(grid.shape[1])]
    assert 3 <= obstacle_coords.shape[0] <= 5

    for upper_r, lower_r in zip(marker_rows, marker_rows[1:]):
        clear_cols, blocked_cols = _pair_clear_blocked_columns(grid, upper_r, lower_r)
        assert clear_cols, "each adjacent marker pair needs at least one clear corridor column"
        assert blocked_cols, "each adjacent marker pair needs at least one blocked column"


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    markers = _extract_markers(inp)
    pre_corridor = _paint_frames(inp, markers)
    marker_rows = [r for r, _c, _color in markers]

    assert np.array_equal(out[inp == OBSTACLE], inp[inp == OBSTACLE]), "obstacles must stay unchanged"

    frame_mask = pre_corridor != inp
    assert np.array_equal(out[frame_mask & (pre_corridor != BACKGROUND)], pre_corridor[frame_mask & (pre_corridor != BACKGROUND)])

    expected = pre_corridor.copy()
    for upper_r, lower_r in zip(marker_rows, marker_rows[1:]):
        for c in range(inp.shape[1]):
            segment = pre_corridor[upper_r + 1 : lower_r, c]
            if segment.size == 0:
                continue
            if np.all(segment == BACKGROUND):
                expected[upper_r + 1 : lower_r, c] = CORRIDOR
            else:
                assert np.array_equal(out[upper_r + 1 : lower_r, c], pre_corridor[upper_r + 1 : lower_r, c])

    assert np.array_equal(out, expected), "output must match frame + corridor semantics"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    marker_count = random.choice(MARKER_COUNT_CHOICES)
    marker_rows = _sample_marker_rows(height, marker_count)
    if marker_rows is None:
        return None

    interior_cols = list(range(COL_MARGIN, width - COL_MARGIN))
    if len(interior_cols) < marker_count + 2:
        return None

    marker_cols = random.sample(interior_cols, marker_count)
    marker_colors = random.sample(MARKER_COLORS, marker_count)
    grid = np.zeros((height, width), dtype=int)

    for r, c, color in zip(marker_rows, marker_cols, marker_colors):
        grid[r, c] = int(color)

    used_obstacle_positions: set[tuple[int, int]] = set()
    reserved_clear: dict[int, set[int]] = {}

    for upper_r, lower_r in zip(marker_rows, marker_rows[1:]):
        corridor_rows = list(range(upper_r + 1, lower_r))
        corridor_cols = list(range(1, width - 1))
        if not corridor_rows or len(corridor_cols) < 2:
            return None

        clear_col, blocked_col = random.sample(corridor_cols, 2)
        reserved_clear.setdefault((upper_r, lower_r), set()).add(clear_col)
        obstacle_r = random.choice(corridor_rows)
        used_obstacle_positions.add((obstacle_r, blocked_col))

    obstacle_target = random.randint(*OBSTACLE_COUNT_RANGE)
    candidate_cells = [
        (r, c)
        for r in range(height)
        for c in range(1, width - 1)
        if r not in marker_rows and grid[r, c] == BACKGROUND
    ]
    random.shuffle(candidate_cells)

    for r, c in candidate_cells:
        if len(used_obstacle_positions) >= obstacle_target:
            break
        blocked = False
        for upper_r, lower_r in zip(marker_rows, marker_rows[1:]):
            if upper_r < r < lower_r and c in reserved_clear[(upper_r, lower_r)]:
                blocked = True
                break
        if blocked:
            continue
        used_obstacle_positions.add((r, c))

    if not (OBSTACLE_COUNT_RANGE[0] <= len(used_obstacle_positions) <= OBSTACLE_COUNT_RANGE[1]):
        return None

    for r, c in used_obstacle_positions:
        grid[r, c] = OBSTACLE

    try:
        _validate_generator_constraints(grid)
        out = main(grid)
        _validate_output_semantics(grid, out)
    except AssertionError:
        return None

    # Reject rare cases where every adjacent gap produces exactly the same corridor layout.
    pair_signatures = []
    for upper_r, lower_r in zip(marker_rows, marker_rows[1:]):
        pair_signatures.append(tuple(int(c) for c in np.where(np.all(out[upper_r + 1 : lower_r] == CORRIDOR, axis=0))[0]))
    if len(pair_signatures) > 1 and len(set(pair_signatures)) == 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
