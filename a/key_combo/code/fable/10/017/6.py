from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
INTERSECTION_COLOR = 9
MARKER_LENGTHS = tuple(range(1, INTERSECTION_COLOR))
VERTICAL_SIDES = ("top", "bottom")
HORIZONTAL_SIDES = ("left", "right")
OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

GRID_ROWS = (10, 18)
GRID_COLS = (10, 18)
EXTRA_VERTICAL_MARKERS = (0, 2)
EXTRA_HORIZONTAL_MARKERS = (0, 2)
MAX_GENERATION_ATTEMPTS = 400
MAX_INTERSECTIONS = 10


@dataclass(frozen=True)
class Marker:
    side: str
    pos: int
    length: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _vertical_distance(side: str, row: int, height: int) -> int:
    if side == "top":
        return int(row)
    if side == "bottom":
        return int(height - 1 - row)
    raise ValueError(f"not a vertical side: {side}")


def _horizontal_distance(side: str, col: int, width: int) -> int:
    if side == "left":
        return int(col)
    if side == "right":
        return int(width - 1 - col)
    raise ValueError(f"not a horizontal side: {side}")


def _vertical_interval(marker: Marker, height: int) -> tuple[int, int]:
    if marker.side == "top":
        return 1, min(marker.length, height - 2)
    if marker.side == "bottom":
        return max(1, height - 1 - marker.length), height - 2
    raise ValueError(f"not a vertical marker: {marker.side}")


def _horizontal_interval(marker: Marker, width: int) -> tuple[int, int]:
    if marker.side == "left":
        return 1, min(marker.length, width - 2)
    if marker.side == "right":
        return max(1, width - 1 - marker.length), width - 2
    raise ValueError(f"not a horizontal marker: {marker.side}")


def _parse_markers(grid: np.ndarray, bg: int) -> list[Marker]:
    h, w = grid.shape
    markers: list[Marker] = []
    seen: set[tuple[str, int]] = set()
    coords = np.argwhere(grid != bg)
    assert len(coords) > 0, "expected edge markers"

    for raw_r, raw_c in coords:
        r = int(raw_r)
        c = int(raw_c)
        color = int(grid[r, c])
        assert color in MARKER_LENGTHS, "marker colors must encode lengths and exclude the fixed intersection color"
        assert r in (0, h - 1) or c in (0, w - 1), "all non-background cells must sit on the border"
        assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "corner markers are ambiguous"

        if r == 0:
            marker = Marker("top", c, color)
        elif r == h - 1:
            marker = Marker("bottom", c, color)
        elif c == 0:
            marker = Marker("left", r, color)
        else:
            marker = Marker("right", r, color)

        key = (marker.side, marker.pos)
        assert key not in seen, "duplicate markers on the same border cell are invalid"
        seen.add(key)
        markers.append(marker)

    vertical = [m for m in markers if m.side in VERTICAL_SIDES]
    horizontal = [m for m in markers if m.side in HORIZONTAL_SIDES]
    assert vertical and horizontal, "need both vertical and horizontal rays"
    return markers


def _intersection_cells(markers: list[Marker], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    vertical = [m for m in markers if m.side in VERTICAL_SIDES]
    horizontal = [m for m in markers if m.side in HORIZONTAL_SIDES]
    cells: set[tuple[int, int]] = set()

    for v_marker in vertical:
        r0, r1 = _vertical_interval(v_marker, h)
        for h_marker in horizontal:
            c0, c1 = _horizontal_interval(h_marker, w)
            row = h_marker.pos
            col = v_marker.pos
            if r0 <= row <= r1 and c0 <= col <= c1:
                cells.add((row, col))
    return cells


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    markers = _parse_markers(g, bg)
    out = g.copy()
    for r, c in _intersection_cells(markers, g.shape):
        out[r, c] = INTERSECTION_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark only the cells where invisible vertical edge-rays and horizontal edge-rays overlap.

    Input structure:
    - A mostly uniform background contains single-pixel markers placed on the four borders.
    - Each marker color is a number from 1 to 8, so the visible color value is also that ray's length.
    - Markers on the top or bottom emit invisible vertical rays; markers on the left or right emit invisible horizontal rays.

    Rule:
    - Keep every border marker unchanged.
    - For each top/bottom marker, extend an invisible inward ray of exactly that many cells; do the analogous inward extension for each left/right marker.
    - Paint only the interior cells covered by both a vertical ray and a horizontal ray, using the fixed intersection color.

    Key insight:
    - The rays themselves are never drawn, so the only way to find the new pixels is to decode each marker's length from its color value and test vertical-horizontal coverage overlaps.
    """
    return _expected_output(grid)


def _choose(seq):
    return seq[int(np.random.randint(0, len(seq)))]


def _randint(bounds: tuple[int, int]) -> int:
    return int(np.random.randint(bounds[0], bounds[1] + 1))


def _candidate_rows(height: int, side: str, min_dist: int) -> list[int]:
    return [
        r
        for r in range(1, height - 1)
        if min_dist <= _vertical_distance(side, r, height) <= MARKER_LENGTHS[-1]
    ]


def _candidate_cols(width: int, side: str, min_dist: int) -> list[int]:
    return [
        c
        for c in range(1, width - 1)
        if min_dist <= _horizontal_distance(side, c, width) <= MARKER_LENGTHS[-1]
    ]


def _sample_pair_cell(
    height: int,
    width: int,
    v_side: str,
    h_side: str,
    used_positions: dict[str, set[int]],
    min_v_dist: int,
    min_h_dist: int,
    forbidden_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    rows = [r for r in _candidate_rows(height, v_side, min_v_dist) if r not in used_positions[h_side]]
    cols = [c for c in _candidate_cols(width, h_side, min_h_dist) if c not in used_positions[v_side]]
    if not rows or not cols:
        return None

    cells = [(r, c) for r in rows for c in cols if (r, c) not in forbidden_cells]
    if not cells:
        return None
    return _choose(cells)


def _sample_free_position(side: str, height: int, width: int, used_positions: dict[str, set[int]]) -> int | None:
    if side in VERTICAL_SIDES:
        positions = [c for c in range(1, width - 1) if c not in used_positions[side]]
    else:
        positions = [r for r in range(1, height - 1) if r not in used_positions[side]]
    if not positions:
        return None
    return _choose(positions)


def _place_marker(grid: np.ndarray, marker: Marker) -> None:
    if marker.side == "top":
        grid[0, marker.pos] = marker.length
    elif marker.side == "bottom":
        grid[-1, marker.pos] = marker.length
    elif marker.side == "left":
        grid[marker.pos, 0] = marker.length
    else:
        grid[marker.pos, -1] = marker.length


def _marker_reaches_crossing(marker: Marker, row: int, col: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    if marker.side in VERTICAL_SIDES:
        return marker.length >= _vertical_distance(marker.side, row, h)
    return marker.length >= _horizontal_distance(marker.side, col, w)


def _has_exact_shortfall(markers: list[Marker], shape: tuple[int, int]) -> bool:
    h, w = shape
    vertical = [m for m in markers if m.side in VERTICAL_SIDES]
    horizontal = [m for m in markers if m.side in HORIZONTAL_SIDES]
    for v_marker in vertical:
        for h_marker in horizontal:
            row = h_marker.pos
            col = v_marker.pos
            dv = _vertical_distance(v_marker.side, row, h)
            dh = _horizontal_distance(h_marker.side, col, w)
            v_reaches = v_marker.length >= dv
            h_reaches = h_marker.length >= dh
            if v_reaches and dh >= 2 and h_marker.length == dh - 1:
                return True
            if h_reaches and dv >= 2 and v_marker.length == dv - 1:
                return True
    return False


def _case_is_valid(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        if bg != BACKGROUND:
            return False

        markers = _parse_markers(grid, bg)
        vertical = [m for m in markers if m.side in VERTICAL_SIDES]
        horizontal = [m for m in markers if m.side in HORIZONTAL_SIDES]
        if len(vertical) < 2 or len(horizontal) < 2:
            return False
        if len({m.side for m in markers}) < 3:
            return False

        hits = _intersection_cells(markers, grid.shape)
        if not (1 <= len(hits) <= MAX_INTERSECTIONS):
            return False

        pair_total = 0
        negative_pairs = 0
        for v_marker in vertical:
            for h_marker in horizontal:
                pair_total += 1
                row = h_marker.pos
                col = v_marker.pos
                if not (_marker_reaches_crossing(v_marker, row, col, grid.shape) and _marker_reaches_crossing(h_marker, row, col, grid.shape)):
                    negative_pairs += 1
        if pair_total == 0 or negative_pairs == 0:
            return False
        if not _has_exact_shortfall(markers, grid.shape):
            return False

        out = _expected_output(grid)
        if np.array_equal(out, grid):
            return False
        if int(np.sum(out == INTERSECTION_COLOR)) != len(hits):
            return False
        if not np.array_equal(out, main(grid)):
            return False
        return True
    except AssertionError:
        return False


def _attempt_generate() -> np.ndarray | None:
    height = _randint(GRID_ROWS)
    width = _randint(GRID_COLS)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    used_positions = {side: set() for side in (*VERTICAL_SIDES, *HORIZONTAL_SIDES)}
    markers: list[Marker] = []

    v_hit_side = _choose(VERTICAL_SIDES)
    h_hit_side = _choose(HORIZONTAL_SIDES)
    hit_cell = _sample_pair_cell(
        height,
        width,
        v_hit_side,
        h_hit_side,
        used_positions,
        min_v_dist=1,
        min_h_dist=1,
        forbidden_cells=set(),
    )
    if hit_cell is None:
        return None
    hit_row, hit_col = hit_cell
    hit_v_dist = _vertical_distance(v_hit_side, hit_row, height)
    hit_h_dist = _horizontal_distance(h_hit_side, hit_col, width)
    markers.append(Marker(v_hit_side, hit_col, _randint((hit_v_dist, MARKER_LENGTHS[-1]))))
    markers.append(Marker(h_hit_side, hit_row, _randint((hit_h_dist, MARKER_LENGTHS[-1]))))
    used_positions[v_hit_side].add(hit_col)
    used_positions[h_hit_side].add(hit_row)

    v_miss_side = OPPOSITE_SIDE[v_hit_side] if np.random.random() < 0.75 else _choose(VERTICAL_SIDES)
    h_miss_side = OPPOSITE_SIDE[h_hit_side] if np.random.random() < 0.75 else _choose(HORIZONTAL_SIDES)
    short_axis = _choose(("vertical", "horizontal"))
    miss_cell = _sample_pair_cell(
        height,
        width,
        v_miss_side,
        h_miss_side,
        used_positions,
        min_v_dist=2 if short_axis == "vertical" else 1,
        min_h_dist=2 if short_axis == "horizontal" else 1,
        forbidden_cells={hit_cell},
    )
    if miss_cell is None:
        return None
    miss_row, miss_col = miss_cell
    miss_v_dist = _vertical_distance(v_miss_side, miss_row, height)
    miss_h_dist = _horizontal_distance(h_miss_side, miss_col, width)
    miss_v_len = miss_v_dist - 1 if short_axis == "vertical" else _randint((miss_v_dist, MARKER_LENGTHS[-1]))
    miss_h_len = miss_h_dist - 1 if short_axis == "horizontal" else _randint((miss_h_dist, MARKER_LENGTHS[-1]))
    markers.append(Marker(v_miss_side, miss_col, miss_v_len))
    markers.append(Marker(h_miss_side, miss_row, miss_h_len))
    used_positions[v_miss_side].add(miss_col)
    used_positions[h_miss_side].add(miss_row)

    extra_vertical = _randint(EXTRA_VERTICAL_MARKERS)
    extra_horizontal = _randint(EXTRA_HORIZONTAL_MARKERS)

    for _ in range(extra_vertical):
        preferred = [side for side in VERTICAL_SIDES if side not in {m.side for m in markers}]
        if preferred and np.random.random() < 0.7:
            side = _choose(preferred)
        else:
            side = _choose(VERTICAL_SIDES)
        pos = _sample_free_position(side, height, width, used_positions)
        if pos is None:
            continue
        markers.append(Marker(side, pos, _choose(MARKER_LENGTHS)))
        used_positions[side].add(pos)

    for _ in range(extra_horizontal):
        preferred = [side for side in HORIZONTAL_SIDES if side not in {m.side for m in markers}]
        if preferred and np.random.random() < 0.7:
            side = _choose(preferred)
        else:
            side = _choose(HORIZONTAL_SIDES)
        pos = _sample_free_position(side, height, width, used_positions)
        if pos is None:
            continue
        markers.append(Marker(side, pos, _choose(MARKER_LENGTHS)))
        used_positions[side].add(pos)

    for marker in markers:
        _place_marker(grid, marker)

    if _case_is_valid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-ray intersection task")
