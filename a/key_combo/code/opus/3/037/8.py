from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER_POOL = (1, 2, 7, 9)
LEGEND_POOL = (3, 4, 5, 6, 8)

BODY_MIN = 21
BODY_MAX = 29
MIN_ZONE_SIDE = 9
COUNT_PATTERN_SEQUENCE = [
    (2, 3, 4),
    (2, 2, 3, 4),
    (2, 4, 4),
    (2, 3, 3, 4),
    (2, 2, 4),
    (2, 2, 4, 4),
]
GENERATOR_ATTEMPTS = 500
CHAIN_ATTEMPTS = 200

Coord = tuple[int, int]
Box = tuple[int, int, int, int]

_pattern_index = 0


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _line_points(start: Coord, stop: Coord) -> set[Coord]:
    dr = stop[0] - start[0]
    dc = stop[1] - start[1]
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "edges must be axial or 45-degree"
    steps = max(abs(dr), abs(dc))
    sr = _sign(dr)
    sc = _sign(dc)
    return {
        (start[0] + step * sr, start[1] + step * sc)
        for step in range(steps + 1)
    }


def _pair_outline(a: Coord, b: Coord) -> set[Coord]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert dr != 0 and dc != 0, "same-color markers must differ in both row and column"
    assert abs(dr) != abs(dc), "equal row/column offsets would degenerate the parallelogram"

    diag_len = min(abs(dr), abs(dc))
    diag = (_sign(dr) * diag_len, _sign(dc) * diag_len)
    axis = (dr - diag[0], dc - diag[1])
    assert axis[0] == 0 or axis[1] == 0
    assert axis != (0, 0)

    c = (a[0] + diag[0], a[1] + diag[1])
    d = (a[0] + axis[0], a[1] + axis[1])
    assert (c[0] + axis[0], c[1] + axis[1]) == b
    assert (d[0] + diag[0], d[1] + diag[1]) == b

    cells = set()
    for p0, p1 in ((a, c), (c, b), (b, d), (d, a)):
        cells |= _line_points(p0, p1)
    return cells


def _segment_sets(points: list[Coord]) -> list[set[Coord]]:
    return [_pair_outline(a, b) for a, b in zip(points, points[1:])]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each row-major same-color marker chain with count-selected parallelogram outlines, then drop the legend row.

    Input structure:
    - The background is black and the last row is a three-cell legend for counts 2, 3, and 4.
    - The remaining rows contain 3 or 4 marker colors, each appearing 2 to 4 times as singleton pixels.
    - Markers of one color are read in row-major order and consecutive markers differ in both row and column by unequal magnitudes.

    Rule:
    - Remove the legend row from the output canvas.
    - For each marker color, count its frequency n and read the outline color from the legend column n-2.
    - Connect every adjacent row-major pair of that color as opposite vertices of a parallelogram whose edges are the 45-degree diagonal part and the leftover axial part of the offset vector.
    - Draw the four outline edges, resolve outline overlaps by larger frequency first, and finally restore all original marker pixels unchanged.

    Key insight:
    - The same frequency value is used twice: it chooses the chain length through n-1 adjacent pairs and also chooses the visible outline color through the legend lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3, "expected a 2D grid with a legend row"

    legend = g[-1]
    body = g[:-1]
    out = body.copy()
    best_count = np.full(body.shape, -1, dtype=int)
    best_legend_idx = np.full(body.shape, -1, dtype=int)

    marker_colors = [int(color) for color in np.unique(body) if int(color) != BG]
    assert 3 <= len(marker_colors) <= 4, "expected 3 or 4 marker colors"

    for marker_color in marker_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(body == marker_color)]
        coords.sort()
        count = len(coords)
        assert 2 <= count <= 4, f"marker color {marker_color} must appear 2 to 4 times"

        legend_idx = count - 2
        draw_color = int(legend[legend_idx])
        assert draw_color != BG and draw_color != marker_color, "legend color must be distinct from the marker"

        for cells in _segment_sets(coords):
            for r, c in cells:
                if count > best_count[r, c] or (
                    count == best_count[r, c] and legend_idx > best_legend_idx[r, c]
                ):
                    out[r, c] = draw_color
                    best_count[r, c] = count
                    best_legend_idx[r, c] = legend_idx

    marker_mask = body != BG
    out[marker_mask] = body[marker_mask]
    return out


def _next_count_pattern() -> list[int]:
    global _pattern_index

    if _pattern_index < len(COUNT_PATTERN_SEQUENCE):
        pattern = COUNT_PATTERN_SEQUENCE[_pattern_index]
    else:
        pattern = random.choice(COUNT_PATTERN_SEQUENCE)
    _pattern_index += 1
    return list(pattern)


def _quadrants(rows: int, cols: int) -> list[Box]:
    mid_r = random.randint(MIN_ZONE_SIDE + 1, rows - MIN_ZONE_SIDE - 2)
    mid_c = random.randint(MIN_ZONE_SIDE + 1, cols - MIN_ZONE_SIDE - 2)
    zones = [
        (1, 1, mid_r - 1, mid_c - 1),
        (1, mid_c + 1, mid_r - 1, cols - 2),
        (mid_r + 1, 1, rows - 2, mid_c - 1),
        (mid_r + 1, mid_c + 1, rows - 2, cols - 2),
    ]
    for r0, c0, r1, c1 in zones:
        assert r1 - r0 + 1 >= MIN_ZONE_SIDE
        assert c1 - c0 + 1 >= MIN_ZONE_SIDE
    return zones


def _assign_modes(counts: list[int]) -> list[list[str]]:
    modes = [["h"] * (count - 1) for count in counts]
    group_options = [group_index for group_index, count in enumerate(counts) if count >= 2]
    pick_count = 1 + int(len(group_options) > 1 and random.random() < 0.4)
    for group_index in random.sample(group_options, pick_count):
        seg_index = random.randrange(counts[group_index] - 1)
        modes[group_index][seg_index] = "v"
    return modes


def _sample_rows(zone: Box, count: int, modes: list[str]) -> list[int] | None:
    r0, c0, r1, c1 = zone
    choices = list(range(r0, r1 + 1))
    max_h_gap = (c1 - c0 + 1) - 2
    if len(choices) < count:
        return None

    for _ in range(CHAIN_ATTEMPTS):
        rows = sorted(random.sample(choices, count))
        gaps = [rows[i + 1] - rows[i] for i in range(count - 1)]
        ok = True
        for gap, mode in zip(gaps, modes):
            min_gap = 4 if mode == "v" else 2
            if gap < min_gap:
                ok = False
                break
            if mode == "h" and gap > max_h_gap:
                ok = False
                break
        if ok:
            return rows
    return None


def _sample_cols(zone: Box, rows: list[int], modes: list[str]) -> list[int] | None:
    _r0, c0, _r1, c1 = zone
    choices = list(range(c0, c1 + 1))

    def search(index: int, cols: list[int]) -> list[int] | None:
        if index == len(rows):
            return cols.copy()

        prev_col = cols[-1]
        dr = rows[index] - rows[index - 1]
        mode = modes[index - 1]
        candidates = []
        for col in choices:
            dc = col - prev_col
            abs_dc = abs(dc)
            if abs_dc == 0 or abs_dc == dr:
                continue
            if mode == "h":
                if abs_dc <= dr or abs_dc < 4:
                    continue
            else:
                if dr < 4 or not (0 < abs_dc < dr):
                    continue

            point = (rows[index], col)
            if any(
                max(abs(point[0] - rows[j]), abs(point[1] - cols[j])) < 4
                for j in range(index)
            ):
                continue
            candidates.append(col)

        random.shuffle(candidates)
        for col in candidates:
            cols.append(col)
            found = search(index + 1, cols)
            if found is not None:
                return found
            cols.pop()
        return None

    for _ in range(CHAIN_ATTEMPTS):
        first_col = random.choice(choices)
        found = search(1, [first_col])
        if found is not None:
            return found
    return None


def _chain_is_clear(points: list[Coord]) -> bool:
    segments = _segment_sets(points)
    point_set = set(points)

    for idx, cells in enumerate(segments):
        allowed = {points[idx], points[idx + 1]}
        if (point_set - allowed) & cells:
            return False

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            inter = segments[i] & segments[j]
            allowed = {points[j]} if j == i + 1 else set()
            if inter - allowed:
                return False
    return True


def _sample_chain(zone: Box, count: int, modes: list[str]) -> list[Coord] | None:
    for _ in range(CHAIN_ATTEMPTS):
        rows = _sample_rows(zone, count, modes)
        if rows is None:
            continue
        cols = _sample_cols(zone, rows, modes)
        if cols is None:
            continue
        points = list(zip(rows, cols))
        if _chain_is_clear(points):
            return points
    return None


def _build_case() -> np.ndarray | None:
    body_rows = random.randint(BODY_MIN, BODY_MAX)
    body_cols = random.randint(BODY_MIN, BODY_MAX)
    counts = _next_count_pattern()
    group_count = len(counts)

    zones = _quadrants(body_rows, body_cols)
    chosen_zones = random.sample(zones, group_count)
    chosen_zones.sort(key=lambda box: (box[2] - box[0] + 1) * (box[3] - box[1] + 1), reverse=True)

    marker_colors = random.sample(MARKER_POOL, group_count)
    legend_colors = random.sample([color for color in LEGEND_POOL if color not in marker_colors], 3)
    group_modes = _assign_modes(counts)
    group_specs = list(zip(marker_colors, counts, group_modes))
    group_specs.sort(key=lambda item: item[1], reverse=True)

    grid = np.zeros((body_rows + 1, body_cols), dtype=int)
    grid[-1, :3] = legend_colors

    for zone, (marker_color, count, modes) in zip(chosen_zones, group_specs):
        points = _sample_chain(zone, count, modes)
        if points is None:
            return None
        for r, c in points:
            grid[r, c] = marker_color

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30 or grid.shape[0] < 2 or grid.shape[1] < 3:
        return False

    body = grid[:-1]
    legend = grid[-1]
    marker_colors = [int(color) for color in np.unique(body) if int(color) != BG]
    if not (3 <= len(marker_colors) <= 4):
        return False

    total_markers = int(np.count_nonzero(body))
    if not (8 <= total_markers <= 12):
        return False

    legend_colors = list(map(int, legend[:3]))
    if len(set(legend_colors)) != 3 or any(color == BG for color in legend_colors):
        return False
    if any(color in marker_colors for color in legend_colors):
        return False

    has_horizontal_residual = False
    has_vertical_residual = False
    for color in marker_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(body == color)]
        coords.sort()
        if not (2 <= len(coords) <= 4):
            return False
        for a, b in zip(coords, coords[1:]):
            dr = b[0] - a[0]
            dc = b[1] - a[1]
            if dr == 0 or dc == 0 or abs(dr) == abs(dc):
                return False
            if abs(dc) > abs(dr):
                has_horizontal_residual = True
            if abs(dr) > abs(dc):
                has_vertical_residual = True

    if not (has_horizontal_residual and has_vertical_residual):
        return False

    out = main(grid)
    if out.shape != body.shape or np.array_equal(out, body):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_ATTEMPTS):
        grid = _build_case()
        if grid is not None and _validate_generated(grid):
            return grid
    raise AssertionError("failed to generate a valid marker-chain parallelogram puzzle")
