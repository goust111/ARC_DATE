from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg


BG = Color.BLACK
CONNECTOR = Color.BLUE
CAP = Color.YELLOW
MARKER_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 400
LAYOUT_ATTEMPTS = 200
LENGTH_POOL = (3, 5, 7, 9)
MIN_CENTER_GAP = 6
MIN_SIZE = 20
MAX_SIZE = 29


def _marker_centers(grid: np.ndarray, bg: int, marker_color: int) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for coords in components_mask(grid == marker_color, connectivity=8):
        r0, c0, r1, c1 = bbox_coords(coords)
        assert coords.shape[0] == 4, "each marker must have four arm pixels"
        assert (r1 - r0, c1 - c0) == (2, 2), "each marker must span a 3x3 box"

        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        expected = {(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)}
        assert set(map(tuple, coords)) == expected, "foreground objects must be hollow plus markers"
        assert int(grid[cr, cc]) == bg, "marker centers must stay empty"
        centers.append((int(cr), int(cc)))
    return centers


def _segments_from_centers(centers: list[tuple[int, int]]) -> list[dict]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in centers:
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))

    segments: list[dict] = []
    for r, cols_in_row in rows.items():
        ordered = sorted(cols_in_row)
        for c0, c1 in zip(ordered, ordered[1:]):
            blue_cells = [(int(r), c) for c in range(int(c0) + 2, int(c1) - 1)]
            segments.append(
                {
                    "orientation": "H",
                    "length": len(blue_cells),
                    "blue_cells": blue_cells,
                    "endpoints": [((int(r), int(c0) + 1), -1), ((int(r), int(c1) - 1), 1)],
                    "sort_key": (int(r), int(c0) + 1, 0),
                }
            )

    for c, rows_in_col in cols.items():
        ordered = sorted(rows_in_col)
        for r0, r1 in zip(ordered, ordered[1:]):
            blue_cells = [(r, int(c)) for r in range(int(r0) + 2, int(r1) - 1)]
            segments.append(
                {
                    "orientation": "V",
                    "length": len(blue_cells),
                    "blue_cells": blue_cells,
                    "endpoints": [((int(r0) + 1, int(c)), -1), ((int(r1) - 1, int(c)), 1)],
                    "sort_key": (int(r0) + 1, int(c), 1),
                }
            )
    return segments


def _point_span(center: int, length: int, bias: int) -> tuple[int, int]:
    half = int(length) // 2
    if int(length) % 2 == 1:
        return int(center) - half, int(center) + half
    if int(bias) < 0:
        return int(center) - half, int(center) + half - 1
    return int(center) - half + 1, int(center) + half


def _paint_row(out: np.ndarray, row: int, c0: int, c1: int, marker_color: int) -> None:
    if not 0 <= row < out.shape[0]:
        return
    for col in range(int(c0), int(c1) + 1):
        if 0 <= col < out.shape[1] and int(out[row, col]) != int(marker_color):
            out[row, col] = CAP


def _paint_col(out: np.ndarray, col: int, r0: int, r1: int, marker_color: int) -> None:
    if not 0 <= col < out.shape[1]:
        return
    for row in range(int(r0), int(r1) + 1):
        if 0 <= row < out.shape[0] and int(out[row, col]) != int(marker_color):
            out[row, col] = CAP


def _paint_caps(
    out: np.ndarray,
    orientation: str,
    endpoint: tuple[int, int],
    length: int,
    bias: int,
    marker_color: int,
) -> None:
    r, c = endpoint
    current = int(length)
    distance = 1
    while current > 0:
        if orientation == "H":
            lo, hi = _point_span(c, current, bias)
            _paint_row(out, int(r) - distance, lo, hi, marker_color)
            _paint_row(out, int(r) + distance, lo, hi, marker_color)
        else:
            lo, hi = _point_span(r, current, bias)
            _paint_col(out, int(c) - distance, lo, hi, marker_color)
            _paint_col(out, int(c) + distance, lo, hi, marker_color)
        current -= 2
        distance += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect adjacent hollow-plus markers and use each connector length to drive perpendicular shrinking caps.

    Input structure:
    - The background is black.
    - The foreground contains several same-color 3x3 hollow-plus markers: only the north, south, east, and west arm pixels are present and the center stays empty.
    - Some marker centers share a row or a column, with background cells between the opposing arms of adjacent markers.

    Rule:
    - Recover the empty center of every marker.
    - For each row and each column, sort centers and connect every adjacent aligned pair with blue pixels across the background gap between their opposing arms.
    - Let L be the number of blue pixels in one connector. From each endpoint arm, extend yellow caps in both perpendicular directions: first a length-L bar, then L-2, then L-4, clipped at the grid edge.
    - Yellow may overwrite blue or earlier yellow, but original marker pixels keep their input color.

    Key insight:
    - The hollow-plus markers define a visibility graph, and each connector gap length is reused as the seed width for a perpendicular shrinking cap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected one marker color"
    marker_color = int(foreground_colors[0])

    centers = _marker_centers(g, bg, marker_color)
    segments = _segments_from_centers(centers)

    out = g.copy()
    for segment in segments:
        for r, c in segment["blue_cells"]:
            assert int(g[r, c]) == bg, "connector paths must be empty in the input"
            out[r, c] = CONNECTOR

    for segment in sorted(segments, key=lambda item: item["sort_key"]):
        for endpoint, bias in segment["endpoints"]:
            _paint_caps(out, segment["orientation"], endpoint, segment["length"], bias, marker_color)
    return out


def _pairwise_clear(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(int(r0) - int(r1)) < MIN_CENTER_GAP and abs(int(c0) - int(c1)) < MIN_CENTER_GAP:
                return False
    return True


def _render_grid(shape: tuple[int, int], centers: list[tuple[int, int]], marker_color: int) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in centers:
        for dr, dc in MARKER_OFFSETS:
            grid[int(r) + dr, int(c) + dc] = int(marker_color)
    return grid


def _make_t4(rows: int, cols: int, lengths: tuple[int, ...]) -> list[tuple[int, int]] | None:
    left_len, right_len, vert_len = random.sample(list(lengths), 3)
    left_step, right_step, vert_step = left_len + 3, right_len + 3, vert_len + 3
    direction = random.choice((-1, 1))

    r_low = 1 + (vert_step if direction < 0 else 0)
    r_high = rows - 2 - (vert_step if direction > 0 else 0)
    c_low = 1 + left_step
    c_high = cols - 2 - right_step
    if r_low > r_high or c_low > c_high:
        return None

    r = random.randint(r_low, r_high)
    c = random.randint(c_low, c_high)
    return [(r, c - left_step), (r, c), (r, c + right_step), (r + direction * vert_step, c)]


def _make_chain_pair(rows: int, cols: int, lengths: tuple[int, ...]) -> list[tuple[int, int]] | None:
    h0_len, h1_len, v_len = random.sample(list(lengths), 3)
    h0_step, h1_step, v_step = h0_len + 3, h1_len + 3, v_len + 3
    if cols < h0_step + h1_step + 3 or rows < v_step + 3:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        row = random.randint(1, rows - 2)
        c0 = random.randint(1, cols - 2 - h0_step - h1_step)
        h_points = [(row, c0), (row, c0 + h0_step), (row, c0 + h0_step + h1_step)]

        candidates = [c for c in range(1, cols - 1) if all(abs(c - hc) >= MIN_CENTER_GAP for _, hc in h_points)]
        if not candidates:
            continue
        col = random.choice(candidates)
        r0 = random.randint(1, rows - 2 - v_step)
        v_points = [(r0, col), (r0 + v_step, col)]
        points = h_points + v_points
        if any(vr == row for vr, _ in v_points):
            continue
        if _pairwise_clear(points):
            return points
    return None


def _make_row3_col3(rows: int, cols: int, lengths: tuple[int, ...]) -> list[tuple[int, int]] | None:
    h0_len, h1_len, v0_len, v1_len = random.sample(list(lengths), 4)
    h0_step, h1_step = h0_len + 3, h1_len + 3
    v0_step, v1_step = v0_len + 3, v1_len + 3
    if cols < h0_step + h1_step + 3 or rows < v0_step + v1_step + 3:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        row = random.randint(1, rows - 2)
        c0 = random.randint(1, cols - 2 - h0_step - h1_step)
        h_points = [(row, c0), (row, c0 + h0_step), (row, c0 + h0_step + h1_step)]

        candidates = [c for c in range(1, cols - 1) if all(abs(c - hc) >= MIN_CENTER_GAP for _, hc in h_points)]
        if not candidates:
            continue
        col = random.choice(candidates)
        r0 = random.randint(1, rows - 2 - v0_step - v1_step)
        v_points = [(r0, col), (r0 + v0_step, col), (r0 + v0_step + v1_step, col)]
        points = h_points + v_points
        if any(vr == row for vr, _ in v_points):
            continue
        if _pairwise_clear(points):
            return points
    return None


CORE_BUILDERS = {
    "t4": (4, _make_t4),
    "chain_pair": (5, _make_chain_pair),
    "row3_col3": (6, _make_row3_col3),
}


def _available_isolated_points(existing: list[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int]]:
    used_rows = {int(r) for r, _ in existing}
    used_cols = {int(c) for _, c in existing}
    candidates: list[tuple[int, int]] = []
    for r in range(1, rows - 1):
        if r in used_rows:
            continue
        for c in range(1, cols - 1):
            if c in used_cols:
                continue
            point = (r, c)
            if _pairwise_clear(existing + [point]):
                candidates.append(point)
    return candidates


def _add_isolated_points(
    points: list[tuple[int, int]], target_count: int, rows: int, cols: int
) -> list[tuple[int, int]] | None:
    out = list(points)
    while len(out) < int(target_count):
        candidates = _available_isolated_points(out, rows, cols)
        if not candidates:
            return None
        out.append(random.choice(candidates))
    return out


def _choose_lengths(count: int) -> tuple[int, ...]:
    return tuple(random.sample(LENGTH_POOL, int(count)))


def _minimum_size(template_name: str, lengths: tuple[int, ...]) -> tuple[int, int]:
    steps = sorted([int(length) + 3 for length in lengths], reverse=True)
    if template_name == "t4":
        return max(MIN_SIZE, steps[0] + 3), max(MIN_SIZE, steps[0] + steps[1] + 3)
    if template_name == "chain_pair":
        return max(MIN_SIZE, steps[0] + 3), max(MIN_SIZE, steps[0] + steps[1] + 3)
    return max(MIN_SIZE, steps[0] + steps[1] + 3), max(MIN_SIZE, steps[0] + steps[1] + 3)


def _attempt_grid() -> np.ndarray | None:
    target_count = random.randint(4, 8)
    template_name = random.choice([name for name, (core_size, _builder) in CORE_BUILDERS.items() if core_size <= target_count])
    core_size, builder = CORE_BUILDERS[template_name]
    lengths = _choose_lengths(3 if template_name != "row3_col3" else 4)
    min_rows, min_cols = _minimum_size(template_name, lengths)
    if min_rows > MAX_SIZE or min_cols > MAX_SIZE:
        return None

    rows = random.randint(min_rows, MAX_SIZE)
    cols = random.randint(min_cols, MAX_SIZE)
    points = builder(rows, cols, lengths)
    if points is None:
        return None
    points = _add_isolated_points(points, target_count, rows, cols)
    if points is None:
        return None

    marker_color = Color.RED
    grid = _render_grid((rows, cols), points, marker_color)

    centers = _marker_centers(grid, BG, marker_color)
    segments = _segments_from_centers(centers)
    if len(segments) < 3:
        return None
    lengths_seen = [segment["length"] for segment in segments]
    if len(lengths_seen) != len(set(lengths_seen)):
        return None
    if {segment["orientation"] for segment in segments} != {"H", "V"}:
        return None
    if any(int(grid[r, c]) != BG for segment in segments for r, c in segment["blue_cells"]):
        return None

    out = main(grid)
    yellow_count = int(np.count_nonzero(out == CAP))
    if yellow_count < 18 or yellow_count > (rows * cols) // 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-cross connector input")
