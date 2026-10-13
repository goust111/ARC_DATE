from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    module_name = "_arc_utils_key_combo_012_2"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
get_bg = _ARC_UTILS.get_bg


BACKGROUND = int(Color.GREY)
CROSS_COLOR = int(Color.RED)
MIN_FILL_COLOR = int(Color.GREEN)
MAX_FILL_COLOR = int(Color.BLUE)
MARKER_PALETTE = (
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
)

DIM_MIN = 11
DIM_MAX = 21
SEGMENT_MIN = 1
EDGE_SEGMENT_MIN = 2
SEGMENT_MAX = 6
GEN_ATTEMPTS = 500
BRANCHES = ("unique_extremes", "tied_extremes", "all_equal")

_GEN_CALLS = 0

Marker = tuple[int, int, int]


def _extract_markers(grid: np.ndarray) -> list[Marker]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected grey to be the dominant background"

    marker_colors = colors_present(g, exclude={bg})
    assert 2 <= len(marker_colors) <= 3, "expected 2 or 3 marker colors"
    assert CROSS_COLOR not in marker_colors, "input must not already contain red"
    assert int(np.count_nonzero(g != bg)) == len(marker_colors), "markers must be isolated singleton pixels"

    h, w = g.shape
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()
    markers: list[Marker] = []
    for color in marker_colors:
        pts = np.argwhere(g == int(color))
        assert pts.shape == (1, 2), "each marker color must appear exactly once"
        r, c = map(int, pts[0])
        assert 2 <= r <= h - 3 and 2 <= c <= w - 3, "markers must stay away from the border"
        assert r not in seen_rows and c not in seen_cols, "marker rows and columns must be distinct"

        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        assert int(np.count_nonzero(g[r0:r1, c0:c1] != bg)) == 1, "markers must be isolated"

        seen_rows.add(r)
        seen_cols.add(c)
        markers.append((r, c, int(color)))

    rows = sorted(seen_rows)
    cols = sorted(seen_cols)
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:])), "divider rows must leave nonzero strips"
    assert all(b - a >= 2 for a, b in zip(cols, cols[1:])), "divider columns must leave nonzero strips"
    return markers


def _region_segments(dividers: list[int], size: int) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    prev = -1
    for divider in dividers:
        start = prev + 1
        end = int(divider) - 1
        length = end - start + 1
        assert length > 0, "region height/width must be positive"
        segments.append((start, end, length))
        prev = int(divider)

    start = prev + 1
    end = int(size) - 1
    length = end - start + 1
    assert length > 0, "region height/width must be positive"
    segments.append((start, end, length))
    return segments


def _area_summary(heights: list[int], widths: list[int]) -> dict[str, int | bool]:
    areas = [int(h) * int(w) for h in heights for w in widths]
    min_area = min(areas)
    max_area = max(areas)
    return {
        "all_equal": min_area == max_area,
        "min_area": min_area,
        "max_area": max_area,
        "min_count": areas.count(min_area),
        "max_count": areas.count(max_area),
    }


def _solve_from_markers(shape: tuple[int, int], markers: list[Marker]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BACKGROUND, dtype=int)

    for r, c, color in markers:
        out[r, :] = int(color)
        out[:, c] = int(color)

    for i, (row_i, _col_i, _color_i) in enumerate(markers):
        for j, (_row_j, col_j, _color_j) in enumerate(markers):
            if i != j:
                out[row_i, col_j] = CROSS_COLOR

    for r, c, color in markers:
        out[r, c] = int(color)

    row_segments = _region_segments(sorted(r for r, _c, _color in markers), h)
    col_segments = _region_segments(sorted(c for _r, c, _color in markers), w)
    heights = [length for _start, _end, length in row_segments]
    widths = [length for _start, _end, length in col_segments]
    summary = _area_summary(heights, widths)
    if bool(summary["all_equal"]):
        return out

    min_area = int(summary["min_area"])
    max_area = int(summary["max_area"])
    for r0, r1, height in row_segments:
        for c0, c1, width in col_segments:
            area = int(height) * int(width)
            if area == min_area:
                out[r0 : r1 + 1, c0 : c1 + 1] = MIN_FILL_COLOR
            elif area == max_area:
                out[r0 : r1 + 1, c0 : c1 + 1] = MAX_FILL_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker into a full row/column cross, mark mixed-source intersections red, and recolor the smallest and largest grey partitions.

    Input structure:
    - The grid has a grey background and 2 or 3 isolated singleton marker pixels.
    - Marker colors are all distinct, non-grey, non-red, and no two markers share a row or a column.
    - Every marker stays at least two cells away from the border, and divider rows/columns leave only positive-area grey regions.

    Rule:
    - For each marker, paint its entire row and entire column with that marker's color, keeping the marker pixel itself unchanged.
    - Any crossing formed by a row from one marker and a column from another marker becomes red; same-source row/column crossings keep the marker color.
    - Treat all painted rows and columns as dividers, compute the grey-cell area of every resulting rectangular region, and fill all minimum-area regions green and all maximum-area regions blue.
    - If all region areas are equal, leave every region grey; divider pixels are never recolored by the area step.

    Key insight:
    - The markers simultaneously encode visible cross lines and an implicit row/column partition whose region areas determine the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    markers = _extract_markers(g)
    return _solve_from_markers(g.shape, markers)


def _segments_to_dividers(segments: list[int]) -> list[int]:
    acc = 0
    dividers: list[int] = []
    for length in segments[:-1]:
        acc += int(length)
        dividers.append(acc)
        acc += 1
    return dividers


def _equal_segments(marker_count: int) -> list[int]:
    segment_count = marker_count + 1
    min_value = max(EDGE_SEGMENT_MIN, (DIM_MIN - marker_count + segment_count - 1) // segment_count)
    max_value = min(SEGMENT_MAX, (DIM_MAX - marker_count) // segment_count)
    value = random.randint(min_value, max_value)
    return [value] * segment_count


def _random_segments(marker_count: int, *, require_not_all_equal: bool = False, require_unique_extremes: bool = False) -> list[int]:
    segment_count = marker_count + 1
    for _ in range(200):
        values = [random.randint(EDGE_SEGMENT_MIN, SEGMENT_MAX)]
        values.extend(random.randint(SEGMENT_MIN, SEGMENT_MAX) for _ in range(segment_count - 2))
        values.append(random.randint(EDGE_SEGMENT_MIN, SEGMENT_MAX))

        dim = sum(values) + marker_count
        if not (DIM_MIN <= dim <= DIM_MAX):
            continue
        if require_not_all_equal and len(set(values)) == 1:
            continue
        if require_unique_extremes and (
            values.count(min(values)) != 1 or values.count(max(values)) != 1
        ):
            continue
        return values
    raise RuntimeError("failed to sample valid divider segments")


def _branch_segments(branch: str, marker_count: int) -> tuple[list[int], list[int]]:
    if branch == "all_equal":
        return _equal_segments(marker_count), _equal_segments(marker_count)

    if branch == "tied_extremes":
        for _ in range(200):
            if random.random() < 0.5:
                heights = _equal_segments(marker_count)
                widths = _random_segments(marker_count, require_not_all_equal=True)
            else:
                heights = _random_segments(marker_count, require_not_all_equal=True)
                widths = _equal_segments(marker_count)
            summary = _area_summary(heights, widths)
            if not bool(summary["all_equal"]) and (
                int(summary["min_count"]) > 1 or int(summary["max_count"]) > 1
            ):
                return heights, widths
        raise RuntimeError("failed to sample tied-extremes layout")

    if branch == "unique_extremes":
        for _ in range(400):
            heights = _random_segments(marker_count, require_unique_extremes=True)
            widths = _random_segments(marker_count, require_unique_extremes=True)
            summary = _area_summary(heights, widths)
            if (
                not bool(summary["all_equal"])
                and int(summary["min_count"]) == 1
                and int(summary["max_count"]) == 1
            ):
                return heights, widths
        raise RuntimeError("failed to sample unique-extremes layout")

    raise ValueError(f"unknown branch: {branch}")


def _render_input(branch: str, marker_count: int) -> np.ndarray:
    heights, widths = _branch_segments(branch, marker_count)
    rows = _segments_to_dividers(heights)
    cols = _segments_to_dividers(widths)
    h = sum(heights) + marker_count
    w = sum(widths) + marker_count

    grid = np.full((h, w), BACKGROUND, dtype=int)
    colors = random.sample(MARKER_PALETTE, marker_count)
    col_order = random.sample(range(marker_count), marker_count)
    for i, color in enumerate(colors):
        grid[rows[i], cols[col_order[i]]] = int(color)
    return grid


def _validate_case(grid: np.ndarray, expected_branch: str) -> bool:
    try:
        markers = _extract_markers(grid)
        out = main(grid)
    except AssertionError:
        return False

    h, w = grid.shape
    rows = sorted(r for r, _c, _color in markers)
    cols = sorted(c for _r, c, _color in markers)
    row_segments = _region_segments(rows, h)
    col_segments = _region_segments(cols, w)
    heights = [length for _start, _end, length in row_segments]
    widths = [length for _start, _end, length in col_segments]
    summary = _area_summary(heights, widths)

    if expected_branch == "all_equal":
        if not bool(summary["all_equal"]):
            return False
    elif expected_branch == "tied_extremes":
        if bool(summary["all_equal"]) or (
            int(summary["min_count"]) == 1 and int(summary["max_count"]) == 1
        ):
            return False
    elif expected_branch == "unique_extremes":
        if bool(summary["all_equal"]) or int(summary["min_count"]) != 1 or int(summary["max_count"]) != 1:
            return False
    else:
        return False

    marker_rows = {r for r, _c, _color in markers}
    marker_cols = {c for _r, c, _color in markers}
    for r, c, color in markers:
        if int(out[r, c]) != int(color):
            return False

        for cc in range(w):
            expected = int(color)
            if cc != c and cc in marker_cols:
                expected = CROSS_COLOR
            if int(out[r, cc]) != expected:
                return False

        for rr in range(h):
            expected = int(color)
            if rr != r and rr in marker_rows:
                expected = CROSS_COLOR
            if int(out[rr, c]) != expected:
                return False

    min_area = int(summary["min_area"])
    max_area = int(summary["max_area"])
    for r0, r1, height in row_segments:
        for c0, c1, width in col_segments:
            area = int(height) * int(width)
            region = out[r0 : r1 + 1, c0 : c1 + 1]
            if bool(summary["all_equal"]):
                expected = BACKGROUND
            elif area == min_area:
                expected = MIN_FILL_COLOR
            elif area == max_area:
                expected = MAX_FILL_COLOR
            else:
                expected = BACKGROUND
            if not np.all(region == expected):
                return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALLS

    call_id = _GEN_CALLS
    branch = BRANCHES[call_id % len(BRANCHES)]
    marker_count = 2 if call_id % 2 == 0 else 3
    for _ in range(GEN_ATTEMPTS):
        grid = _render_input(branch, marker_count)
        if _validate_case(grid, branch):
            _GEN_CALLS += 1
            return grid

    raise RuntimeError("failed to generate a valid input")
