from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg  # noqa: E402


Coord = tuple[int, int]
Box = tuple[int, int, int, int]


CORNER_SPECS = (
    ((0, 0), Color.RED, (1, 1)),
    ((0, -1), Color.YELLOW, (1, -1)),
    ((-1, 0), Color.BLUE, (-1, 1)),
    ((-1, -1), Color.PINK, (-1, -1)),
)

CELL_SIZE_OPTIONS = (2, 3, 4, 5)
DIRECT_CELL_SIZE_OPTIONS = (2, 3, 4)
GRID_COUNT_OPTIONS = (4, 5)
MAX_GENERATION_ATTEMPTS = 500


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = [-1, *lines, limit]
    regions = [(a + 1, b - 1) for a, b in zip(bounds, bounds[1:])]
    assert all(r0 <= r1 for r0, r1 in regions)
    return regions


def _cell_center(box: Box) -> Coord:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _parse_table(grid: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK
    palette = set(map(int, np.unique(g)))
    assert palette.issubset({Color.BLACK, Color.GREEN})
    assert Color.GREEN in palette

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == Color.GREEN)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == Color.GREEN)]
    assert full_rows and full_cols

    expected = np.zeros(g.shape, dtype=bool)
    expected[full_rows, :] = True
    expected[:, full_cols] = True
    assert np.array_equal(g == Color.GREEN, expected)

    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) in GRID_COUNT_OPTIONS
    assert len(col_regions) in GRID_COUNT_OPTIONS
    assert len(row_regions) >= 4 and len(col_regions) >= 4
    return bg, full_rows, full_cols, row_regions, col_regions


def _cell_box(row_regions: list[tuple[int, int]], col_regions: list[tuple[int, int]], row_i: int, col_i: int) -> Box:
    r0, r1 = row_regions[row_i]
    c0, c1 = col_regions[col_i]
    return r0, c0, r1, c1


def _interior_boxes(row_regions: list[tuple[int, int]], col_regions: list[tuple[int, int]]) -> list[Box]:
    boxes = []
    last_row = len(row_regions) - 1
    last_col = len(col_regions) - 1
    for row_i in range(1, last_row):
        for col_i in range(1, last_col):
            boxes.append(_cell_box(row_regions, col_regions, row_i, col_i))
    return boxes


def _point_in_box(point: Coord, box: Box) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _bresenham_line(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    points: list[Coord] = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        err2 = 2 * err
        if err2 > -dc:
            err -= dc
            r0 += sr
        if err2 < dr:
            err += dr
            c0 += sc
    return points


def _segment_points(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _join_segments(first: list[Coord], second: list[Coord]) -> list[Coord]:
    if not first:
        return second[:]
    if not second:
        return first[:]
    if first[-1] == second[0]:
        return first + second[1:]
    return first + second


def _first_interior_hit(path: list[Coord], interior_boxes: list[Box]) -> tuple[int, Coord, Box] | None:
    for idx, point in enumerate(path):
        for box in interior_boxes:
            if _point_in_box(point, box):
                return idx, point, box
    return None


def _nearest_interior_box(
    row_regions: list[tuple[int, int]],
    col_regions: list[tuple[int, int]],
    corner_row_idx: int,
    corner_col_idx: int,
) -> Box:
    last_row = len(row_regions) - 1
    last_col = len(col_regions) - 1
    target_row = 1 if corner_row_idx == 0 else last_row - 1
    target_col = 1 if corner_col_idx == 0 else last_col - 1
    return _cell_box(row_regions, col_regions, target_row, target_col)


def _fallback_endpoint(target_box: Box, dr: int, dc: int) -> Coord:
    r0, c0, r1, c1 = target_box
    end_r = r0 - 1 if dr > 0 else r1 + 1
    end_c = c0 - 1 if dc > 0 else c1 + 1
    return end_r, end_c


def _count_non_background(grid: np.ndarray, coords: list[Coord]) -> int:
    return sum(int(grid[r, c] != Color.BLACK) for r, c in coords)


def _choose_fallback_path(base: np.ndarray, start: Coord, endpoint: Coord) -> tuple[list[Coord], str]:
    bend_h = (start[0], endpoint[1])
    bend_v = (endpoint[0], start[1])

    h_first = _join_segments(_segment_points(start, bend_h), _segment_points(bend_h, endpoint))
    v_first = _join_segments(_segment_points(start, bend_v), _segment_points(bend_v, endpoint))

    h_first_head = _segment_points(start, bend_h)[1:]
    v_first_head = _segment_points(start, bend_v)[1:]
    h_score = (_count_non_background(base, h_first_head), _count_non_background(base, h_first[1:]), 0)
    v_score = (_count_non_background(base, v_first_head), _count_non_background(base, v_first[1:]), 1)
    if h_score <= v_score:
        return h_first, "hv"
    return v_first, "vh"


def _ray_metadata(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    _bg, _rows, _cols, row_regions, col_regions = _parse_table(g)

    table_center = ((g.shape[0] - 1) // 2, (g.shape[1] - 1) // 2)
    interior_boxes = _interior_boxes(row_regions, col_regions)
    info: list[dict] = []

    for (row_idx, col_idx), color, (dr, dc) in CORNER_SPECS:
        real_row_idx = row_idx if row_idx >= 0 else len(row_regions) + row_idx
        real_col_idx = col_idx if col_idx >= 0 else len(col_regions) + col_idx
        start_box = _cell_box(row_regions, col_regions, real_row_idx, real_col_idx)
        start = _cell_center(start_box)

        diagonal = _bresenham_line(start, table_center)
        hit = _first_interior_hit(diagonal, interior_boxes)
        if hit is not None:
            hit_idx, _hit_point, hit_box = hit
            path = diagonal[:hit_idx]
            info.append(
                {
                    "color": color,
                    "start": start,
                    "path": path,
                    "mode": "direct",
                    "target_box": hit_box,
                    "corner_indices": (real_row_idx, real_col_idx),
                }
            )
            continue

        target_box = _nearest_interior_box(row_regions, col_regions, real_row_idx, real_col_idx)
        endpoint = _fallback_endpoint(target_box, dr, dc)
        path, order = _choose_fallback_path(g, start, endpoint)
        info.append(
            {
                "color": color,
                "start": start,
                "path": path,
                "mode": f"fallback_{order}",
                "target_box": target_box,
                "corner_indices": (real_row_idx, real_col_idx),
            }
        )
    return info


def _paint_base(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, _rows, _cols, row_regions, col_regions = _parse_table(g)

    last_row = len(row_regions) - 1
    last_col = len(col_regions) - 1
    for row_i in range(len(row_regions)):
        for col_i in range(len(col_regions)):
            box = _cell_box(row_regions, col_regions, row_i, col_i)
            fill = None
            if row_i == 0 and col_i == 0:
                fill = Color.RED
            elif row_i == 0 and col_i == last_col:
                fill = Color.YELLOW
            elif row_i == last_row and col_i == 0:
                fill = Color.BLUE
            elif row_i == last_row and col_i == last_col:
                fill = Color.PINK
            elif 0 < row_i < last_row and 0 < col_i < last_col:
                fill = Color.ORANGE
            if fill is not None:
                r0, c0, r1, c1 = box
                out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = _paint_base(grid)
    for ray in _ray_metadata(grid):
        for r, c in ray["path"]:
            out[r, c] = int(ray["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color the corners and strict interior of a green table, then connect each corner inward with a ray.

    Input structure
    - The input is a black grid whose only foreground color is green, and the green pixels form complete horizontal and vertical separator lines.
    - Those separator lines partition the canvas into a 4x4 or 5x5 table of rectangular black cells.

    Rule
    - Preserve the table geometry, fill the four corner cells red/yellow/blue/pink by corner position, and fill every strict interior cell orange while leaving the remaining edge cells black.
    - From the center pixel of each colored corner cell, trace a ray toward the geometric center of the table. If that discrete diagonal path reaches a strict interior cell, color every traversed black or green pixel with the corner color and stop before entering the interior cell.
    - If the diagonal never reaches any strict interior cell, connect the corner to the nearest inward strict interior cell by a two-segment Manhattan path ending at that interior cell's outer corner; choose horizontal-then-vertical or vertical-then-horizontal by whichever first segment crosses fewer non-background input pixels, breaking ties in favor of horizontal first.

    Key insight
    - The separator scaffold defines three positional roles at once: corners become emitters, strict interior cells become stopping targets, and the untouched edge cells form the transit corridor for the rays.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _line_positions(widths: list[int]) -> list[int]:
    positions = []
    pos = widths[0]
    for width in widths[1:]:
        positions.append(pos)
        pos += 1 + width
    return positions


def _sample_band_widths(count: int, rng: np.random.Generator) -> list[int]:
    return [int(rng.choice(CELL_SIZE_OPTIONS)) for _ in range(count)]


def _sample_direct_band_widths(count: int, rng: np.random.Generator) -> list[int]:
    return [int(rng.choice(DIRECT_CELL_SIZE_OPTIONS)) for _ in range(count)]


def _sample_fallback_axis(count: int, rng: np.random.Generator) -> list[int]:
    if count == 4:
        widths = [5, 2, 2, 2]
    else:
        widths = [5, 2, 2, 2, int(rng.choice(CELL_SIZE_OPTIONS))]
    if bool(rng.integers(0, 2)):
        widths = list(reversed(widths))
    return widths


def _build_grid(row_widths: list[int], col_widths: list[int]) -> np.ndarray:
    height = sum(row_widths) + len(row_widths) - 1
    width = sum(col_widths) + len(col_widths) - 1
    grid = np.zeros((height, width), dtype=int)
    grid[_line_positions(row_widths), :] = Color.GREEN
    grid[:, _line_positions(col_widths)] = Color.GREEN
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        _bg, full_rows, full_cols, row_regions, col_regions = _parse_table(inp)
    except Exception:
        return False

    if out.shape != inp.shape:
        return False

    last_row = len(row_regions) - 1
    last_col = len(col_regions) - 1
    line_mask = np.zeros(inp.shape, dtype=bool)
    line_mask[full_rows, :] = True
    line_mask[:, full_cols] = True

    expected_corner_colors = {
        (0, 0): Color.RED,
        (0, last_col): Color.YELLOW,
        (last_row, 0): Color.BLUE,
        (last_row, last_col): Color.PINK,
    }

    for row_i in range(len(row_regions)):
        for col_i in range(len(col_regions)):
            r0, c0, r1, c1 = _cell_box(row_regions, col_regions, row_i, col_i)
            cell = out[r0 : r1 + 1, c0 : c1 + 1]
            if (row_i, col_i) in expected_corner_colors:
                if not np.all(cell == expected_corner_colors[(row_i, col_i)]):
                    return False
            elif 0 < row_i < last_row and 0 < col_i < last_col:
                if not np.all(cell == Color.ORANGE):
                    return False
            else:
                if not np.all(cell == Color.BLACK):
                    return False

    allowed_line_colors = {Color.GREEN, Color.RED, Color.YELLOW, Color.BLUE, Color.PINK}
    if not set(map(int, np.unique(out[line_mask]))).issubset(allowed_line_colors):
        return False

    for ray in _ray_metadata(inp):
        color = int(ray["color"])
        path = ray["path"]
        if not path:
            return False
        for r, c in path:
            if int(out[r, c]) != color:
                return False
        target_box = ray["target_box"]
        if any(_point_in_box(point, target_box) for point in path):
            return False

    if np.any((inp != Color.GREEN) & (inp != Color.BLACK)):
        return False
    return True


def _branch_signature(grid: np.ndarray) -> tuple[bool, bool]:
    modes = [ray["mode"] for ray in _ray_metadata(grid)]
    has_direct = any(mode == "direct" for mode in modes)
    has_fallback = any(mode.startswith("fallback") for mode in modes)
    return has_direct, has_fallback


def _branch_counts(grid: np.ndarray) -> tuple[int, int]:
    modes = [ray["mode"] for ray in _ray_metadata(grid)]
    direct = sum(mode == "direct" for mode in modes)
    fallback = sum(mode.startswith("fallback") for mode in modes)
    return direct, fallback


def _gen_attempt(desired_mode: str, rng: np.random.Generator) -> np.ndarray | None:
    if desired_mode == "direct_only":
        row_count = int(rng.choice(GRID_COUNT_OPTIONS))
        col_count = int(rng.choice(GRID_COUNT_OPTIONS))
        row_widths = _sample_direct_band_widths(row_count, rng)
        col_widths = _sample_direct_band_widths(col_count, rng)
    else:
        fixed_on_rows = bool(rng.integers(0, 2))
        fixed_count = int(rng.choice(GRID_COUNT_OPTIONS))
        free_count = int(rng.choice(GRID_COUNT_OPTIONS))
        fixed_widths = _sample_fallback_axis(fixed_count, rng)
        free_widths = _sample_band_widths(free_count, rng)
        if fixed_on_rows:
            row_widths, col_widths = fixed_widths, free_widths
        else:
            row_widths, col_widths = free_widths, fixed_widths
    grid = _build_grid(row_widths, col_widths)

    has_direct, has_fallback = _branch_signature(grid)
    direct_count, fallback_count = _branch_counts(grid)
    if desired_mode == "direct_only" and not (has_direct and not has_fallback):
        return None
    if desired_mode == "mixed" and not (has_direct and has_fallback):
        return None
    if desired_mode == "fallback_heavy" and not (fallback_count >= 2 and direct_count >= 1):
        return None

    out = main(grid)
    if not _validate_semantics(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    desired_modes = ("direct_only", "mixed")
    weights = np.array([0.45, 0.55], dtype=float)
    weights /= weights.sum()

    desired_mode = str(rng.choice(desired_modes, p=weights))
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(desired_mode, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
