from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


BLUE = Color.BLUE
RED = Color.RED
BG = Color.BLACK

GRID_RANGE = (13, 18)
GUIDE_ROW_MARGIN = 3
GUIDE_COL_MARGIN = 2
GUIDE_LEN_RANGE = (8, 12)
GAP_RUN_COUNT = (1, 2)
MAX_BLUE_COUNT = 7
MAX_ATTEMPTS = 400


def _intervals(limit: int, walls: list[int]) -> list[tuple[int, int]]:
    walls = sorted(int(w) for w in walls)
    intervals: list[tuple[int, int]] = []
    start = 0
    for wall in walls:
        if start <= wall - 1:
            intervals.append((start, wall - 1))
        start = wall + 1
    if start <= limit - 1:
        intervals.append((start, limit - 1))
    return intervals


def _contiguous_runs(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    runs = [[int(values[0])]]
    for value in values[1:]:
        value = int(value)
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def _guide_info(grid: np.ndarray) -> dict:
    red = np.argwhere(grid == RED)
    assert red.size > 0, "expected a red guide bar"

    rows = sorted({int(r) for r in red[:, 0]})
    cols = sorted({int(c) for c in red[:, 1]})
    if len(rows) == 1:
        row = rows[0]
        c0 = min(cols)
        c1 = max(cols)
        span = list(range(c0, c1 + 1))
        gaps = [c for c in span if int(grid[row, c]) != RED]
        solids = [c for c in span if int(grid[row, c]) == RED]
        assert 0 < row < grid.shape[0] - 1, "guide needs lanes on both sides"
        assert all(int(grid[row, c]) in {BG, RED} for c in span), "guide row must contain only red cells and gaps"
        return {
            "orientation": "h",
            "fixed": int(row),
            "axis_min": int(c0),
            "axis_max": int(c1),
            "gap_axes": gaps,
            "solid_segments": _contiguous_runs(solids),
            "limit": int(grid.shape[1]),
        }
    if len(cols) == 1:
        col = cols[0]
        r0 = min(rows)
        r1 = max(rows)
        span = list(range(r0, r1 + 1))
        gaps = [r for r in span if int(grid[r, col]) != RED]
        solids = [r for r in span if int(grid[r, col]) == RED]
        assert 0 < col < grid.shape[1] - 1, "guide needs lanes on both sides"
        assert all(int(grid[r, col]) in {BG, RED} for r in span), "guide column must contain only red cells and gaps"
        return {
            "orientation": "v",
            "fixed": int(col),
            "axis_min": int(r0),
            "axis_max": int(r1),
            "gap_axes": gaps,
            "solid_segments": _contiguous_runs(solids),
            "limit": int(grid.shape[0]),
        }
    raise AssertionError("guide must be a straight one-cell-thick bar")


def _compressed_survivors(grid: np.ndarray, info: dict) -> tuple[dict[int, list[int]], list[int]]:
    survivors = {-1: [], 1: []}
    erased: list[int] = []

    for r_raw, c_raw in np.argwhere(grid == BLUE):
        r = int(r_raw)
        c = int(c_raw)
        if info["orientation"] == "h":
            guide_row = int(info["fixed"])
            if r == guide_row:
                continue
            if not (int(info["axis_min"]) <= c <= int(info["axis_max"])):
                continue
            side = -1 if r < guide_row else 1
            if int(grid[guide_row, c]) == RED:
                survivors[side].append(c)
            else:
                erased.append(c)
        else:
            guide_col = int(info["fixed"])
            if c == guide_col:
                continue
            if not (int(info["axis_min"]) <= r <= int(info["axis_max"])):
                continue
            side = -1 if c < guide_col else 1
            if int(grid[r, guide_col]) == RED:
                survivors[side].append(r)
            else:
                erased.append(r)

    for side in (-1, 1):
        survivors[side].sort()
    erased.sort()
    return survivors, erased


def _stack_axes(axes: list[int], walls: list[int], limit: int) -> list[int]:
    packed: list[int] = []
    for start, end in _intervals(limit, walls):
        inside = [int(a) for a in axes if start <= int(a) <= end]
        count = len(inside)
        for axis in range(end - count + 1, end + 1):
            packed.append(int(axis))
    return packed


def _render_blue_positions(shape: tuple[int, int], info: dict, axes_by_side: dict[int, list[int]]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    if info["orientation"] == "h":
        row = int(info["fixed"])
        for side in (-1, 1):
            lane_row = row - 1 if side == -1 else row + 1
            for axis in axes_by_side[side]:
                pixels.add((lane_row, int(axis)))
    else:
        col = int(info["fixed"])
        for side in (-1, 1):
            lane_col = col - 1 if side == -1 else col + 1
            for axis in axes_by_side[side]:
                pixels.add((int(axis), lane_col))
    return pixels


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _guide_info(g)
    bg = int(get_bg(g))
    survivors, _erased = _compressed_survivors(g, info)

    out = np.full_like(g, bg)
    out[g == RED] = RED
    for side in (-1, 1):
        packed_axes = _stack_axes(survivors[side], info["gap_axes"], int(info["limit"]))
        if info["orientation"] == "h":
            lane_row = int(info["fixed"]) - 1 if side == -1 else int(info["fixed"]) + 1
            out[lane_row, packed_axes] = BLUE
        else:
            lane_col = int(info["fixed"]) - 1 if side == -1 else int(info["fixed"]) + 1
            out[packed_axes, lane_col] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress blue pixels onto the cells beside the red guide bar, erase those aimed at guide gaps, then stack the survivors along the guide axis.

    Input structure:
    - The background is black.
    - One red one-cell-thick guide bar is horizontal or vertical and may contain one or two black gap runs along its axis.
    - Blue single pixels are scattered on both sides of the guide, with only background between each blue pixel and the guide.

    Rule:
    - Keep the red guide cells unchanged.
    - Orthogonally project each blue pixel onto the cell immediately beside the guide while preserving which side of the guide it came from.
    - If that projection points to a black gap cell of the guide, erase the blue pixel instead of keeping it.
    - Slide all surviving projected blue pixels simultaneously along the guide axis: rightward for a horizontal guide and downward for a vertical guide.
    - The cells beside guide gaps act as walls, so blue pixels cannot cross those positions, other stopped blue pixels, or the grid boundary.

    Key insight:
    - The same guide gaps play two roles at once: they first filter projected blue pixels out of existence, then they split the adjacent lane into independent one-dimensional gravity chambers.
    """
    return _solve(grid)


def _sample_run_lengths(total_len: int, gap_runs: int, rng: np.random.Generator) -> tuple[list[int], list[int]] | None:
    solid_lengths = [2] * (gap_runs + 1)
    gap_lengths = [1] * gap_runs
    if gap_runs == 1:
        gap_lengths[0] = 2
    remaining = total_len - sum(solid_lengths) - sum(gap_lengths)
    if remaining < 0:
        return None

    buckets = solid_lengths + gap_lengths
    for _ in range(remaining):
        buckets[int(rng.integers(0, len(buckets)))] += 1
    solid_lengths = [int(v) for v in buckets[: gap_runs + 1]]
    gap_lengths = [int(v) for v in buckets[gap_runs + 1 :]]
    if sum(gap_lengths) < 2:
        return None
    return solid_lengths, gap_lengths


def _pick_blue_axes(
    solid_segments: list[list[int]],
    gap_axes: list[int],
    width: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]] | None:
    all_solids = [axis for segment in solid_segments for axis in segment]
    max_survivors = min(len(all_solids), MAX_BLUE_COUNT - 2)
    if len(gap_axes) < 2 or max_survivors < 3:
        return None

    erased_count = int(rng.integers(2, min(len(gap_axes), 4) + 1))
    survivor_count = int(rng.integers(3, max_survivors + 1))

    for _ in range(80):
        erased = sorted(int(v) for v in rng.choice(gap_axes, size=erased_count, replace=False))
        survive = sorted(int(v) for v in rng.choice(all_solids, size=survivor_count, replace=False))
        touched_segments = [idx for idx, segment in enumerate(solid_segments) if any(axis in segment for axis in survive)]
        if len(touched_segments) < 2:
            continue
        if min(touched_segments) == len(solid_segments) - 1:
            continue
        if sorted(_stack_axes(survive, gap_axes, width)) == survive:
            continue
        if _stack_axes(survive, gap_axes, width) == _stack_axes(survive, [], width):
            continue
        return survive, erased
    return None


def _isolated4(cells: list[tuple[int, int]]) -> bool:
    seen = set(cells)
    if len(seen) != len(cells):
        return False
    for r, c in seen:
        if (r - 1, c) in seen or (r + 1, c) in seen or (r, c - 1) in seen or (r, c + 1) in seen:
            return False
    return True


def _place_blue_pixels_horizontal(
    shape: tuple[int, int],
    guide_row: int,
    axes: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    height, _width = shape
    top_max = min(5, guide_row)
    bottom_max = min(5, height - 1 - guide_row)
    if top_max < 2 or bottom_max < 2:
        return None

    for _ in range(80):
        side_flags = rng.choice(np.array([-1, 1], dtype=int), size=len(axes), replace=True)
        if -1 not in side_flags or 1 not in side_flags:
            continue
        cells: list[tuple[int, int]] = []
        for axis, side in zip(axes, side_flags):
            max_dist = top_max if int(side) == -1 else bottom_max
            dist = int(rng.integers(2, max_dist + 1))
            row = guide_row - dist if int(side) == -1 else guide_row + dist
            cells.append((int(row), int(axis)))
        if _isolated4(cells):
            return cells
    return None


def _horizontal_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_RANGE[0], GRID_RANGE[1] + 1))
    width = int(rng.integers(GRID_RANGE[0], GRID_RANGE[1] + 1))
    row = int(rng.integers(GUIDE_ROW_MARGIN, height - GUIDE_ROW_MARGIN))

    max_len = min(GUIDE_LEN_RANGE[1], width - 2 * GUIDE_COL_MARGIN)
    if max_len < GUIDE_LEN_RANGE[0]:
        return None
    guide_len = int(rng.integers(GUIDE_LEN_RANGE[0], max_len + 1))
    gap_runs = int(rng.choice(np.array(GAP_RUN_COUNT, dtype=int)))
    run_lengths = _sample_run_lengths(guide_len, gap_runs, rng)
    if run_lengths is None:
        return None
    solid_lengths, gap_lengths = run_lengths

    start = int(rng.integers(GUIDE_COL_MARGIN, width - guide_len - GUIDE_COL_MARGIN + 1))
    grid = np.full((height, width), BG, dtype=int)

    solid_segments: list[list[int]] = []
    gap_axes: list[int] = []
    cursor = start
    for idx, solid_len in enumerate(solid_lengths):
        segment = list(range(cursor, cursor + solid_len))
        solid_segments.append(segment)
        grid[row, segment] = RED
        cursor += solid_len
        if idx < len(gap_lengths):
            gap_segment = list(range(cursor, cursor + gap_lengths[idx]))
            gap_axes.extend(gap_segment)
            cursor += gap_lengths[idx]

    chosen_axes = _pick_blue_axes(solid_segments, gap_axes, width, rng)
    if chosen_axes is None:
        return None
    survive_axes, erased_axes = chosen_axes
    blue_cells = _place_blue_pixels_horizontal((height, width), row, survive_axes + erased_axes, rng)
    if blue_cells is None:
        return None

    for cell in blue_cells:
        grid[cell] = BLUE
    return grid


def _segment_hits(segments: list[list[int]], axes: list[int]) -> int:
    return sum(1 for segment in segments if any(axis in segment for axis in axes))


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, BLUE, RED}:
        return False
    if int(get_bg(grid)) != BG:
        return False

    info = _guide_info(grid)
    gap_runs = _contiguous_runs(info["gap_axes"])
    if len(gap_runs) not in {1, 2}:
        return False
    if len(info["solid_segments"]) < 2:
        return False

    blue_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == BLUE)]
    if len(blue_coords) < 5:
        return False
    if len(components_mask(grid == BLUE, connectivity=4)) != len(blue_coords):
        return False

    survivors, erased = _compressed_survivors(grid, info)
    survivor_axes = survivors[-1] + survivors[1]
    if len(erased) < 2 or len(survivor_axes) < 3:
        return False
    if _segment_hits(info["solid_segments"], survivor_axes) < 2:
        return False

    if info["orientation"] == "h":
        upper = any(r < int(info["fixed"]) for r, _c in blue_coords)
        lower = any(r > int(info["fixed"]) for r, _c in blue_coords)
        if not (upper and lower):
            return False
    else:
        left = any(c < int(info["fixed"]) for _r, c in blue_coords)
        right = any(c > int(info["fixed"]) for _r, c in blue_coords)
        if not (left and right):
            return False

    compressed_pixels = _render_blue_positions(grid.shape, info, survivors)
    actual = _solve(grid)
    actual_pixels = {tuple(map(int, coord)) for coord in np.argwhere(actual == BLUE)}
    if len(actual_pixels) != len(survivor_axes):
        return False
    if actual_pixels == compressed_pixels:
        return False

    no_wall_pixels = _render_blue_positions(
        grid.shape,
        info,
        {
            -1: _stack_axes(survivors[-1], [], int(info["limit"])),
            1: _stack_axes(survivors[1], [], int(info["limit"])),
        },
    )
    if actual_pixels == no_wall_pixels:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _horizontal_attempt(rng)
        if grid is None:
            continue
        if bool(rng.integers(0, 2)):
            grid = grid.T.copy()
        if _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid guide-bar stacking instance")
