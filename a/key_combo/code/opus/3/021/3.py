from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


try:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_task_3",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    Color = _ARC_UTILS.Color
except Exception:  # pragma: no cover - fallback keeps the file directly loadable
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2


BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)

GRID_HEIGHT_RANGE = (11, 17)
GRID_WIDTH_RANGE = (16, 22)
WALL_ROW_COUNT_RANGE = (3, 5)
GENERATION_ATTEMPTS = 400
ODD_SPANS = (3, 5, 7)


def _complete_wall_gaps(grid: np.ndarray) -> np.ndarray:
    """Fill horizontal single-cell gaps between blue cells in one simultaneous pass."""
    out = np.array(grid, dtype=int, copy=True)
    middle = out[:, 1:-1]
    fill_mask = (middle == BLACK) & (out[:, :-2] == BLUE) & (out[:, 2:] == BLUE)
    middle[fill_mask] = BLUE
    return out


def _row_interval(blue_cols: np.ndarray, ref_col: int, width: int, *, nearest: bool) -> tuple[int, int] | None:
    if blue_cols.size == 0:
        return None

    left = blue_cols[blue_cols < int(ref_col)]
    right = blue_cols[blue_cols >= int(ref_col)]

    if nearest:
        if left.size and right.size:
            start, end = int(left[-1]) + 1, int(right[0]) - 1
        elif left.size:
            start, end = int(left[-1]) + 1, int(width) - 1
        else:
            start, end = int(blue_cols[0]) + 1, int(right[0]) - 1
    else:
        if left.size and right.size:
            start, end = int(blue_cols[0]) + 1, int(blue_cols[-1]) - 1
        elif left.size:
            start, end = int(blue_cols[0]) + 1, int(width) - 1
        else:
            start, end = int(blue_cols[0]) + 1, int(blue_cols[-1]) - 1

    if start > end:
        return None
    return start, end


def _solve(
    grid: np.ndarray,
    *,
    complete_gaps: bool,
    fill_non_wall_rows: bool,
    nearest_walls: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=False)
    assert g.ndim == 2, "expected a 2D grid"

    marker_coords = np.argwhere(g == RED)
    assert marker_coords.shape[0] == 1, "expected exactly one red marker"
    ref_col = int(marker_coords[0, 1])

    walls = _complete_wall_gaps(g) if complete_gaps else g.copy()
    out = walls.copy()
    height, width = out.shape

    wall_rows = [int(r) for r in range(height) if np.any(walls[r] == BLUE)]
    assert wall_rows, "expected at least one wall row"

    intervals: dict[int, tuple[int, int] | None] = {}
    for r in wall_rows:
        blue_cols = np.flatnonzero(walls[r] == BLUE)
        interval = _row_interval(blue_cols, ref_col, width, nearest=nearest_walls)
        intervals[r] = interval
        if interval is None:
            continue
        start, end = interval
        row_slice = out[r, start : end + 1]
        row_slice[row_slice == BLACK] = RED

    if fill_non_wall_rows:
        top_row = wall_rows[0]
        bottom_row = wall_rows[-1]
        wall_row_set = set(wall_rows)
        for r in range(height):
            if r in wall_row_set:
                continue
            if top_row < r < bottom_row:
                row = out[r]
                row[row == BLACK] = RED
                continue

            ref_row = top_row if r < top_row else bottom_row
            interval = intervals.get(ref_row)
            if interval is None:
                continue
            start, end = interval
            row_slice = out[r, start : end + 1]
            row_slice[row_slice == BLACK] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete dashed blue wall fragments, then scan-fill by the red marker column.

    Input structure:
    - The grid has a black background with several rows of horizontal blue dashed segments.
    - Each dashed segment alternates blue and black cells, so every internal dash gap is exactly one cell wide.
    - Exactly one red marker pixel appears on a row without blue cells, and blue wall rows exist both above and below it.

    Rule:
    - First, fill every black cell whose immediate left and right neighbors are blue, turning each dashed segment into a continuous blue wall segment.
    - Let c be the marker column. On each wall row, find the nearest blue wall pixel left of c and the nearest blue wall pixel at or right of c, then fill the black cells between them red; if only the left side exists, fill from that wall's right side to the grid's right edge, and if only the right side exists, use the rule's left-missing interval.
    - On rows without blue walls, fill the whole row red when it lies strictly between the topmost and bottommost wall rows; outside that vertical span, reuse the scan interval of the nearest end wall row.
    - Never overwrite the blue walls or the original red marker.

    Key insight:
    - The blue dashes are encoded walls rather than final shapes: solving requires first repairing the one-cell dash gaps, then using the repaired walls relative to the marker column to decide where red scanline fill belongs.
    """
    return _solve(
        grid,
        complete_gaps=True,
        fill_non_wall_rows=True,
        nearest_walls=True,
    )


def _left_options(ref_col: int, *, reserve_left: int = 0) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for distance in (1, 2, 3):
        end = int(ref_col) - int(distance)
        for span in ODD_SPANS:
            start = end - span + 1
            if start >= int(reserve_left):
                options.append((start, end))
    return options


def _right_options(width: int, ref_col: int, *, reserve_right: int = 0) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for distance in (1, 2, 3):
        start = int(ref_col) + int(distance)
        for span in ODD_SPANS:
            end = start + span - 1
            if end <= int(width) - 1 - int(reserve_right):
                options.append((start, end))
    return options


def _far_left_options(before_start: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for end in range(2, int(before_start) - 2):
        for span in ODD_SPANS:
            start = end - span + 1
            if start >= 0:
                options.append((start, end))
    return options


def _far_right_options(after_end: int, width: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for start in range(int(after_end) + 3, int(width) - 2):
        for span in ODD_SPANS:
            end = start + span - 1
            if end < int(width):
                options.append((start, end))
    return options


def _choose_row_segments(kind: str, width: int, ref_col: int) -> list[tuple[int, int]] | None:
    if kind == "enclosed":
        pairs = [
            (left, right)
            for left in _left_options(ref_col)
            for right in _right_options(width, ref_col)
            if right[0] - left[1] - 1 >= 2
        ]
        if not pairs:
            return None
        left, right = random.choice(pairs)
        return [left, right]

    if kind == "enclosed_multi_left":
        pairs = [
            (left, right)
            for left in _left_options(ref_col, reserve_left=4)
            for right in _right_options(width, ref_col)
            if right[0] - left[1] - 1 >= 2
        ]
        if not pairs:
            return None
        near_left, right = random.choice(pairs)
        far_left_choices = _far_left_options(near_left[0])
        if not far_left_choices:
            return None
        far_left = random.choice(far_left_choices)
        return [far_left, near_left, right]

    if kind == "enclosed_multi_right":
        pairs = [
            (left, right)
            for left in _left_options(ref_col)
            for right in _right_options(width, ref_col, reserve_right=4)
            if right[0] - left[1] - 1 >= 2
        ]
        if not pairs:
            return None
        left, near_right = random.choice(pairs)
        far_right_choices = _far_right_options(near_right[1], width)
        if not far_right_choices:
            return None
        far_right = random.choice(far_right_choices)
        return [left, near_right, far_right]

    if kind == "right_open":
        choices = _left_options(ref_col)
        if not choices:
            return None
        return [random.choice(choices)]

    if kind == "right_open_multi":
        near_left_choices = _left_options(ref_col, reserve_left=4)
        if not near_left_choices:
            return None
        near_left = random.choice(near_left_choices)
        far_left_choices = _far_left_options(near_left[0])
        if not far_left_choices:
            return None
        far_left = random.choice(far_left_choices)
        return [far_left, near_left]

    return None


def _paint_dashed_segment(grid: np.ndarray, row: int, start: int, end: int) -> None:
    grid[int(row), int(start) : int(end) + 1 : 2] = BLUE


def _continuous_segments(row: np.ndarray) -> list[tuple[int, int]]:
    cols = np.flatnonzero(row == BLUE)
    if cols.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        segments.append((start, prev))
        start = prev = col
    segments.append((start, prev))
    return segments


def _validate_input_structure(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    if height > 30 or width > 30:
        return False

    marker_coords = np.argwhere(g == RED)
    if marker_coords.shape[0] != 1:
        return False
    marker_row, ref_col = map(int, marker_coords[0])

    if np.any(g[marker_row] == BLUE):
        return False

    completed = _complete_wall_gaps(g)
    wall_rows = [int(r) for r in range(height) if np.any(completed[r] == BLUE)]
    if not (WALL_ROW_COUNT_RANGE[0] <= len(wall_rows) <= WALL_ROW_COUNT_RANGE[1]):
        return False
    if not any(r < marker_row for r in wall_rows):
        return False
    if not any(r > marker_row for r in wall_rows):
        return False

    top_row = wall_rows[0]
    bottom_row = wall_rows[-1]
    if top_row <= 0 or bottom_row >= height - 1:
        return False

    non_wall_between = [
        r
        for r in range(top_row + 1, bottom_row)
        if not np.any(completed[r] == BLUE)
    ]
    if len(non_wall_between) < 2:
        return False
    if marker_row not in non_wall_between:
        return False

    has_enclosed = False
    has_right_open = False
    blue_count_increase = int(np.count_nonzero(completed == BLUE) - np.count_nonzero(g == BLUE))
    if blue_count_increase <= 0:
        return False

    for r in wall_rows:
        completed_segments = _continuous_segments(completed[r])
        if not completed_segments:
            return False

        previous_end = None
        for start, end in completed_segments:
            if (end - start + 1) < 3:
                return False
            pattern = g[r, start : end + 1]
            if np.any(pattern[::2] != BLUE):
                return False
            if pattern.shape[0] < 3 or np.any(pattern[1::2] != BLACK):
                return False
            if previous_end is not None and start - previous_end - 1 < 2:
                return False
            previous_end = end

        blue_cols = np.flatnonzero(completed[r] == BLUE)
        left = blue_cols[blue_cols < ref_col]
        right = blue_cols[blue_cols >= ref_col]
        if left.size and right.size and int(right[0]) - int(left[-1]) - 1 >= 2:
            has_enclosed = True
        if left.size and not right.size and width - (int(left[-1]) + 1) >= 2:
            has_right_open = True

    if not has_enclosed or not has_right_open:
        return False

    correct = main(g)
    if np.array_equal(correct, g):
        return False

    without_completion = _solve(
        g,
        complete_gaps=False,
        fill_non_wall_rows=True,
        nearest_walls=True,
    )
    if np.array_equal(correct, without_completion):
        return False

    without_non_wall_fill = _solve(
        g,
        complete_gaps=True,
        fill_non_wall_rows=False,
        nearest_walls=True,
    )
    if np.array_equal(correct, without_non_wall_fill):
        return False

    outermost_shortcut = _solve(
        g,
        complete_gaps=True,
        fill_non_wall_rows=True,
        nearest_walls=False,
    )
    if np.array_equal(correct, outermost_shortcut):
        return False

    return True


def _attempt_generate_input() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    ref_col = random.randint(6, width - 7)
    wall_row_count = random.randint(*WALL_ROW_COUNT_RANGE)

    max_span = min(height - 2, wall_row_count + 4)
    min_span = wall_row_count + 2
    if min_span > max_span:
        return None

    span_len = random.randint(min_span, max_span)
    top_row = random.randint(1, height - span_len - 1)
    bottom_row = top_row + span_len - 1
    marker_row = random.randint(top_row + 1, bottom_row - 1)

    candidate_wall_rows = [r for r in range(top_row, bottom_row + 1) if r != marker_row]
    wall_rows = sorted(random.sample(candidate_wall_rows, wall_row_count))
    if not any(r < marker_row for r in wall_rows) or not any(r > marker_row for r in wall_rows):
        return None

    if sum(1 for r in range(top_row + 1, bottom_row) if r not in wall_rows) < 2:
        return None

    row_kinds = [
        "enclosed",
        "enclosed_multi_left",
        "right_open_multi",
    ]
    extra_pool = [
        "enclosed",
        "enclosed_multi_right",
        "right_open",
        "enclosed_multi_left",
        "right_open_multi",
    ]
    while len(row_kinds) < wall_row_count:
        row_kinds.append(random.choice(extra_pool))
    random.shuffle(row_kinds)

    row_segments: dict[int, list[tuple[int, int]]] = {}
    for row, kind in zip(wall_rows, row_kinds):
        segments = _choose_row_segments(kind, width, ref_col)
        if segments is None:
            return None
        row_segments[int(row)] = segments

    grid = np.full((height, width), BLACK, dtype=int)
    grid[marker_row, ref_col] = RED
    for row in wall_rows:
        for start, end in row_segments[row]:
            _paint_dashed_segment(grid, row, start, end)

    return grid if _validate_input_structure(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dashed-wall scan-fill input")
