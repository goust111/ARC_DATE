from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

GEN_ATTEMPTS = 400


def _wall_interval(wall_cols: np.ndarray, marker_col: int, width: int) -> tuple[int, int]:
    left = wall_cols[wall_cols < marker_col]
    right = wall_cols[wall_cols > marker_col]
    assert left.size or right.size, "each wall row needs at least one wall"

    if left.size and right.size:
        start = int(left.max()) + 1
        stop = int(right.min()) - 1
    elif left.size:
        start = int(left.max()) + 1
        stop = int(width) - 1
    else:
        start = 0
        stop = int(right.min()) - 1

    assert 0 <= start <= marker_col <= stop < int(width)
    return start, stop


def _half_mask(half: np.ndarray, marker_color: int) -> tuple[np.ndarray, dict[int, tuple[int, int]], tuple[int, int]]:
    h, w = half.shape
    marker_coords = np.argwhere(half == marker_color)
    assert marker_coords.shape == (1, 2), "expected one marker"
    marker_row, marker_col = map(int, marker_coords[0])

    wall_rows = [int(r) for r in range(h) if np.any(half[r] == YELLOW)]
    assert wall_rows, "expected at least one wall row"
    top = min(wall_rows)
    bottom = max(wall_rows)
    assert top <= marker_row <= bottom, "marker must lie inside the wall band"

    intervals: dict[int, tuple[int, int]] = {}
    for r in wall_rows:
        wall_cols = np.flatnonzero(half[r] == YELLOW)
        intervals[r] = _wall_interval(wall_cols, marker_col, w)

    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        if r in intervals:
            start, stop = intervals[r]
        elif top < r < bottom:
            start, stop = 0, w - 1
        elif r < top:
            start, stop = intervals[top]
        else:
            start, stop = intervals[bottom]
        mask[r, start : stop + 1] = True
    return mask, intervals, (marker_row, marker_col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge two separator-split scanline fills into one orange logical-OR mask.

    Input structure:
    - A black grid is split into equal left/right halves by one full-height red separator column.
    - The left half contains one blue marker and yellow wall pixels; the right half contains one green marker and yellow wall pixels.
    - In each half, wall rows define scanline intervals relative to that half's marker column, with empty rows inside the wall band and reused endpoint intervals outside it.

    Rule:
    - Find the red separator and split the grid into the aligned left and right halves.
    - Solve each half independently: on wall rows fill between the nearest wall left of the marker column and the nearest wall right of it, extending to an edge when only one side exists; empty rows inside the wall band fill the whole row; rows outside the wall band reuse the nearest endpoint interval.
    - Remove the separator and align column k of the left half with column k of the right half; output orange wherever either half's solved fill is active, otherwise black.
    - Marker and wall pixels are only cues and do not appear in the output.

    Key insight:
    - The output is not copied from either half: each side first produces its own scanline-fill mask, and only then are the two aligned masks combined by a pixelwise OR and recolored orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_cols = np.flatnonzero(np.all(g == RED, axis=0))
    assert separator_cols.size == 1, "expected one full-height red separator"
    sep = int(separator_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape and left.size > 0, "expected equal non-empty halves"

    left_mask, _left_intervals, _left_marker = _half_mask(left, BLUE)
    right_mask, _right_intervals, _right_marker = _half_mask(right, GREEN)

    out = np.full(left.shape, BLACK, dtype=int)
    out[left_mask | right_mask] = ORANGE
    return out


def _center_empty_rows(top: int, band_len: int, empty_count: int) -> tuple[int, ...]:
    mid = top + band_len // 2
    if empty_count == 1:
        return (mid,)
    if band_len % 2 == 0:
        return (mid - 1, mid)
    return random.choice(((mid - 1, mid), (mid, mid + 1)))


def _sample_mode() -> str:
    return random.choices(
        ["both", "left_only", "right_only"],
        weights=[5, 2, 2],
        k=1,
    )[0]


def _sample_marker_col(width: int, top_mode: str, bottom_mode: str) -> int:
    mid = width // 2
    if top_mode == bottom_mode == "left_only":
        return random.randint(max(2, mid), width - 2)
    if top_mode == bottom_mode == "right_only":
        return random.randint(1, min(width - 3, mid))
    if "left_only" in (top_mode, bottom_mode) and "right_only" not in (top_mode, bottom_mode):
        return random.randint(max(2, mid - 1), width - 2)
    if "right_only" in (top_mode, bottom_mode) and "left_only" not in (top_mode, bottom_mode):
        return random.randint(1, min(width - 3, mid + 1))
    if width >= 7:
        return random.randint(2, width - 3)
    return random.randint(1, width - 2)


def _sample_endpoint_interval(width: int, marker_col: int, mode: str) -> tuple[int, int]:
    if mode == "both":
        choices = [
            (marker_col - left_reach, marker_col + right_reach)
            for left_reach in range(0, min(2, marker_col - 1) + 1)
            for right_reach in range(0, min(2, width - 2 - marker_col) + 1)
            if left_reach + right_reach >= 1
        ]
    elif mode == "left_only":
        choices = [
            (start, width - 1)
            for start in range(max(1, marker_col - 2), marker_col + 1)
            if width - start >= 2
        ]
    else:
        choices = [
            (0, stop)
            for stop in range(marker_col, min(width - 2, marker_col + 2) + 1)
            if stop + 1 >= 2
        ]
    assert choices
    return random.choice(choices)


def _widen_segment(width: int, intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = [intervals[0]]
    for _ in range(1, len(intervals)):
        start, stop = out[-1]
        can_left = start > 0
        can_right = stop < width - 1
        if can_left and can_right:
            step_left, step_right = random.choices(
                [(1, 0), (0, 1), (1, 1)],
                weights=[4, 4, 1],
                k=1,
            )[0]
        elif can_left:
            step_left, step_right = (1, 0)
        elif can_right:
            step_left, step_right = (0, 1)
        else:
            step_left, step_right = (0, 0)
        next_start = start - step_left
        next_stop = stop + step_right
        if next_start == 0 and next_stop == width - 1:
            if step_left and step_right:
                if random.random() < 0.5:
                    next_start = start
                else:
                    next_stop = stop
            elif step_left:
                next_start = start
            else:
                next_stop = stop
        out.append((next_start, next_stop))
    return out


def _build_half_profile(
    height: int,
    width: int,
    wall_rows: tuple[int, ...],
    empty_rows: tuple[int, ...],
) -> dict[str, object] | None:
    top_mode = _sample_mode()
    bottom_mode = _sample_mode()
    marker_col = _sample_marker_col(width, top_mode, bottom_mode)
    band_rows = tuple(range(min(wall_rows), max(wall_rows) + 1))

    if empty_rows and random.random() < 0.5:
        marker_row = random.choice(empty_rows)
    else:
        marker_row = random.choice(band_rows)

    top_interval = _sample_endpoint_interval(width, marker_col, top_mode)
    bottom_interval = _sample_endpoint_interval(width, marker_col, bottom_mode)

    rows_above_empty = [r for r in wall_rows if r < min(empty_rows)]
    rows_below_empty = [r for r in wall_rows if r > max(empty_rows)]
    assert rows_above_empty and rows_below_empty

    upper = _widen_segment(width, [top_interval] * len(rows_above_empty))
    lower_from_bottom = _widen_segment(width, [bottom_interval] * len(rows_below_empty))

    intervals: dict[int, tuple[int, int]] = {}
    for r, interval in zip(rows_above_empty, upper):
        intervals[int(r)] = interval
    for r, interval in zip(reversed(rows_below_empty), lower_from_bottom):
        intervals[int(r)] = interval

    for start, stop in intervals.values():
        if start == 0 and stop == width - 1:
            return None
        if not (0 <= start <= marker_col <= stop < width):
            return None

    mask = np.zeros((height, width), dtype=bool)
    top = min(wall_rows)
    bottom = max(wall_rows)
    for r in range(height):
        if r in intervals:
            start, stop = intervals[r]
        elif top < r < bottom:
            start, stop = 0, width - 1
        elif r < top:
            start, stop = intervals[top]
        else:
            start, stop = intervals[bottom]
        mask[r, start : stop + 1] = True

    return {
        "marker_row": marker_row,
        "marker_col": marker_col,
        "intervals": intervals,
        "mask": mask,
        "top_mode": top_mode,
        "bottom_mode": bottom_mode,
    }


def _render_half(
    height: int,
    width: int,
    profile: dict[str, object],
    marker_color: int,
) -> np.ndarray:
    half = np.full((height, width), BLACK, dtype=int)
    intervals: dict[int, tuple[int, int]] = profile["intervals"]  # type: ignore[assignment]
    for r, (start, stop) in intervals.items():
        if start > 0:
            half[r, start - 1] = YELLOW
        if stop < width - 1:
            half[r, stop + 1] = YELLOW
    half[int(profile["marker_row"]), int(profile["marker_col"])] = marker_color
    return half


def _validate_profile(
    grid: np.ndarray,
    left_profile: dict[str, object],
    right_profile: dict[str, object],
    wall_rows: tuple[int, ...],
    empty_rows: tuple[int, ...],
) -> bool:
    h, total_w = grid.shape
    w = (total_w - 1) // 2
    left = grid[:, :w]
    right = grid[:, w + 1 :]

    left_mask, left_intervals, _ = _half_mask(left, BLUE)
    right_mask, right_intervals, _ = _half_mask(right, GREEN)

    if not np.array_equal(left_mask, left_profile["mask"]):
        return False
    if not np.array_equal(right_mask, right_profile["mask"]):
        return False
    if left_intervals != left_profile["intervals"]:
        return False
    if right_intervals != right_profile["intervals"]:
        return False

    if tuple(sorted(left_intervals)) != wall_rows or tuple(sorted(right_intervals)) != wall_rows:
        return False

    top = min(wall_rows)
    bottom = max(wall_rows)
    for r in empty_rows:
        if not (top < r < bottom):
            return False
        if not left_mask[r].all() or not right_mask[r].all():
            return False
    for r in range(top):
        if not np.array_equal(left_mask[r], left_mask[top]) or not np.array_equal(right_mask[r], right_mask[top]):
            return False
    for r in range(bottom + 1, h):
        if not np.array_equal(left_mask[r], left_mask[bottom]) or not np.array_equal(right_mask[r], right_mask[bottom]):
            return False

    overlap = int(np.count_nonzero(left_mask & right_mask))
    left_only = int(np.count_nonzero(left_mask & ~right_mask))
    right_only = int(np.count_nonzero(right_mask & ~left_mask))
    if overlap < 3 or left_only < 3 or right_only < 3:
        return False

    union = left_mask | right_mask
    active = int(np.count_nonzero(union))
    if active <= overlap or active > int(0.8 * h * w):
        return False

    expected = np.full((h, w), BLACK, dtype=int)
    expected[union] = ORANGE
    try:
        produced = main(grid)
    except Exception:
        return False
    return np.array_equal(produced, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(11, 14)
        width = random.randint(6, 9)
        band_len = random.randint(5, min(7, height - 4))
        top = random.randint(2, height - band_len - 2)
        empty_count = random.choice((1, 2))
        empty_rows = tuple(_center_empty_rows(top, band_len, empty_count))
        wall_rows = tuple(r for r in range(top, top + band_len) if r not in empty_rows)

        left_profile = _build_half_profile(height, width, wall_rows, empty_rows)
        right_profile = _build_half_profile(height, width, wall_rows, empty_rows)
        if left_profile is None or right_profile is None:
            continue
        if (
            left_profile["top_mode"] == "both"
            and left_profile["bottom_mode"] == "both"
            and right_profile["top_mode"] == "both"
            and right_profile["bottom_mode"] == "both"
        ):
            continue

        if np.array_equal(left_profile["mask"], right_profile["mask"]):
            continue

        left_half = _render_half(height, width, left_profile, BLUE)
        right_half = _render_half(height, width, right_profile, GREEN)
        separator = np.full((height, 1), RED, dtype=int)
        grid = np.concatenate([left_half, separator, right_half], axis=1)

        if _validate_profile(grid, left_profile, right_profile, wall_rows, empty_rows):
            return grid

    raise RuntimeError("failed to generate a valid dual-scanline ARC input")
