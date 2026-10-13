from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
TRACE_COLOR = 8
SQUARE_COLOR = 4

HEIGHT_RANGE = (12, 24)
WIDTH_RANGE = (14, 28)
OBJECT_COUNT_RANGE = (2, 4)
SIDE_RANGE = (2, 6)
MAX_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _segments_1d(values: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(np.asarray(values, dtype=int).tolist()):
        if int(value) == int(color):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _interval_is_clear(existing: list[tuple[int, int]], start: int, end: int, gap: int = 1) -> bool:
    for other_start, other_end in existing:
        if not (end + gap < other_start or other_end + gap < start):
            return False
    return True


def _box_is_clear(
    existing: list[tuple[int, int, int, int]],
    candidate: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = candidate
    for rr0, cc0, rr1, cc1 in existing:
        rows_sep = r1 + gap < rr0 or rr1 + gap < r0
        cols_sep = c1 + gap < cc0 or cc1 + gap < c0
        if not (rows_sep or cols_sep):
            return False
    return True


def _expected_output(grid: np.ndarray, placements: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for size, row0, col0, _offset, _trace_col0 in placements:
        out[row0 : row0 + size, col0 : col0 + size] = SQUARE_COLOR
    out[grid == TRACE_COLOR] = TRACE_COLOR
    return out


def _candidate_placements(
    height: int,
    width: int,
    size: int,
    shadow_intervals: list[tuple[int, int]],
    trace_intervals: list[tuple[int, int]],
    square_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for row0 in range(0, height - size):
        row1 = row0 + size - 1
        offset = height - 1 - row1
        if offset < 1:
            continue
        if not _interval_is_clear(shadow_intervals, row0, row1):
            continue
        max_trace_col0 = width - size - offset
        for trace_col0 in range(1, max_trace_col0 + 1):
            trace_col1 = trace_col0 + size - 1
            if not _interval_is_clear(trace_intervals, trace_col0, trace_col1):
                continue
            col0 = trace_col0 + offset
            col1 = col0 + size - 1
            box = (row0, col0, row1, col1)
            if _box_is_clear(square_boxes, box):
                candidates.append((size, row0, col0, offset, trace_col0))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each hidden square from a left-side height shadow and a diagonally shifted bottom trace.

    Input structure
    - A black background contains only cyan trace pixels.
    - The bottom row has several separated horizontal cyan segments.
    - The leftmost column has the same number of vertical cyan segments, and each square side length appears exactly once.

    Rule
    - Pair every bottom trace with the left shadow segment of the same length.
    - The left shadow gives the square's row span; its distance from the bottom row is the diagonal slide offset.
    - Shift the matched bottom trace right by that offset, fill the recovered square with yellow, and keep the cyan traces unchanged.

    Key insight
    - The bottom trace is not directly below the hidden square: it has slid down-left by exactly the square's height above the floor, so the left shadow must be solved first to invert the offset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)

    bg = _dominant_color(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert len(colors) == 1
    trace_color = colors[0]

    bottom_segments = _segments_1d(g[height - 1, :], trace_color)
    left_segments = _segments_1d(g[:, 0], trace_color)
    assert bottom_segments
    assert len(bottom_segments) == len(left_segments)

    bottom_by_size: dict[int, tuple[int, int]] = {}
    for c0, c1 in bottom_segments:
        size = c1 - c0 + 1
        assert size not in bottom_by_size
        bottom_by_size[size] = (c0, c1)

    left_by_size: dict[int, tuple[int, int]] = {}
    for r0, r1 in left_segments:
        size = r1 - r0 + 1
        assert size not in left_by_size
        left_by_size[size] = (r0, r1)

    assert set(bottom_by_size) == set(left_by_size)

    out = g.copy()
    for size in sorted(bottom_by_size):
        trace_c0, _trace_c1 = bottom_by_size[size]
        row0, row1 = left_by_size[size]
        offset = height - 1 - row1
        col0 = trace_c0 + offset
        col1 = col0 + size - 1
        assert 0 <= row0 <= row1 < height
        assert 0 <= col0 <= col1 < width
        out[row0 : row1 + 1, col0 : col1 + 1] = SQUARE_COLOR

    out[g == trace_color] = trace_color
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    max_side = min(SIDE_RANGE[1], height - 2, width - 3)
    if max_side < SIDE_RANGE[0]:
        return None

    size_choices = list(range(SIDE_RANGE[0], max_side + 1))
    max_objects = min(OBJECT_COUNT_RANGE[1], len(size_choices))
    if max_objects < OBJECT_COUNT_RANGE[0]:
        return None

    if min(height, width) < 18:
        max_objects = min(max_objects, 3)
    object_count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    sizes = sorted(random.sample(size_choices, object_count), reverse=True)

    placements: list[tuple[int, int, int, int, int]] = []
    shadow_intervals: list[tuple[int, int]] = []
    trace_intervals: list[tuple[int, int]] = []
    square_boxes: list[tuple[int, int, int, int]] = []

    for size in sizes:
        candidates = _candidate_placements(height, width, size, shadow_intervals, trace_intervals, square_boxes)
        if not candidates:
            return None
        placement = random.choice(candidates)
        size, row0, col0, offset, trace_col0 = placement
        placements.append(placement)
        shadow_intervals.append((row0, row0 + size - 1))
        trace_intervals.append((trace_col0, trace_col0 + size - 1))
        square_boxes.append((row0, col0, row0 + size - 1, col0 + size - 1))

    if max(offset for _size, _row0, _col0, offset, _trace_col0 in placements) < 2:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for size, row0, _col0, _offset, trace_col0 in placements:
        grid[row0 : row0 + size, 0] = TRACE_COLOR
        grid[height - 1, trace_col0 : trace_col0 + size] = TRACE_COLOR

    if int(np.sum(grid == BG_COLOR)) <= int(np.sum(grid == TRACE_COLOR)):
        return None

    expected = _expected_output(grid, placements)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-trace reconstruction input")
