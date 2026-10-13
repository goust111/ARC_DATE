from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
AZURE = 8

NWSE = "nwse"
NESW = "nesw"

GRID_MIN = 11
GRID_MAX = 19
SEGMENT_MIN_LEN = 2
SEGMENT_MAX_LEN = 5
MAJOR_SEGMENT_COUNT_RANGE = (2, 4)
MINOR_SEGMENT_COUNT_RANGE = (1, 3)
PLACE_TRIES = 120
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _orientation_steps(orientation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if orientation == NWSE:
        return (-1, -1), (1, 1)
    if orientation == NESW:
        return (-1, 1), (1, -1)
    raise ValueError(f"unknown orientation: {orientation}")


def _other_orientation(orientation: str) -> str:
    return NESW if orientation == NWSE else NWSE


def _line_id(cell: tuple[int, int], orientation: str) -> int:
    r, c = cell
    return r - c if orientation == NWSE else r + c


def _line_cells(shape: tuple[int, int], orientation: str, line_id: int) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    if orientation == NWSE:
        for r in range(h):
            c = r - line_id
            if 0 <= c < w:
                cells.append((r, c))
    else:
        for r in range(h):
            c = line_id - r
            if 0 <= c < w:
                cells.append((r, c))
    return cells


def _all_line_ids(shape: tuple[int, int], orientation: str, min_length: int) -> list[int]:
    h, w = shape
    ids: list[int] = []
    if orientation == NWSE:
        candidates = range(-(w - 1), h)
    else:
        candidates = range(h + w - 1)
    for line_id in candidates:
        if len(_line_cells(shape, orientation, line_id)) >= min_length:
            ids.append(line_id)
    return ids


def _line_intersection(shape: tuple[int, int], nwse_id: int, nesw_id: int) -> tuple[int, int] | None:
    if (nwse_id + nesw_id) % 2 != 0:
        return None
    r = (nwse_id + nesw_id) // 2
    c = (nesw_id - nwse_id) // 2
    if _in_bounds(shape, r, c):
        return (r, c)
    return None


def _segment_intervals(
    line_length: int,
    min_len: int,
    max_len: int,
    *,
    must_contain: int | None = None,
    avoid: set[int] | None = None,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    upper = min(max_len, line_length - 1)
    for length in range(min_len, upper + 1):
        for start in range(0, line_length - length + 1):
            stop = start + length - 1
            if must_contain is not None and not (start <= must_contain <= stop):
                continue
            if avoid is not None and any(start <= idx <= stop for idx in avoid):
                continue
            intervals.append((start, length))
    return intervals


def _neighbor8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _can_place_segment(cells: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    if len(cells) < SEGMENT_MIN_LEN:
        return False
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    for cell in cells:
        if cell in occupied:
            return False
        if any(neighbor in occupied for neighbor in _neighbor8(cell)):
            return False
    return True


def _cell_orientation(mask: np.ndarray, r: int, c: int) -> str:
    nwse = False
    nesw = False
    for dr, dc in _orientation_steps(NWSE):
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            nwse = True
    for dr, dc in _orientation_steps(NESW):
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            nesw = True
    assert nwse != nesw, "each foreground pixel must belong to exactly one diagonal orientation"
    return NWSE if nwse else NESW


def _extract_segments(grid: np.ndarray) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    mask = np.asarray(grid == AZURE)
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    if not coords:
        return []

    orientations = {
        cell: _cell_orientation(mask, cell[0], cell[1])
        for cell in coords
    }
    visited: set[tuple[int, int]] = set()
    segments: list[tuple[str, tuple[tuple[int, int], ...]]] = []

    for cell in coords:
        if cell in visited:
            continue
        orientation = orientations[cell]
        stack = [cell]
        component: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            assert orientations[cur] == orientation
            visited.add(cur)
            component.append(cur)
            for dr, dc in _orientation_steps(orientation):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in orientations and nxt not in visited:
                    assert orientations[nxt] == orientation
                    stack.append(nxt)

        assert len(component) >= SEGMENT_MIN_LEN, "segments must have length at least 2"
        if orientation == NWSE:
            component.sort(key=lambda p: (p[0], p[1]))
            step = (1, 1)
        else:
            component.sort(key=lambda p: (p[0], -p[1]))
            step = (1, -1)
        for prev, cur in zip(component, component[1:]):
            assert (cur[0] - prev[0], cur[1] - prev[1]) == step
        segments.append((orientation, tuple(component)))
    return segments


def _full_line_from_segment(
    shape: tuple[int, int],
    orientation: str,
    segment_cells: tuple[tuple[int, int], ...],
) -> set[tuple[int, int]]:
    line_id = _line_id(segment_cells[0], orientation)
    return set(_line_cells(shape, orientation, line_id))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the globally dominant diagonal orientation to the grid edge, then mark
    crossings with the other orientation in red.
    """
    g = np.asarray(grid, dtype=int)
    try:
        segments = _extract_segments(g)
    except AssertionError:
        return g.copy()
    if not segments:
        return g.copy()

    nwse_count = sum(len(cells) for orientation, cells in segments if orientation == NWSE)
    nesw_count = sum(len(cells) for orientation, cells in segments if orientation == NESW)
    if nwse_count == 0 or nesw_count == 0 or nwse_count == nesw_count:
        return g.copy()

    major = NWSE if nwse_count > nesw_count else NESW
    major_cells: set[tuple[int, int]] = set()
    minor_cells: set[tuple[int, int]] = set()
    for orientation, cells in segments:
        if orientation == major:
            major_cells |= _full_line_from_segment(g.shape, orientation, cells)
        else:
            minor_cells |= set(cells)

    out = np.full_like(g, BLACK)
    for r, c in major_cells:
        out[r, c] = AZURE
    for r, c in minor_cells:
        out[r, c] = AZURE
    for r, c in major_cells & minor_cells:
        out[r, c] = RED
    return out


def _segment_spec(
    orientation: str,
    line_id: int,
    cells: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    return {"orientation": orientation, "line_id": line_id, "cells": cells}


def _build_grid(shape: tuple[int, int], segments: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for segment in segments:
        for r, c in segment["cells"]:
            grid[r, c] = AZURE
    return grid


def _place_anchor_pair(
    shape: tuple[int, int],
    major_orientation: str,
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    minor_orientation = _other_orientation(major_orientation)
    major_ids = _all_line_ids(shape, major_orientation, min_length=4)
    minor_ids = _all_line_ids(shape, minor_orientation, min_length=3)
    if not major_ids or not minor_ids:
        return None

    pairs: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for major_id in major_ids:
        major_line = _line_cells(shape, major_orientation, major_id)
        for minor_id in minor_ids:
            if major_orientation == NWSE:
                cross = _line_intersection(shape, major_id, minor_id)
            else:
                cross = _line_intersection(shape, minor_id, major_id)
            if cross is None:
                continue

            minor_line = _line_cells(shape, minor_orientation, minor_id)
            major_index = major_line.index(cross)
            minor_index = minor_line.index(cross)
            major_intervals = _segment_intervals(
                len(major_line),
                SEGMENT_MIN_LEN,
                SEGMENT_MAX_LEN,
                avoid={major_index - 1, major_index, major_index + 1},
            )
            minor_intervals = _segment_intervals(
                len(minor_line),
                SEGMENT_MIN_LEN,
                SEGMENT_MAX_LEN,
                must_contain=minor_index,
            )
            if major_intervals and minor_intervals:
                pairs.append((major_id, minor_id, major_intervals, minor_intervals))

    if not pairs:
        return None

    for pair_index in rng.permutation(len(pairs)):
        major_id, minor_id, major_intervals, minor_intervals = pairs[int(pair_index)]
        major_line = _line_cells(shape, major_orientation, major_id)
        minor_line = _line_cells(shape, minor_orientation, minor_id)

        for _ in range(12):
            major_start, major_len = major_intervals[int(rng.integers(len(major_intervals)))]
            major_cells = tuple(major_line[major_start : major_start + major_len])
            if not _can_place_segment(major_cells, set()):
                continue
            occupied = set(major_cells)
            minor_start, minor_len = minor_intervals[int(rng.integers(len(minor_intervals)))]
            minor_cells = tuple(minor_line[minor_start : minor_start + minor_len])
            if not _can_place_segment(minor_cells, occupied):
                continue
            return [
                _segment_spec(major_orientation, major_id, major_cells),
                _segment_spec(minor_orientation, minor_id, minor_cells),
            ]
    return None


def _place_random_segment(
    shape: tuple[int, int],
    orientation: str,
    used_lines: set[int],
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[str, object] | None:
    available = [line_id for line_id in _all_line_ids(shape, orientation, min_length=3) if line_id not in used_lines]
    if not available:
        return None

    for _ in range(PLACE_TRIES):
        line_id = int(available[int(rng.integers(len(available)))])
        line = _line_cells(shape, orientation, line_id)
        intervals = _segment_intervals(len(line), SEGMENT_MIN_LEN, SEGMENT_MAX_LEN)
        if not intervals:
            continue
        start, length = intervals[int(rng.integers(len(intervals)))]
        cells = tuple(line[start : start + length])
        if _can_place_segment(cells, occupied):
            return _segment_spec(orientation, line_id, cells)
    return None


def _scene_is_valid(grid: np.ndarray, major_orientation: str) -> bool:
    segments = _extract_segments(grid)
    nwse_count = sum(len(cells) for orientation, cells in segments if orientation == NWSE)
    nesw_count = sum(len(cells) for orientation, cells in segments if orientation == NESW)
    if nwse_count == 0 or nesw_count == 0 or nwse_count == nesw_count:
        return False
    inferred_major = NWSE if nwse_count > nesw_count else NESW
    if inferred_major != major_orientation:
        return False

    out = main(grid)
    if not np.any(out == RED):
        return False
    if not np.any((grid == BLACK) & (out == AZURE)):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(GRID_MIN, GRID_MAX + 1)),
        int(rng.integers(GRID_MIN, GRID_MAX + 1)),
    )
    major_orientation = NWSE if int(rng.integers(2)) == 0 else NESW
    minor_orientation = _other_orientation(major_orientation)

    anchor = _place_anchor_pair(shape, major_orientation, rng)
    if anchor is None:
        return None

    segments = list(anchor)
    occupied = {
        cell
        for segment in segments
        for cell in segment["cells"]
    }
    used_lines = {
        NWSE: {int(segment["line_id"]) for segment in segments if segment["orientation"] == NWSE},
        NESW: {int(segment["line_id"]) for segment in segments if segment["orientation"] == NESW},
    }

    total_major = int(rng.integers(MAJOR_SEGMENT_COUNT_RANGE[0], MAJOR_SEGMENT_COUNT_RANGE[1] + 1))
    total_minor = int(rng.integers(MINOR_SEGMENT_COUNT_RANGE[0], MINOR_SEGMENT_COUNT_RANGE[1] + 1))

    while len([s for s in segments if s["orientation"] == major_orientation]) < total_major:
        segment = _place_random_segment(shape, major_orientation, used_lines[major_orientation], occupied, rng)
        if segment is None:
            return None
        segments.append(segment)
        used_lines[major_orientation].add(int(segment["line_id"]))
        occupied.update(segment["cells"])

    while len([s for s in segments if s["orientation"] == minor_orientation]) < total_minor:
        segment = _place_random_segment(shape, minor_orientation, used_lines[minor_orientation], occupied, rng)
        if segment is None:
            return None
        segments.append(segment)
        used_lines[minor_orientation].add(int(segment["line_id"]))
        occupied.update(segment["cells"])

    grid = _build_grid(shape, segments)
    if not _scene_is_valid(grid, major_orientation):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
