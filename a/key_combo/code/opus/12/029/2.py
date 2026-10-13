from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

MIN_SIZE = 12
MAX_SIZE = 20
MIN_SEGMENTS = 2
MAX_SEGMENTS = 4
MIN_LENGTH = 4
MAX_LENGTH = 8
MIN_COL_GAP = 2
MAX_MARKER_CLEARANCE = 4
MAX_ATTEMPTS = 400


def _split_total(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _extract_runs(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    blue = grid == BLUE
    height, width = grid.shape

    starts: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(blue):
        r = int(rr)
        c = int(cc)
        assert not (
            (r > 0 and blue[r - 1, c])
            or (r + 1 < height and blue[r + 1, c])
            or (c > 0 and blue[r, c - 1])
            or (c + 1 < width and blue[r, c + 1])
        ), "blue pixels must form one-cell-wide down-right runs"
        prev_diag = r > 0 and c > 0 and blue[r - 1, c - 1]
        next_diag = r + 1 < height and c + 1 < width and blue[r + 1, c + 1]
        assert not (
            (r > 0 and c + 1 < width and blue[r - 1, c + 1])
            or (r + 1 < height and c > 0 and blue[r + 1, c - 1])
        ), "blue pixels must not branch across opposite diagonals"
        if not prev_diag:
            starts.append((r, c))
        assert prev_diag or next_diag, "every blue pixel must belong to a diagonal run"

    assert starts, "expected at least one blue diagonal run"

    seen = np.zeros_like(blue, dtype=bool)
    runs: list[list[tuple[int, int]]] = []
    for start_r, start_c in sorted(starts):
        run: list[tuple[int, int]] = []
        r = start_r
        c = start_c
        while r < height and c < width and blue[r, c]:
            assert not seen[r, c], "diagonal runs must be disjoint"
            seen[r, c] = True
            run.append((r, c))
            r += 1
            c += 1
        runs.append(run)

    assert np.array_equal(seen, blue), "all blue pixels must belong to a run"
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each red marker onto its blue down-right diagonal and use that landing index to phase-shift the alternating recolor.

    Input structure:
    - A black background contains several one-cell-wide blue down-right diagonal segments.
    - Each red marker sits somewhere above one blue pixel in the same column, with only black cells between them.
    - Blue segments do not overlap, and each marked column identifies exactly one blue landing pixel.

    Rule:
    - Remove all red markers from the output.
    - For each red marker, follow its column downward until it lands on the unique blue pixel in that column.
    - If that landing pixel is index k within its diagonal run, keep pixels before and at k blue, then recolor indices k+1, k+3, ... to yellow until the run ends.

    Key insight:
    - The marker's column encodes a specific index on one diagonal run, so gravity converts a vertical cue into the alternating pattern's starting phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    runs = _extract_runs(g)
    column_map: dict[int, tuple[int, int, int]] = {}
    for run_idx, run in enumerate(runs):
        for index, (r, c) in enumerate(run):
            assert c not in column_map, "each blue column must belong to exactly one landing pixel"
            column_map[c] = (run_idx, index, r)

    out = g.copy()
    out[out == RED] = BLACK

    marker_rows = np.argwhere(g == RED)
    run_to_index: dict[int, int] = {}
    for rr, cc in marker_rows:
        r = int(rr)
        c = int(cc)
        assert c in column_map, "every marker column must hit a blue pixel"
        run_idx, index, target_r = column_map[c]
        assert target_r >= r + 2, "marker must have at least one black cell before the landing pixel"
        assert np.all(g[r + 1 : target_r, c] == BLACK), "marker path to the diagonal must be unobstructed"
        assert run_idx not in run_to_index, "each run must receive at most one marker"
        run_to_index[run_idx] = index

    for run_idx, index in run_to_index.items():
        run = runs[run_idx]
        for pos in range(index + 1, len(run), 2):
            r, c = run[pos]
            out[r, c] = YELLOW

    return out


def _choose_behaviors(count: int) -> list[str]:
    assert MIN_SEGMENTS <= count <= MAX_SEGMENTS
    if count == 2:
        behaviors = ["start", random.choice(["middle", "end"])]
    elif count == 3:
        behaviors = ["start", "middle", "end"]
    else:
        behaviors = ["start", "middle", "end", random.choice(["start", "middle", "end"])]
    random.shuffle(behaviors)
    return behaviors


def _choose_marker_index(length: int, behavior: str) -> int:
    if behavior == "start":
        return random.choice([0, 1])
    if behavior == "middle":
        return random.randint(1, length - 2)
    if behavior == "end":
        return length - 1
    raise ValueError(f"unknown behavior: {behavior}")


def _expected_from_design(shape: tuple[int, int], segments: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for segment in segments:
        cells = segment["cells"]
        marker_index = int(segment["marker_index"])
        for idx, (r, c) in enumerate(cells):
            out[r, c] = YELLOW if idx > marker_index and (idx - marker_index) % 2 == 1 else BLUE
    return out


def _design_is_valid(grid: np.ndarray, segments: list[dict[str, object]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.count_nonzero(grid == RED)) != len(segments):
        return False

    try:
        runs = _extract_runs(grid)
    except Exception:
        return False
    if len(runs) != len(segments):
        return False

    all_blue_cols: set[int] = set()
    changed_segments = 0
    non_start_segments = 0
    end_segments = 0

    for segment in segments:
        cells = segment["cells"]
        marker = segment["marker"]
        marker_index = int(segment["marker_index"])
        if not (MIN_LENGTH <= len(cells) <= MAX_LENGTH):
            return False

        for idx, (r, c) in enumerate(cells):
            if grid[r, c] != BLUE:
                return False
            if c in all_blue_cols:
                return False
            all_blue_cols.add(c)
            if idx > 0:
                pr, pc = cells[idx - 1]
                if (r, c) != (pr + 1, pc + 1):
                    return False

        mr, mc = marker
        if grid[mr, mc] != RED:
            return False
        target_r, target_c = cells[marker_index]
        if mc != target_c or mr >= target_r - 1:
            return False
        if not np.all(grid[mr + 1 : target_r, mc] == BLACK):
            return False

        if marker_index < len(cells) - 1:
            changed_segments += 1
        if marker_index > 0:
            non_start_segments += 1
        if marker_index == len(cells) - 1:
            end_segments += 1

    if changed_segments == 0 or non_start_segments == 0:
        return False
    if len(segments) >= 3 and end_segments == 0:
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    expected = _expected_from_design(grid.shape, segments)
    return bool(np.array_equal(solved, expected))


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)
    count = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    behaviors = _choose_behaviors(count)

    lengths = [random.randint(MIN_LENGTH, MAX_LENGTH) for _ in range(count)]
    min_needed = sum(lengths) + MIN_COL_GAP * (count - 1)
    if min_needed > width:
        return None

    extra = width - min_needed
    outer_gaps = _split_total(extra, count + 1)
    starts: list[int] = []
    cursor = outer_gaps[0]
    for idx, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if idx < count - 1:
            cursor += MIN_COL_GAP + outer_gaps[idx + 1]

    grid = np.full((height, width), BLACK, dtype=int)
    segments: list[dict[str, object]] = []

    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        length = lengths[idx]
        col0 = starts[idx]
        max_row0 = height - length
        if max_row0 < 2:
            return None
        row0 = random.randint(2, max_row0)
        cells = [(row0 + step, col0 + step) for step in range(length)]
        marker_index = _choose_marker_index(length, behaviors[idx])
        target_r, target_c = cells[marker_index]
        max_clearance = min(MAX_MARKER_CLEARANCE, target_r - 1)
        if max_clearance < 1:
            return None
        clearance = random.randint(1, max_clearance)
        marker = (target_r - clearance - 1, target_c)

        for r, c in cells:
            grid[r, c] = BLUE
        grid[marker] = RED
        segments.append(
            {
                "cells": cells,
                "marker_index": marker_index,
                "marker": marker,
            }
        )

    if not _design_is_valid(grid, segments):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-phase diagonal input")
