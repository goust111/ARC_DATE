from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
UP_COLOR = 1
DOWN_COLOR = 3
SEGMENT_COLOR = 5
REFERENCE_PALETTE = (2, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flatten()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _segment_interval(height: int, width: int, band_row: int, row: int, length: int) -> tuple[int, int, str]:
    space_above = row - band_row - 1
    space_below = height - row - 1
    assert space_above >= 1 and space_below >= 1
    upper_depth = min(space_above, width - length)
    lower_depth = min(space_below, length - 1)
    branch = "width" if width - length < space_above else "band"
    return row - upper_depth, row + lower_depth, branch


def _parse_input(grid: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BACKGROUND, "background must be black"

    occupied_rows = [int(r) for r in range(g.shape[0]) if np.any(g[r] != bg)]
    assert occupied_rows, "expected a reference row and grey segments"

    full_rows = [r for r in occupied_rows if np.all(g[r] != bg)]
    assert len(full_rows) == 1, "expected exactly one full-width reference row"
    band_row = full_rows[0]
    reference = g[band_row].copy()
    assert SEGMENT_COLOR not in set(map(int, reference)), "reference row must not reuse the grey segment color"
    assert len(set(map(int, reference))) >= 2, "reference row should visibly encode a color sequence"

    segments: list[tuple[int, int]] = []
    for r in occupied_rows:
        if r == band_row:
            continue
        nz = np.flatnonzero(g[r] != bg)
        assert nz.size > 0, "non-reference occupied rows must contain a segment"
        length = int(nz.size)
        assert np.array_equal(nz, np.arange(length)), "each segment must be contiguous and left-aligned"
        assert np.all(g[r, :length] == SEGMENT_COLOR), "segments must be grey"
        assert np.all(g[r, length:] == bg), "segment rows must contain only one grey prefix"
        segments.append((int(r), length))

    assert 2 <= len(segments) <= 3, "expected 2-3 grey line segments"
    segments.sort()
    lengths = [length for _, length in segments]
    assert len(set(lengths)) == len(lengths), "segment lengths must be distinct"
    assert min(lengths) >= 2, "segment lengths must be at least 2"
    assert max(lengths) < g.shape[1], "segments must be shorter than the reference row"
    assert band_row < min(r for r, _ in segments), "reference row must be above all segments"

    colors_elsewhere = _colors_present(g, exclude={bg, SEGMENT_COLOR})
    assert colors_elsewhere, "expected a non-black reference sequence"
    assert set(colors_elsewhere) == set(map(int, reference)), "all non-grey colors should belong to the reference row"

    return band_row, reference, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each grey prefix length from the top reference strip and expand it into an upward/downward wedge.

    Input structure:
    - A black background contains exactly one full-width single-row reference strip near the top.
    - The reference strip is a left-to-right sequence of non-black, non-grey colors.
    - Below it sit 2-3 separate grey horizontal segments, each on its own row, left-aligned at column 0 and with a distinct length.

    Rule:
    - Read the top strip as the global reference sequence.
    - For each grey segment of length L, preserve the segment itself.
    - Above the segment, paint blue rows at distance d with left-aligned width L+d, stopping before the first row where that width would exceed the grid or would hit existing non-black cells.
    - Below the segment, paint green rows at distance d with left-aligned width L-d, stopping when the width drops below 1.

    Key insight:
    - The grey bars are only length markers: each bar stands for a prefix length of the top reference strip, and that recovered length parameter controls both the expanding upper wedge and the shrinking lower wedge.
    """
    g = np.asarray(grid, dtype=int)
    _, _, segments = _parse_input(g)
    height, width = map(int, g.shape)

    out = g.copy()
    for row, length in segments:
        d = 1
        while True:
            target_row = row - d
            fill_width = length + d
            if target_row < 0 or fill_width > width:
                break
            if np.any(out[target_row, :fill_width] != BACKGROUND):
                break
            out[target_row, :fill_width] = UP_COLOR
            d += 1

        d = 1
        while True:
            fill_width = length - d
            if fill_width < 1:
                break
            target_row = row + d
            if target_row >= height:
                break
            assert np.all(out[target_row, :fill_width] == BACKGROUND), "downward wedge would overlap existing content"
            out[target_row, :fill_width] = DOWN_COLOR
            d += 1

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _make_reference_row(width: int, lengths: list[int]) -> np.ndarray:
    max_prefix = max(lengths)
    for _ in range(200):
        run_count = random.randint(4, min(8, width))
        run_lengths = _random_partition(width, run_count)
        colors = [random.choice(REFERENCE_PALETTE)]
        while len(colors) < run_count:
            choices = [c for c in REFERENCE_PALETTE if c != colors[-1]]
            colors.append(random.choice(choices))

        row = np.zeros(width, dtype=int)
        c0 = 0
        for run_len, color in zip(run_lengths, colors):
            row[c0 : c0 + run_len] = color
            c0 += run_len

        prefix = row[:max_prefix]
        if len(set(map(int, prefix))) < 2:
            continue
        if not np.any(prefix[:-1] != prefix[1:]):
            continue
        if len(set(map(int, row))) < 3:
            continue
        return row
    raise RuntimeError("failed to build a varied reference row")


def _choose_lengths(width: int, n_segments: int) -> list[int] | None:
    max_length = min(6, width - 1)
    candidates = list(range(2, max_length + 1))
    if len(candidates) < n_segments:
        return None
    return random.sample(candidates, n_segments)


def _rows_valid(
    height: int,
    width: int,
    band_row: int,
    lengths: list[int],
    goal: str,
) -> list[int] | None:
    lengths = list(lengths)
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], random.random()))
    chosen: dict[int, int] = {}
    intervals: list[tuple[int, int]] = []

    def branch_summary() -> set[str]:
        kinds = set()
        for idx, row in chosen.items():
            _, _, branch = _segment_interval(height, width, band_row, row, lengths[idx])
            kinds.add(branch)
        return kinds

    def compatible(start: int, end: int, row: int) -> bool:
        for other_start, other_end in intervals:
            if not (end < other_start - 1 or other_end < start - 1):
                return False
        for other_row in chosen.values():
            if abs(row - other_row) < 4:
                return False
        return True

    def search(pos: int) -> bool:
        if pos == len(order):
            kinds = branch_summary()
            if goal == "mixed":
                return kinds == {"band", "width"}
            if goal == "band":
                return kinds == {"band"}
            if goal == "width":
                return kinds == {"width"}
            return True

        idx = order[pos]
        length = lengths[idx]
        row_min = band_row + length + 1
        row_max = height - length - 1
        if row_min > row_max:
            return False

        rows = list(range(row_min, row_max + 1))
        random.shuffle(rows)
        for row in rows:
            start, end, _ = _segment_interval(height, width, band_row, row, length)
            if not compatible(start, end, row):
                continue
            chosen[idx] = row
            intervals.append((start, end))
            if search(pos + 1):
                return True
            intervals.pop()
            del chosen[idx]
        return False

    if not search(0):
        return None
    return [chosen[i] for i in range(len(lengths))]


def _expected_output(inp: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    height, width = map(int, inp.shape)
    for row, length in sorted(segments):
        for d in range(1, width + 1):
            target_row = row - d
            fill_width = length + d
            if target_row < 0 or fill_width > width:
                break
            if np.any(out[target_row, :fill_width] != BACKGROUND):
                break
            out[target_row, :fill_width] = UP_COLOR
        for d in range(1, length):
            target_row = row + d
            fill_width = length - d
            if target_row >= height or fill_width < 1:
                break
            if np.any(out[target_row, :fill_width] != BACKGROUND):
                raise AssertionError("generator created overlapping downward wedges")
            out[target_row, :fill_width] = DOWN_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.choice([2, 3])
    height = random.randint(18 if n_segments == 3 else 15, 26)
    width = random.randint(10, 16)
    band_row = random.randint(0, 2)

    lengths = _choose_lengths(width, n_segments)
    if lengths is None:
        return None

    goals = ["mixed", "band", "width", "any"]
    goal_weights = [4, 2, 2, 2]
    goal = random.choices(goals, weights=goal_weights, k=1)[0]
    rows = _rows_valid(height, width, band_row, lengths, goal)
    if rows is None and goal != "any":
        rows = _rows_valid(height, width, band_row, lengths, "any")
    if rows is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    inp[band_row] = _make_reference_row(width, lengths)
    segments = sorted(zip(rows, lengths))
    for row, length in segments:
        inp[row, :length] = SEGMENT_COLOR

    occupied_rows = {band_row, *rows}
    if len(occupied_rows) != n_segments + 1:
        return None

    for row, length in segments:
        space_above = row - band_row - 1
        space_below = height - row - 1
        if space_above < length or space_below < length:
            return None

    try:
        parsed_band_row, reference, parsed_segments = _parse_input(inp)
        assert parsed_band_row == band_row
        assert np.array_equal(reference, inp[band_row])
        assert parsed_segments == segments
        expected = _expected_output(inp, segments)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
