from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))
BAR_COUNT_RANGE = (3, 6)
SEGMENT_LENGTH_POOL = tuple(range(1, 9))
OUTPUT_HEADROOM_RANGE = (0, 3)
MAX_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _contiguous_segments(row: np.ndarray, bg: int) -> list[dict[str, int]]:
    assert row.ndim == 1
    assert np.all(row != bg), "palette row must be fully non-background"

    segments: list[dict[str, int]] = []
    start = 0
    current = int(row[0])
    for c in range(1, len(row) + 1):
        if c == len(row) or int(row[c]) != current:
            segments.append(
                {
                    "color": current,
                    "start": start,
                    "end": c - 1,
                    "length": c - start,
                }
            )
            if c < len(row):
                start = c
                current = int(row[c])
    lengths = [seg["length"] for seg in segments]
    assert len(segments) >= 2, "palette must contain multiple segments"
    assert len(lengths) == len(set(lengths)), "palette segment lengths must be distinct"
    return segments


def _body_bar_color(body: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, body[body != bg].tolist())))
    assert len(colors) == 1, "body must contain exactly one bar color"
    return int(colors[0])


def _vertical_bars(body: np.ndarray, bg: int) -> list[dict[str, int]]:
    bar_color = _body_bar_color(body, bg)
    h, w = body.shape
    cols = np.flatnonzero(np.any(body != bg, axis=0))
    assert cols.size > 0, "expected at least one vertical bar"

    bars: list[dict[str, int]] = []
    occupied = np.zeros_like(body, dtype=bool)
    for c in map(int, cols):
        rows = np.flatnonzero(body[:, c] != bg)
        assert rows.size > 0
        r0, r1 = int(rows[0]), int(rows[-1])
        assert r1 == h - 1, "bars must be bottom-aligned"
        assert np.array_equal(rows, np.arange(r0, h)), "bars must be solid vertical runs"
        assert np.all(body[r0:, c] == bar_color), "all bars must share one input color"
        occupied[r0:, c] = True
        bars.append({"col": c, "top": r0, "height": h - r0})

    assert np.array_equal(body != bg, occupied), "body may contain only the vertical bars"
    heights = [bar["height"] for bar in bars]
    assert len(heights) == len(set(heights)), "bar heights must be distinct"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Pair palette segments to vertical bars by rank, then rewrite each bar's color and height.

    Input structure:
    - The first row is a full non-background palette row made of consecutive color segments.
    - Every lower foreground cell belongs to a one-cell-wide vertical bar on a black background.
    - The palette segment lengths are all distinct, the bar heights are all distinct, and every bar touches the last row.

    Rule:
    - Rank the palette segments from shortest to longest.
    - Rank the vertical bars from shortest to tallest.
    - Repaint the k-th shortest bar with the k-th shortest segment's color and change that bar's height to the segment's length, keeping the bar's column and bottom cell fixed.
    - Erase the entire palette row to black.

    Key insight:
    - The matching key is not left-to-right order or column alignment; it is the independent length ranking of the two families.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3

    bg = _most_common_color(g)
    assert bg == BG, "this puzzle uses black as background"

    palette = _contiguous_segments(g[0], bg)
    body = g[1:, :]
    bars = _vertical_bars(body, bg)
    assert len(palette) == len(bars), "segment count must match bar count"

    ranked_segments = sorted(palette, key=lambda seg: seg["length"])
    ranked_bars = sorted(bars, key=lambda bar: bar["height"])

    out = np.full((h, w), bg, dtype=int)
    for segment, bar in zip(ranked_segments, ranked_bars):
        new_height = int(segment["length"])
        assert 1 <= new_height <= h - 1, "segment length must fit below the palette row"
        col = int(bar["col"])
        out[h - new_height :, col] = int(segment["color"])
    return out


def _ranks(values: Iterable[int]) -> list[int]:
    vals = list(map(int, values))
    order = {value: rank for rank, value in enumerate(sorted(vals))}
    return [order[value] for value in vals]


def _sample_distinct_lengths(count: int, rng: random.Random) -> list[int]:
    for _ in range(200):
        lengths = sorted(rng.sample(SEGMENT_LENGTH_POOL, count))
        total = sum(lengths)
        if total >= max(10, 2 * count - 1) and total <= 24:
            return lengths
    raise RuntimeError("failed to sample segment lengths")


def _sample_spaced_columns(width: int, count: int, rng: random.Random) -> list[int]:
    assert width >= 2 * count - 1
    bases = sorted(rng.sample(range(width - count + 1), count))
    return [base + idx for idx, base in enumerate(bases)]


def _segment_index_at(segments: list[dict[str, int]], col: int) -> int:
    for idx, segment in enumerate(segments):
        if int(segment["start"]) <= col <= int(segment["end"]):
            return idx
    raise AssertionError(f"column {col} not covered by any palette segment")


def _expected_output(shape: tuple[int, int], segments: list[dict[str, int]], bars: list[dict[str, int]]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    ranked_segments = sorted(segments, key=lambda seg: seg["length"])
    ranked_bars = sorted(bars, key=lambda bar: bar["height"])
    for segment, bar in zip(ranked_segments, ranked_bars):
        out[h - int(segment["length"]) :, int(bar["col"])] = int(segment["color"])
    return out


def _semantic_checks(inp: np.ndarray) -> bool:
    h, w = map(int, inp.shape)
    if not (3 <= h <= 30 and 3 <= w <= 30):
        return False
    if np.any(inp[0] == BG):
        return False

    try:
        segments = _contiguous_segments(inp[0], BG)
        bars = _vertical_bars(inp[1:, :], BG)
    except AssertionError:
        return False

    if len(segments) != len(bars):
        return False

    segment_lengths = [int(seg["length"]) for seg in segments]
    bar_heights = [int(bar["height"]) for bar in bars]
    if sorted(segment_lengths) == sorted(bar_heights):
        return False

    seg_pos_ranks = _ranks(segment_lengths)
    bar_col_ranks = _ranks([bar["height"] for bar in sorted(bars, key=lambda bar: bar["col"])])
    if seg_pos_ranks == list(range(len(segments))):
        return False
    if bar_col_ranks == list(range(len(bars))):
        return False
    if seg_pos_ranks == bar_col_ranks:
        return False

    segment_rank_by_index = _ranks(segment_lengths)
    aligned_matches = 0
    for bar in bars:
        seg_idx = _segment_index_at(segments, int(bar["col"]))
        if segment_rank_by_index[seg_idx] == _ranks(bar_heights)[bars.index(bar)]:
            aligned_matches += 1
    if aligned_matches >= len(bars) - 1:
        return False

    ranked_segments = sorted(segments, key=lambda seg: seg["length"])
    ranked_bars = sorted(bars, key=lambda bar: bar["height"])
    changed_heights = sum(
        int(seg["length"]) != int(bar["height"]) for seg, bar in zip(ranked_segments, ranked_bars)
    )
    if changed_heights < 2:
        return False

    expected = _expected_output(inp.shape, segments, bars)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual[0] != BG):
        return False
    if np.count_nonzero(actual[1:, :]) != sum(segment_lengths):
        return False

    return True


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    bar_count = rng.randint(*BAR_COUNT_RANGE)
    segment_lengths = _sample_distinct_lengths(bar_count, rng)
    width = sum(segment_lengths)
    if width > 30 or width < 2 * bar_count - 1:
        return None

    segment_colors = rng.sample(list(NON_BG_COLORS), bar_count + 1)
    bar_color = segment_colors[-1]
    palette_order = rng.sample(range(bar_count), bar_count)
    segments: list[dict[str, int]] = []
    row = np.empty(width, dtype=int)
    cursor = 0
    for idx in palette_order:
        length = int(segment_lengths[idx])
        color = int(segment_colors[idx])
        row[cursor : cursor + length] = color
        segments.append(
            {"color": color, "start": cursor, "end": cursor + length - 1, "length": length}
        )
        cursor += length

    max_height = max(segment_lengths)
    height_limit = min(12, 29 - max_height)
    if height_limit < bar_count:
        return None
    bar_heights = rng.sample(range(1, height_limit + 1), bar_count)
    grid_height = max(max(segment_lengths), max(bar_heights)) + 1 + rng.randint(*OUTPUT_HEADROOM_RANGE)
    grid_height = min(grid_height, 30)
    if max(segment_lengths) > grid_height - 1 or max(bar_heights) > grid_height - 1:
        return None

    cols = _sample_spaced_columns(width, bar_count, rng)
    inp = np.zeros((grid_height, width), dtype=int)
    inp[0, :] = row
    for col, height in zip(cols, bar_heights):
        inp[grid_height - height :, col] = bar_color

    return inp if _semantic_checks(inp) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-to-bar ranking instance")
