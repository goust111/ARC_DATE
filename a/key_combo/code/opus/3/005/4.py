from __future__ import annotations

import random
from typing import Any

import numpy as np

BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CODE_COLORS = (RED, GREEN, YELLOW, MAGENTA)
MAX_GENERATION_ATTEMPTS = 400


def _boundary_ring(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    ring: list[tuple[int, int]] = []
    for c in range(w):
        ring.append((0, c))
    for r in range(1, h):
        ring.append((r, w - 1))
    for c in range(w - 2, -1, -1):
        ring.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        ring.append((r, 0))
    return ring


def _single_run(mask: np.ndarray) -> tuple[int, int]:
    starts = [i for i in range(len(mask)) if mask[i] and (i == 0 or not mask[i - 1])]
    assert len(starts) == 1, "expected a unique linear run"
    start = starts[0]
    length = 0
    while start + length < len(mask) and mask[start + length]:
        length += 1
    assert length == int(np.count_nonzero(mask))
    return int(start), int(length)


def _single_circular_run(mask: np.ndarray) -> tuple[int, int]:
    n = len(mask)
    starts = [i for i in range(n) if mask[i] and not mask[i - 1]]
    assert len(starts) == 1, "expected a unique circular run"
    start = starts[0]
    length = 0
    while length < n and mask[(start + length) % n]:
        length += 1
    assert length == int(np.count_nonzero(mask))
    return int(start), int(length)


def _find_divider_row(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for r in range(1, h - 1):
        if w < 3 or not np.all(grid[r, 1:-1] == GREY):
            continue
        if not np.any(grid[r - 1, 1:-1] == ORANGE):
            continue
        if not np.any(np.isin(grid[r + 1, 1:-1], CODE_COLORS)):
            continue
        candidates.append(r)
    assert len(candidates) == 1, "expected one divider row"
    return candidates[0]


def _marker_length(row: np.ndarray) -> int:
    mask = row[1:-1] == ORANGE
    _start, length = _single_run(mask)
    return length


def _palette_for_threshold(row: np.ndarray, threshold: int) -> list[int]:
    hits = [
        color
        for color in CODE_COLORS
        if int(np.count_nonzero(row == color)) == threshold and threshold > 0
    ]
    hits.sort(key=lambda color: int(np.flatnonzero(row == color)[0]))
    return hits


def _extract_parameters(grid: np.ndarray) -> dict[str, Any]:
    ring = _boundary_ring(grid.shape)
    ring_vals = np.array([grid[r, c] for r, c in ring], dtype=int)
    start, seglen = _single_circular_run(ring_vals == BLUE)
    divider = _find_divider_row(grid)
    n = _marker_length(grid[divider - 1])
    palette = _palette_for_threshold(grid[divider + 1], n)
    assert len(palette) >= 2, "expected at least two palette colors"
    return {
        "divider": divider,
        "marker_len": n,
        "palette": palette,
        "start": start,
        "segment_len": seglen,
        "ring": ring,
    }


def _tile_starts(ring_len: int, start: int, segment_len: int, gap: int) -> list[int]:
    return [start + offset for offset in range(0, ring_len - segment_len + 1, gap)]


def _paint_ring_segment(
    out: np.ndarray, ring: list[tuple[int, int]], start_unwrapped: int, length: int, color: int
) -> None:
    ring_len = len(ring)
    for step in range(length):
        r, c = ring[(start_unwrapped + step) % ring_len]
        out[r, c] = color


def _render_from_parameters(
    grid: np.ndarray, start: int, segment_len: int, gap: int, palette: list[int]
) -> np.ndarray:
    ring = _boundary_ring(grid.shape)
    starts = _tile_starts(len(ring), start, segment_len, gap)
    out = grid.copy()
    for i, start_unwrapped in enumerate(starts):
        color = BLUE if i == 0 else palette[(i - 1) % len(palette)]
        _paint_ring_segment(out, ring, start_unwrapped, segment_len, color)
    _paint_ring_segment(out, ring, start, segment_len, BLUE)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the orange marker, then use that number to space boundary copies and choose their colors.

    Input structure:
    - The background is black.
    - The outer boundary contains one continuous blue run on the perimeter.
    - Inside the grid, one grey horizontal divider spans the full interior width; the row above it contains one orange marker run, and the row below it is an encoding row using colors {2,3,4,6}.

    Rule:
    - Let N be the number of orange cells in the marker run.
    - Traverse the rectangular boundary clockwise from the top-left corner and read the original blue run as a segment on that ring.
    - Starting from the blue run's first boundary position, place copies of the same segment every 2N boundary steps for one full lap, stopping before a copy would exceed that lap.
    - In the encoding row, collect the colors whose frequency is exactly N; sort them by the column of their first appearance, then recolor the copied segments cyclically with that palette while keeping the original segment blue.
    - Leave the divider row, marker row, and encoding row unchanged.

    Key insight:
    - The same marker count N drives two different subsystems at once: the perimeter tiling gap is 2N, and the encoding-row frequency threshold is N.
    """
    g = np.asarray(grid, dtype=int)
    params = _extract_parameters(g)
    gap = 2 * int(params["marker_len"])
    return _render_from_parameters(
        g,
        start=int(params["start"]),
        segment_len=int(params["segment_len"]),
        gap=gap,
        palette=list(params["palette"]),
    )


def _choose_palette_order(colors: list[int]) -> list[int]:
    palette = colors[:]
    while True:
        random.shuffle(palette)
        if palette != sorted(palette):
            return palette


def _sample_code_spec(n: int, segment_len: int) -> tuple[dict[int, int], list[int]]:
    colors = list(CODE_COLORS)
    random.shuffle(colors)
    feasible = [count for count in (2, 3) if count * n + segment_len <= 14]
    matched_count = random.choice(feasible)
    matched = colors[:matched_count]
    remaining = colors[matched_count:]

    counts = {color: 0 for color in CODE_COLORS}
    for color in matched:
        counts[color] = n

    counts[remaining[0]] = segment_len
    base_total = matched_count * n + segment_len
    budget = 14 - base_total
    if len(remaining) > 1 and budget > 0 and random.random() < 0.7:
        options = [
            value
            for value in (1, n - 1, n + 1, n + 2)
            if value > 0 and value not in {n, segment_len} and value <= budget
        ]
        if options:
            counts[remaining[1]] = random.choice(options)

    palette = _choose_palette_order(matched)
    return counts, palette


def _build_code_row(
    inner_width: int, counts: dict[int, int], palette: list[int]
) -> tuple[np.ndarray, list[int]]:
    active = [color for color in CODE_COLORS if counts[color] > 0]
    unmatched = [color for color in active if color not in palette]

    first_order = list(palette)
    for color in unmatched:
        insert_at = random.randint(0, len(first_order))
        first_order.insert(insert_at, color)

    sequence = first_order[:]
    remainder: list[int] = []
    for color in active:
        remainder.extend([color] * (counts[color] - 1))
    random.shuffle(remainder)
    sequence.extend(remainder)

    colored_cols = sorted(random.sample(range(inner_width), len(sequence)))
    row = np.zeros(inner_width, dtype=int)
    for col, color in zip(colored_cols, sequence):
        row[col] = color
    return row, first_order


def _candidate_shapes(min_inner_width: int, n: int, segment_len: int, palette_len: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for h in range(8, 15):
        for inner_w in range(min_inner_width, 17):
            w = inner_w + 2
            ring_len = 2 * (h + w) - 4
            copy_count = 1 + (ring_len - segment_len) // (2 * n)
            if copy_count < 3:
                continue
            if copy_count - 1 <= palette_len:
                continue
            candidates.append((h, w))
    return candidates


def _available_boundary_starts(h: int, w: int, segment_len: int) -> list[int]:
    starts: list[int] = []
    starts.extend(offset for offset in range(w - segment_len + 1))
    starts.extend((w - 1) + offset for offset in range(h - segment_len + 1))
    starts.extend((w + h - 2) + offset for offset in range(w - segment_len + 1))
    left_len = h - 1
    if left_len >= segment_len:
        starts.extend((2 * w + h - 3) + offset for offset in range(left_len - segment_len + 1))
    return starts


def _make_case() -> tuple[np.ndarray, dict[str, Any]] | None:
    n = random.choice([2, 3, 4])
    segment_options = [length for length in (3, 4, 5) if length < 2 * n and length != n]
    segment_len = random.choice(segment_options)
    counts, palette = _sample_code_spec(n, segment_len)
    total_cells = sum(counts.values())
    min_inner_width = total_cells + 2
    shape_candidates = _candidate_shapes(min_inner_width, n, segment_len, len(palette))
    if not shape_candidates:
        return None

    h, w = random.choice(shape_candidates)
    inner_w = random.randint(min_inner_width, min(16, w - 2))
    w = inner_w + 2
    ring_len = 2 * (h + w) - 4
    copy_count = 1 + (ring_len - segment_len) // (2 * n)
    if copy_count < 3 or copy_count - 1 <= len(palette):
        return None

    sep_r = random.randint(2, h - 3)
    marker_start = random.randint(1, w - 1 - n)
    start = random.choice(_available_boundary_starts(h, w, segment_len))

    code_inner, first_order = _build_code_row(inner_w, counts, palette)
    grid = np.zeros((h, w), dtype=int)
    grid[sep_r, 1:-1] = GREY
    grid[sep_r - 1, marker_start : marker_start + n] = ORANGE
    grid[sep_r + 1, 1:-1] = code_inner
    _paint_ring_segment(grid, _boundary_ring(grid.shape), start, segment_len, BLUE)

    meta = {
        "marker_len": n,
        "segment_len": segment_len,
        "counts": counts,
        "palette": palette,
        "start": start,
        "divider": sep_r,
        "copy_count": copy_count,
        "first_order": first_order,
    }
    return grid, meta


def _validate_case(grid: np.ndarray, meta: dict[str, Any]) -> bool:
    expected = _render_from_parameters(
        grid,
        start=int(meta["start"]),
        segment_len=int(meta["segment_len"]),
        gap=2 * int(meta["marker_len"]),
        palette=list(meta["palette"]),
    )
    if np.array_equal(expected, grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    params = _extract_parameters(grid)
    if int(params["marker_len"]) != int(meta["marker_len"]):
        return False
    if int(params["segment_len"]) != int(meta["segment_len"]):
        return False
    if int(params["start"]) != int(meta["start"]):
        return False
    if list(params["palette"]) != list(meta["palette"]):
        return False

    counts = meta["counts"]
    n = int(meta["marker_len"])
    segment_len = int(meta["segment_len"])
    matched = [color for color in CODE_COLORS if counts[color] == n]
    active_unmatched = [color for color in CODE_COLORS if counts[color] not in {0, n}]
    if len(matched) < 2 or not active_unmatched:
        return False
    if counts[active_unmatched[0]] != segment_len:
        return False
    if list(meta["palette"]) == sorted(meta["palette"]):
        return False

    wrong_gap_out = _render_from_parameters(
        grid,
        start=int(meta["start"]),
        segment_len=segment_len,
        gap=segment_len,
        palette=list(meta["palette"]),
    )
    if np.array_equal(wrong_gap_out, expected):
        return False

    wrong_palette = sorted(meta["palette"])
    wrong_palette_out = _render_from_parameters(
        grid,
        start=int(meta["start"]),
        segment_len=segment_len,
        gap=2 * n,
        palette=wrong_palette,
    )
    if np.array_equal(wrong_palette_out, expected):
        return False

    wrong_threshold_palette = _palette_for_threshold(grid[int(meta["divider"]) + 1], segment_len)
    if wrong_threshold_palette == list(meta["palette"]):
        return False
    if wrong_threshold_palette:
        wrong_threshold_out = _render_from_parameters(
            grid,
            start=int(meta["start"]),
            segment_len=segment_len,
            gap=2 * n,
            palette=wrong_threshold_palette,
        )
        if np.array_equal(wrong_threshold_out, expected):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _make_case()
        if built is None:
            continue
        grid, meta = built
        if _validate_case(grid, meta):
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
