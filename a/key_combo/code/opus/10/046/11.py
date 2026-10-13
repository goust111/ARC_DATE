from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEGMENT_COLORS = (1, 2, 3, 4)

MIN_INNER_SIDE = 2
MAX_INNER_SIDE = 6
MAX_GENERATION_ATTEMPTS = 500

SEGMENT_COUNT_OPTIONS = (3, 4)
GAP_SUM_TO_SEG_SUM_OPTIONS = {
    3: [((3, 4, 5), 12)],
    4: [((1, 2, 4, 5), 12), ((1, 3, 4, 5), 13), ((2, 3, 4, 5), 14)],
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected at least one foreground pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _outer_cycle(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    assert len(coords) == 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    return coords


def _locate_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_mask = grid == GREY
    r0, c0, r1, c1 = _bbox(frame_mask)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    expected = _rectangle_border_mask(grid.shape, (r0, c0, r1, c1))
    assert np.array_equal(frame_mask, expected), "gray pixels must form exactly one complete rectangular frame"

    assert r0 >= 1 and c0 >= 1, "frame must have a full exterior ring"
    assert r1 + 1 < grid.shape[0] and c1 + 1 < grid.shape[1], "frame must have a full exterior ring"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0 and np.all(interior == BLACK), "frame interior must start black"
    return r0, c0, r1, c1


def _segments_and_gaps(values: list[int]) -> tuple[list[tuple[int, int, int]], list[int]]:
    n = len(values)
    assert n >= 8, "ring must be large enough to contain multiple segments"
    assert any(v == BLACK for v in values), "segments must be separated by black gaps"

    segments: list[tuple[int, int, int]] = []
    for i, value in enumerate(values):
        if value == BLACK or values[i - 1] != BLACK:
            continue

        color = int(value)
        length = 0
        j = i
        while values[j] != BLACK:
            assert values[j] == color, "each border segment must be monochrome"
            length += 1
            j = (j + 1) % n
            assert length <= n, "ring cannot be fully colored"
        segments.append((i, length, color))

    assert len(segments) >= 3, "expected at least three colored border segments"

    gaps: list[int] = []
    for idx, (start, length, _color) in enumerate(segments):
        next_start = segments[(idx + 1) % len(segments)][0]
        gap_start = (start + length) % n
        gap_len = (next_start - gap_start) % n
        assert gap_len > 0, "adjacent segments must be separated by black pixels"
        for step in range(gap_len):
            assert values[(gap_start + step) % n] == BLACK, "gaps must be black"
        gaps.append(gap_len)

    return segments, gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of the gray frame with the color of the segment that follows the longest exterior black gap.

    Input structure:
    - The grid has a black background, one complete 1-pixel-thick gray rectangular frame, and a black interior inside that frame.
    - On the 1-cell-thick ring immediately outside the frame, several separated colored segments lie on a clockwise perimeter cycle.
    - The colored segments are separated by black gaps, and the rest of the grid is unchanged background.

    Rule:
    - Locate the gray frame and unwrap the exterior ring around it clockwise starting from the outer top-left corner.
    - Identify every monochrome colored segment on that ring and the black gap after each segment.
    - Find the unique longest black gap, then take the color of the first colored segment that appears immediately after that gap clockwise.
    - Fill every black cell inside the gray frame with that decoded color, preserving the frame and the outer ring.

    Key insight:
    - The outer colored segments are only candidates; the actual instruction is encoded by the size of the black interval between segments, so the longest silence selects the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _locate_frame(g)
    ring_box = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
    cycle = _outer_cycle(ring_box)
    values = [int(g[r, c]) for r, c in cycle]

    segments, gaps = _segments_and_gaps(values)
    longest_gap = max(gaps)
    assert gaps.count(longest_gap) == 1, "expected a unique longest gap"
    chosen_idx = (gaps.index(longest_gap) + 1) % len(segments)
    fill_color = segments[chosen_idx][2]

    out = g.copy()
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    interior[interior == BLACK] = fill_color
    return out


def _ordered_compositions(total: int, parts: int, low: int, high: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, prefix: list[int]) -> None:
        if slots == 0:
            if remaining == 0:
                out.append(tuple(prefix))
            return
        min_rest = low * (slots - 1)
        max_rest = high * (slots - 1)
        for value in range(low, high + 1):
            next_remaining = remaining - value
            if min_rest <= next_remaining <= max_rest:
                prefix.append(value)
                rec(next_remaining, slots - 1, prefix)
                prefix.pop()

    rec(total, parts, [])
    return out


def _sample_lengths_and_gaps() -> tuple[tuple[int, ...], tuple[int, ...]]:
    segment_count = random.choice(SEGMENT_COUNT_OPTIONS)
    gap_set, segment_sum = random.choice(GAP_SUM_TO_SEG_SUM_OPTIONS[segment_count])
    gap_lengths = tuple(random.sample(gap_set, k=segment_count))

    chosen_idx = (gap_lengths.index(max(gap_lengths)) + 1) % segment_count
    candidates = _ordered_compositions(segment_sum, segment_count, 2, 4)
    informative = [lengths for lengths in candidates if lengths[chosen_idx] < max(lengths)]
    segment_lengths = random.choice(informative or candidates)
    return segment_lengths, gap_lengths


def _sample_inner_size(perimeter: int) -> tuple[int, int]:
    total = (perimeter - 12) // 2
    candidates = [
        (h, total - h)
        for h in range(MIN_INNER_SIDE, MAX_INNER_SIDE + 1)
        if MIN_INNER_SIDE <= total - h <= MAX_INNER_SIDE
    ]
    assert candidates, "perimeter must map to a feasible interior size"
    return random.choice(candidates)


def _gap_contains_start(perimeter: int, shift: int, segment_lengths: tuple[int, ...], gap_lengths: tuple[int, ...]) -> bool:
    pos = shift
    max_gap = max(gap_lengths)
    for seg_len, gap_len in zip(segment_lengths, gap_lengths):
        pos = (pos + seg_len) % perimeter
        if gap_len == max_gap:
            return any((pos + step) % perimeter == 0 for step in range(gap_len))
        pos = (pos + gap_len) % perimeter
    raise AssertionError("expected a unique longest gap")


def _sample_shift(perimeter: int, segment_lengths: tuple[int, ...], gap_lengths: tuple[int, ...]) -> int:
    wrap_longest = bool(random.getrandbits(1))
    candidates = [
        shift
        for shift in range(perimeter)
        if _gap_contains_start(perimeter, shift, segment_lengths, gap_lengths) == wrap_longest
    ]
    assert candidates, "expected at least one valid phase shift"
    return random.choice(candidates)


def _render_input(
    inner_h: int,
    inner_w: int,
    segment_lengths: tuple[int, ...],
    gap_lengths: tuple[int, ...],
    colors: tuple[int, ...],
    shift: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    outer_h = inner_h + 4
    outer_w = inner_w + 4
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)

    grid = np.zeros((outer_h + top_margin + bottom_margin, outer_w + left_margin + right_margin), dtype=int)
    frame_box = (
        top_margin + 1,
        left_margin + 1,
        top_margin + inner_h + 2,
        left_margin + inner_w + 2,
    )

    frame_mask = _rectangle_border_mask(grid.shape, frame_box)
    grid[frame_mask] = GREY

    ring_box = (frame_box[0] - 1, frame_box[1] - 1, frame_box[2] + 1, frame_box[3] + 1)
    cycle = _outer_cycle(ring_box)
    assert len(cycle) == sum(segment_lengths) + sum(gap_lengths)

    pos = shift
    for color, seg_len, gap_len in zip(colors, segment_lengths, gap_lengths):
        for step in range(seg_len):
            r, c = cycle[(pos + step) % len(cycle)]
            grid[r, c] = color
        pos = (pos + seg_len + gap_len) % len(cycle)

    chosen_idx = (gap_lengths.index(max(gap_lengths)) + 1) % len(gap_lengths)
    fill_color = colors[chosen_idx]
    return grid, frame_box, fill_color


def _expected_output(grid: np.ndarray, frame_box: tuple[int, int, int, int], fill_color: int) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = frame_box
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    interior[interior == BLACK] = fill_color
    return out


def _semantic_checks(
    segment_lengths: tuple[int, ...],
    gap_lengths: tuple[int, ...],
    colors: tuple[int, ...],
) -> bool:
    chosen_idx = (gap_lengths.index(max(gap_lengths)) + 1) % len(gap_lengths)
    before_idx = gap_lengths.index(max(gap_lengths))
    assert colors[chosen_idx] != colors[before_idx], "distinct segment colors are required"

    if len(segment_lengths) == 4:
        assert segment_lengths[chosen_idx] < max(segment_lengths), "generator should include a counterexample to longest-segment heuristics"
    return True


def _gen_attempt() -> np.ndarray | None:
    segment_lengths, gap_lengths = _sample_lengths_and_gaps()
    perimeter = sum(segment_lengths) + sum(gap_lengths)
    if perimeter % 2 != 0:
        return None

    inner_h, inner_w = _sample_inner_size(perimeter)
    colors = tuple(random.sample(SEGMENT_COLORS, k=len(segment_lengths)))
    shift = _sample_shift(perimeter, segment_lengths, gap_lengths)
    grid, frame_box, fill_color = _render_input(inner_h, inner_w, segment_lengths, gap_lengths, colors, shift)

    if not _semantic_checks(segment_lengths, gap_lengths, colors):
        return None

    expected = _expected_output(grid, frame_box, fill_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
