from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import get_bg
import random


def _segments(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    spans: list[tuple[int, int]] = []
    start = end = int(ordered[0])
    for idx in ordered[1:]:
        idx = int(idx)
        if idx == end + 1:
            end = idx
        else:
            spans.append((start, end))
            start = end = idx
    spans.append((start, end))
    return spans


def _barrier_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.ravel()))
    foreground = [color for color in counts if color != int(bg)]
    assert foreground, "expected foreground pixels"
    largest = max(counts[color] for color in foreground)
    candidates = [color for color in foreground if counts[color] == largest]
    assert len(candidates) == 1, "expected a unique most frequent foreground color"
    return int(candidates[0])


def _line_moves(values: np.ndarray, barrier: int, bg: int) -> list[tuple[int, int, int]]:
    vals = [int(v) for v in values]
    barrier_spans = _segments([i for i, v in enumerate(vals) if v == int(barrier)])
    if not barrier_spans:
        return []

    marker_pos = [i for i, v in enumerate(vals) if v not in (int(bg), int(barrier))]
    if len(barrier_spans) != 1:
        assert not marker_pos, "marker line must cross exactly one barrier segment"
        return []

    start, end = barrier_spans[0]
    if start == 0 or end == len(vals) - 1:
        assert not marker_pos, "paired markers need space on both sides of the barrier"
        return []

    left = [i for i in range(start) if vals[i] not in (int(bg), int(barrier))]
    right = [i for i in range(end + 1, len(vals)) if vals[i] not in (int(bg), int(barrier))]
    if not left or not right:
        return []

    assert len(left) == 1 and len(right) == 1, "expected one marker on each side"
    left_src, right_src = left[0], right[0]
    left_dst, right_dst = start - 1, end + 1
    assert vals[left_dst] == int(bg) and vals[right_dst] == int(bg), "destination cells must be background"
    return [
        (left_src, right_dst, vals[left_src]),
        (right_src, left_dst, vals[right_src]),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap paired marker pixels across the largest foreground-color barrier, placing them just outside the barrier segment.

    Input structure:
    - The background is the dominant color.
    - One foreground color is the unique most frequent non-background color; its pixels form the stationary barrier.
    - Other foreground pixels are single markers, sometimes paired on opposite sides of one contiguous barrier segment in the same row or column.

    Rule:
    - Scan each row and column that crosses exactly one contiguous barrier segment.
    - If the line has exactly one marker on each side of that segment, erase both source markers and place each marker in the cell immediately beyond the opposite side of the segment.
    - Lines with markers on only one side are unchanged; the barrier remains unchanged.

    Key insight:
    - The marker pairing is local to a row or column, and the destination is defined by the nearest cells outside the intervening barrier segment, not by the original distance from the barrier.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    barrier = _barrier_color(g, bg)

    clears: list[tuple[int, int]] = []
    paints: list[tuple[int, int, int]] = []
    height, width = g.shape

    for r in range(height):
        for src, dst, color in _line_moves(g[r, :], barrier, bg):
            clears.append((int(r), int(src)))
            paints.append((int(r), int(dst), int(color)))

    for c in range(width):
        for src, dst, color in _line_moves(g[:, c], barrier, bg):
            clears.append((int(src), int(c)))
            paints.append((int(dst), int(c), int(color)))

    assert clears, "expected at least one marker pair"
    assert len(clears) == len(set(clears)), "a marker cannot move in two directions"
    destinations = [(r, c) for r, c, _ in paints]
    assert len(destinations) == len(set(destinations)), "destination cells must be unique"
    assert all(int(g[r, c]) not in (int(bg), int(barrier)) for r, c in clears), "sources must be markers"
    assert all(int(g[r, c]) == int(bg) for r, c, _ in paints), "destinations must start as background"

    out = g.copy()
    for r, c in clears:
        out[r, c] = int(bg)
    for r, c, color in paints:
        out[r, c] = int(color)
    return out

# --------------- generator (o5x) ---------------

# Grid and palette bounds.
GRID_HEIGHT = 12
GRID_WIDTH_RANGE = (12, 14)
COLOR_POOL = range(1, 10)
MARKER_COLOR_COUNT_RANGE = (3, 4)
HORIZONTAL = "H"
VERTICAL = "V"
ORIENTATIONS = (HORIZONTAL, VERTICAL)

# Barrier profile controls.
H_SPINE_RANGE = (4, 7)
V_SPINE_MIN = 4
V_SPINE_RIGHT_MARGIN = 5
BARRIER_PIXEL_MAX = 26
BARRIER_EXTRA_MIN = 3
VERTICAL_EXTRA_MAX = 14

# Marker layout and retry limits.
PAIR_COUNT_RANGE = range(2, 7)
UNPAIRED_COUNT_RANGE = range(3, 7)
MARKER_TOTAL_RANGE = (11, 16)
MARKER_BARRIER_GAP = 2
GENERATION_RETRIES = 300


def _overlapping_intervals(length: int, total: int) -> tuple[tuple[int, int], tuple[int, int]]:
    splits = [
        (a, total - a)
        for a in range(1, total)
        if a <= length and total - a <= length
    ]
    a_len, b_len = random.choice(splits)
    a0 = random.randint(0, length - a_len)
    a1 = a0 + a_len - 1

    lo = max(0, a0 - b_len + 1)
    hi = min(length - b_len, a1)
    b0 = random.randint(lo, hi)
    return (a0, a1), (b0, b0 + b_len - 1)


def _color_sequence(count: int, palette: list[int]) -> list[int]:
    if len(palette) == 1:
        return [palette[0]] * count
    seq = palette[:]
    random.shuffle(seq)
    while len(seq) < count:
        seq.append(random.choice(palette))
    random.shuffle(seq)
    return seq


def _side_positions(start: int, end: int, limit: int, side: int) -> list[int]:
    if side == 0:
        return list(range(0, start - MARKER_BARRIER_GAP + 1))
    return list(range(end + MARKER_BARRIER_GAP, limit))


def _choose_counts(line_count: int) -> tuple[int, int]:
    choices = [
        (pairs, unpaired)
        for pairs in PAIR_COUNT_RANGE
        for unpaired in UNPAIRED_COUNT_RANGE
        if MARKER_TOTAL_RANGE[0] <= 2 * pairs + unpaired <= MARKER_TOTAL_RANGE[1]
        and pairs + unpaired <= line_count
    ]
    return random.choice(choices)


def _spans_from_offsets(line_count: int, spine: int, before: tuple[int, int], after: tuple[int, int]):
    return [
        (
            spine - int(before[0] <= line <= before[1]),
            spine + int(after[0] <= line <= after[1]),
        )
        for line in range(line_count)
    ]


def _build_barrier(grid: np.ndarray, orientation: str, color: int) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    if orientation == HORIZONTAL:
        spine = random.randint(*H_SPINE_RANGE)
        max_extra = min(BARRIER_PIXEL_MAX - w, 2 * w)
        extra = random.randint(BARRIER_EXTRA_MIN, max_extra)
        top, bottom = _overlapping_intervals(w, extra)
        grid[spine, :] = color
        grid[spine - 1, top[0] : top[1] + 1] = color
        grid[spine + 1, bottom[0] : bottom[1] + 1] = color
        return _spans_from_offsets(w, spine, top, bottom), w

    spine = random.randint(V_SPINE_MIN, w - V_SPINE_RIGHT_MARGIN)
    max_extra = min(VERTICAL_EXTRA_MAX, 2 * h)
    extra = random.randint(BARRIER_EXTRA_MIN, max_extra)
    left, right = _overlapping_intervals(h, extra)
    grid[:, spine] = color
    grid[left[0] : left[1] + 1, spine - 1] = color
    grid[right[0] : right[1] + 1, spine + 1] = color
    return _spans_from_offsets(h, spine, left, right), h


def _side_palettes(marker_colors: list[int]) -> tuple[list[int], list[int]]:
    if len(marker_colors) == 2:
        return [marker_colors[0]], [marker_colors[1]]
    if random.randrange(2) == 0:
        return [marker_colors[0], marker_colors[1]], [marker_colors[2]]
    return [marker_colors[0]], [marker_colors[1], marker_colors[2]]


def _put_marker(grid: np.ndarray, orientation: str, line: int, pos: int, color: int) -> None:
    if orientation == HORIZONTAL:
        grid[pos, line] = color
    else:
        grid[line, pos] = color


def _place_markers(inp, out, orientation, spans, pair_lines, unpaired_lines, side0_colors, side1_colors, limit):
    """Apply the paired swap layout plus unchanged one-sided markers."""
    side0_pair_colors = _color_sequence(len(pair_lines), side0_colors)
    side1_pair_colors = _color_sequence(len(pair_lines), side1_colors)
    for line, color0, color1 in zip(pair_lines, side0_pair_colors, side1_pair_colors):
        start, end = spans[line]
        side0_choices = _side_positions(start, end, limit, 0)
        side1_choices = _side_positions(start, end, limit, 1)
        if not side0_choices or not side1_choices:
            return False
        _put_marker(out, orientation, line, start - 1, color1)
        _put_marker(out, orientation, line, end + 1, color0)
        _put_marker(inp, orientation, line, random.choice(side0_choices), color0)
        _put_marker(inp, orientation, line, random.choice(side1_choices), color1)

    for line in unpaired_lines:
        start, end = spans[line]
        side = random.randrange(2)
        choices = _side_positions(start, end, limit, side)
        if not choices:
            return False
        pos = random.choice(choices)
        palette = side0_colors if side == 0 else side1_colors
        color = random.choice(palette)
        _put_marker(out, orientation, line, pos, color)
        _put_marker(inp, orientation, line, pos, color)
    return True


def _gen_attempt() -> np.ndarray | None:
    w = random.randint(*GRID_WIDTH_RANGE)
    orientation = random.choice(ORIENTATIONS)
    colors = random.sample(COLOR_POOL, random.randint(*MARKER_COLOR_COUNT_RANGE))
    barrier_color = colors[0]
    marker_colors = colors[1:]

    out = np.zeros((GRID_HEIGHT, w), dtype=int)
    spans, line_count = _build_barrier(out, orientation, barrier_color)
    pair_count, unpaired_count = _choose_counts(line_count)
    side0_colors, side1_colors = _side_palettes(marker_colors)

    lines = random.sample(range(line_count), pair_count + unpaired_count)
    pair_lines = lines[:pair_count]
    unpaired_lines = lines[pair_count:]
    inp = out.copy()
    limit = GRID_HEIGHT if orientation == HORIZONTAL else w

    if not _place_markers(inp, out, orientation, spans, pair_lines, unpaired_lines, side0_colors, side1_colors, limit):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid 1efba499 input")
