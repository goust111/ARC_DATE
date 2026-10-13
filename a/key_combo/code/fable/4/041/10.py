from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_row_segment(row: np.ndarray, bg: int) -> tuple[str, np.ndarray]:
    width = int(row.shape[0])
    left_marker = int(row[0]) != bg
    right_marker = int(row[-1]) != bg
    assert left_marker ^ right_marker, "each row must have exactly one edge marker"

    marker_col = 0 if left_marker else width - 1
    mask = row != bg
    mask[marker_col] = False
    segment_cols = np.flatnonzero(mask)
    assert segment_cols.size > 0, "each row must contain one foreground segment"
    assert np.all(np.diff(segment_cols) == 1), "segment must be a single contiguous run"

    if left_marker:
        assert int(segment_cols[0]) >= 2, "left marker must be separated from the segment"
        direction = "left"
    else:
        assert int(segment_cols[-1]) <= width - 3, "right marker must be separated from the segment"
        direction = "right"

    start = int(segment_cols[0])
    end = int(segment_cols[-1]) + 1
    return direction, np.array(row[start:end], dtype=int)


def _settled_start(
    prev_span: tuple[int, int] | None, seg_len: int, width: int, direction: str
) -> int:
    if prev_span is None:
        return 0 if direction == "left" else int(width - seg_len)

    prev_left, prev_right = prev_span
    lo = max(0, int(prev_left - seg_len + 1))
    hi = min(int(width - seg_len), int(prev_right))
    assert lo <= hi, "current row must be able to overlap the previous settled row"
    return lo if direction == "left" else hi


def main(grid: np.ndarray) -> np.ndarray:
    """Cascade each row segment toward its edge marker while keeping column overlap with the row above.

    Input structure:
    - The grid has one continuous non-background segment in every row plus one isolated
      edge marker on the leftmost or rightmost column of that row.
    - The marker is separated from the row segment by background cells, and rows start
      at different horizontal offsets.

    Rule:
    - Process rows from top to bottom.
    - Slide the top-row segment toward its own marker until it touches the matching edge.
    - For every later row, slide its segment toward its own marker as far as possible
      while still sharing at least one column with the previous row's settled segment.
    - Erase all markers and return the settled row segments.

    Key insight:
    - Each marker gives only a direction; the actual landing column of a row is the
      directional extreme that still overlaps the row above, so the whole layout is a
      top-down cascade rather than a set of independent row moves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    bg = _get_bg(g)

    out = np.full((height, width), bg, dtype=int)
    prev_span: tuple[int, int] | None = None

    for r in range(height):
        direction, pattern = _extract_row_segment(g[r], bg)
        seg_len = int(pattern.shape[0])
        start = _settled_start(prev_span, seg_len, width, direction)
        out[r, start : start + seg_len] = pattern
        prev_span = (start, start + seg_len - 1)

    return out


# --------------- generator ---------------

BG = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
HEIGHT_RANGE = (4, 9)
WIDTH_RANGE = (9, 18)
SEG_LEN_MIN = 2
SEG_LEN_MAX = 7
MAX_ATTEMPTS = 400


def _random_pattern(length: int) -> np.ndarray:
    branch = random.random()
    if branch < 0.35:
        color = random.choice(PALETTE)
        return np.full(length, color, dtype=int)

    if branch < 0.7:
        c1, c2 = random.sample(PALETTE, 2)
        pattern = np.array(
            [c1 if i % 2 == 0 else c2 for i in range(length)],
            dtype=int,
        )
        if length >= 3 and random.random() < 0.5:
            pattern[length // 2] = random.choice((c1, c2))
        return pattern

    color_count = random.randint(2, min(3, len(PALETTE)))
    colors = random.sample(PALETTE, color_count)
    pattern = np.array([random.choice(colors) for _ in range(length)], dtype=int)
    if len(set(map(int, pattern))) == 1:
        pattern[random.randrange(length)] = colors[-1]
    return pattern


def _random_directions(height: int) -> list[str]:
    roll = random.random()
    if roll < 0.2:
        direction = random.choice(("left", "right"))
        return [direction] * height
    if roll < 0.45:
        first = random.choice(("left", "right"))
        second = "right" if first == "left" else "left"
        return [first if i % 2 == 0 else second for i in range(height)]
    return [random.choice(("left", "right")) for _ in range(height)]


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    max_seg_len = min(SEG_LEN_MAX, width - 2)
    if max_seg_len < SEG_LEN_MIN:
        return None

    directions = _random_directions(height)
    lengths = [random.randint(SEG_LEN_MIN, max_seg_len) for _ in range(height)]

    settled_starts: list[int] = []
    prev_span: tuple[int, int] | None = None
    for direction, seg_len in zip(directions, lengths):
        start = _settled_start(prev_span, seg_len, width, direction)
        if direction == "left":
            if start > width - seg_len - 1:
                return None
        else:
            if start < 1 or width - seg_len < 2:
                return None
        settled_starts.append(start)
        prev_span = (start, start + seg_len - 1)

    patterns = [_random_pattern(seg_len) for seg_len in lengths]

    inp = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    for r, (direction, seg_len, settled_start, pattern) in enumerate(
        zip(directions, lengths, settled_starts, patterns)
    ):
        if direction == "left":
            input_lo = max(2, settled_start + 1)
            input_hi = width - seg_len
            if input_lo > input_hi:
                return None
            input_start = random.randint(input_lo, input_hi)
            inp[r, 0] = MARKER
        else:
            input_lo = 0
            input_hi = min(settled_start - 1, width - seg_len - 2)
            if input_lo > input_hi:
                return None
            input_start = random.randint(input_lo, input_hi)
            inp[r, width - 1] = MARKER

        inp[r, input_start : input_start + seg_len] = pattern
        expected[r, settled_start : settled_start + seg_len] = pattern

    if np.array_equal(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid chained-row ARC input")
