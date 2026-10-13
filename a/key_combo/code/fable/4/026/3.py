import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = list(range(1, 10))

SEGMENT_COUNT_RANGE = (3, 5)
HEIGHT_RANGE = (10, 18)
LEFT_WIDTH_RANGE = (8, 13)
RIGHT_WIDTH_RANGE = (9, 12)
MAX_ATTEMPTS = 400
DECOY_ORDER_PROB = 0.35


def _mode_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    candidates = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(candidates) == 1, "expected one full non-background separator column"
    return int(candidates[0])


def _extract_segments(left_panel: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    wall_color = int(left_panel[-1, 0])
    assert wall_color != bg and np.all(left_panel[-1] == wall_color), "expected one solid bottom wall"

    body = left_panel[:-1]
    segments: list[tuple[int, int, int, int]] = []
    for color in sorted(int(c) for c in np.unique(body) if int(c) != bg):
        coords = np.argwhere(body == color)
        assert coords.size > 0, "segment color must appear in the upper panel"
        rows = np.unique(coords[:, 0])
        assert len(rows) == 1, "each segment must occupy exactly one row"
        row = int(rows[0])
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        assert coords.shape[0] == c1 - c0 + 1, "segments must be solid horizontal runs"
        segments.append((row, color, c0, c1))

    assert len(segments) >= 1, "expected at least one segment"
    return sorted(segments)


def _right_color_counts(right_panel: np.ndarray, bg: int) -> dict[int, int]:
    counts = {
        int(color): int(np.sum(right_panel == color))
        for color in np.unique(right_panel)
        if int(color) != bg
    }
    assert counts, "right panel must contain colored tally pixels"
    assert len(set(counts.values())) == len(counts), "tally counts must be unique"
    return counts


def _paint_projections(
    base: np.ndarray,
    segments: list[tuple[int, int, int, int]],
    color_counts: dict[int, int],
    bar_row: int,
) -> np.ndarray:
    out = base.copy()
    for row, color, c0, c1 in sorted(segments, key=lambda item: color_counts[item[1]]):
        out[row:bar_row, c0 : c1 + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each left-panel segment downward, but use the right-panel dot counts to decide which strip overwrites the others.

    Input structure:
    - A full monochrome vertical separator splits the grid into a left drawing panel and a right tally panel.
    - In the left panel, the bottom row is one solid bar and the rows above contain several distinct-color one-cell-thick horizontal segments.
    - In the right panel, the background is black and each segment color appears as a unique number of isolated tally pixels.

    Rule:
    - For every left-panel segment, paint the rectangle from that segment's row down to the row just above the bottom bar, using the segment color.
    - Draw those strips in ascending order of the matching color's tally count from the right panel; later strips overwrite earlier ones where they overlap.
    - Keep the separator column, the bottom bar, and the entire right panel unchanged.

    Key insight:
    - The visible overlap order is not determined by which segment is closer to the wall; it is encoded by the hidden per-color counts in the right-side tally panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    sep_col = _find_separator_col(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.size > 0 and right.size > 0

    segments = _extract_segments(left, bg)
    color_counts = _right_color_counts(right, bg)
    segment_colors = {color for _row, color, _c0, _c1 in segments}
    assert set(color_counts) == segment_colors, "left segment colors must match right tally colors"

    return _paint_projections(g, segments, color_counts, left.shape[0] - 1)


def _candidate_tally_slots(height: int, right_width: int) -> list[tuple[int, int]]:
    slots = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, right_width - 1)
        if (r + c) % 2 == 0
    ]
    if slots:
        return slots
    return [(r, c) for r in range(height) for c in range(right_width)]


def _sample_rows(height: int, n_segments: int) -> list[int]:
    rows = sorted(random.sample(range(1, height - 2), n_segments))
    return [int(r) for r in rows]


def _sample_spans(left_width: int, n_segments: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    overlap_count = random.randint(2, min(3, n_segments))
    overlap_ids = sorted(random.sample(range(n_segments), overlap_count))
    shared_col = random.randint(1, left_width - 2)

    spans: list[tuple[int, int]] = []
    for idx in range(n_segments):
        length = random.randint(2, left_width - 2)
        if idx in overlap_ids:
            lo = max(0, shared_col - length + 1)
            hi = min(shared_col, left_width - length)
        else:
            lo = 0
            hi = left_width - length
        start = random.randint(lo, hi)
        spans.append((int(start), int(start + length - 1)))

    return spans, (int(overlap_ids[0]), int(overlap_ids[1]))


def _sample_count_values(n_segments: int, capacity: int) -> list[int]:
    pool = list(range(1, 8))
    for _ in range(100):
        values = sorted(random.sample(pool, n_segments))
        if sum(values) <= capacity:
            return [int(v) for v in values]
    values = list(range(1, n_segments + 1))
    assert sum(values) <= capacity
    return values


def _assign_counts(
    base_counts: list[int],
    overlap_pair: tuple[int, int],
    keep_decoy_order: bool,
) -> list[int]:
    if keep_decoy_order:
        return base_counts[:]

    upper, lower = overlap_pair
    for _ in range(200):
        perm = base_counts[:]
        random.shuffle(perm)
        if perm[upper] > perm[lower]:
            return [int(v) for v in perm]

    perm = base_counts[::-1]
    assert perm[upper] > perm[lower]
    return [int(v) for v in perm]


def _scatter_tallies(height: int, right_width: int, color_counts: dict[int, int]) -> np.ndarray | None:
    slots = _candidate_tally_slots(height, right_width)
    if len(slots) < sum(color_counts.values()):
        return None

    random.shuffle(slots)
    panel = np.zeros((height, right_width), dtype=int)
    idx = 0
    items = list(color_counts.items())
    random.shuffle(items)
    for color, count in items:
        for _ in range(count):
            r, c = slots[idx]
            idx += 1
            panel[r, c] = int(color)
    return panel


def _render_input(
    height: int,
    left_width: int,
    right_width: int,
    separator_color: int,
    wall_color: int,
    segments: list[tuple[int, int, int, int]],
    color_counts: dict[int, int],
) -> np.ndarray | None:
    left = np.zeros((height, left_width), dtype=int)
    left[-1, :] = wall_color
    for row, color, c0, c1 in segments:
        left[row, c0 : c1 + 1] = color

    right = _scatter_tallies(height, right_width, color_counts)
    if right is None:
        return None

    divider = np.full((height, 1), separator_color, dtype=int)
    return np.concatenate([left, divider, right], axis=1)


def _expected_output(
    inp: np.ndarray,
    segments: list[tuple[int, int, int, int]],
    color_counts: dict[int, int],
    left_width: int,
) -> np.ndarray:
    expected = inp.copy()
    return _paint_projections(expected, segments, color_counts, expected.shape[0] - 1)


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(*SEGMENT_COUNT_RANGE)
    height = random.randint(max(HEIGHT_RANGE[0], n_segments + 5), HEIGHT_RANGE[1])
    left_width = random.randint(*LEFT_WIDTH_RANGE)
    right_width = random.randint(*RIGHT_WIDTH_RANGE)

    colors = random.sample(PALETTE, n_segments + 2)
    separator_color = int(colors[0])
    wall_color = int(colors[1])
    segment_colors = [int(c) for c in colors[2:]]

    rows = _sample_rows(height, n_segments)
    spans, overlap_pair = _sample_spans(left_width, n_segments)

    tally_capacity = len(_candidate_tally_slots(height, right_width))
    base_counts = _sample_count_values(n_segments, tally_capacity)
    keep_decoy_order = random.random() < DECOY_ORDER_PROB
    assigned_counts = _assign_counts(base_counts, overlap_pair, keep_decoy_order)

    segments = [
        (rows[idx], segment_colors[idx], spans[idx][0], spans[idx][1])
        for idx in range(n_segments)
    ]
    color_counts = {segment_colors[idx]: assigned_counts[idx] for idx in range(n_segments)}

    inp = _render_input(
        height,
        left_width,
        right_width,
        separator_color,
        wall_color,
        segments,
        color_counts,
    )
    if inp is None:
        return None

    expected = _expected_output(inp, segments, color_counts, left_width)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
