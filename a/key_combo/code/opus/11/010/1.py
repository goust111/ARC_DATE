from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
STRIPE = 5
MARKER = 6
PALETTE = (1, 2, 3, 4, 7, 8)
MIN_SEGMENTS = 4
MAX_SEGMENTS = 6
MIN_LENGTH = 2
MAX_LENGTH = 10
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).ravel())
    return int(counts.most_common(1)[0][0])


def _find_stripe_row(grid: np.ndarray, bg: int) -> int:
    stripe_rows = [
        r
        for r in range(grid.shape[0])
        if np.unique(grid[r]).size == 1 and int(grid[r, 0]) != bg
    ]
    assert len(stripe_rows) == 1, "expected exactly one full-width non-background stripe"
    stripe_row = int(stripe_rows[0])
    assert stripe_row == grid.shape[0] - 1, "stripe must be on the bottom row"
    return stripe_row


def _extract_segments(grid: np.ndarray, bg: int, stripe_row: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for r in range(stripe_row):
        cols = np.flatnonzero(grid[r] != bg)
        if cols.size == 0:
            continue
        assert np.all(np.diff(cols) == 1), "each foreground row must contain one contiguous segment"
        color = int(grid[r, cols[0]])
        assert color != STRIPE, "moving segments must not use the stripe color"
        assert np.all(grid[r, cols] == color), "each segment must be monochrome"
        segments.append((int(cols.size), color))

    assert MIN_SEGMENTS <= len(segments) <= MAX_SEGMENTS, "expected 4-6 segments above the stripe"
    lengths = [length for length, _color in segments]
    assert len(set(lengths)) == len(lengths), "segment lengths must be unique"
    colors = [color for _length, color in segments]
    assert colors.count(MARKER) == 1, "expected exactly one magenta marker segment"
    assert len(set(c for c in colors if c != MARKER)) == len(colors) - 1, "non-marker segment colors must be distinct"
    return segments


def _stack_segments(shape: tuple[int, int], segments: list[tuple[int, int]], stripe_row: int, bg: int) -> np.ndarray:
    height, width = shape
    out = np.full((height, width), bg, dtype=int)
    for idx, (length, color) in enumerate(sorted(segments, key=lambda item: item[0])):
        r = stripe_row - 1 - idx
        out[r, width - length : width] = int(color)
    return out


def _reflection_orbit(r: int, c: int, height: int, width: int) -> tuple[tuple[int, int], ...]:
    return (
        (int(r), int(c)),
        (int(height - 1 - r), int(c)),
        (int(r), int(width - 1 - c)),
        (int(height - 1 - r), int(width - 1 - c)),
    )


def _fill_marker_holes(stacked_upper: np.ndarray, bg: int) -> np.ndarray:
    upper = np.asarray(stacked_upper, dtype=int)
    height, width = upper.shape
    out = upper.copy()
    marker_cells = np.argwhere(upper == MARKER)
    assert marker_cells.size > 0, "expected at least one marker cell"

    for r, c in marker_cells:
        orbit = _reflection_orbit(int(r), int(c), height, width)
        known = {
            int(upper[rr, cc])
            for rr, cc in orbit
            if int(upper[rr, cc]) not in (bg, MARKER)
        }
        assert len(known) == 1, "each marker orbit must reveal exactly one non-marker color"
        out[int(r), int(c)] = known.pop()

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered horizontal segments into a right-aligned stack, then recolor the magenta holes from their two-axis reflection orbits.

    Input structure:
    - The grid has a black background and one solid grey stripe on the bottom row.
    - Above the stripe are 4-6 one-cell-thick horizontal segments on distinct rows.
    - Segment lengths are all different; exactly one segment is magenta and acts as the hole marker.

    Rule:
    - Remove the scattered segments from their original rows, sort them by length from short to long, and stack them from the stripe upward with every right end flush to the grid edge.
    - Treat the rows above the stripe as the symmetry canvas.
    - For every magenta cell in the stacked canvas, look at its four-way reflection orbit under horizontal mirror, vertical mirror, and 180-degree rotation.
    - Ignore black background and magenta marker cells inside that orbit; the remaining visible color is unique and replaces the magenta cell.
    - Keep the bottom grey stripe unchanged.

    Key insight:
    - The scattered input does not expose the symmetry relation directly; only after length-sorting and right-aligned stacking do the magenta cells land on reflection orbits that reveal their fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    stripe_row = _find_stripe_row(g, bg)
    stripe_color = int(g[stripe_row, 0])
    assert stripe_color == STRIPE, "expected the bottom stripe to be grey"

    segments = _extract_segments(g, bg, stripe_row)
    stacked_upper = _stack_segments(g.shape, segments, stripe_row, bg)
    filled_upper = _fill_marker_holes(stacked_upper[:stripe_row, :], bg)

    out = np.full_like(g, bg)
    out[:stripe_row, :] = filled_upper
    out[stripe_row, :] = stripe_color
    return out


def _choose_lengths(n_segments: int) -> list[int]:
    return sorted(random.sample(range(MIN_LENGTH, MAX_LENGTH + 1), n_segments))


def _choose_geometry(n_segments: int, lengths: list[int]) -> tuple[int, int]:
    overlap_mode = random.choice((2, 3))
    upper_height = 2 * n_segments - overlap_mode
    max_len = max(lengths)
    min_width = max(max_len + 2, 7)
    max_width = min(2 * max_len - 1, 16)
    if max_width < min_width:
        max_width = min_width
    width = random.randint(min_width, max_width)
    return upper_height, width


def _build_stacked_input(n_segments: int) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    lengths = _choose_lengths(n_segments)
    upper_height, width = _choose_geometry(n_segments, lengths)
    stripe_row = upper_height
    height = upper_height + 1

    colors = random.sample(PALETTE, n_segments - 1)
    overlap_rows = 2 * n_segments - upper_height
    marker_stack_index = overlap_rows - 1

    sorted_pairs: list[tuple[int, int]] = []
    color_iter = iter(colors)
    for stack_index, length in enumerate(reversed(lengths)):
        if stack_index == marker_stack_index:
            sorted_pairs.append((length, MARKER))
        else:
            sorted_pairs.append((length, next(color_iter)))

    stacked = np.full((height, width), BG, dtype=int)
    stacked[stripe_row, :] = STRIPE
    for stack_index, (length, color) in enumerate(sorted_pairs):
        r = stripe_row - n_segments + stack_index
        stacked[r, width - length : width] = int(color)

    upper = stacked[:stripe_row, :]
    filled = _fill_marker_holes(upper, BG)

    # The marker row must actually change, and the source row must not be empty.
    marker_cells = np.argwhere(upper == MARKER)
    assert marker_cells.size > 0
    assert np.any(filled[upper == MARKER] != MARKER)

    # Every orbit in the stacked canvas should carry at most one visible non-black color.
    for r in range(upper.shape[0]):
        for c in range(upper.shape[1]):
            orbit = _reflection_orbit(r, c, upper.shape[0], upper.shape[1])
            visible = {int(upper[rr, cc]) for rr, cc in orbit if int(upper[rr, cc]) not in (BG, MARKER)}
            assert len(visible) <= 1, "generated stack must keep reflection colors unambiguous"

    segments = [(length, color) for length, color in sorted_pairs]
    return stacked, segments, stripe_row


def _scatter_segments(shape: tuple[int, int], segments: list[tuple[int, int]], stripe_row: int) -> np.ndarray:
    height, width = shape
    grid = np.full((height, width), BG, dtype=int)
    grid[stripe_row, :] = STRIPE

    rows = random.sample(range(stripe_row), len(segments))
    random.shuffle(rows)
    shuffled_segments = segments[:]
    random.shuffle(shuffled_segments)

    placed = []
    for row, (length, color) in zip(rows, shuffled_segments):
        c0 = random.randint(0, width - length)
        grid[row, c0 : c0 + length] = int(color)
        placed.append((row, c0, length, color))

    target_rows = list(range(stripe_row - len(segments), stripe_row))
    target_signature = sorted((r, width - length, length, color) for r, (length, color) in zip(target_rows, segments))
    input_signature = sorted(placed)
    assert input_signature != target_signature, "input must not already be the solved stack"
    return grid


def _semantic_checks(inp: np.ndarray) -> None:
    out = main(inp)
    bg = _get_bg(inp)
    stripe_row = _find_stripe_row(inp, bg)
    segments = _extract_segments(inp, bg, stripe_row)
    stacked = _stack_segments(inp.shape, segments, stripe_row, bg)
    expected = np.full_like(inp, bg)
    expected[:stripe_row, :] = _fill_marker_holes(stacked[:stripe_row, :], bg)
    expected[stripe_row, :] = STRIPE
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out), "generation should not collapse to an identity task"
    assert len(np.unique(out)) >= 4, "output should stay visually informative"
    assert np.count_nonzero(out == MARKER) == 0, "marker cells must be fully resolved in the output"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
            stacked, segments, stripe_row = _build_stacked_input(n_segments)
            inp = _scatter_segments(stacked.shape, segments, stripe_row)
            _semantic_checks(inp)
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for the new ARC task")
