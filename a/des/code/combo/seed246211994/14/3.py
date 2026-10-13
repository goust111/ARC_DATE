from __future__ import annotations

import random

import numpy as np


ORANGE = 7
RED = 2
GREY = 5
PALETTE = (ORANGE, RED, GREY)
SIGNAL_COLORS = (RED, GREY)
MAX_GENERATION_ATTEMPTS = 300


def _flip_color(color: int, row: int) -> int:
    """Odd rows swap red/grey; background and other colors stay unchanged."""
    if row % 2 == 0:
        return int(color)
    if color == RED:
        return GREY
    if color == GREY:
        return RED
    return int(color)


def _flip_row(values: np.ndarray, row: int) -> np.ndarray:
    out = values.astype(int, copy=True)
    if row % 2 == 0:
        return out
    red_mask = values == RED
    grey_mask = values == GREY
    out[red_mask] = GREY
    out[grey_mask] = RED
    return out


def _span_bounds(top_row: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(top_row != ORANGE)
    assert cols.size > 0, "expected at least one foreground pixel on the top row"
    left = int(cols[0])
    right = int(cols[-1])
    assert (right - left + 1) % 2 == 1, "expected an odd-width foreground span"
    return left, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a top-row red/grey span downward as a shrinking diagonal pattern.

    Input structure:
    - Orange background.
    - All non-background pixels are red or grey and appear only on the top row.
    - The first and last foreground pixels define an odd-width span.

    Rule:
    - Copy the input.
    - For each row r below the top, fill columns [left+r, right-r].
    - Cell (r, c) copies top-row color (0, c-r), with red/grey flipped on odd r.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert bool(np.isin(g, PALETTE).all()), "unexpected colors"
    assert bool(np.all(g[1:] == ORANGE)), "all foreground pixels must be on the top row"

    top_row = g[0]
    left, right = _span_bounds(top_row)

    out = g.copy()
    for row in range(1, g.shape[0]):
        lo = left + row
        hi = right - row
        if lo > hi:
            break
        source = top_row[left : right - 2 * row + 1]
        out[row, lo : hi + 1] = _flip_row(source, row)
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    """Independent diagonal view of the same rule, used by the generator."""
    g = np.asarray(grid, dtype=int)
    top_row = g[0]
    left, right = _span_bounds(top_row)
    out = g.copy()
    for col in range(left, right + 1):
        color = int(top_row[col])
        max_row = min(g.shape[0] - 1, (right - col) // 2)
        for row in range(1, max_row + 1):
            out[row, col + row] = _flip_color(color, row)
    return out


def _other_signal(color: int) -> int:
    return GREY if color == RED else RED


def _dense_random_span(width: int) -> np.ndarray:
    return np.array([random.choice(SIGNAL_COLORS) for _ in range(width)], dtype=int)


def _alternating_span(width: int) -> np.ndarray:
    start = random.choice(SIGNAL_COLORS)
    other = _other_signal(start)
    return np.array([start if idx % 2 == 0 else other for idx in range(width)], dtype=int)


def _run_span(width: int) -> np.ndarray:
    run_count = random.randint(2, min(5, width))
    cuts = sorted(random.sample(range(1, width), run_count - 1))
    lengths: list[int] = []
    prev = 0
    for cut in cuts + [width]:
        lengths.append(cut - prev)
        prev = cut

    color = random.choice(SIGNAL_COLORS)
    values: list[int] = []
    for length in lengths:
        values.extend([color] * length)
        color = _other_signal(color)
    return np.array(values, dtype=int)


def _sparse_span(width: int) -> np.ndarray:
    values = _dense_random_span(width)
    interior = list(range(1, width - 1))
    if interior:
        hole_count = random.randint(1, min(len(interior), max(1, width // 4)))
        for idx in random.sample(interior, hole_count):
            values[idx] = ORANGE
    return values


def _build_top_span(width: int) -> np.ndarray:
    mode = random.choice(("dense", "alternating", "runs", "sparse"))
    if mode == "dense":
        values = _dense_random_span(width)
    elif mode == "alternating":
        values = _alternating_span(width)
    elif mode == "runs":
        values = _run_span(width)
    else:
        values = _sparse_span(width)

    values[0] = random.choice(SIGNAL_COLORS)
    values[-1] = random.choice(SIGNAL_COLORS)
    return values


def _choose_height(span_width: int) -> int:
    full_height = span_width // 2 + 1
    mode = random.choice(("short", "exact", "tall"))
    if mode == "short" and full_height > 2:
        return random.randint(2, full_height - 1)
    if mode == "tall":
        return full_height + random.randint(1, 3)
    return full_height


def _generate_attempt() -> np.ndarray | None:
    span_width = random.randrange(3, 18, 2)
    height = _choose_height(span_width)
    left_pad = random.randint(0, 4)
    right_pad = random.randint(0, 4)
    width = span_width + left_pad + right_pad

    grid = np.full((height, width), ORANGE, dtype=int)
    grid[0, left_pad : left_pad + span_width] = _build_top_span(span_width)

    expected = _expected_output(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
