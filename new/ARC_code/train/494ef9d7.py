from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg
import random


MOVING_PAIRS = (
    frozenset({Color.BLUE, Color.TEAL}),
    frozenset({Color.YELLOW, Color.ORANGE}),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each blue/teal or yellow/orange row pair together while preserving its left-to-right order.

    Input structure
    - The grid has a black background.
    - Each nonempty row contains exactly two distinct foreground pixels.
    - Blue/teal and yellow/orange row pairs are the moving pairs; other color pairs are distractors.

    Rule
    - For each moving pair, keep the left pixel fixed and move the right pixel left until it touches the left pixel.
    - Clear the moved pixel's original location, and leave non-moving rows unchanged.

    Key insight
    - Only same-hue-family pixel pairs attract horizontally; separated unrelated colors do not compact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black to be the background"

    out = g.copy()
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] != bg)
        assert len(cols) in (0, 2), "each nonempty row must have exactly two foreground pixels"
        if len(cols) == 0:
            continue

        c0, c1 = map(int, cols)
        colors = frozenset({int(g[r, c0]), int(g[r, c1])})
        assert len(colors) == 2, "row foreground pixels must have distinct colors"

        if colors in MOVING_PAIRS:
            dst_c = c0 + 1
            assert dst_c < g.shape[1], "moving pair must have room to touch"
            if c1 != dst_c:
                assert g[r, dst_c] == bg, "movement path must begin with background"
                out[r, dst_c] = g[r, c1]
                out[r, c1] = bg

    return out

# --------------- generator (att2, o5x) ---------------

BLACK = 0
# Canvas bounds; numpy randint highs are exclusive.
ROW_RANGE, COL_RANGE = (3, 11), (4, 11)
# Row composition ranges; all highs are exclusive.
MOVING_COUNT_RANGE = range(1, 5)
DISTRACTOR_COUNT_RANGE = range(2, 6)
EMPTY_COUNT_RANGE = range(0, 3)
ROW_MOVING, ROW_DISTRACTOR, ROW_EMPTY = "moving", "distractor", "empty"
# Pair palettes and allowed empty-cell gaps between the two row pixels.
GEN_MOVING_PAIRS = ((1, 8), (4, 7))
MOVING_PAIR_SETS = {frozenset(pair) for pair in GEN_MOVING_PAIRS}
FOREGROUND_COLORS = range(1, 10)
DISTRACTOR_PAIRS = tuple(
    (a, b) for a in FOREGROUND_COLORS for b in FOREGROUND_COLORS
    if b > a and frozenset((a, b)) not in MOVING_PAIR_SETS
)
MOVING_MAX_GAP, DISTRACTOR_MAX_GAP = 7, 6
# Retry budget for oracle-checked generation.
MAX_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _ordered_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if bool(_RNG.integers(0, 2)):
        return pair[1], pair[0]
    return pair


def _separated_columns(cols: int, max_gap: int) -> tuple[int, int]:
    gap = int(_RNG.integers(1, min(max_gap, cols - 2) + 1))
    c0 = int(_RNG.integers(0, cols - gap - 1))
    return c0, c0 + gap + 1


def _choose_counts(rows: int) -> tuple[int, int, int]:
    choices = [
        (moving, distractor, empty)
        for moving in MOVING_COUNT_RANGE
        for distractor in DISTRACTOR_COUNT_RANGE
        for empty in EMPTY_COUNT_RANGE
        if moving + distractor + empty == rows
    ]
    return choices[int(_RNG.integers(0, len(choices)))]


def _row_types(rows: int) -> list[str]:
    moving_count, distractor_count, empty_count = _choose_counts(rows)
    row_types = [ROW_MOVING] * moving_count
    row_types += [ROW_DISTRACTOR] * distractor_count
    row_types += [ROW_EMPTY] * empty_count
    _RNG.shuffle(row_types)
    return row_types


def _place_row(out: np.ndarray, row: int, row_type: str, cols: int, moving_sources) -> None:
    if row_type == ROW_EMPTY:
        return
    if row_type == ROW_MOVING:
        pairs, max_gap = GEN_MOVING_PAIRS, MOVING_MAX_GAP
    else:
        pairs, max_gap = DISTRACTOR_PAIRS, DISTRACTOR_MAX_GAP

    pair = pairs[int(_RNG.integers(0, len(pairs)))]
    left_color, right_color = _ordered_pair(pair)
    c0, c1 = _separated_columns(cols, max_gap)
    out[row, c0] = left_color
    if row_type == ROW_MOVING:
        out[row, c0 + 1] = right_color
        moving_sources.append((row, c0, c1, right_color))
    else:
        out[row, c1] = right_color


def _expand_input(out: np.ndarray, moving_sources) -> np.ndarray:
    inp = out.copy()
    for row, c0, c1, right_color in moving_sources:
        inp[row, c0 + 1] = BLACK
        inp[row, c1] = right_color
    return inp


def _gen_attempt() -> np.ndarray | None:
    rows = int(_RNG.integers(*ROW_RANGE))
    cols = int(_RNG.integers(*COL_RANGE))
    out = np.full((rows, cols), BLACK, dtype=int)
    moving_sources = []
    for row, row_type in enumerate(_row_types(rows)):
        _place_row(out, row, row_type, cols, moving_sources)

    inp = _expand_input(out, moving_sources)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
