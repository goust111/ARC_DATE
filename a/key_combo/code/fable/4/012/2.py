from __future__ import annotations

import random

import numpy as np


YELLOW = 4
GREEN = 3
RED = 2
BACKGROUND_CHOICES = (0, 1, 5, 6, 7, 8, 9)
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_left_segment(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == YELLOW)
    assert pts.size > 0, "expected a yellow segment"
    assert np.all(pts[:, 1] == 0), "yellow source must lie on the left edge"

    rows = np.sort(pts[:, 0].astype(int))
    start = int(rows[0])
    length = int(rows.size)
    assert np.array_equal(rows, np.arange(start, start + length)), "yellow source must be contiguous"
    return start, length


def _find_top_segment(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.size > 0, "expected a green segment"
    assert np.all(pts[:, 0] == 0), "green source must lie on the top edge"

    cols = np.sort(pts[:, 1].astype(int))
    start = int(cols[0])
    length = int(cols.size)
    assert np.array_equal(cols, np.arange(start, start + length)), "green source must be contiguous"
    return start, length


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the red interference points created by two invisible wavefront families.

    Input structure:
    - The grid has a dominant background and exactly two visible source objects.
    - A yellow vertical segment sits on the left border, spanning consecutive rows.
    - A green horizontal segment sits on the top border, spanning consecutive columns.

    Rule:
    - Let the yellow segment length be Ly. For every positive multiple of Ly, place an invisible vertical wavefront at that column, but only across the yellow segment's row span.
    - Let the green segment length be Lg. For every positive multiple of Lg, place an invisible horizontal wavefront at that row, but only across the green segment's column span.
    - Keep the wavefronts themselves invisible and paint only their intersections red.
    - Preserve the original background and the two source segments.

    Key insight:
    - Each border segment is a wave source whose own length becomes the propagation period; the visible output is the Cartesian product of the two hidden periodic schedules inside their overlapping corridor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == RED), "red must be absent from the input"

    bg = _get_bg(g)
    colors = set(map(int, np.unique(g))) - {bg}
    assert colors == {YELLOW, GREEN}, "expected only yellow and green foreground sources"

    yellow_start, yellow_len = _find_left_segment(g)
    green_start, green_len = _find_top_segment(g)

    h, w = g.shape
    out = g.copy()

    hit_rows = [
        row
        for row in range(green_len, h, green_len)
        if yellow_start <= row < yellow_start + yellow_len
    ]
    hit_cols = [
        col
        for col in range(yellow_len, w, yellow_len)
        if green_start <= col < green_start + green_len
    ]

    for row in hit_rows:
        for col in hit_cols:
            out[row, col] = RED
    return out


def _count_hits(start: int, span_len: int, step: int) -> int:
    end = start + span_len - 1
    return int(end // step - (start - 1) // step)


def _valid_starts(limit: int, span_len: int, step: int, counts: set[int]) -> list[int]:
    return [
        start
        for start in range(1, limit - span_len)
        if _count_hits(start, span_len, step) in counts
    ]


def _choose_lengths(mode: str) -> tuple[int, int]:
    if mode == "single":
        yellow_len = random.randint(3, 9)
        green_len = random.randint(3, 9)
    elif mode == "row_rich":
        green_len = random.randint(3, 6)
        yellow_len = random.randint(green_len + 1, min(2 * green_len + 2, 10))
    else:
        yellow_len = random.randint(3, 6)
        green_len = random.randint(yellow_len + 1, min(2 * yellow_len + 2, 10))
    return yellow_len, green_len


def _attempt_input() -> np.ndarray | None:
    mode = random.choices(
        ("single", "row_rich", "col_rich"),
        weights=(2, 3, 3),
        k=1,
    )[0]
    yellow_len, green_len = _choose_lengths(mode)

    min_h = max(2 * green_len + 1, yellow_len + 3, 10)
    min_w = max(2 * yellow_len + 1, green_len + 3, 10)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, min(30, min_h + 8))
    w = random.randint(min_w, min(30, min_w + 8))

    if mode == "single":
        yellow_start_choices = _valid_starts(h, yellow_len, green_len, {1})
        green_start_choices = _valid_starts(w, green_len, yellow_len, {1})
    elif mode == "row_rich":
        yellow_start_choices = _valid_starts(h, yellow_len, green_len, {2, 3})
        green_start_choices = _valid_starts(w, green_len, yellow_len, {1})
    else:
        yellow_start_choices = _valid_starts(h, yellow_len, green_len, {1})
        green_start_choices = _valid_starts(w, green_len, yellow_len, {2, 3})

    if not yellow_start_choices or not green_start_choices:
        return None

    yellow_start = random.choice(yellow_start_choices)
    green_start = random.choice(green_start_choices)
    bg = random.choice(BACKGROUND_CHOICES)

    grid = np.full((h, w), bg, dtype=int)
    grid[yellow_start : yellow_start + yellow_len, 0] = YELLOW
    grid[0, green_start : green_start + green_len] = GREEN

    out = main(grid)
    red_count = int(np.sum(out == RED))
    if red_count == 0 or red_count > 6:
        return None
    if _get_bg(grid) != bg:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
