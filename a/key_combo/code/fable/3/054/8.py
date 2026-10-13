from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
BACKGROUND = 0

MAX_GENERATION_ATTEMPTS = 400
SEGMENT_COUNT_RANGE = (3, 8)
STEP_LENGTH_RANGE = (1, 6)
MAX_TAPE_CELLS = 26
MAX_EXTRA_HEIGHT = 6
MAX_EXTRA_WIDTH = 8


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _other_tape_color(color: int) -> int:
    return BLUE if int(color) == RED else RED


def _parse_tape_segments(top_row: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cols = np.flatnonzero(np.asarray(top_row, dtype=int) != int(bg))
    assert cols.size > 0, "expected a non-background tape on the top row"

    left = int(cols[0])
    right = int(cols[-1])
    assert right - left + 1 == int(cols.size), "tape must be one contiguous strip"

    tape = np.asarray(top_row[left : right + 1], dtype=int)
    assert set(map(int, np.unique(tape))) <= {BLUE, RED}, "tape must use only red and blue"

    segments: list[tuple[int, int]] = []
    cur = int(tape[0])
    length = 1
    for value in map(int, tape[1:]):
        if value == cur:
            length += 1
            continue
        segments.append((cur, int(length)))
        cur = int(value)
        length = 1
    segments.append((cur, int(length)))
    return segments, (left, right)


def _draw_trace(
    out: np.ndarray,
    segments: list[tuple[int, int]],
    start: tuple[int, int],
) -> None:
    r, c = map(int, start)
    h, w = out.shape

    for index, (color, length) in enumerate(segments):
        step = 1 if int(color) == RED else -1
        dr, dc = (0, step) if index % 2 == 0 else (step, 0)
        for _ in range(int(length)):
            r += dr
            c += dc
            assert 0 <= r < h and 0 <= c < w, "decoded trace left the grid"
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top red/blue edge strip into a colored orthogonal trace from the green start marker.

    Input structure:
    - The top row contains one contiguous strip made of red and blue runs, with background on both sides.
    - A single green pixel below the strip marks the trace start.
    - The remaining cells are background.

    Rule:
    - Read the strip left to right as maximal same-color runs.
    - Odd-numbered runs move horizontally and even-numbered runs move vertically; red means the positive direction (right or down), while blue means the negative direction (left or up).
    - The run length is the number of unit steps. Clear the original strip, keep the green start marker, and color every walked cell with the current run color.

    Key insight:
    - The top strip is a one-dimensional program tape: run parity chooses the axis, run color chooses the sign, and run length chooses the distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    segments, (left, right) = _parse_tape_segments(g[0], bg)

    starts = np.argwhere(g == GREEN)
    assert starts.shape == (1, 2), "expected exactly one green start marker"
    start = tuple(map(int, starts[0]))

    out = g.copy()
    out[0, left : right + 1] = bg
    _draw_trace(out, segments, start)
    out[start] = GREEN
    return out


def _sample_segments() -> list[tuple[int, int]] | None:
    count = random.randint(*SEGMENT_COUNT_RANGE)
    first_color = random.choice((BLUE, RED))
    lengths = [random.randint(*STEP_LENGTH_RANGE) for _ in range(count)]
    if sum(lengths) > MAX_TAPE_CELLS:
        return None

    colors = [first_color]
    for _ in range(1, count):
        colors.append(_other_tape_color(colors[-1]))
    return [(int(color), int(length)) for color, length in zip(colors, lengths)]


def _path_extents(segments: list[tuple[int, int]]) -> tuple[int, int]:
    lengths = [length for _color, length in segments]
    horiz_total = sum(lengths[0::2])
    vert_total = sum(lengths[1::2])
    return int(horiz_total), int(vert_total)


def _build_input(segments: list[tuple[int, int]]) -> np.ndarray | None:
    tape_len = sum(length for _color, length in segments)
    horiz_total, vert_total = _path_extents(segments)
    path_h = vert_total + 1
    path_w = horiz_total + 1

    max_gap = min(4, 29 - path_h)
    if max_gap < 1:
        return None
    gap = random.randint(1, max_gap)

    min_h = 1 + gap + path_h
    min_w = max(path_w, tape_len)
    if min_h > 30 or min_w > 30:
        return None

    max_h = min(30, min_h + MAX_EXTRA_HEIGHT)
    max_w = min(30, min_w + MAX_EXTRA_WIDTH)
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    path_top = random.randint(1 + gap, h - path_h)
    path_left = random.randint(0, w - path_w)

    if segments[0][0] == RED:
        start_r = path_top + vert_total
        start_c = path_left
    else:
        start_r = path_top
        start_c = path_left + horiz_total

    tape_left = random.randint(0, w - tape_len)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    cursor = tape_left
    for color, length in segments:
        grid[0, cursor : cursor + length] = int(color)
        cursor += int(length)
    grid[start_r, start_c] = GREEN
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        segments = _sample_segments()
        if segments is None:
            continue
        grid = _build_input(segments)
        if grid is None:
            continue
        out = main(grid)
        if out.shape == grid.shape and not np.array_equal(out, grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
