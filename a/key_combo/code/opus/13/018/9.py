from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3

GRID_MIN = 13
GRID_MAX = 18
DIAG_LEN_MIN = 2
STRAIGHT_LEN_MIN = 2
MAX_GEN_ATTEMPTS = 500


def _sign(x: int) -> int:
    return int((x > 0) - (x < 0))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _segment_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    step: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(int(length)):
        r = int(start[0] + i * step[0])
        c = int(start[1] + i * step[1])
        if _in_bounds(shape, r, c):
            cells.append((r, c))
    return cells


def _segment_fully_in_bounds(
    shape: tuple[int, int],
    start: tuple[int, int],
    step: tuple[int, int],
    length: int,
) -> bool:
    end = (int(start[0] + (length - 1) * step[0]), int(start[1] + (length - 1) * step[1]))
    return _in_bounds(shape, start[0], start[1]) and _in_bounds(shape, end[0], end[1])


def _decompose_offset(p: tuple[int, int], q: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr = int(q[0] - p[0])
    dc = int(q[1] - p[1])
    assert dr != 0 and dc != 0
    m = min(abs(dr), abs(dc))
    diag = (m * _sign(dr), m * _sign(dc))
    axis = (dr - diag[0], dc - diag[1])
    assert (axis[0] == 0) ^ (axis[1] == 0)
    return diag, axis


def _line_parameters(axis: tuple[int, int]) -> tuple[tuple[int, int], int]:
    step = (_sign(axis[0]), _sign(axis[1]))
    length = abs(axis[0]) + abs(axis[1]) + 1
    assert step != (0, 0)
    return step, int(length)


def _visible_translation_indices(
    shape: tuple[int, int],
    start: tuple[int, int],
    diag: tuple[int, int],
    axis: tuple[int, int],
) -> list[int]:
    step, length = _line_parameters(axis)
    bound = int(sum(shape) + length + 2)
    visible: list[int] = []
    for k in range(-bound, bound + 1):
        shifted_start = (int(start[0] + k * diag[0]), int(start[1] + k * diag[1]))
        if _segment_cells(shape, shifted_start, step, length):
            visible.append(k)
    return visible


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a periodic array of green axis-aligned line segments driven by diagonal decomposition of the two red markers.

    Input structure:
    - The grid is a black background with exactly two red marker pixels.
    - The two markers are in different rows and different columns, and their row and column offsets have unequal magnitudes.

    Rule:
    - Compute the offset from one marker to the other and decompose it into a diagonal vector D and an axis-aligned vector A.
    - Use A to define a horizontal or vertical seed line segment starting at the first marker, with length |A| + 1 cells including the start.
    - Copy that same segment by repeatedly translating it by +D and -D, clipping any out-of-bounds cells, until translated segments are fully outside the grid.
    - Paint all segment cells green, then repaint the two original marker pixels red.

    Key insight:
    - The diagonal component does not draw part of the segment itself; it acts as the period vector that replicates an axis-aligned seed segment across the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, RED}

    markers = np.argwhere(g == RED)
    assert markers.shape == (2, 2)
    p = tuple(map(int, markers[0]))
    q = tuple(map(int, markers[1]))

    diag, axis = _decompose_offset(p, q)
    step, length = _line_parameters(axis)

    out = np.full(g.shape, BLACK, dtype=int)
    for k in _visible_translation_indices(g.shape, p, diag, axis):
        shifted_start = (int(p[0] + k * diag[0]), int(p[1] + k * diag[1]))
        for r, c in _segment_cells(g.shape, shifted_start, step, length):
            out[r, c] = GREEN

    out[p] = RED
    out[q] = RED
    return out


def _make_input_grid(shape: tuple[int, int], p: tuple[int, int], q: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[p] = RED
    grid[q] = RED
    return grid


def _candidate_starts(
    shape: tuple[int, int],
    diag: tuple[int, int],
    axis: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    step, length = _line_parameters(axis)
    starts: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            p = (r, c)
            q = (int(r + diag[0] + axis[0]), int(c + diag[1] + axis[1]))
            if not _in_bounds(shape, q[0], q[1]):
                continue
            if max(abs(q[0] - r), abs(q[1] - c)) <= 1:
                continue
            if any(
                not _segment_fully_in_bounds(
                    shape,
                    (int(r + k * diag[0]), int(c + k * diag[1])),
                    step,
                    length,
                )
                for k in (-1, 0, 1)
            ):
                continue
            starts.append(p)
    return starts


def _random_dimensions() -> tuple[int, int]:
    return random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)


def _random_vectors(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, shape)
    dominant_rows = random.random() < 0.5
    sr = random.choice((-1, 1))
    sc = random.choice((-1, 1))

    max_diag = min(h, w) // 3
    if max_diag < DIAG_LEN_MIN:
        return None
    diag_len = random.randint(DIAG_LEN_MIN, max_diag)

    if dominant_rows:
        max_straight = min(6, h - diag_len - 2)
        if max_straight < STRAIGHT_LEN_MIN:
            return None
        straight_len = random.randint(STRAIGHT_LEN_MIN, max_straight)
        axis = (sr * straight_len, 0)
    else:
        max_straight = min(6, w - diag_len - 2)
        if max_straight < STRAIGHT_LEN_MIN:
            return None
        straight_len = random.randint(STRAIGHT_LEN_MIN, max_straight)
        axis = (0, sc * straight_len)

    diag = (sr * diag_len, sc * diag_len)
    return diag, axis


def _is_valid_example(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    markers = np.argwhere(grid == RED)
    p = tuple(map(int, markers[0]))
    q = tuple(map(int, markers[1]))
    diag, axis = _decompose_offset(p, q)
    visible = _visible_translation_indices(grid.shape, p, diag, axis)
    if len(visible) < 3:
        return False

    green_count = int(np.count_nonzero(out == GREEN))
    return green_count >= 6 and not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    shape = _random_dimensions()
    vectors = _random_vectors(shape)
    if vectors is None:
        return None
    diag, axis = vectors

    starts = _candidate_starts(shape, diag, axis)
    if not starts:
        return None

    p = random.choice(starts)
    q = (int(p[0] + diag[0] + axis[0]), int(p[1] + diag[1] + axis[1]))
    grid = _make_input_grid(shape, p, q)
    if _is_valid_example(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
