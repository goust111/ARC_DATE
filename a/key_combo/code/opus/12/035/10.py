from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
MARKER_PAIR = (PINK, GREY)
HEADER_COLOR_POOL = (1, 2, 3, 4, 7, 8, 9)

GRID_SIDE_RANGE = (4, 6)
ACTIVE_HEADER_RANGE = (3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _diag_fill(header: np.ndarray) -> np.ndarray:
    size = int(header.size)
    idx = np.maximum.outer(np.arange(size), np.arange(size))
    return header[idx]


def _extract_active_header(top_row: np.ndarray) -> np.ndarray:
    blank_positions = np.flatnonzero(top_row == BLACK)
    assert blank_positions.size > 0, "expected a trailing black suffix in the header"
    first_blank = int(blank_positions[0])
    assert first_blank >= ACTIVE_HEADER_RANGE[0], "active header is too short"
    assert first_blank <= ACTIVE_HEADER_RANGE[1], "active header is too long"
    assert np.all(top_row[first_blank:] == BLACK), "black header cells must form one trailing suffix"

    active = np.asarray(top_row[:first_blank], dtype=int)
    assert np.all(active != BLACK), "active header colors must be non-black"
    assert np.all(active != GREY) and np.all(active != PINK), "marker colors cannot appear in the header"
    assert len(set(map(int, active.tolist()))) >= 2, "active header must use at least two colors"
    return active


def _validate_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n_rows, n_cols = map(int, g.shape)
    assert n_rows == n_cols, "expected a square input"
    assert GRID_SIDE_RANGE[0] <= n_rows <= GRID_SIDE_RANGE[1], "unexpected input side length"
    assert n_cols >= 2, "grid is too small for the marker pair"

    top_row = g[0, :].copy()
    left_col = g[:, 0].copy()
    assert np.array_equal(top_row, left_col), "top row and left column must share the same header"

    active = _extract_active_header(top_row)

    assert tuple(map(int, g[-1, -2:].tolist())) == MARKER_PAIR, "missing bottom-right marker pair"
    assert int(np.count_nonzero(g == PINK)) == 1, "pink may only appear in the marker pair"
    assert int(np.count_nonzero(g == GREY)) == 1, "grey may only appear in the marker pair"

    expected = _diag_fill(top_row)
    mask = np.ones_like(g, dtype=bool)
    mask[-1, -2:] = False
    assert np.array_equal(g[mask], expected[mask]), "non-marker cells must follow the diagonal-index rule"

    return top_row, active


def _build_output(active_header: np.ndarray, input_side: int) -> np.ndarray:
    out_side = input_side * 2
    repeated = np.resize(np.asarray(active_header, dtype=int), out_side)
    out = _diag_fill(repeated)
    out[:, ::3] = np.where(out[:, ::3] == BLACK, BLACK, PINK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Double a diagonal-index header table and repaint every third column pink.

    Input structure:
    - The input is a square grid with side length 4-6.
    - The top row and left column share a header made of 3-4 non-black colors
      followed by a trailing black suffix.
    - Every non-marker cell obeys the diagonal-index rule
      `cell(r, c) = header[max(r, c)]`.
    - The last row's final two cells are a fixed special tag `[6, 5]` and are
      not part of the diagonal table.

    Rule:
    - Read the active non-black header prefix from the top row.
    - Repeat that prefix cyclically until it reaches length `2N`, where `N` is
      the input side length.
    - Fill a `2N x 2N` square with the same diagonal-index rule based on the
      repeated header.
    - In columns `0, 3, 6, ...`, recolor every non-black cell to pink(6).

    Key insight:
    - The output comes from two coupled structures: a one-dimensional header
      projected through `max(r, c)`, then a periodic every-third-column stripe
      override. The bottom-right `[6, 5]` tag is only a cue and must be ignored
      when reconstructing the header table.
    """
    header, active = _validate_input(grid)
    return _build_output(active, int(header.size))


def _sample_active_header(active_len: int) -> list[int]:
    assert ACTIVE_HEADER_RANGE[0] <= active_len <= ACTIVE_HEADER_RANGE[1]
    if random.random() < 0.5:
        colors = random.sample(HEADER_COLOR_POOL, active_len)
        random.shuffle(colors)
        return colors

    palette_size = random.randint(2, min(3, active_len))
    palette = random.sample(HEADER_COLOR_POOL, palette_size)
    while True:
        active = [random.choice(palette) for _ in range(active_len)]
        if len(set(active)) >= 2 and len(set(active)) < active_len:
            return active


def _expected_input_from_header(header: np.ndarray) -> np.ndarray:
    grid = _diag_fill(header)
    grid[-1, -2:] = np.array(MARKER_PAIR, dtype=int)
    return grid


def _attempt_generate() -> np.ndarray | None:
    n = random.randint(*GRID_SIDE_RANGE)
    max_active = min(ACTIVE_HEADER_RANGE[1], n - 1)
    active_len = random.randint(ACTIVE_HEADER_RANGE[0], max_active)
    active = _sample_active_header(active_len)
    header = np.array(active + [BLACK] * (n - active_len), dtype=int)

    grid = _expected_input_from_header(header)
    expected = _build_output(np.array(active, dtype=int), n)

    if int(np.unique(expected).size) < 3:
        return None
    if not np.any(expected[:, ::3] == PINK):
        return None
    if np.any(expected[:, 1:] == GREY):
        return None
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
