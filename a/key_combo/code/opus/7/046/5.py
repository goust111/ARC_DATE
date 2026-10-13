from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4)
GREY = 5

SEED_DIMENSIONS = ((2, 2), (2, 3), (3, 2), (3, 3))
MARKER_COUNT_CHOICES = (2, 3)
LEFT_COL_REPEAT_RANGE = (3, 5)
ROW_REPEAT_RANGE = (3, 6)
RIGHT_COL_REPEAT_RANGE = (2, 5)
GENERATION_RETRIES = 400


def _find_separator_col(grid: np.ndarray) -> int:
    sep_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert len(sep_cols) == 1, "expected exactly one solid grey separator column"
    sep = int(sep_cols[0])
    assert 0 < sep < grid.shape[1] - 1
    return sep


def _smallest_row_period(pattern: np.ndarray) -> int:
    h = pattern.shape[0]
    for period in range(1, h + 1):
        if np.array_equal(pattern, pattern[np.arange(h) % period, :]):
            return period
    raise AssertionError("left zone must have a row period")


def _smallest_col_period(pattern: np.ndarray) -> int:
    w = pattern.shape[1]
    for period in range(1, w + 1):
        if np.array_equal(pattern, pattern[:, np.arange(w) % period]):
            return period
    raise AssertionError("left zone must have a column period")


def _extract_seed_tile(left_zone: np.ndarray) -> np.ndarray:
    seed_h = _smallest_row_period(left_zone)
    seed_w = _smallest_col_period(left_zone)
    seed = left_zone[:seed_h, :seed_w].copy()
    assert np.any(seed == BLUE), "seed tile must contain blue foreground"
    return seed


def _marker_cells(grid: np.ndarray, sep: int) -> list[tuple[int, int, int]]:
    right_zone = grid[:, sep + 1 :]
    coords = np.argwhere(np.isin(right_zone, MARKER_COLORS))
    markers = [
        (int(r), int(sep + 1 + c), int(right_zone[r, c]))
        for r, c in coords
    ]
    assert markers, "expected at least one marker pixel in the right zone"
    return markers


def _stamp_seed(
    out: np.ndarray, seed: np.ndarray, center_r: int, center_c: int, color: int, sep: int
) -> None:
    seed_h, seed_w = seed.shape
    anchor_r = seed_h // 2
    anchor_c = seed_w // 2

    for tile_r, tile_c in np.argwhere(seed == BLUE):
        rr = int(center_r + tile_r - anchor_r)
        cc = int(center_c + tile_c - anchor_c)
        if 0 <= rr < out.shape[0] and sep < cc < out.shape[1] and out[rr, cc] == BLACK:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the repeated left seed tile and stamp it around each right-side marker in that marker's color.

    Input structure:
    - A black-background grid is split by one solid grey vertical separator column.
    - The left zone is tiled by repetitions of a small blue-and-black seed pattern whose unshifted copy is at the upper-left corner.
    - The right zone contains isolated single-pixel markers colored red, green, or yellow.

    Rule:
    - Find the smallest row and column periods that exactly reproduce the left zone, then extract that top-left seed tile.
    - Treat the blue cells of the seed tile as a stamp whose anchor is at floor(height/2), floor(width/2).
    - For each marker in the right zone, place the stamp centered on the marker and recolor only black target cells to the marker's color, preserving the left zone, the separator, and all existing non-black cells.

    Key insight:
    - The stamp shape is hidden as the minimal repeating tile on the left; the right-side markers contribute only the paint color and placement anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _find_separator_col(g)
    left_zone = g[:, :sep]
    seed = _extract_seed_tile(left_zone)
    markers = _marker_cells(g, sep)

    out = g.copy()
    for r, c, color in markers:
        _stamp_seed(out, seed, r, c, color, sep)
    return out


def _make_seed_tile(seed_h: int, seed_w: int) -> np.ndarray | None:
    coords = [(r, c) for r in range(seed_h) for c in range(seed_w)]
    max_blue = min(4, seed_h * seed_w - 1)
    if max_blue < 2:
        return None

    for _ in range(100):
        blue_count = random.randint(2, max_blue)
        blue_cells = random.sample(coords, blue_count)
        seed = np.zeros((seed_h, seed_w), dtype=int)
        for r, c in blue_cells:
            seed[r, c] = BLUE
        if (
            _smallest_row_period(seed) == seed_h
            and _smallest_col_period(seed) == seed_w
        ):
            return seed
    return None


def _choose_layout(seed_h: int, seed_w: int, marker_count: int) -> tuple[int, int, int, int] | None:
    for _ in range(100):
        row_repeats = random.randint(max(ROW_REPEAT_RANGE[0], marker_count), ROW_REPEAT_RANGE[1])
        left_col_repeats = random.randint(*LEFT_COL_REPEAT_RANGE)
        right_min = max(RIGHT_COL_REPEAT_RANGE[0], marker_count)
        right_col_repeats = random.randint(right_min, RIGHT_COL_REPEAT_RANGE[1])

        height = seed_h * row_repeats
        left_width = seed_w * left_col_repeats
        right_width = seed_w * right_col_repeats
        total_width = left_width + 1 + right_width
        if height <= 30 and total_width <= 30:
            return height, left_width, right_width, row_repeats
    return None


def _place_markers(
    grid: np.ndarray, left_width: int, seed_h: int, seed_w: int, marker_count: int
) -> list[tuple[int, int, int]]:
    height, _width = grid.shape
    row_slots = list(range(0, height // seed_h))
    col_slots = list(range(0, (grid.shape[1] - left_width - 1) // seed_w))
    assert len(row_slots) >= marker_count
    assert len(col_slots) >= marker_count

    chosen_rows = random.sample(row_slots, marker_count)
    chosen_cols = random.sample(col_slots, marker_count)
    random.shuffle(chosen_cols)
    colors = random.sample(list(MARKER_COLORS), marker_count)

    anchor_r = seed_h // 2
    anchor_c = seed_w // 2
    markers: list[tuple[int, int, int]] = []
    for row_slot, col_slot, color in zip(chosen_rows, chosen_cols, colors):
        r = row_slot * seed_h + anchor_r
        c = left_width + 1 + col_slot * seed_w + anchor_c
        markers.append((r, c, color))
        grid[r, c] = color
    return markers


def _expected_output(inp: np.ndarray, seed: np.ndarray) -> np.ndarray:
    sep = _find_separator_col(inp)
    out = inp.copy()
    for r, c, color in _marker_cells(inp, sep):
        _stamp_seed(out, seed, r, c, color, sep)
    return out


def _gen_attempt() -> np.ndarray | None:
    seed_h, seed_w = random.choice(SEED_DIMENSIONS)
    marker_count = random.choice(MARKER_COUNT_CHOICES)
    layout = _choose_layout(seed_h, seed_w, marker_count)
    if layout is None:
        return None

    height, left_width, right_width, _row_repeats = layout
    seed = _make_seed_tile(seed_h, seed_w)
    if seed is None:
        return None

    left_zone = np.tile(seed, (height // seed_h, left_width // seed_w))
    grid = np.zeros((height, left_width + 1 + right_width), dtype=int)
    grid[:, :left_width] = left_zone
    grid[:, left_width] = GREY
    _place_markers(grid, left_width, seed_h, seed_w, marker_count)

    if not np.array_equal(_extract_seed_tile(grid[:, :left_width]), seed):
        return None
    expected = _expected_output(grid, seed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
