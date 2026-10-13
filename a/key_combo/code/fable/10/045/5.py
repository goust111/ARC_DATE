from __future__ import annotations

import random

import numpy as np


BG = 0
TURN_RIGHT_COLOR = 1
TURN_LEFT_COLOR = 2
SEED_COLOR = 3
MARKER_COLOR = 4

DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR_TO_INDEX = {direction: i for i, direction in enumerate(DIRS)}
TURN_BY_COLOR = {TURN_RIGHT_COLOR: 1, TURN_LEFT_COLOR: -1}

MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trace_program(
    lengths: list[int], turns: list[int], direction_index: int, start: tuple[int, int] = (0, 0)
) -> list[tuple[int, int]]:
    r, c = start
    path: list[tuple[int, int]] = []
    for length, turn in zip(lengths, turns):
        dr, dc = DIRS[direction_index]
        for _ in range(length):
            r += dr
            c += dc
            path.append((r, c))
        direction_index = (direction_index + turn) % 4
    return path


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path:
        out[r, c] = SEED_COLOR
    return out


def _bar_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] != bg)
        if cols.size == 0:
            continue
        colors = set(map(int, grid[r, cols]))
        if colors <= set(TURN_BY_COLOR):
            assert cols.size >= 2, "each bar must have length at least 2"
            assert len(colors) == 1, "each source row must contain exactly one bar color"
            assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), "bars must be contiguous"
            rows.append(int(r))
    assert rows, "expected at least one program bar row"
    assert all(b - a == 2 for a, b in zip(rows, rows[1:])), "bar rows must be staggered every other row"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """Read the left barcode as a turn program and draw the resulting polyline from the seed.

    Input structure:
    - The grid has a black background, a left-side list of solid horizontal bars on staggered rows,
      and a seed pixel in the open right-side area.
    - The bars use exactly two colors and encode a top-to-bottom program: bar length is the step count
      and bar color is the turn instruction.
    - One adjacent marker pixel touches the seed and gives the initial heading.

    Rule:
    - Start from the seed and face from the seed toward the marker.
    - Execute the bar rows from top to bottom: move forward by that bar's length while painting the traversed
      cells with the seed color, then rotate 90 degrees according to the bar color.
    - Keep the bar list unchanged; the marker is overwritten by the first painted step.

    Key insight:
    - The horizontal bars are not targets to copy: together with the seed and heading marker they form an
      executable drawing program whose lengths control segment sizes and whose colors control left/right turns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    seed_coords = np.argwhere(g == SEED_COLOR)
    marker_coords = np.argwhere(g == MARKER_COLOR)
    assert seed_coords.shape == (1, 2), "expected exactly one seed pixel"
    assert marker_coords.shape == (1, 2), "expected exactly one heading marker"
    seed_r, seed_c = map(int, seed_coords[0])
    marker_r, marker_c = map(int, marker_coords[0])

    heading = (marker_r - seed_r, marker_c - seed_c)
    assert heading in DIR_TO_INDEX, "heading marker must be 4-neighbor adjacent to the seed"
    direction_index = DIR_TO_INDEX[heading]

    rows = _bar_rows(g, bg)
    out = g.copy()
    r, c = seed_r, seed_c

    for row in rows:
        cols = np.flatnonzero(g[row] != bg)
        length = int(cols.size)
        turn_color = int(g[row, cols[0]])
        turn = TURN_BY_COLOR[turn_color]
        dr, dc = DIRS[direction_index]

        for _ in range(length):
            r += dr
            c += dc
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "program path left the canvas"
            cell = int(g[r, c])
            assert cell in (bg, SEED_COLOR, MARKER_COLOR), "program path must stay in the right-side empty area"
            out[r, c] = SEED_COLOR

        direction_index = (direction_index + turn) % 4

    return out


def _alt_path_set(
    lengths: list[int], turns: list[int], direction_index: int, seed_pos: tuple[int, int], grid: np.ndarray
) -> frozenset[tuple[int, int]] | None:
    rel_path = _trace_program(lengths, turns, direction_index)
    if len(set(rel_path)) != len(rel_path) or (0, 0) in rel_path:
        return None

    seed_r, seed_c = seed_pos
    abs_path = [(seed_r + dr, seed_c + dc) for dr, dc in rel_path]
    all_pts = [(seed_r, seed_c)] + abs_path
    rows = [r for r, _ in all_pts]
    cols = [c for _, c in all_pts]
    if min(rows) < 0 or max(rows) >= grid.shape[0] or min(cols) < 0 or max(cols) >= grid.shape[1]:
        return None

    for r, c in abs_path:
        if int(grid[r, c]) not in (BG, SEED_COLOR, MARKER_COLOR):
            return None
    return frozenset(abs_path)


def _sample_program() -> tuple[list[int], list[int], int] | None:
    bar_count = random.randint(4, 6)
    lengths = [random.randint(2, 6) for _ in range(bar_count)]
    if len(set(lengths)) < 2:
        return None

    turns = [random.choice([1, -1]) for _ in range(bar_count)]
    if 1 not in turns[:-1] or -1 not in turns[:-1]:
        return None

    direction_index = random.randrange(4)
    rel_path = _trace_program(lengths, turns, direction_index)
    rel_set = set(rel_path)
    if len(rel_set) != len(rel_path) or (0, 0) in rel_set:
        return None

    all_pts = [(0, 0)] + rel_path
    row_span = max(r for r, _ in all_pts) - min(r for r, _ in all_pts)
    col_span = max(c for _, c in all_pts) - min(c for _, c in all_pts)
    if row_span < 4 or col_span < 4:
        return None
    return lengths, turns, direction_index


def _build_input() -> np.ndarray | None:
    sampled = _sample_program()
    if sampled is None:
        return None
    lengths, turns, direction_index = sampled

    first_bar_row = random.choice([1, 2])
    bar_rows = [first_bar_row + 2 * i for i in range(len(lengths))]
    bar_starts = [random.randint(0, 2) for _ in lengths]
    max_bar_end = max(start + length - 1 for start, length in zip(bar_starts, lengths))

    rel_path = _trace_program(lengths, turns, direction_index)
    all_rel = [(0, 0)] + rel_path
    min_rel_r = min(r for r, _ in all_rel)
    max_rel_r = max(r for r, _ in all_rel)
    min_rel_c = min(c for _, c in all_rel)
    max_rel_c = max(c for _, c in all_rel)

    path_left = max_bar_end + random.randint(3, 5)
    seed_c = path_left - min_rel_c
    width = seed_c + max_rel_c + random.randint(2, 4) + 1
    height = max(bar_rows[-1] + 2, max_rel_r - min_rel_r + random.randint(3, 6))

    heading_dr, heading_dc = DIRS[direction_index]
    valid_seed_rows = [
        seed_r
        for seed_r in range(1 - min_rel_r, height - max_rel_r)
        if seed_r not in bar_rows and seed_r + heading_dr not in bar_rows
    ]
    if not valid_seed_rows:
        return None
    seed_r = random.choice(valid_seed_rows)

    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for row, start, length, turn in zip(bar_rows, bar_starts, lengths, turns):
        bar_color = TURN_RIGHT_COLOR if turn == 1 else TURN_LEFT_COLOR
        grid[row, start : start + length] = bar_color

    seed_pos = (seed_r, seed_c)
    marker_pos = (seed_r + heading_dr, seed_c + heading_dc)
    abs_path = [(seed_r + dr, seed_c + dc) for dr, dc in rel_path]
    all_abs = [seed_pos] + abs_path
    if min(r for r, _ in all_abs) < 0 or max(r for r, _ in all_abs) >= height:
        return None
    if min(c for _, c in all_abs) < 0 or max(c for _, c in all_abs) >= width:
        return None
    if abs_path[0] != marker_pos:
        return None
    if grid[seed_pos] != BG or grid[marker_pos] != BG:
        return None

    for r, c in abs_path:
        if int(grid[r, c]) != BG:
            return None

    grid[seed_pos] = SEED_COLOR
    grid[marker_pos] = MARKER_COLOR
    expected = _paint_path(grid, abs_path)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    correct_path = frozenset(abs_path)
    alternatives = [
        _alt_path_set(lengths, [-turn for turn in turns], direction_index, seed_pos, grid),
        _alt_path_set(lengths, turns, (direction_index + 2) % 4, seed_pos, grid),
        _alt_path_set(lengths[::-1], turns[::-1], direction_index, seed_pos, grid),
        _alt_path_set(lengths, [1] * len(turns), direction_index, seed_pos, grid),
        _alt_path_set(lengths, [-1] * len(turns), direction_index, seed_pos, grid),
    ]
    if any(alt == correct_path for alt in alternatives):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid barcode-polyline input")
