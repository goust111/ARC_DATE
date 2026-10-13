from __future__ import annotations

from typing import Iterable

import numpy as np


Coord = tuple[int, int]

BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
START_SQUARE_OFFSETS: dict[int, tuple[Coord, Coord]] = {
    0: ((1, 0), (1, -1)),    # first segment goes up, square sits below
    1: ((0, -2), (-1, -2)),  # first segment goes right, square sits left
    2: ((-2, 0), (-2, -1)),  # first segment goes down, square sits above
    3: ((0, 1), (-1, 1)),    # first segment goes left, square sits right
}

GRID_BASE_RANGE = (14, 31)
GRID_EXTRA_PER_OBJECT = 4
PATH_COUNT_OPTIONS = (1, 2, 3)
TURN_RANGE = (2, 6)
SEGMENT_CELL_RANGE = (2, 6)
PLACEMENT_TRIES = 120
SHAPE_TRIES = 200
GENERATION_TRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(cell: Coord) -> tuple[Coord, Coord, Coord, Coord]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _all_2x2_blocks(cells: set[Coord]) -> list[Coord]:
    blocks: list[Coord] = []
    for r, c in cells:
        if (r + 1, c) in cells and (r, c + 1) in cells and (r + 1, c + 1) in cells:
            blocks.append((r, c))
    return sorted(set(blocks))


def _square_from_endpoint(end: Coord, direction: Coord, size: int) -> set[Coord]:
    if size <= 0:
        return set()

    er, ec = end
    if direction == (0, 1):
        r0, c0 = er, ec
    elif direction == (0, -1):
        r0, c0 = er - size + 1, ec - size + 1
    elif direction == (1, 0):
        r0, c0 = er, ec - size + 1
    elif direction == (-1, 0):
        r0, c0 = er - size + 1, ec
    else:
        raise AssertionError("invalid direction")

    return {
        (r, c)
        for r in range(r0, r0 + size)
        for c in range(c0, c0 + size)
    }


def _trace_colored_path(cells: Iterable[Coord]) -> dict[str, object]:
    cell_set = set(cells)
    blocks = _all_2x2_blocks(cell_set)
    assert len(blocks) == 1, "expected exactly one 2x2 start marker"

    sr0, sc0 = blocks[0]
    start_square = {
        (sr0, sc0),
        (sr0 + 1, sc0),
        (sr0, sc0 + 1),
        (sr0 + 1, sc0 + 1),
    }
    path_cells = cell_set - start_square
    assert path_cells, "expected a non-empty path outside the start square"

    adjacent_to_square = [
        cell
        for cell in path_cells
        if any(neighbor in start_square for neighbor in _neighbors4(cell))
    ]
    assert len(adjacent_to_square) == 1, "expected exactly one path cell touching the start square"
    start = adjacent_to_square[0]

    square_contacts = [neighbor for neighbor in _neighbors4(start) if neighbor in start_square]
    assert len(square_contacts) == 1, "path must touch a single square corner"
    start_dir = (start[0] - square_contacts[0][0], start[1] - square_contacts[0][1])

    degrees = {
        cell: sum(neighbor in path_cells for neighbor in _neighbors4(cell))
        for cell in path_cells
    }
    assert all(degree <= 2 for degree in degrees.values()), "path cannot branch"

    if len(path_cells) == 1:
        ordered = [start]
    else:
        endpoints = [cell for cell, degree in degrees.items() if degree == 1]
        assert len(endpoints) == 2, "path should have exactly two endpoints"
        assert start in endpoints, "start-adjacent pixel must be one path endpoint"

        ordered = [start]
        prev: Coord | None = None
        cur = start
        while True:
            next_cells = [
                neighbor
                for neighbor in _neighbors4(cur)
                if neighbor in path_cells and neighbor != prev
            ]
            if not next_cells:
                break
            assert len(next_cells) == 1, "path traversal must stay unambiguous"
            prev, cur = cur, next_cells[0]
            ordered.append(cur)
        assert len(ordered) == len(path_cells), "path traversal missed cells"

    if len(ordered) == 1:
        step_dirs: list[Coord] = []
        last_dir = start_dir
    else:
        step_dirs = [
            (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
            for i in range(len(ordered) - 1)
        ]
        assert all(direction in DIRS for direction in step_dirs)
        last_dir = step_dirs[-1]

    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, len(step_dirs)))
    return {
        "start_square": start_square,
        "path": ordered,
        "turns": turns,
        "last_dir": last_dir,
        "end": ordered[-1],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an endpoint square on each colored orthogonal path using that path's turn count.

    Input structure:
    - The background is black.
    - Each foreground color forms one 4-connected object made of a 2x2 solid start marker
      and a one-pixel-wide orthogonal polyline attached to one corner of that marker.
    - The path does not branch; its far end is a single-pixel endpoint, and different
      colored paths do not touch or cross.

    Rule:
    - For each colored object, locate the unique 2x2 start marker and trace the attached
      path from the marker-adjacent endpoint to the opposite endpoint.
    - Count how many times the step direction changes between horizontal and vertical
      while following the path.
    - At the path's terminal endpoint, add a same-colored filled square whose side length
      equals that turn count; the endpoint acts as the corner determined by the final path
      direction: right -> top-left, left -> bottom-right, down -> top-right, up -> bottom-left.
    - Keep the original paths and start markers unchanged.

    Key insight:
    - The 2x2 block identifies which end is the start, so the important hidden attribute is
      the ordered path itself: only by tracing it can you recover both the turn count and the
      direction of the final segment, which together determine the endpoint square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    out = g.copy()
    for color in sorted(int(value) for value in np.unique(g) if int(value) != bg):
        cells = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        info = _trace_colored_path(cells)
        turns = int(info["turns"])
        if turns <= 0:
            continue

        square = _square_from_endpoint(
            info["end"],
            info["last_dir"],
            turns,
        )
        for r, c in square:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "grown square must stay in bounds"
            out[r, c] = color

    return out


def _build_random_path_shape() -> dict[str, object] | None:
    turns = int(np.random.randint(*TURN_RANGE))
    first_dir = int(np.random.randint(0, 4))

    directions = [first_dir]
    for _ in range(turns):
        spin = int(np.random.choice((-1, 1)))
        directions.append((directions[-1] + spin) % 4)

    segment_cells = [
        int(np.random.randint(*SEGMENT_CELL_RANGE))
        for _ in range(turns + 1)
    ]

    path: list[Coord] = [(0, 0)]
    cur_r, cur_c = 0, 0
    for direction, count in zip(directions, segment_cells):
        dr, dc = DIRS[direction]
        for _ in range(count - 1):
            cur_r += dr
            cur_c += dc
            path.append((cur_r, cur_c))

    path_set = set(path)
    if len(path_set) != len(path):
        return None

    index = {cell: idx for idx, cell in enumerate(path)}
    for idx, cell in enumerate(path):
        for neighbor in _neighbors4(cell):
            if neighbor not in path_set:
                continue
            if abs(index[neighbor] - idx) != 1:
                return None

    square_offset = START_SQUARE_OFFSETS[first_dir][int(np.random.randint(0, 2))]
    sr0 = path[0][0] + square_offset[0]
    sc0 = path[0][1] + square_offset[1]
    start_square = {
        (sr0, sc0),
        (sr0 + 1, sc0),
        (sr0, sc0 + 1),
        (sr0 + 1, sc0 + 1),
    }
    if start_square & path_set:
        return None

    square_neighbors = {
        cell
        for cell in path_set
        if any(neighbor in start_square for neighbor in _neighbors4(cell))
    }
    if square_neighbors != {path[0]}:
        return None
    if sum(cell in start_square for cell in _neighbors4(path[0])) != 1:
        return None

    future_square = _square_from_endpoint(path[-1], DIRS[directions[-1]], turns)
    if not future_square:
        return None
    if future_square & start_square:
        return None
    overlap_with_path = future_square & path_set
    if overlap_with_path != {path[-1]}:
        return None

    input_cells = start_square | path_set
    input_without_end = input_cells - {path[-1]}
    for cell in future_square:
        if cell == path[-1]:
            continue
        for neighbor in _neighbors4(cell):
            if neighbor in input_without_end:
                return None

    return {
        "turns": turns,
        "start_square": start_square,
        "path": path,
        "input_cells": input_cells,
        "future_square": future_square,
        "reserved": input_cells | future_square,
    }


def _shift_cells(cells: Iterable[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def _bounds(cells: Iterable[Coord]) -> tuple[int, int, int, int]:
    points = list(cells)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mark_blocked(blocked: np.ndarray, cells: Iterable[Coord]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        blocked[r, c] = True
        for nr, nc in _neighbors4((r, c)):
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_shape(
    shape: dict[str, object],
    blocked: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[set[Coord], set[Coord]] | None:
    h, w = grid_shape
    reserved = set(shape["reserved"])
    rmin, cmin, rmax, cmax = _bounds(reserved)

    min_dr = -rmin
    max_dr = h - 1 - rmax
    min_dc = -cmin
    max_dc = w - 1 - cmax
    if min_dr > max_dr or min_dc > max_dc:
        return None

    for _ in range(PLACEMENT_TRIES):
        dr = int(np.random.randint(min_dr, max_dr + 1))
        dc = int(np.random.randint(min_dc, max_dc + 1))
        shifted_reserved = _shift_cells(reserved, dr, dc)
        if any(blocked[r, c] for r, c in shifted_reserved):
            continue

        shifted_input = _shift_cells(shape["input_cells"], dr, dc)
        shifted_future = _shift_cells(shape["future_square"], dr, dc)
        return shifted_input, shifted_future
    return None


def _gen_attempt() -> np.ndarray | None:
    path_count = int(np.random.choice(PATH_COUNT_OPTIONS))
    min_dim = GRID_BASE_RANGE[0] + GRID_EXTRA_PER_OBJECT * (path_count - 1)
    h = int(np.random.randint(min_dim, GRID_BASE_RANGE[1]))
    w = int(np.random.randint(min_dim, GRID_BASE_RANGE[1]))

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    colors = [int(color) for color in np.random.choice(COLOR_POOL, size=path_count, replace=False)]

    for color in colors:
        placed = False
        for _ in range(SHAPE_TRIES):
            shape = _build_random_path_shape()
            if shape is None:
                continue
            placed_cells = _place_shape(shape, blocked, grid.shape)
            if placed_cells is None:
                continue

            input_cells, future_cells = placed_cells
            for r, c in input_cells:
                grid[r, c] = color
            _mark_blocked(blocked, input_cells | future_cells)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 3")
