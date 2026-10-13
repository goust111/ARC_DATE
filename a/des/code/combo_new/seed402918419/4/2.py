from __future__ import annotations

import random

import numpy as np


BG = 0
START_COLOR = 2
STRAIGHT_COLOR = 3
LEFT_COLOR = 1
RIGHT_COLOR = 4
SEGMENT_COLORS = (5, 6, 7, 8, 9)

GRID_H_RANGE = (12, 28)
GRID_W_RANGE = (12, 28)
SEGMENT_COUNT_RANGE = (3, 5)
SEGMENT_LEN_RANGE = (2, 5)
GENERATION_ATTEMPTS = 400

DIR_FROM_SIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

TURN_OPTIONS = ("straight", "left", "right")


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (int(a[0] + b[0]), int(a[1] + b[1]))


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _right_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _apply_turn(direction: tuple[int, int], turn: str) -> tuple[int, int]:
    if turn == "straight":
        return direction
    if turn == "left":
        return _left_of(direction)
    if turn == "right":
        return _right_of(direction)
    raise AssertionError(f"unknown turn {turn}")


def _turn_color(prev_dir: tuple[int, int], curr_dir: tuple[int, int]) -> int:
    if curr_dir == prev_dir:
        return STRAIGHT_COLOR
    if curr_dir == _left_of(prev_dir):
        return LEFT_COLOR
    if curr_dir == _right_of(prev_dir):
        return RIGHT_COLOR
    raise AssertionError("segments must continue straight or turn 90 degrees")


def _in_bounds(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _segment_cells(
    entry: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(entry[0] + dr * k, entry[1] + dc * k) for k in range(length)]


def _parse_segment(color: int, coords: np.ndarray) -> dict:
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert 2 <= len(coords) <= 5

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])

    if len(rows) == 1:
        row = int(rows[0])
        ordered_cols = np.sort(coords[:, 1])
        assert np.array_equal(
            ordered_cols,
            np.arange(int(ordered_cols[0]), int(ordered_cols[-1]) + 1),
        )
        cells = [(row, int(c)) for c in ordered_cols]
    elif len(cols) == 1:
        col = int(cols[0])
        ordered_rows = np.sort(coords[:, 0])
        assert np.array_equal(
            ordered_rows,
            np.arange(int(ordered_rows[0]), int(ordered_rows[-1]) + 1),
        )
        cells = [(int(r), col) for r in ordered_rows]
    else:
        raise AssertionError("every segment must be a straight horizontal or vertical run")

    return {
        "color": int(color),
        "coords": np.asarray(cells, dtype=int),
        "endpoints": (cells[0], cells[-1]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect colored line segments into one snake path and recolor each later segment by how it turns.

    Input structure:
    - The background is black.
    - A single red 1x1 start marker sits on the grid edge.
    - Every other foreground color forms exactly one horizontal or vertical segment of length 2-5.
    - Consecutive segments in the hidden snake touch only by endpoint adjacency in the 4-neighborhood.

    Rule:
    - Find the unique segment endpoint adjacent to the red start marker; that segment is first.
    - Traverse from each segment's far endpoint to the next endpoint-adjacent segment until the whole snake order is recovered.
    - Remove the red start marker from the output.
    - Keep the first segment's original color; recolor every later segment green for straight, blue for a left turn, and yellow for a right turn relative to the previous segment's travel direction.

    Key insight:
    - The visible colors identify individual straight pieces, but the real task is reconstructing their hidden topological order first; only then can turn directions be decoded into output colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start_pixels = np.argwhere(g == START_COLOR)
    assert len(start_pixels) == 1, "expected one red start marker"
    start = tuple(map(int, start_pixels[0]))
    h, w = g.shape
    assert start[0] in {0, h - 1} or start[1] in {0, w - 1}, "start marker must lie on the edge"

    segment_colors = [int(color) for color in np.unique(g) if int(color) not in {BG, START_COLOR}]
    assert segment_colors, "expected at least one colored segment"

    segments: list[dict] = []
    for color in segment_colors:
        coords = np.argwhere(g == color)
        segments.append(_parse_segment(color, coords))

    links: dict[tuple[int, int], list[tuple[str, int, int]]] = {}
    for i, segment in enumerate(segments):
        for endpoint_index, endpoint in enumerate(segment["endpoints"]):
            connectors: list[tuple[str, int, int]] = []
            if _manhattan(endpoint, start) == 1:
                connectors.append(("start", -1, -1))
            for j, other in enumerate(segments):
                if j == i:
                    continue
                for other_endpoint_index, other_endpoint in enumerate(other["endpoints"]):
                    if _manhattan(endpoint, other_endpoint) == 1:
                        connectors.append(("segment", j, other_endpoint_index))
            assert len(connectors) <= 1, "each segment endpoint must have at most one external neighbor"
            links[(i, endpoint_index)] = connectors

    first_candidates = [
        (segment_index, endpoint_index)
        for (segment_index, endpoint_index), connectors in links.items()
        if connectors and connectors[0][0] == "start"
    ]
    assert len(first_candidates) == 1, "start marker must connect to exactly one segment endpoint"

    ordered_indices: list[int] = []
    ordered_dirs: list[tuple[int, int]] = []
    visited: set[int] = set()
    current_index, entry_endpoint_index = first_candidates[0]

    while True:
        assert current_index not in visited, "snake path must not revisit a segment"
        visited.add(current_index)
        ordered_indices.append(int(current_index))

        segment = segments[current_index]
        entry = segment["endpoints"][entry_endpoint_index]
        exit_endpoint_index = 1 - entry_endpoint_index
        exit_point = segment["endpoints"][exit_endpoint_index]
        direction = (
            int(np.sign(exit_point[0] - entry[0])),
            int(np.sign(exit_point[1] - entry[1])),
        )
        assert abs(direction[0]) + abs(direction[1]) == 1
        ordered_dirs.append(direction)

        connectors = links[(current_index, exit_endpoint_index)]
        if not connectors:
            break

        kind, next_index, next_endpoint_index = connectors[0]
        assert kind == "segment", "only the first endpoint may connect to the start marker"
        current_index = int(next_index)
        entry_endpoint_index = int(next_endpoint_index)

    assert len(visited) == len(segments), "all segments must belong to the same snake"

    out = np.full_like(g, BG)
    for order_index, segment_index in enumerate(ordered_indices):
        if order_index == 0:
            paint_color = int(segments[segment_index]["color"])
        else:
            paint_color = _turn_color(ordered_dirs[order_index - 1], ordered_dirs[order_index])
        coords = segments[segment_index]["coords"]
        out[coords[:, 0], coords[:, 1]] = paint_color

    return out


def _sample_start(h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return (0, random.randint(1, w - 2))
    if side == "bottom":
        return (h - 1, random.randint(1, w - 2))
    if side == "left":
        return (random.randint(1, h - 2), 0)
    if side == "right":
        return (random.randint(1, h - 2), w - 1)
    raise AssertionError(f"unknown side {side}")


def _valid_new_segment(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    allowed_neighbor: tuple[int, int],
    h: int,
    w: int,
) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    if any(not _in_bounds(cell, h, w) for cell in cells):
        return False
    if any(cell in occupied for cell in cells):
        return False

    for index, cell in enumerate(cells):
        for neighbor in _neighbors4(cell):
            if neighbor in occupied and not (index == 0 and neighbor == allowed_neighbor):
                return False

    return True


def _build_segments(
    h: int,
    w: int,
    start: tuple[int, int],
    directions: list[tuple[int, int]],
    colors: list[int],
) -> list[dict] | None:
    occupied = {start}
    segments: list[dict] = []

    def rec(index: int, prev_exit: tuple[int, int]) -> bool:
        if index == len(directions):
            return True

        direction = directions[index]
        entry = _add(start, direction) if index == 0 else _add(prev_exit, direction)
        allowed_neighbor = start if index == 0 else prev_exit

        lengths = list(range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1))
        random.shuffle(lengths)
        for length in lengths:
            cells = _segment_cells(entry, direction, length)
            if not _valid_new_segment(cells, occupied, allowed_neighbor, h, w):
                continue

            occupied.update(cells)
            segments.append(
                {
                    "color": int(colors[index]),
                    "cells": list(cells),
                    "dir": direction,
                }
            )
            if rec(index + 1, cells[-1]):
                return True
            segments.pop()
            for cell in cells:
                occupied.remove(cell)

        return False

    if rec(0, start):
        return segments
    return None


def _render_input(h: int, w: int, start: tuple[int, int], segments: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[start] = START_COLOR
    for segment in segments:
        for r, c in segment["cells"]:
            grid[r, c] = int(segment["color"])
    return grid


def _render_expected_output(h: int, w: int, segments: list[dict]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for index, segment in enumerate(segments):
        if index == 0:
            color = int(segment["color"])
        else:
            color = _turn_color(segments[index - 1]["dir"], segment["dir"])
        for r, c in segment["cells"]:
            out[r, c] = color
    return out


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    segment_count = random.randint(*SEGMENT_COUNT_RANGE)
    side = random.choice(tuple(DIR_FROM_SIDE))
    start = _sample_start(h, w, side)

    turns = [random.choice(TURN_OPTIONS) for _ in range(segment_count - 1)]
    directions = [DIR_FROM_SIDE[side]]
    for turn in turns:
        directions.append(_apply_turn(directions[-1], turn))

    colors = random.sample(list(SEGMENT_COLORS), segment_count)
    segments = _build_segments(h, w, start, directions, colors)
    if segments is None:
        return None

    inp = _render_input(h, w, start, segments)
    expected = _render_expected_output(h, w, segments)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-segment ARC input")
