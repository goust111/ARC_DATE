from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG_COLOR = 8
SOURCE_COLOR = 4
PATH_COLOR_POOL = [1, 2, 3, 5, 6, 7, 9]
ROOM_BORDER_OFFSETS = [
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bg_color(grid: np.ndarray) -> int:
    values = [int(v) for v in grid.ravel()]
    return Counter(values).most_common(1)[0][0]


def _find_rooms(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    h, w = grid.shape
    rooms: list[dict] = []
    border_to_room: dict[tuple[int, int], dict] = {}

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            block = grid[r0 : r0 + 3, c0 : c0 + 3]
            if int(block[1, 1]) != bg:
                continue
            border_vals = [int(block[dr, dc]) for dr, dc in ROOM_BORDER_OFFSETS]
            color = border_vals[0]
            if color in (bg, SOURCE_COLOR):
                continue
            if any(v != color for v in border_vals):
                continue

            border = {(r0 + dr, c0 + dc) for dr, dc in ROOM_BORDER_OFFSETS}
            if any(cell in border_to_room for cell in border):
                continue

            room = {
                "color": color,
                "center": (r0 + 1, c0 + 1),
                "border": border,
            }
            rooms.append(room)
            for cell in border:
                border_to_room[cell] = room

    return rooms, border_to_room


def _path_neighbors(grid: np.ndarray, pos: tuple[int, int], bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = pos
    result = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (bg, SOURCE_COLOR):
            result.append((nr, nc))
    return result


def _trace_signal(
    grid: np.ndarray,
    source: tuple[int, int],
    bg: int,
    border_to_room: dict[tuple[int, int], dict],
) -> tuple[dict, int]:
    h, w = grid.shape
    sr, sc = source

    starts = []
    for nr, nc in _neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (bg, SOURCE_COLOR):
            starts.append((nr, nc))
    assert len(starts) == 1, f"expected one path start next to source, got {starts}"

    prev = source
    cur = starts[0]
    signal = int(grid[cur])
    seen = {(prev, cur)}

    while True:
        room = border_to_room.get(cur)
        if room is not None and prev not in room["border"]:
            return room, signal

        nbrs = _path_neighbors(grid, cur, bg)
        if len(nbrs) == 4:
            cell_color = int(grid[cur])
            if cell_color != signal:
                signal = cell_color

            dr = cur[0] - prev[0]
            dc = cur[1] - prev[1]
            nxt = (cur[0] + dr, cur[1] + dc)
            assert nxt in nbrs, f"crossing must continue straight from {prev} through {cur}"
        else:
            options = [cell for cell in nbrs if cell != prev]
            assert len(options) == 1, f"expected one successor from {cur}, got {options}"
            nxt = options[0]

        state = (cur, nxt)
        assert state not in seen, f"loop detected while tracing from {source}"
        seen.add(state)
        prev, cur = cur, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each yellow source along its orthogonal path, recoloring the signal
    at visible four-way crossings, and fill the reached room center.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    rooms, border_to_room = _find_rooms(g, bg)
    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == SOURCE_COLOR)]

    assert rooms, "expected at least one room"
    assert sources, "expected at least one yellow source"
    assert len(rooms) == len(sources), "source and room counts must match"

    out = g.copy()
    used_centers: set[tuple[int, int]] = set()

    for source in sorted(sources):
        room, final_color = _trace_signal(g, source, bg, border_to_room)
        center = room["center"]
        assert center not in used_centers, f"multiple sources reached room {center}"
        used_centers.add(center)
        out[center] = final_color

    return out


def _ordered_path_cells(source_row: int, room_row: int, bend_col: int, room_col: int) -> list[tuple[int, int]]:
    cells = [(source_row, c) for c in range(1, bend_col + 1)]

    step = 1 if room_row > source_row else -1
    for r in range(source_row + step, room_row + step, step):
        cells.append((r, bend_col))

    for c in range(bend_col + 1, room_col + 1):
        cells.append((room_row, c))

    return cells


def _room_border(room_row: int, room_col: int) -> list[tuple[int, int]]:
    r0 = room_row - 1
    c0 = room_col
    return [(r0 + dr, c0 + dc) for dr, dc in ROOM_BORDER_OFFSETS]


def _simulate_final_color(path: dict, crossing_colors: dict[tuple[int, int], int]) -> int:
    signal = path["color"]
    for cell in path["cells"]:
        cross_color = crossing_colors.get(cell)
        if cross_color is not None and cross_color != signal:
            signal = cross_color
    return signal


def _build_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_paths = rng.choice([2, 3])
    h_min = 18 if n_paths == 3 else 15
    h = rng.randint(h_min, 27)
    w = rng.randint(18, 28)

    room_col = w - 3
    row_slots = list(range(1, h - 1, 3))
    bend_candidates = list(range(4, room_col - 1, 2))
    if len(row_slots) < 2 * n_paths or len(bend_candidates) < n_paths:
        return None

    chosen_rows = rng.sample(row_slots, 2 * n_paths)
    rng.shuffle(chosen_rows)
    source_rows = chosen_rows[:n_paths]
    room_rows = chosen_rows[n_paths:]

    bends = rng.sample(bend_candidates, n_paths)
    colors = rng.sample(PATH_COLOR_POOL, n_paths)

    paths = []
    for idx in range(n_paths):
        path = {
            "id": idx,
            "color": colors[idx],
            "source": (source_rows[idx], 0),
            "room_center": (room_rows[idx], room_col + 1),
            "room_row": room_rows[idx],
            "cells": _ordered_path_cells(source_rows[idx], room_rows[idx], bends[idx], room_col),
        }
        paths.append(path)

    coord_to_paths: dict[tuple[int, int], list[int]] = defaultdict(list)
    for path in paths:
        for cell in path["cells"]:
            coord_to_paths[cell].append(path["id"])

    if any(len(ids) > 2 for ids in coord_to_paths.values()):
        return None

    crossings = [cell for cell, ids in coord_to_paths.items() if len(ids) == 2]
    if not crossings:
        return None

    crossing_colors: dict[tuple[int, int], int] = {}
    for cell in crossings:
        a, b = coord_to_paths[cell]
        crossing_colors[cell] = rng.choice([paths[a]["color"], paths[b]["color"]])

    final_colors = []
    for path in paths:
        final_color = _simulate_final_color(path, crossing_colors)
        path["final_color"] = final_color
        final_colors.append(final_color)

    if not any(path["final_color"] != path["color"] for path in paths):
        return None
    if not any(path["final_color"] == path["color"] for path in paths):
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for path in paths:
        grid[path["source"]] = SOURCE_COLOR
        for cell in _room_border(path["room_row"], room_col):
            grid[cell] = path["color"]

    for cell, ids in coord_to_paths.items():
        if len(ids) == 1:
            grid[cell] = paths[ids[0]]["color"]
        else:
            grid[cell] = crossing_colors[cell]

    expected = grid.copy()
    for path in paths:
        expected[path["room_center"]] = path["final_color"]

    k = rng.randrange(4)
    if k:
        grid = np.rot90(grid, k)
        expected = np.rot90(expected, k)
    if rng.random() < 0.5:
        grid = np.fliplr(grid)
        expected = np.fliplr(expected)

    bg = _bg_color(grid)
    rooms, _ = _find_rooms(grid, bg)
    if len(rooms) != n_paths:
        return None
    if len(np.argwhere(grid == SOURCE_COLOR)) != n_paths:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    """
    Generate a random path-crossing puzzle with 2-3 colored paths.
    """
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt(rng)
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
