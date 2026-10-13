from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 28
LINE_MARGIN = 4
TRACK_START = 2
TRACK_END_MARGIN = 3
MIN_GAP = 3
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if _in_bounds(shape, rr, cc):
            out.append((rr, cc))
    return out


def _non_bg_degree(grid: np.ndarray, bg: int, cell: tuple[int, int]) -> int:
    r, c = cell
    return sum(int(grid[rr, cc]) != bg for rr, cc in _neighbors4(grid.shape, r, c))


def _find_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    height, width = map(int, grid.shape)

    for r in range(height - 1):
        for c in range(width - 1):
            color = int(grid[r, c])
            if color == bg:
                continue
            patch = grid[r : r + 2, c : c + 2]
            if not bool(np.all(patch == color)):
                continue

            cells = frozenset({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)})
            if cells in seen:
                continue
            seen.add(cells)

            anchors: list[tuple[int, int]] = []
            for rr, cc in cells:
                if any(int(grid[nr, nc]) != bg and (nr, nc) not in cells for nr, nc in _neighbors4(grid.shape, rr, cc)):
                    anchors.append((rr, cc))

            assert len(anchors) == 1, "each 2x2 piece must touch the track at exactly one anchor cell"
            anchor = anchors[0]
            offsets = sorted((int(rr - anchor[0]), int(cc - anchor[1])) for rr, cc in cells)
            pieces.append({"color": color, "cells": cells, "anchor": anchor, "offsets": offsets})

    colors = [int(piece["color"]) for piece in pieces]
    assert pieces and len(colors) == len(set(colors)), "expected one source piece per color"
    return pieces


def _strip_source_pieces(grid: np.ndarray, pieces: list[dict], bg: int) -> np.ndarray:
    track = np.array(grid, dtype=int, copy=True)
    for piece in pieces:
        for r, c in piece["cells"]:
            track[r, c] = bg
        ar, ac = piece["anchor"]
        track[ar, ac] = int(piece["color"])
    return track


def _find_marker(track: np.ndarray, bg: int, color: int, anchor: tuple[int, int]) -> tuple[int, int]:
    endpoints: list[tuple[int, int]] = []
    for r, c in np.argwhere(track == color):
        cell = (int(r), int(c))
        if _non_bg_degree(track, bg, cell) == 1:
            endpoints.append(cell)
    assert anchor in endpoints and len(endpoints) == 2, "each color needs exactly one source endpoint and one destination marker"
    return endpoints[0] if endpoints[1] == anchor else endpoints[1]


def _successor_states(
    track: np.ndarray,
    bg: int,
    color: int,
    prev: tuple[int, int],
    cur: tuple[int, int],
) -> list[tuple[int, int]]:
    neighbors = [cell for cell in _neighbors4(track.shape, cur[0], cur[1]) if int(track[cell]) != bg and cell != prev]
    if not neighbors:
        return []

    dr = int(cur[0] - prev[0])
    dc = int(cur[1] - prev[1])
    forward = (int(cur[0] + dr), int(cur[1] + dc))
    orth = [cell for cell in neighbors if cell != forward]

    if orth and int(track[cur]) == color:
        return sorted(orth)
    if forward in neighbors:
        return [forward]
    if len(neighbors) == 1:
        return neighbors
    return []


def _trace_route(
    track: np.ndarray,
    bg: int,
    color: int,
    anchor: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    first_steps = [cell for cell in _neighbors4(track.shape, anchor[0], anchor[1]) if int(track[cell]) != bg]
    assert len(first_steps) == 1, "source anchor must meet the track at exactly one neighboring cell"
    start_state = (anchor, first_steps[0])

    queue = deque([start_state])
    parent: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]] | None] = {
        start_state: None
    }
    goal_state: tuple[tuple[int, int], tuple[int, int]] | None = None

    while queue:
        prev, cur = queue.popleft()
        if cur == target and _non_bg_degree(track, bg, cur) == 1:
            goal_state = (prev, cur)
            break

        for nxt in _successor_states(track, bg, color, prev, cur):
            state = (cur, nxt)
            if state in parent:
                continue
            parent[state] = (prev, cur)
            queue.append(state)

    assert goal_state is not None, "failed to find a routed path to the destination marker"

    route = [goal_state[1], goal_state[0]]
    state = parent[goal_state]
    while state is not None:
        route.append(state[0])
        state = parent[state]
    route.reverse()
    assert route[0] == anchor and route[-1] == target
    return route


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored 2x2 piece through the crossed track network until it lands on its routed endpoint marker.

    Input structure:
    - The background is the dominant color.
    - Several orthogonal one-pixel-wide colored tracks run horizontally and vertically, and different track colors can cross on one shared cell.
    - Each moving color has one solid 2x2 source piece attached to one track endpoint and one same-colored single-pixel destination marker at a reachable endpoint.

    Rule:
    - Treat every non-background cell as part of the track network, but first remove each 2x2 source piece back to its single anchor endpoint cell.
    - For each piece color, start from that anchor and follow the track direction. At a crossing, a piece turns onto the perpendicular track exactly when the shared crossing cell has the piece's own color; otherwise it continues straight through the crossing.
    - Continue until the path reaches the other degree-1 endpoint of that color, which is the destination marker.
    - Leave all track pixels unchanged, erase the source overhang of the piece, and stamp the same 2x2 piece shape around the destination marker.

    Key insight:
    - The crossing color is a routing switch: a piece reacts to the color painted on the shared crossing cell, not to the color of the track segment it is currently traveling along.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    pieces = _find_pieces(g, bg)
    track = _strip_source_pieces(g, pieces, bg)
    out = np.array(track, dtype=int, copy=True)

    for piece in pieces:
        color = int(piece["color"])
        anchor = piece["anchor"]
        target = _find_marker(track, bg, color, anchor)
        _trace_route(track, bg, color, anchor, target)

        for dr, dc in piece["offsets"]:
            rr = int(target[0] + dr)
            cc = int(target[1] + dc)
            assert _in_bounds(track.shape, rr, cc), "destination piece must stay inside the grid"
            out[rr, cc] = color

    return out


H_SOURCE_OFFSETS = (
    ((0, 0), (0, -1), (1, 0), (1, -1)),
    ((0, 0), (0, -1), (-1, 0), (-1, -1)),
)
V_SOURCE_OFFSETS = (
    ((0, 0), (-1, 0), (0, 1), (-1, 1)),
    ((0, 0), (-1, 0), (0, -1), (-1, -1)),
)


def _sample_spaced(low: int, high: int, count: int, gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(120):
        picks = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _piece_cells(anchor: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(int(ar + dr), int(ac + dc)) for dr, dc in offsets]


def _transform(array: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return np.array(array, dtype=int, copy=True)
    if code == 1:
        return np.flipud(array)
    if code == 2:
        return np.fliplr(array)
    if code == 3:
        return np.flipud(np.fliplr(array))
    if code == 4:
        return np.transpose(array)
    if code == 5:
        return np.flipud(np.transpose(array))
    if code == 6:
        return np.fliplr(np.transpose(array))
    if code == 7:
        return np.flipud(np.fliplr(np.transpose(array)))
    raise AssertionError(f"unknown transform code: {code}")


def _build_core(
    height: int,
    width: int,
    bg: int,
    h_rows: list[int],
    v_cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    right = width - TRACK_END_MARGIN
    bottom = height - TRACK_END_MARGIN

    for row, color in zip(h_rows, h_colors):
        grid[row, TRACK_START : right + 1] = color
    for col, color in zip(v_cols, v_colors):
        grid[TRACK_START : bottom + 1, col] = color

    cross_colors = (
        (h_colors[0], h_colors[0], v_colors[2]),
        (v_colors[0], h_colors[1], v_colors[2]),
        (v_colors[0], v_colors[1], h_colors[2]),
    )
    for i, row in enumerate(h_rows):
        for j, col in enumerate(v_cols):
            grid[row, col] = int(cross_colors[i][j])

    right_markers = (v_colors[2], v_colors[0], v_colors[1])
    for row, color in zip(h_rows, right_markers):
        grid[row, right] = int(color)
    for col, color in zip(v_cols, h_colors):
        grid[bottom, col] = int(color)

    return grid


def _render_example() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    h_rows = _sample_spaced(LINE_MARGIN, height - LINE_MARGIN - 1, 3, MIN_GAP)
    v_cols = _sample_spaced(LINE_MARGIN, width - LINE_MARGIN - 1, 3, MIN_GAP)
    if h_rows is None or v_cols is None:
        return None

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], 6)
    h_colors = colors[:3]
    v_colors = colors[3:]
    core = _build_core(height, width, bg, h_rows, v_cols, h_colors, v_colors)
    inp = np.array(core, dtype=int, copy=True)
    out = np.array(core, dtype=int, copy=True)

    left = TRACK_START
    top = TRACK_START
    right = width - TRACK_END_MARGIN
    bottom = height - TRACK_END_MARGIN

    h_offsets: list[tuple[tuple[int, int], ...]] = [random.choice(H_SOURCE_OFFSETS) for _ in range(3)]
    v_offsets: list[tuple[tuple[int, int], ...]] = [random.choice(V_SOURCE_OFFSETS) for _ in range(3)]

    for row, color, offsets in zip(h_rows, h_colors, h_offsets):
        cells = _piece_cells((row, left), offsets)
        if not all(_in_bounds(inp.shape, r, c) for r, c in cells):
            return None
        _paint_cells(inp, cells, color)

    for col, color, offsets in zip(v_cols, v_colors, v_offsets):
        cells = _piece_cells((top, col), offsets)
        if not all(_in_bounds(inp.shape, r, c) for r, c in cells):
            return None
        _paint_cells(inp, cells, color)

    for col, color, offsets in zip(v_cols, h_colors, h_offsets):
        cells = _piece_cells((bottom, col), offsets)
        if not all(_in_bounds(out.shape, r, c) for r, c in cells):
            return None
        _paint_cells(out, cells, color)

    right_dest_rows = (h_rows[1], h_rows[2], h_rows[0])
    for row, color, offsets in zip(right_dest_rows, v_colors, v_offsets):
        cells = _piece_cells((row, right), offsets)
        if not all(_in_bounds(out.shape, r, c) for r, c in cells):
            return None
        _paint_cells(out, cells, color)

    code = random.randrange(8)
    inp = _transform(inp, code)
    out = _transform(out, code)
    return np.array(inp, dtype=int, copy=True), np.array(out, dtype=int, copy=True)


def _gen_attempt() -> np.ndarray | None:
    rendered = _render_example()
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        got = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
