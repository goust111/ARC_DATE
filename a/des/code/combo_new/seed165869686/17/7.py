from __future__ import annotations

import random

import numpy as np


BG = 0
PIPE = 5
SOURCE_COLORS = (1, 2, 3)  # blue, red, green
EXIT_MARKER = 7  # stand-in for white, which ARC does not provide
MIX_COLORS = {
    frozenset((1, 2)): 6,  # blue + red -> magenta
    frozenset((2, 3)): 4,  # red + green -> yellow
    frozenset((1, 3)): 8,  # blue + green -> azure
}

GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 26
MIN_PIPE_LEN = 3
MAX_PIPE_LEN = 12
MIN_PIPE_COUNT = 2
MAX_PIPE_COUNT = 5
MAX_GENERATION_ATTEMPTS = 500
MODES = ("no_cross", "rb_cross", "rg_cross", "bg_cross")


def _role_of_endpoint(color: int, exit_colors: set[int]) -> str | None:
    if color in SOURCE_COLORS:
        return "source"
    if color in exit_colors:
        return "exit"
    return None


def _pipe_cells(pipe: dict) -> list[tuple[int, int]]:
    if pipe["orientation"] == "H":
        row = pipe["row"]
        return [(row, c) for c in range(pipe["start"], pipe["end"] + 1)]
    col = pipe["col"]
    return [(r, col) for r in range(pipe["start"], pipe["end"] + 1)]


def _pipe_endpoints(pipe: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    if pipe["orientation"] == "H":
        left = (pipe["row"], pipe["start"] - 1)
        right = (pipe["row"], pipe["end"] + 1)
        if pipe["source_side"] == "left":
            return left, right
        return right, left
    top = (pipe["start"] - 1, pipe["col"])
    bottom = (pipe["end"] + 1, pipe["col"])
    if pipe["source_side"] == "top":
        return top, bottom
    return bottom, top


def _build_pipe_from_run(
    grid: np.ndarray,
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    exit_colors: set[int],
) -> dict | None:
    height, width = grid.shape
    if orientation == "H":
        if start == 0 or end == width - 1:
            return None
        left = int(grid[fixed, start - 1])
        right = int(grid[fixed, end + 1])
        left_role = _role_of_endpoint(left, exit_colors)
        right_role = _role_of_endpoint(right, exit_colors)
        if {left_role, right_role} != {"source", "exit"}:
            return None
        if left_role == "source":
            source_pos = (fixed, start - 1)
            exit_pos = (fixed, end + 1)
            source_color = left
        else:
            source_pos = (fixed, end + 1)
            exit_pos = (fixed, start - 1)
            source_color = right
        return {
            "orientation": "H",
            "row": fixed,
            "start": start,
            "end": end,
            "color": int(source_color),
            "source_pos": source_pos,
            "exit_pos": exit_pos,
            "cells": [(fixed, c) for c in range(start, end + 1)],
        }

    if start == 0 or end == height - 1:
        return None
    top = int(grid[start - 1, fixed])
    bottom = int(grid[end + 1, fixed])
    top_role = _role_of_endpoint(top, exit_colors)
    bottom_role = _role_of_endpoint(bottom, exit_colors)
    if {top_role, bottom_role} != {"source", "exit"}:
        return None
    if top_role == "source":
        source_pos = (start - 1, fixed)
        exit_pos = (end + 1, fixed)
        source_color = top
    else:
        source_pos = (end + 1, fixed)
        exit_pos = (start - 1, fixed)
        source_color = bottom
    return {
        "orientation": "V",
        "col": fixed,
        "start": start,
        "end": end,
        "color": int(source_color),
        "source_pos": source_pos,
        "exit_pos": exit_pos,
        "cells": [(r, fixed) for r in range(start, end + 1)],
    }


def _extract_pipes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    colors = set(map(int, np.unique(g)))
    assert BG in colors and PIPE in colors, "expected black background and grey pipes"
    exit_colors = colors - {BG, PIPE, *SOURCE_COLORS}
    assert exit_colors, "expected at least one exit-marker color"

    pipes: list[dict] = []
    for r in range(height):
        c = 0
        while c < width:
            if g[r, c] != PIPE:
                c += 1
                continue
            start = c
            while c + 1 < width and g[r, c + 1] == PIPE:
                c += 1
            pipe = _build_pipe_from_run(g, "H", r, start, c, exit_colors)
            if pipe is not None:
                pipes.append(pipe)
            c += 1

    for c in range(width):
        r = 0
        while r < height:
            if g[r, c] != PIPE:
                r += 1
                continue
            start = r
            while r + 1 < height and g[r + 1, c] == PIPE:
                r += 1
            pipe = _build_pipe_from_run(g, "V", c, start, r, exit_colors)
            if pipe is not None:
                pipes.append(pipe)
            r += 1

    assert len(pipes) >= 2, "expected at least two pipes"

    covered = np.zeros_like(g, dtype=bool)
    endpoint_cells: set[tuple[int, int]] = set()
    for pipe in pipes:
        for pos in (pipe["source_pos"], pipe["exit_pos"]):
            assert pos not in endpoint_cells, "endpoint markers must be unique"
            endpoint_cells.add(pos)
        for cell in pipe["cells"]:
            covered[cell] = True

    assert np.array_equal(covered, g == PIPE), "every grey pixel must belong to a detected pipe"

    marker_mask = (g != BG) & (g != PIPE)
    marker_cells = {tuple(map(int, pos)) for pos in np.argwhere(marker_mask)}
    assert marker_cells == endpoint_cells, "all non-pipe colored pixels must be endpoints"
    return pipes


def _mix_color(colors: set[int]) -> int:
    if len(colors) == 1:
        return next(iter(colors))
    key = frozenset(colors)
    assert key in MIX_COLORS, "unexpected crossing color combination"
    return MIX_COLORS[key]


def main(grid: np.ndarray) -> np.ndarray:
    """Transmit each source color through its grey pipe and mix only the crossing pixels.

    Input structure:
    - The background is black.
    - Grey pixels form one-cell-wide horizontal or vertical straight pipes, and a
      horizontal pipe may share one pixel with a vertical pipe at a cross.
    - Each pipe has one red/blue/green source pixel just outside one end and one
      dedicated exit-marker pixel just outside the opposite end.

    Rule:
    - Identify every maximal horizontal or vertical grey run whose two outer
      neighbors are exactly one source pixel and one exit marker.
    - Recolor every grey pixel on a pipe to that pipe's source color.
    - Recolor each exit marker to its own pipe's source color, keep source pixels
      unchanged, and if two differently colored pipes share one crossing pixel,
      recolor only that shared pixel to the pairwise mix color.

    Key insight:
    - The straight grey runs define the pipes, while the colored endpoint outside
      each run fixes direction only locally; crossings are just shared pixels, so
      the mixed color is a local overlay and does not alter either pipe's exit.
    """
    g = np.asarray(grid, dtype=int)
    pipes = _extract_pipes(g)

    out = g.copy()
    pixel_colors: dict[tuple[int, int], set[int]] = {}
    for pipe in pipes:
        color = int(pipe["color"])
        exit_r, exit_c = pipe["exit_pos"]
        out[exit_r, exit_c] = color
        for cell in pipe["cells"]:
            pixel_colors.setdefault(cell, set()).add(color)

    for cell, colors in pixel_colors.items():
        out[cell] = _mix_color(colors)
    return out


def _crossing_point(a: dict, b: dict) -> tuple[int, int] | None:
    if a["orientation"] == b["orientation"]:
        return None
    if a["orientation"] == "H":
        h_pipe, v_pipe = a, b
    else:
        h_pipe, v_pipe = b, a
    row = h_pipe["row"]
    col = v_pipe["col"]
    if h_pipe["start"] <= col <= h_pipe["end"] and v_pipe["start"] <= row <= v_pipe["end"]:
        return (row, col)
    return None


def _occupied_sets(pipes: list[dict]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    pipe_cells: set[tuple[int, int]] = set()
    markers: set[tuple[int, int]] = set()
    for pipe in pipes:
        pipe_cells.update(_pipe_cells(pipe))
        source_pos, exit_pos = _pipe_endpoints(pipe)
        markers.add(source_pos)
        markers.add(exit_pos)
    return pipe_cells, markers


def _four_neighbors(pos: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    height, width = shape
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _can_add_pipe(pipe: dict, pipes: list[dict], shape: tuple[int, int], allow_cross: bool) -> bool:
    height, width = shape
    source_pos, exit_pos = _pipe_endpoints(pipe)
    for r, c in (*_pipe_cells(pipe), source_pos, exit_pos):
        if not (0 <= r < height and 0 <= c < width):
            return False

    candidate_cells = set(_pipe_cells(pipe))
    candidate_markers = {source_pos, exit_pos}
    occupied_cells, occupied_markers = _occupied_sets(pipes)

    if candidate_markers & occupied_markers:
        return False
    if candidate_markers & occupied_cells:
        return False
    if candidate_cells & occupied_markers:
        return False

    if pipe["orientation"] == "H" and any(other["orientation"] == "H" and other["row"] == pipe["row"] for other in pipes):
        return False
    if pipe["orientation"] == "V" and any(other["orientation"] == "V" and other["col"] == pipe["col"] for other in pipes):
        return False

    for other in pipes:
        shared = candidate_cells & set(_pipe_cells(other))
        if not shared:
            continue
        if not allow_cross:
            return False
        if pipe["orientation"] == other["orientation"]:
            return False
        if len(shared) != 1:
            return False
        cross = _crossing_point(pipe, other)
        if cross is None or cross not in shared:
            return False
        if cross in {source_pos, exit_pos}:
            return False
        other_source, other_exit = _pipe_endpoints(other)
        if cross in {other_source, other_exit}:
            return False

    own_adjacent = {
        source_pos: _pipe_cells(pipe)[0] if pipe["source_side"] in {"left", "top"} else _pipe_cells(pipe)[-1],
        exit_pos: _pipe_cells(pipe)[-1] if pipe["source_side"] in {"left", "top"} else _pipe_cells(pipe)[0],
    }
    for marker in candidate_markers:
        for neighbor in _four_neighbors(marker, shape):
            if neighbor == own_adjacent[marker]:
                continue
            if neighbor in occupied_cells or neighbor in occupied_markers:
                return False

    for cell in candidate_cells:
        for neighbor in _four_neighbors(cell, shape):
            if neighbor in occupied_markers:
                return False
    return True


def _random_pipe(shape: tuple[int, int], orientation: str, color: int) -> dict | None:
    height, width = shape
    if orientation == "H":
        max_len = min(MAX_PIPE_LEN, width - 2)
        if max_len < MIN_PIPE_LEN:
            return None
        length = random.randint(MIN_PIPE_LEN, max_len)
        start = random.randint(1, width - length - 1)
        row = random.randint(1, height - 2)
        return {
            "orientation": "H",
            "row": row,
            "start": start,
            "end": start + length - 1,
            "color": color,
            "source_side": random.choice(("left", "right")),
        }

    max_len = min(MAX_PIPE_LEN, height - 2)
    if max_len < MIN_PIPE_LEN:
        return None
    length = random.randint(MIN_PIPE_LEN, max_len)
    start = random.randint(1, height - length - 1)
    col = random.randint(1, width - 2)
    return {
        "orientation": "V",
        "col": col,
        "start": start,
        "end": start + length - 1,
        "color": color,
        "source_side": random.choice(("top", "bottom")),
    }


def _cross_pair(shape: tuple[int, int], colors: tuple[int, int]) -> list[dict] | None:
    height, width = shape
    if height < 8 or width < 8:
        return None

    row = random.randint(3, height - 4)
    col = random.randint(3, width - 4)

    left_len = random.randint(2, min(MAX_PIPE_LEN // 2, col - 1))
    right_len = random.randint(2, min(MAX_PIPE_LEN // 2, width - 2 - col))
    up_len = random.randint(2, min(MAX_PIPE_LEN // 2, row - 1))
    down_len = random.randint(2, min(MAX_PIPE_LEN // 2, height - 2 - row))

    horizontal_color, vertical_color = colors
    if random.random() < 0.5:
        horizontal_color, vertical_color = vertical_color, horizontal_color

    horizontal = {
        "orientation": "H",
        "row": row,
        "start": col - left_len,
        "end": col + right_len,
        "color": horizontal_color,
        "source_side": random.choice(("left", "right")),
    }
    vertical = {
        "orientation": "V",
        "col": col,
        "start": row - up_len,
        "end": row + down_len,
        "color": vertical_color,
        "source_side": random.choice(("top", "bottom")),
    }
    return [horizontal, vertical]


def _render_grid(shape: tuple[int, int], pipes: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for pipe in pipes:
        for r, c in _pipe_cells(pipe):
            grid[r, c] = PIPE
    for pipe in pipes:
        source_pos, exit_pos = _pipe_endpoints(pipe)
        grid[source_pos] = pipe["color"]
        grid[exit_pos] = EXIT_MARKER
    return grid


def _visual_clear(grid: np.ndarray) -> bool:
    height, width = grid.shape
    marker_positions = np.argwhere((grid != BG) & (grid != PIPE))
    for r, c in marker_positions:
        gray_neighbors = 0
        other_neighbors = 0
        for rr, cc in _four_neighbors((int(r), int(c)), (height, width)):
            if grid[rr, cc] == PIPE:
                gray_neighbors += 1
            elif grid[rr, cc] != BG:
                other_neighbors += 1
        if gray_neighbors != 1 or other_neighbors != 0:
            return False

    for r, c in np.argwhere(grid == PIPE):
        neighbors = [(rr, cc) for rr, cc in _four_neighbors((int(r), int(c)), (height, width)) if grid[rr, cc] == PIPE]
        if len(neighbors) not in {1, 2, 4}:
            return False
        if len(neighbors) == 2:
            same_row = neighbors[0][0] == neighbors[1][0] == int(r)
            same_col = neighbors[0][1] == neighbors[1][1] == int(c)
            if not (same_row or same_col):
                return False
        if len(neighbors) == 4:
            expected = {
                (int(r) - 1, int(c)),
                (int(r) + 1, int(c)),
                (int(r), int(c) - 1),
                (int(r), int(c) + 1),
            }
            if set(neighbors) != expected:
                return False
    return True


def _mode_colors(mode: str) -> tuple[int, int] | None:
    if mode == "rb_cross":
        return (2, 1)
    if mode == "rg_cross":
        return (2, 3)
    if mode == "bg_cross":
        return (1, 3)
    return None


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX),
        random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX),
    )
    mode = random.choice(MODES)
    pipes: list[dict] = []

    pair_colors = _mode_colors(mode)
    if pair_colors is not None:
        pair = _cross_pair(shape, pair_colors)
        if pair is None:
            return None
        if not _can_add_pipe(pair[0], pipes, shape, allow_cross=False):
            return None
        pipes.append(pair[0])
        if not _can_add_pipe(pair[1], pipes, shape, allow_cross=True):
            return None
        pipes.append(pair[1])
        target_count = random.randint(2, MAX_PIPE_COUNT)
    else:
        target_count = random.randint(MIN_PIPE_COUNT, MAX_PIPE_COUNT)

    placement_attempts = 0
    while len(pipes) < target_count and placement_attempts < 200:
        orientation = random.choice(("H", "V"))
        color = random.choice(SOURCE_COLORS)
        candidate = _random_pipe(shape, orientation, color)
        if candidate is None:
            return None
        if _can_add_pipe(candidate, pipes, shape, allow_cross=False):
            pipes.append(candidate)
        placement_attempts += 1
    if len(pipes) < target_count:
        return None

    if len({pipe["color"] for pipe in pipes}) == 1:
        pipes[random.randrange(len(pipes))]["color"] = random.choice([c for c in SOURCE_COLORS if c != pipes[0]["color"]])

    grid = _render_grid(shape, pipes)
    if not _visual_clear(grid):
        return None

    try:
        parsed = _extract_pipes(grid)
    except AssertionError:
        return None
    if len(parsed) != len(pipes):
        return None

    cross_count = sum(
        1
        for i in range(len(pipes))
        for j in range(i + 1, len(pipes))
        if _crossing_point(pipes[i], pipes[j]) is not None
    )
    if pair_colors is None and cross_count != 0:
        return None
    if pair_colors is not None and cross_count < 1:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-routing puzzle")
