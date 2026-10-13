from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PATH_COLORS = (1, 2, 3, 6, 7, 8, 9)

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
MAIN_PATH_COUNT_RANGE = (1, 3)
TERMINAL_SEGMENT_LEN_RANGE = (1, 3)
MAIN_AXIS_GAP = 5
MAIN_AXIS_MARGIN = 4
AUX_SEGMENT_LEN = 1
MAX_GENERATION_ATTEMPTS = 500

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_path_color(value: int) -> bool:
    return value not in {BG, YELLOW, GREY}


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = direction
    return [(r0 + dr * step, c0 + dc * step) for step in range(1, length + 1)]


def _cells_clear(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG for r, c in cells)


def _collect_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    node_positions = [tuple(map(int, pos)) for pos in np.argwhere((g == YELLOW) | (g == GREY))]
    node_set = set(node_positions)

    assert node_positions, "expected grey endpoints or yellow intersections"

    segments: list[dict] = []
    seen: set[tuple] = set()
    covered_cells: set[tuple[int, int]] = set()
    node_degree = {node: 0 for node in node_positions}

    for start in node_positions:
        sr, sc = start
        for dr, dc in DIRS:
            nr, nc = sr + dr, sc + dc
            if not _in_bounds(g.shape, nr, nc):
                continue

            color = int(g[nr, nc])
            if not _is_path_color(color):
                continue

            cells = [(nr, nc)]
            cr, cc = nr, nc
            while True:
                rr, cc2 = cr + dr, cc + dc
                assert _in_bounds(g.shape, rr, cc2), "path exited the grid before reaching a node"
                value = int(g[rr, cc2])
                if value == color:
                    cells.append((rr, cc2))
                    cr, cc = rr, cc2
                    continue

                end = (rr, cc2)
                assert end in node_set, "segment must terminate at a grey or yellow node"
                break

            orientation = "h" if dr == 0 else "v"
            key = (color, orientation, min(start, end), max(start, end))
            if key in seen:
                continue
            seen.add(key)

            if orientation == "h":
                assert len({r for r, _ in cells}) == 1, "horizontal segment must stay on one row"
            else:
                assert len({c for _, c in cells}) == 1, "vertical segment must stay on one column"

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "color": color,
                    "length": len(cells),
                    "orientation": orientation,
                    "cells": cells,
                }
            )
            covered_cells.update(cells)
            node_degree[start] += 1
            node_degree[end] += 1

    path_cells = {
        tuple(map(int, pos))
        for pos in np.argwhere(~np.isin(g, np.array([BG, YELLOW, GREY], dtype=int)))
    }
    assert covered_cells == path_cells, "every colored path pixel must belong to a straight node-to-node segment"

    for node in node_positions:
        degree = node_degree[node]
        value = int(g[node[0], node[1]])
        if value == GREY:
            assert degree == 1, "each grey marker must terminate exactly one segment"
        else:
            assert degree >= 2, "each yellow marker must connect at least two segments"

    return segments


def _plan_branches(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    segments = _collect_segments(g)
    branches: list[dict] = []
    claimed_cells: dict[tuple[int, int], int] = {}

    for segment in segments:
        orientation = segment["orientation"]
        length = int(segment["length"])
        color = int(segment["color"])

        if orientation == "h":
            preferred = (1, 0)
            alternate = (-1, 0)
        else:
            preferred = (0, 1)
            alternate = (0, -1)

        for endpoint in (segment["start"], segment["end"]):
            er, ec = endpoint
            if int(g[er, ec]) != GREY:
                continue

            preferred_cells = _ray_cells(endpoint, preferred, length)
            alternate_cells = _ray_cells(endpoint, alternate, length)

            if _cells_clear(g, preferred_cells):
                direction = preferred
                cells = preferred_cells
            else:
                assert _cells_clear(g, alternate_cells), "a branch endpoint must have one clear perpendicular side"
                direction = alternate
                cells = alternate_cells

            for cell in cells:
                prev = claimed_cells.get(cell)
                assert prev in (None, color), "branches may not overlap ambiguously"
                claimed_cells[cell] = color

            branches.append(
                {
                    "endpoint": endpoint,
                    "orientation": orientation,
                    "length": length,
                    "color": color,
                    "direction": direction,
                    "cells": cells,
                }
            )

    return branches


def main(grid: np.ndarray) -> np.ndarray:
    """Split colored paths at yellow crossings, then grow same-length perpendicular branches at grey endpoints."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for branch in _plan_branches(g):
        for r, c in branch["cells"]:
            out[r, c] = int(branch["color"])
    return out


def _sample_spaced_positions(count: int, low: int, high: int, gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if count > len(values):
        return None

    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _paint_horizontal_path(grid: np.ndarray, row: int, left_col: int, right_col: int, color: int) -> bool:
    if left_col < 0 or right_col >= grid.shape[1] or left_col + 1 >= right_col:
        return False
    if not np.all(grid[row, left_col : right_col + 1] == BG):
        return False

    grid[row, left_col] = GREY
    grid[row, right_col] = GREY
    grid[row, left_col + 1 : right_col] = color
    return True


def _paint_vertical_path(grid: np.ndarray, top_row: int, bottom_row: int, col: int, color: int) -> bool:
    if top_row < 0 or bottom_row >= grid.shape[0] or top_row + 1 >= bottom_row:
        return False
    if int(grid[top_row, col]) != BG or int(grid[bottom_row, col]) != BG:
        return False

    grid[top_row, col] = GREY
    grid[bottom_row, col] = GREY

    for row in range(top_row + 1, bottom_row):
        value = int(grid[row, col])
        if value == BG:
            grid[row, col] = color
        elif _is_path_color(value):
            if value == color:
                return False
            grid[row, col] = YELLOW
        else:
            return False

    return True


def _paint_vertical_aux(grid: np.ndarray, top_row: int, col: int, color: int) -> bool:
    cells = [(top_row, col, GREY), (top_row + 1, col, color), (top_row + 2, col, GREY)]
    if not all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG for r, c, _ in cells):
        return False
    for r, c, value in cells:
        grid[r, c] = value
    return True


def _paint_horizontal_aux(grid: np.ndarray, row: int, left_col: int, color: int) -> bool:
    cells = [(row, left_col, GREY), (row, left_col + 1, color), (row, left_col + 2, GREY)]
    if not all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG for r, c, _ in cells):
        return False
    for r, c, value in cells:
        grid[r, c] = value
    return True


def _random_aux_color(avoid: int) -> int:
    choices = [color for color in PATH_COLORS if color != avoid]
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    num_h = random.randint(*MAIN_PATH_COUNT_RANGE)
    num_v = random.randint(*MAIN_PATH_COUNT_RANGE)

    rows = _sample_spaced_positions(num_h, MAIN_AXIS_MARGIN, height - MAIN_AXIS_MARGIN - 1, MAIN_AXIS_GAP)
    cols = _sample_spaced_positions(num_v, MAIN_AXIS_MARGIN, width - MAIN_AXIS_MARGIN - 1, MAIN_AXIS_GAP)
    if rows is None or cols is None:
        return None

    leftmost_col, rightmost_col = cols[0], cols[-1]
    topmost_row, bottommost_row = rows[0], rows[-1]
    main_colors = random.sample(PATH_COLORS, num_h + num_v)

    horizontals = []
    for idx, row in enumerate(rows):
        left_len = random.randint(*TERMINAL_SEGMENT_LEN_RANGE)
        right_len = random.randint(*TERMINAL_SEGMENT_LEN_RANGE)
        left_col = leftmost_col - left_len - 1
        right_col = rightmost_col + right_len + 1
        if left_col < 0 or right_col >= width:
            return None
        blocked_side = random.choice(("left", "right"))
        horizontals.append(
            {
                "row": row,
                "left_col": left_col,
                "right_col": right_col,
                "color": main_colors[idx],
                "blocked_side": blocked_side,
            }
        )

    verticals = []
    for idx, col in enumerate(cols):
        top_len = random.randint(*TERMINAL_SEGMENT_LEN_RANGE)
        bottom_len = random.randint(*TERMINAL_SEGMENT_LEN_RANGE)
        top_row = topmost_row - top_len - 1
        bottom_row = bottommost_row + bottom_len + 1
        if top_row < 0 or bottom_row >= height:
            return None
        blocked_side = random.choice(("top", "bottom"))
        verticals.append(
            {
                "col": col,
                "top_row": top_row,
                "bottom_row": bottom_row,
                "color": main_colors[num_h + idx],
                "blocked_side": blocked_side,
            }
        )

    grid = np.full((height, width), BG, dtype=int)

    for spec in horizontals:
        if not _paint_horizontal_path(grid, spec["row"], spec["left_col"], spec["right_col"], spec["color"]):
            return None

    for spec in verticals:
        if not _paint_vertical_path(grid, spec["top_row"], spec["bottom_row"], spec["col"], spec["color"]):
            return None

    expected_dirs: dict[tuple[int, int], tuple[int, int]] = {}

    for spec in horizontals:
        row = int(spec["row"])
        left_endpoint = (row, int(spec["left_col"]))
        right_endpoint = (row, int(spec["right_col"]))

        if spec["blocked_side"] == "left":
            if not _paint_vertical_aux(grid, row + 1, int(spec["left_col"]), _random_aux_color(int(spec["color"]))):
                return None
            expected_dirs[left_endpoint] = (-1, 0)
            expected_dirs[right_endpoint] = (1, 0)
        else:
            if not _paint_vertical_aux(grid, row + 1, int(spec["right_col"]), _random_aux_color(int(spec["color"]))):
                return None
            expected_dirs[left_endpoint] = (1, 0)
            expected_dirs[right_endpoint] = (-1, 0)

    for spec in verticals:
        col = int(spec["col"])
        top_endpoint = (int(spec["top_row"]), col)
        bottom_endpoint = (int(spec["bottom_row"]), col)

        if spec["blocked_side"] == "top":
            if not _paint_horizontal_aux(grid, int(spec["top_row"]), col + 1, _random_aux_color(int(spec["color"]))):
                return None
            expected_dirs[top_endpoint] = (0, -1)
            expected_dirs[bottom_endpoint] = (0, 1)
        else:
            if not _paint_horizontal_aux(grid, int(spec["bottom_row"]), col + 1, _random_aux_color(int(spec["color"]))):
                return None
            expected_dirs[top_endpoint] = (0, 1)
            expected_dirs[bottom_endpoint] = (0, -1)

    try:
        branch_specs = _plan_branches(grid)
        branch_dirs = {tuple(spec["endpoint"]): tuple(spec["direction"]) for spec in branch_specs}
        for endpoint, direction in expected_dirs.items():
            if branch_dirs.get(endpoint) != direction:
                return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
