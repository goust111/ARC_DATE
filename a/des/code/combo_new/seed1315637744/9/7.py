from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

NODE_COLORS = (BLUE, RED, YELLOW, MAGENTA, ORANGE)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # right, down, left, up
SPECIAL_COLORS = {BLACK, GREEN, GREY, AZURE, MAROON}

MAX_GENERATION_ATTEMPTS = 400


def _legend_turns(grid: np.ndarray) -> dict[int, int]:
    """Read left-edge legend entries as color -> turn_delta, where -1=left and +1=right."""
    h, w = grid.shape
    turns: dict[int, int] = {}

    for r in range(h - 1):
        color = int(grid[r, 0])
        if color in (BLACK, AZURE):
            continue
        if int(grid[r + 1, 0]) != AZURE:
            continue

        anchor_r, anchor_c = r + 1, 0
        extras: list[tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = anchor_r + dr, anchor_c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == AZURE:
                extras.append((dr, dc))

        if len(extras) != 1:
            continue

        turns[color] = 1 if extras[0] == (1, 0) else -1

    assert turns, "expected at least one left-edge legend entry"
    return turns


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _path_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        segment = _line_cells(points[i], points[i + 1])
        if i:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the route directed by color-keyed turn markers and paint the walked background grey.

    Input structure:
    - The background is black.
    - The left edge contains a legend: a colored pixel is followed one row below by an azure anchor pixel, and a second adjacent azure pixel encodes whether that color means "turn left" or "turn right".
    - Away from the legend, the main area contains one green start pixel, one maroon end pixel, a grey pixel immediately to the right of the start, and several colored turn-trigger nodes whose colors appear in the legend.

    Rule:
    - Start at the green pixel, facing right because of the adjacent grey starter pixel.
    - Walk one cell at a time; every traversed black cell becomes grey.
    - When the walk reaches a colored node, look up that node's color in the legend and rotate the direction by 90 degrees left or right before continuing.
    - Stop after reaching the maroon end pixel or after stepping onto the grid boundary and having no next in-bounds step.

    Key insight:
    - The colored nodes do not themselves show the turn; they only key into the left-edge legend, so the task is "decode legend first, then simulate the orthogonal walk".
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    turns = _legend_turns(g)

    starts = np.argwhere(g == GREEN)
    ends = np.argwhere(g == MAROON)
    assert len(starts) == 1, "expected exactly one green start"
    assert len(ends) == 1, "expected exactly one maroon end"

    r, c = map(int, starts[0])
    assert c + 1 < w and int(g[r, c + 1]) == GREY, "green start must point right with a grey neighbor"

    out = g.copy()
    dir_idx = 0
    seen = {(r, c, dir_idx)}

    while True:
        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break

        r, c = nr, nc
        cell = int(g[r, c])
        if cell == BLACK:
            out[r, c] = GREY
        elif cell == MAROON:
            break
        elif cell in turns:
            dir_idx = (dir_idx + turns[cell]) % 4
        elif cell not in (GREY, GREEN):
            break

        state = (r, c, dir_idx)
        if state in seen:
            break
        seen.add(state)

    return out


def _cover_colors(colors: list[int], count: int) -> list[int]:
    assigned = list(colors)
    while len(assigned) < count:
        assigned.append(random.choice(colors))
    random.shuffle(assigned)
    return assigned[:count]


def _sample_polyline() -> tuple[list[tuple[int, int]], list[int]] | None:
    bends = random.randint(1, 4)
    horizontal_lengths = [random.randint(2, 6) for _ in range(bends + 1)]
    vertical_lengths = [random.randint(2, 5) for _ in range(bends)]
    vertical_signs = [random.choice((-1, 1)) for _ in range(bends)]

    total_width = 1 + sum(horizontal_lengths)
    if total_width > 24:
        return None

    points = [(0, 0)]
    turns: list[int] = []
    row = 0
    col = 0
    for i in range(bends):
        col += horizontal_lengths[i]
        points.append((row, col))
        turn_in = 1 if vertical_signs[i] > 0 else -1
        turns.append(turn_in)

        row += vertical_signs[i] * vertical_lengths[i]
        points.append((row, col))
        turns.append(-turn_in)

    col += horizontal_lengths[-1]
    points.append((row, col))

    rows = [r for r, _ in points]
    if max(rows) - min(rows) > 14:
        return None

    return points, turns


def _sample_turn_colors(turns: list[int]) -> tuple[dict[int, int], list[int]] | None:
    left_count = turns.count(-1)
    right_count = turns.count(1)
    if left_count == 0 or right_count == 0:
        return None

    left_palette_size = random.randint(1, min(2, left_count))
    right_palette_size = random.randint(1, min(2, right_count))
    total_colors = left_palette_size + right_palette_size
    palette = random.sample(list(NODE_COLORS), total_colors)
    random.shuffle(palette)

    left_colors = palette[:left_palette_size]
    right_colors = palette[left_palette_size:]
    mapping = {color: -1 for color in left_colors}
    mapping.update({color: 1 for color in right_colors})

    left_nodes = _cover_colors(left_colors, left_count)
    right_nodes = _cover_colors(right_colors, right_count)
    left_idx = 0
    right_idx = 0
    node_colors: list[int] = []
    for turn in turns:
        if turn == -1:
            node_colors.append(left_nodes[left_idx])
            left_idx += 1
        else:
            node_colors.append(right_nodes[right_idx])
            right_idx += 1

    return mapping, node_colors


def _render_legend(grid: np.ndarray, mapping: dict[int, int], top: int) -> None:
    entries = list(mapping.items())
    random.shuffle(entries)
    for i, (color, turn) in enumerate(entries):
        r = top + 3 * i
        grid[r, 0] = color
        grid[r + 1, 0] = AZURE
        if turn == 1:
            grid[r + 2, 0] = AZURE
        else:
            grid[r + 1, 1] = AZURE


def _expected_output(inp: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    node_set = set(points[1:-1])
    end = points[-1]
    for cell in _path_cells(points)[1:]:
        if cell == end or cell in node_set:
            continue
        out[cell] = GREY
    return out


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_polyline()
    if sampled is None:
        return None
    rel_points, turns = sampled

    sampled_colors = _sample_turn_colors(turns)
    if sampled_colors is None:
        return None
    mapping, node_colors = sampled_colors

    legend_height = 3 * len(mapping)
    min_row = min(r for r, _ in rel_points)
    max_row = max(r for r, _ in rel_points)
    path_height = max_row - min_row + 1

    min_h = max(legend_height, path_height)
    max_h = min(30, min_h + 6)
    if min_h > max_h:
        return None
    height = random.randint(min_h, max_h)

    row_shift_low = -min_row
    row_shift_high = height - 1 - max_row
    if row_shift_low > row_shift_high:
        return None
    row_shift = random.randint(row_shift_low, row_shift_high)

    path_width = rel_points[-1][1] + 1
    start_col = random.randint(2, 5)
    end_on_boundary = random.random() < 0.5
    min_w = start_col + path_width
    if min_w > 30:
        return None
    if end_on_boundary:
        width = min_w
    else:
        max_pad = min(5, 30 - min_w)
        if max_pad < 1:
            return None
        width = min_w + random.randint(1, max_pad)

    if width <= 1:
        return None

    legend_top_max = height - legend_height
    if legend_top_max < 0:
        return None
    legend_top = random.randint(0, legend_top_max)

    points = [(r + row_shift, c + start_col) for r, c in rel_points]
    path_cells = _path_cells(points)
    if len(path_cells) < 3:
        return None

    start = points[0]
    end = points[-1]
    if start[1] + 1 >= width or path_cells[1] != (start[0], start[1] + 1):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    _render_legend(grid, mapping, legend_top)

    grid[start] = GREEN
    grid[path_cells[1]] = GREY
    for point, color in zip(points[1:-1], node_colors, strict=False):
        grid[point] = color
    grid[end] = MAROON

    expected = _expected_output(grid, points)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
