from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300

Segment = tuple[str, int, int]


def _line_coords(grid: np.ndarray) -> tuple[list[int], list[int]]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected grey intersection pixels"
    rows = sorted({int(r) for r, _ in grey})
    cols = sorted({int(c) for _, c in grey})
    assert len(rows) >= 2 and len(cols) >= 2, "expected at least one rectangular room"
    for r in rows:
        for c in cols:
            assert int(grid[r, c]) == GREY, "grey intersections must form a full line grid"
    return rows, cols


def _uniform_color(values: np.ndarray) -> int:
    flat = values.reshape(-1)
    assert flat.size > 0, "empty border segment"
    colors = np.unique(flat)
    assert colors.size == 1, "each border segment must be one color"
    return int(colors[0])


def _paint_segment(grid: np.ndarray, side: tuple[str, int, int, int]) -> None:
    kind, fixed, lo, hi = side
    if kind == "row":
        grid[fixed, lo:hi] = BG
    else:
        grid[lo:hi, fixed] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open room doors whose colored wall segment matches the room marker color.

    Input structure:
    - Black background fills the interiors of rectangular rooms.
    - One-pixel colored horizontal and vertical line segments form a grid of
      rectangular rooms; grey pixels are the fixed intersections between line
      segments.
    - Every room interior contains exactly one single-pixel colored marker, and
      marker colors come from colors used by border segments elsewhere in the maze.

    Rule:
    - For each room, read the color of its interior marker.
    - Compare that color with the four complete border segments around the room.
    - If one or more side segments have the same color, replace those whole
      non-grey segments with black and erase the room marker.
    - If none of the four sides match the marker color, leave that room marker
      and all of its walls unchanged. Grey intersection pixels always remain grey.

    Key insight:
    A marker is a key for an entire same-colored wall segment on that room's
    boundary, not just for the nearest wall pixel or for all pixels of that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rows, cols = _line_coords(g)

    out = g.copy()
    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]
        if r1 - r0 <= 1:
            continue
        for j in range(len(cols) - 1):
            c0, c1 = cols[j], cols[j + 1]
            if c1 - c0 <= 1:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_cells = np.argwhere(interior != BG)
            assert marker_cells.shape[0] == 1, "each room must have exactly one marker"
            mr, mc = map(int, marker_cells[0])
            marker_r, marker_c = r0 + 1 + mr, c0 + 1 + mc
            marker_color = int(g[marker_r, marker_c])

            sides = [
                (_uniform_color(g[r0, c0 + 1 : c1]), ("row", r0, c0 + 1, c1)),
                (_uniform_color(g[r1, c0 + 1 : c1]), ("row", r1, c0 + 1, c1)),
                (_uniform_color(g[r0 + 1 : r1, c0]), ("col", c0, r0 + 1, r1)),
                (_uniform_color(g[r0 + 1 : r1, c1]), ("col", c1, r0 + 1, r1)),
            ]

            matching = [side for color, side in sides if color == marker_color]
            if matching:
                out[marker_r, marker_c] = BG
                for side in matching:
                    _paint_segment(out, side)

    return out


def _segments(room_rows: int, room_cols: int) -> list[Segment]:
    horizontal = [("h", i, j) for i in range(room_rows + 1) for j in range(room_cols)]
    vertical = [("v", i, j) for i in range(room_rows) for j in range(room_cols + 1)]
    return horizontal + vertical


def _room_segment_ids(i: int, j: int) -> list[Segment]:
    return [("h", i, j), ("h", i + 1, j), ("v", i, j), ("v", i, j + 1)]


def _conflict_graph(room_rows: int, room_cols: int) -> dict[Segment, set[Segment]]:
    conflicts = {seg: set() for seg in _segments(room_rows, room_cols)}
    for i in range(room_rows):
        for j in range(room_cols):
            for a, b in combinations(_room_segment_ids(i, j), 2):
                conflicts[a].add(b)
                conflicts[b].add(a)
    return conflicts


def _color_segments(room_rows: int, room_cols: int) -> dict[Segment, int] | None:
    conflicts = _conflict_graph(room_rows, room_cols)
    order = list(conflicts)
    random.shuffle(order)
    order.sort(key=lambda seg: len(conflicts[seg]), reverse=True)

    assigned: dict[Segment, int] = {}
    for seg in order:
        unavailable = {assigned[n] for n in conflicts[seg] if n in assigned}
        choices = [c for c in COLORS if c not in unavailable]
        if not choices:
            return None
        assigned[seg] = random.choice(choices)

    if len(set(assigned.values())) < 5:
        return None
    return assigned


def _line_positions(sizes: list[int]) -> list[int]:
    pos = [0]
    for size in sizes:
        pos.append(pos[-1] + size + 1)
    return pos


def _draw_maze(
    room_heights: list[int],
    room_widths: list[int],
    seg_colors: dict[Segment, int],
) -> tuple[np.ndarray, list[int], list[int]]:
    rows = _line_positions(room_heights)
    cols = _line_positions(room_widths)
    grid = np.full((rows[-1] + 1, cols[-1] + 1), BG, dtype=int)

    for i, r in enumerate(rows):
        for j in range(len(room_widths)):
            grid[r, cols[j] + 1 : cols[j + 1]] = seg_colors[("h", i, j)]
    for i in range(len(room_heights)):
        for j, c in enumerate(cols):
            grid[rows[i] + 1 : rows[i + 1], c] = seg_colors[("v", i, j)]

    for r in rows:
        for c in cols:
            grid[r, c] = GREY
    return grid, rows, cols


def _expected_for_generated(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    room_heights = [random.randint(2, 6) for _ in range(room_rows)]
    room_widths = [random.randint(2, 6) for _ in range(room_cols)]
    if sum(room_heights) + room_rows + 1 > 30 or sum(room_widths) + room_cols + 1 > 30:
        return None

    seg_colors = _color_segments(room_rows, room_cols)
    if seg_colors is None:
        return None

    grid, rows, cols = _draw_maze(room_heights, room_widths, seg_colors)
    global_colors = set(seg_colors.values())
    rooms = [(i, j) for i in range(room_rows) for j in range(room_cols)]
    random.shuffle(rooms)

    match_count = len(rooms) // 2 + (1 if len(rooms) % 2 and random.random() < 0.5 else 0)
    branches = [True] * match_count + [False] * (len(rooms) - match_count)
    random.shuffle(branches)

    for (i, j), should_match in zip(rooms, branches):
        side_ids = _room_segment_ids(i, j)
        side_colors = [seg_colors[seg] for seg in side_ids]
        if should_match:
            marker_color = random.choice(side_colors)
        else:
            choices = list(global_colors - set(side_colors))
            if not choices:
                return None
            marker_color = random.choice(choices)

        r = random.randint(rows[i] + 1, rows[i + 1] - 1)
        c = random.randint(cols[j] + 1, cols[j + 1] - 1)
        grid[r, c] = marker_color

    try:
        expected = _expected_for_generated(grid)
    except Exception:
        return None
    if expected.shape != grid.shape or max(expected.shape) > 30:
        return None
    if not np.all(expected[np.ix_(rows, cols)] == GREY):
        return None
    if np.array_equal(expected, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door maze input")
