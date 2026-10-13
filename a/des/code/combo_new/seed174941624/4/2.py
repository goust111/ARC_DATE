import random
from collections import deque

import numpy as np


BACKGROUND = 0
WALL_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("up", "down", "left", "right")
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
INTERIOR_SIDE_RANGE = (3, 7)
MAX_ROOMS = 4
GENERATION_RETRIES = 300
ROOM_PLACEMENT_RETRIES = 250


def _neighbors4(row: int, col: int, height: int, width: int):
    if row > 0:
        yield row - 1, col
    if row + 1 < height:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < width:
        yield row, col + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if visited[row, col] or int(grid[row, col]) != color:
                continue

            queue = deque([(row, col)])
            visited[row, col] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cur_row, cur_col = queue.popleft()
                coords.append((cur_row, cur_col))
                for nxt_row, nxt_col in _neighbors4(cur_row, cur_col, height, width):
                    if visited[nxt_row, nxt_col] or int(grid[nxt_row, nxt_col]) != color:
                        continue
                    visited[nxt_row, nxt_col] = True
                    queue.append((nxt_row, nxt_col))

            components.append(coords)

    return components


def _gap_side(row: int, col: int, r0: int, c0: int, r1: int, c1: int) -> str | None:
    matches = []
    if row == r0 and c0 < col < c1:
        matches.append("up")
    if row == r1 and c0 < col < c1:
        matches.append("down")
    if col == c0 and r0 < row < r1:
        matches.append("left")
    if col == c1 and r0 < row < r1:
        matches.append("right")
    return matches[0] if len(matches) == 1 else None


def _extract_room(grid: np.ndarray, wall_cells: list[tuple[int, int]], wall_color: int, bg: int):
    rows = [row for row, _ in wall_cells]
    cols = [col for _, col in wall_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    wall_set = set(wall_cells)
    boundary_cells = []
    missing = []
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            on_boundary = row in (r0, r1) or col in (c0, c1)
            if (row, col) in wall_set and not on_boundary:
                return None
            if on_boundary:
                boundary_cells.append((row, col))
                if (row, col) not in wall_set:
                    missing.append((row, col))

    if len(missing) != 1 or len(wall_set) != len(boundary_cells) - 1:
        return None

    gap_row, gap_col = missing[0]
    side = _gap_side(gap_row, gap_col, r0, c0, r1, c1)
    if side is None:
        return None

    seed_cells = []
    seed_colors = set()
    for row in range(r0 + 1, r1):
        for col in range(c0 + 1, c1):
            value = int(grid[row, col])
            if value == bg:
                continue
            if value == wall_color:
                return None
            seed_cells.append((row, col))
            seed_colors.add(value)

    if not seed_cells or len(seed_colors) != 1:
        return None

    seed_color = seed_colors.pop()
    seed_rows = {row for row, _ in seed_cells}
    seed_cols = {col for _, col in seed_cells}
    seed_set = set(seed_cells)

    if len(seed_rows) == 1:
        seed_row = next(iter(seed_rows))
        lo, hi = min(seed_cols), max(seed_cols)
        expected = {(seed_row, col) for col in range(lo, hi + 1)}
    elif len(seed_cols) == 1:
        seed_col = next(iter(seed_cols))
        lo, hi = min(seed_rows), max(seed_rows)
        expected = {(row, seed_col) for row in range(lo, hi + 1)}
    else:
        return None

    if seed_set != expected or len(seed_set) < 2:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (gap_row, gap_col),
        "side": side,
        "seed_color": seed_color,
        "seed_length": len(seed_set),
    }


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    rooms = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for component in _components_of_color(grid, color):
            room = _extract_room(grid, component, color, bg)
            if room is not None:
                rooms.append(room)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _draw_trail(out: np.ndarray, room: dict, bg: int) -> None:
    gap_row, gap_col = room["gap"]
    d_row, d_col = DIRECTION_VECTORS[room["side"]]
    seed_color = int(room["seed_color"])

    for step in range(1, int(room["seed_length"]) + 1):
        row = gap_row + d_row * step
        col = gap_col + d_col * step
        assert 0 <= row < out.shape[0] and 0 <= col < out.shape[1], "trail must stay in bounds"
        assert int(out[row, col]) in (bg, seed_color), "trail must not overwrite another object"
        out[row, col] = seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot each room's interior seed outward through the room's single wall gap, using the seed length as the trail length.

    Input structure:
    - The background is black.
    - One or more gray rectangular room walls appear as 1-pixel-thick rectangle borders with exactly one missing wall cell on one side.
    - Inside each room there is exactly one non-black, non-gray horizontal or vertical seed segment.

    Rule:
    - For each room, locate the missing border cell to determine whether the exit faces up, down, left, or right.
    - Count the seed segment's pixels; this count is the trail length.
    - Keep the room walls, the gap, and the interior seed unchanged, then draw a same-colored straight trail outside the room starting just beyond the gap and extending outward for that many cells.

    Key insight:
    - The output depends on two independent geometric cues: the gap position chooses the direction, while the seed segment length chooses the distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BACKGROUND
    rooms = _find_rooms(g, bg)
    assert rooms, "expected at least one gapped rectangular room"

    out = g.copy()
    for room in rooms:
        _draw_trail(out, room, bg)
    return out


def _footprint_with_padding(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    padded = set()
    for row, col in cells:
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                nxt_row = row + d_row
                nxt_col = col + d_col
                if 0 <= nxt_row < height and 0 <= nxt_col < width:
                    padded.add((nxt_row, nxt_col))
    return padded


def _max_room_count(height: int, width: int) -> int:
    smaller = min(height, width)
    if smaller >= 22:
        return MAX_ROOMS
    if smaller >= 16:
        return 3
    return 2


def _sample_side_sequence(room_count: int) -> list[str]:
    if room_count <= len(SIDES) and random.random() < 0.5:
        return random.sample(SIDES, room_count)
    return [random.choice(SIDES) for _ in range(room_count)]


def _candidate_room(height: int, width: int, seed_color: int, side: str):
    max_interior_h = min(INTERIOR_SIDE_RANGE[1], height - 2)
    max_interior_w = min(INTERIOR_SIDE_RANGE[1], width - 2)
    if max_interior_h < INTERIOR_SIDE_RANGE[0] or max_interior_w < INTERIOR_SIDE_RANGE[0]:
        return None

    interior_h = random.randint(INTERIOR_SIDE_RANGE[0], max_interior_h)
    interior_w = random.randint(INTERIOR_SIDE_RANGE[0], max_interior_w)
    segment_orientation = random.choice(("horizontal", "vertical"))
    max_length = interior_w if segment_orientation == "horizontal" else interior_h
    seed_length = random.randint(2, max_length)

    room_h = interior_h + 2
    room_w = interior_w + 2
    row_min, row_max = 0, height - room_h
    col_min, col_max = 0, width - room_w
    if side == "up":
        row_min = seed_length
    elif side == "down":
        row_max = height - room_h - seed_length
    elif side == "left":
        col_min = seed_length
    else:
        col_max = width - room_w - seed_length

    if row_min > row_max or col_min > col_max:
        return None

    r0 = random.randint(row_min, row_max)
    c0 = random.randint(col_min, col_max)
    r1 = r0 + room_h - 1
    c1 = c0 + room_w - 1

    if side in ("up", "down"):
        gap = (r0 if side == "up" else r1, random.randint(c0 + 1, c1 - 1))
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)

    wall_cells = set()
    room_bbox_cells = set()
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            room_bbox_cells.add((row, col))
            if (row in (r0, r1) or col in (c0, c1)) and (row, col) != gap:
                wall_cells.add((row, col))

    if segment_orientation == "horizontal":
        seed_row = random.randint(r0 + 1, r1 - 1)
        seed_start = random.randint(c0 + 1, c1 - seed_length)
        seed_cells = {(seed_row, seed_start + offset) for offset in range(seed_length)}
    else:
        seed_col = random.randint(c0 + 1, c1 - 1)
        seed_start = random.randint(r0 + 1, r1 - seed_length)
        seed_cells = {(seed_start + offset, seed_col) for offset in range(seed_length)}

    d_row, d_col = DIRECTION_VECTORS[side]
    trail_cells = {
        (gap[0] + d_row * step, gap[1] + d_col * step)
        for step in range(1, seed_length + 1)
    }

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "seed_color": seed_color,
        "seed_length": seed_length,
        "wall_cells": wall_cells,
        "seed_cells": seed_cells,
        "trail_cells": trail_cells,
        "footprint": room_bbox_cells | trail_cells,
    }


def _render_input(height: int, width: int, rooms: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for room in rooms:
        for row, col in room["wall_cells"]:
            grid[row, col] = WALL_COLOR
        for row, col in room["seed_cells"]:
            grid[row, col] = int(room["seed_color"])
    return grid


def _render_output(inp: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        for row, col in room["trail_cells"]:
            assert int(out[row, col]) == BACKGROUND, "generated trail must start on empty cells"
            out[row, col] = int(room["seed_color"])
    return out


def _attempt_generation() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    room_count = random.randint(1, _max_room_count(height, width))
    seed_colors = random.sample(SEED_COLORS, room_count)
    sides = _sample_side_sequence(room_count)

    blocked: set[tuple[int, int]] = set()
    rooms: list[dict] = []
    for seed_color, side in zip(seed_colors, sides):
        placed = False
        for _ in range(ROOM_PLACEMENT_RETRIES):
            room = _candidate_room(height, width, seed_color, side)
            if room is None:
                continue
            if room["footprint"] & blocked:
                continue
            rooms.append(room)
            blocked |= _footprint_with_padding(room["footprint"], height, width)
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(height, width, rooms)
    expected = _render_output(inp, rooms)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generation()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-trail sample")
