from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
ARROW_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 300
MIN_ROOM_SIZE = 5
MAX_ROOM_EXTRA = 3

ARROW_TEMPLATES: dict[str, tuple[tuple[int, int], ...]] = {
    "up": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "down": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    "left": ((0, 0), (1, 0), (1, 1), (1, 2), (2, 0)),
    "right": ((0, 2), (1, 0), (1, 1), (1, 2), (2, 2)),
}
SHAPE_TO_DIRECTION = {frozenset(coords): direction for direction, coords in ARROW_TEMPLATES.items()}
DIRECTIONS = tuple(ARROW_TEMPLATES)


def _partition_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLUE)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLUE)]
    assert rows and cols, "expected blue separator rows and columns"
    assert rows[0] == 0 and rows[-1] == grid.shape[0] - 1, "blue border must frame rooms vertically"
    assert cols[0] == 0 and cols[-1] == grid.shape[1] - 1, "blue border must frame rooms horizontally"
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:])), "room interiors must have positive height"
    assert all(b - a >= 2 for a, b in zip(cols, cols[1:])), "room interiors must have positive width"
    return rows, cols


def _room_bounds(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    row_lines, col_lines = _partition_lines(grid)
    rooms = []
    for top, bottom in zip(row_lines, row_lines[1:]):
        for left, right in zip(col_lines, col_lines[1:]):
            rooms.append((top, bottom, left, right))
    return rooms


def _arrow_direction(coords: np.ndarray) -> str:
    assert coords.shape == (5, 2), "arrow must have exactly five pixels"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    assert tuple((maxs - mins).tolist()) == (2, 2), "arrow bbox must be 3x3"
    shape = frozenset(map(tuple, (coords - mins).tolist()))
    assert shape in SHAPE_TO_DIRECTION, "room foreground must be a valid arrow"
    return SHAPE_TO_DIRECTION[shape]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for top, bottom, left, right in _room_bounds(g):
        room = g[top + 1 : bottom, left + 1 : right]
        colors = [int(color) for color in np.unique(room) if int(color) != BLACK]
        if not colors:
            continue
        assert len(colors) == 1, "each room may contain at most one monochrome arrow"
        color = colors[0]
        assert color != BLUE, "blue should only appear on separator lines"
        coords = np.argwhere(room == color)
        direction = _arrow_direction(coords)

        room_out = out[top + 1 : bottom, left + 1 : right]
        room_out[room == color] = BLACK
        if direction == "up":
            room_out[0, :] = color
        elif direction == "down":
            room_out[-1, :] = color
        elif direction == "left":
            room_out[:, 0] = color
        else:
            room_out[:, -1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each room's arrow and paint the inner wall that the arrow points toward.

    Input structure:
    - The background is black.
    - Blue full-width rows and full-height columns form a rectangular room lattice with a blue outer border.
    - Some rooms contain one non-black, non-blue monochrome arrow made of five pixels: a 3-pixel shaft plus two head wings.

    Rule:
    - Identify the blue separator rows and columns, then split the grid into rectangular room interiors.
    - In each non-empty room, read the arrow color and its direction from the 5-pixel arrow shape.
    - Delete the arrow by restoring its five cells to black.
    - Paint the entire inner wall row or column on the pointed side of that same room with the arrow color.
    - Leave empty rooms and all blue separator lines unchanged.

    Key insight:
    - The arrow's geometry encodes direction, its color encodes the paint color, and the target is the whole inner wall on that side rather than the arrow's local footprint.
    """
    return _solve(grid)


def _sample_axis_sizes(count: int, rng: random.Random) -> list[int]:
    extras = [0] * count
    max_extra = min(MAX_ROOM_EXTRA * count, 30 - (count + 1) - MIN_ROOM_SIZE * count)
    used_extra = rng.randint(0, max_extra)
    for _ in range(used_extra):
        choices = [idx for idx, extra in enumerate(extras) if extra < MAX_ROOM_EXTRA]
        extras[rng.choice(choices)] += 1
    sizes = [MIN_ROOM_SIZE + extra for extra in extras]
    rng.shuffle(sizes)
    return sizes


def _boundary_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _sample_directions(count: int, rng: random.Random) -> list[str]:
    directions = [rng.choice(DIRECTIONS) for _ in range(count)]
    if count >= 2 and len(set(directions)) == 1:
        directions[0] = rng.choice([direction for direction in DIRECTIONS if direction != directions[0]])
    if count >= 4 and rng.random() < 0.5:
        shuffled = list(DIRECTIONS)
        rng.shuffle(shuffled)
        directions[:4] = shuffled
    rng.shuffle(directions)
    return directions


def _sample_colors(count: int, rng: random.Random) -> list[int]:
    colors = [rng.choice(ARROW_COLORS) for _ in range(count)]
    if count >= 2 and len(set(colors)) == 1:
        colors[0] = rng.choice([color for color in ARROW_COLORS if color != colors[0]])
    return colors


def _place_arrow(
    grid: np.ndarray,
    room_bounds: tuple[int, int, int, int],
    direction: str,
    color: int,
    rng: random.Random,
) -> None:
    top, bottom, left, right = room_bounds
    room_height = bottom - top - 1
    room_width = right - left - 1
    start_r = rng.randint(1, room_height - 4)
    start_c = rng.randint(1, room_width - 4)
    base_r = top + 1 + start_r
    base_c = left + 1 + start_c
    for dr, dc in ARROW_TEMPLATES[direction]:
        grid[base_r + dr, base_c + dc] = color


def _generate_attempt(rng: random.Random) -> np.ndarray:
    room_rows = rng.randint(2, 4)
    room_cols = rng.randint(2, 4)
    row_sizes = _sample_axis_sizes(room_rows, rng)
    col_sizes = _sample_axis_sizes(room_cols, rng)

    height = sum(row_sizes) + room_rows + 1
    width = sum(col_sizes) + room_cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_lines = _boundary_positions(row_sizes)
    col_lines = _boundary_positions(col_sizes)
    grid[row_lines, :] = BLUE
    grid[:, col_lines] = BLUE

    rooms = _room_bounds(grid)
    arrow_count = rng.randint(1, len(rooms) - 1)
    arrow_room_indices = rng.sample(range(len(rooms)), arrow_count)
    directions = _sample_directions(arrow_count, rng)
    colors = _sample_colors(arrow_count, rng)

    for idx, direction, color in zip(arrow_room_indices, directions, colors):
        _place_arrow(grid, rooms[idx], direction, color, rng)

    solved = _solve(grid)
    assert solved.shape == grid.shape
    assert np.any(solved != grid), "at least one arrow must cause a visible change"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            return _generate_attempt(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input grid")
