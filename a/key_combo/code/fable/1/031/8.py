from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
TEAL = 8

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_INDEX = {direction: i for i, direction in enumerate(DIRS)}

GRID_MIN = 12
GRID_MAX = 24
GREEN_LEN_RANGE = (2, 4)
RED_LEN_RANGE = (2, 4)
TURN_COUNT_RANGE = (2, 5)
SEGMENT_LEN_RANGE = (2, 6)
FINAL_SEGMENT_LEN_RANGE = (2, 7)
EXTRA_BARRIER_RANGE = (4, 14)
GEN_ATTEMPTS = 400


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _in_bounds(shape: tuple[int, int], coord: tuple[int, int]) -> bool:
    return 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(TURN_INDEX[direction] - 1) % 4]


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(TURN_INDEX[direction] + 1) % 4]


def _other_turn(direction: tuple[int, int], next_direction: tuple[int, int]) -> tuple[int, int]:
    left = _turn_left(direction)
    right = _turn_right(direction)
    return right if next_direction == left else left


def _ordered_line_cells(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"expected color {color} to appear"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if r0 == r1:
        assert coords.shape[0] == c1 - c0 + 1, "line marker must be gap-free"
        return [(int(r0), c) for c in range(int(c0), int(c1) + 1)]
    if c0 == c1:
        assert coords.shape[0] == r1 - r0 + 1, "line marker must be gap-free"
        return [(r, int(c0)) for r in range(int(r0), int(r1) + 1)]
    raise AssertionError("marker must be a straight line")


def _start_state(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    H, W = grid.shape
    green = _ordered_line_cells(grid, GREEN)
    same_row = len({r for r, _ in green}) == 1

    if same_row:
        touches_left = green[0][1] == 0
        touches_right = green[-1][1] == W - 1
        assert touches_left ^ touches_right, "green line must touch exactly one side"
        if touches_left:
            return green[-1], (0, 1)
        return green[0], (0, -1)

    touches_top = green[0][0] == 0
    touches_bottom = green[-1][0] == H - 1
    assert touches_top ^ touches_bottom, "green line must touch exactly one side"
    if touches_top:
        return green[-1], (1, 0)
    return green[0], (-1, 0)


def _simulate_turn_barriers(grid: np.ndarray) -> list[tuple[int, int]]:
    red_cells = set(_ordered_line_cells(grid, RED))
    current, direction = _start_state(grid)
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    causal: list[tuple[int, int]] = []
    step_budget = grid.shape[0] * grid.shape[1] * 4 + 1

    for _ in range(step_budget):
        state = (current, direction)
        assert state not in seen_states, "walk should not loop forever"
        seen_states.add(state)

        front = _add(current, direction)
        assert _in_bounds(grid.shape, front), "path left the grid before reaching the end marker"
        if front in red_cells:
            return causal

        front_value = int(grid[front])
        if front_value == TEAL:
            options = []
            for next_direction in (_turn_left(direction), _turn_right(direction)):
                nxt = _add(current, next_direction)
                if _in_bounds(grid.shape, nxt) and int(grid[nxt]) == BLACK:
                    options.append(next_direction)
            assert len(options) == 1, "blocked step must have exactly one black turning option"
            causal.append(front)
            direction = options[0]
            continue

        assert front_value == BLACK, "walker may only move through black cells until the red end marker"
        current = front

    raise AssertionError("path simulation exceeded the safety budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the teal barriers that truly caused a turn, recolored to yellow.

    Input structure:
    - A black grid contains teal barrier pixels, one short green straight line, and one short red straight line.
    - The green line touches one border and points inward, so its inner endpoint gives the start position and initial travel direction.
    - The red line is the terminal marker that stops the walk when it appears directly ahead.

    Rule:
    - Start from the inner endpoint of the green line and move straight through black cells.
    - Whenever the cell directly ahead is teal, mark that ahead teal pixel as causal and turn into the unique black left-or-right neighbor.
    - Stop as soon as the cell directly ahead belongs to the red line; output the green and red lines unchanged, recolor every causal teal pixel to yellow, and erase all other teal pixels to black.

    Key insight:
    - The output hides the walked path itself and preserves only the obstacle pixels that were directly in front at actual turning moments, so the walk must be simulated rather than inferred from local proximity alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED, GREEN, TEAL}, "unexpected input colors"
    causal = set(_simulate_turn_barriers(g))

    out = np.full(g.shape, BLACK, dtype=int)
    out[g == GREEN] = GREEN
    out[g == RED] = RED
    for r, c in causal:
        out[r, c] = YELLOW
    return out


def _cells_along(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    cells = []
    current = start
    for _ in range(length):
        current = _add(current, direction)
        cells.append(current)
    return cells


def _ray_to_border(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    cells = []
    current = start
    while _in_bounds(shape, current):
        cells.append(current)
        current = _add(current, direction)
    return cells


def _green_marker(
    shape: tuple[int, int],
    side: str,
    length: int,
    offset: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    H, W = shape
    if side == "top":
        cells = [(r, offset) for r in range(length)]
        return cells, cells[-1], (1, 0)
    if side == "bottom":
        cells = [(H - length + r, offset) for r in range(length)]
        return cells, cells[0], (-1, 0)
    if side == "left":
        cells = [(offset, c) for c in range(length)]
        return cells, cells[-1], (0, 1)
    if side == "right":
        cells = [(offset, W - length + c) for c in range(length)]
        return cells, cells[0], (0, -1)
    raise ValueError("invalid side")


def _offset_choices(shape: tuple[int, int], side: str) -> list[int]:
    H, W = shape
    if side in {"top", "bottom"}:
        return list(range(2, W - 2))
    return list(range(2, H - 2))


def _build_walk(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    turns_left: int,
    occupied: set[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]] | None:
    if turns_left == 0:
        lengths = list(range(FINAL_SEGMENT_LEN_RANGE[0], FINAL_SEGMENT_LEN_RANGE[1] + 1))
        random.shuffle(lengths)
        for length in lengths:
            segment = _cells_along(current, direction, length)
            if any((not _in_bounds(shape, cell)) or (cell in occupied) for cell in segment):
                continue
            end_pos = segment[-1]
            red_cells = _ray_to_border(shape, _add(end_pos, direction), direction)
            if not (RED_LEN_RANGE[0] <= len(red_cells) <= RED_LEN_RANGE[1]):
                continue
            if any(cell in occupied or cell in segment for cell in red_cells):
                continue
            return {
                "path": segment,
                "causal": [],
                "decoy": [],
                "red": red_cells,
            }
        return None

    next_directions = [_turn_left(direction), _turn_right(direction)]
    random.shuffle(next_directions)
    lengths = list(range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1))
    random.shuffle(lengths)

    for length in lengths:
        segment = _cells_along(current, direction, length)
        if any((not _in_bounds(shape, cell)) or (cell in occupied) for cell in segment):
            continue
        turn_pos = segment[-1]

        for next_direction in next_directions:
            front = _add(turn_pos, direction)
            decoy = _add(turn_pos, _other_turn(direction, next_direction))
            next_step = _add(turn_pos, next_direction)
            needed = [front, decoy, next_step]
            if any((not _in_bounds(shape, cell)) for cell in needed):
                continue
            if front in occupied or decoy in occupied or next_step in occupied:
                continue
            if front in segment or decoy in segment or next_step in segment:
                continue

            new_occupied = occupied | set(segment) | {front, decoy}
            tail = _build_walk(shape, turn_pos, next_direction, turns_left - 1, new_occupied)
            if tail is None:
                continue
            return {
                "path": segment + tail["path"],
                "causal": [front] + tail["causal"],
                "decoy": [decoy] + tail["decoy"],
                "red": tail["red"],
            }
    return None


def _scatter_extras(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    green: list[tuple[int, int]],
    red: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    blocked = set(reserved)
    for marker in green + red:
        blocked.add(marker)
        for direction in DIRS:
            neighbor = _add(marker, direction)
            if _in_bounds(shape, neighbor):
                blocked.add(neighbor)

    pool = [
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
        if (r, c) not in blocked
    ]
    random.shuffle(pool)
    extra_count = min(random.randint(*EXTRA_BARRIER_RANGE), len(pool))
    return pool[:extra_count]


def _render_grid(
    shape: tuple[int, int],
    green: list[tuple[int, int]],
    red: list[tuple[int, int]],
    teal_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in green:
        grid[r, c] = GREEN
    for r, c in red:
        grid[r, c] = RED
    for r, c in teal_cells:
        grid[r, c] = TEAL
    return grid


def _expected_output(
    shape: tuple[int, int],
    green: list[tuple[int, int]],
    red: list[tuple[int, int]],
    causal: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for r, c in green:
        out[r, c] = GREEN
    for r, c in red:
        out[r, c] = RED
    for r, c in causal:
        out[r, c] = YELLOW
    return out


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    side = random.choice(["top", "bottom", "left", "right"])
    green_len = random.randint(*GREEN_LEN_RANGE)
    offsets = _offset_choices(shape, side)
    if not offsets:
        return None

    offset = random.choice(offsets)
    green, start_pos, start_direction = _green_marker(shape, side, green_len, offset)
    turns = random.randint(*TURN_COUNT_RANGE)
    walk = _build_walk(shape, start_pos, start_direction, turns, set(green))
    if walk is None:
        return None

    reserved = set(green) | set(walk["path"]) | set(walk["causal"]) | set(walk["decoy"]) | set(walk["red"])
    extras = _scatter_extras(shape, reserved, green, walk["red"])
    teal_cells = walk["causal"] + walk["decoy"] + extras
    inp = _render_grid(shape, green, walk["red"], teal_cells)
    expected = _expected_output(shape, green, walk["red"], walk["causal"])

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-turning ARC input")
