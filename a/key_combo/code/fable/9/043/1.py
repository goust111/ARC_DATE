from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREEN = 3
PINK = 6
TEAL = 8
CONTROL_COLORS = (PINK, TEAL)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {direction: idx for idx, direction in enumerate(DIRS)}

GRID_HEIGHT_RANGE = (10, 17)
GRID_WIDTH_RANGE = (10, 17)
TOTAL_TURNS_RANGE = (2, 6)
TURN_SEGMENT_LENGTHS = (1, 2, 3, 4)
FINAL_SEGMENT_RANGE = (2, 7)
TOTAL_PATH_CELLS_RANGE = (9, 26)
MAX_GENERATION_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def add(cell: tuple[int, int], direction: tuple[int, int], step: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * step, cell[1] + direction[1] * step


def left_of(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIR_TO_INDEX[direction] - 1) % 4]


def right_of(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIR_TO_INDEX[direction] + 1) % 4]


def turn_dir(direction: tuple[int, int], turn: int) -> tuple[int, int]:
    return left_of(direction) if turn < 0 else right_of(direction)


def diff(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def opposite_turn(turn: int) -> int:
    return -turn


def path_neighbors(cell: tuple[int, int], green_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    neighbors = []
    for dr, dc in DIRS:
        nxt = (r + dr, c + dc)
        if nxt in green_set:
            neighbors.append(nxt)
    return neighbors


def order_path(green_set: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    current = start
    while True:
        nxts = [cell for cell in path_neighbors(current, green_set) if cell != prev]
        if not nxts:
            return ordered
        assert len(nxts) == 1, "green prefix must be a simple non-branching path"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, current = current, nxt


def infer_order_mapping(
    grid: np.ndarray, ordered: list[tuple[int, int]]
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    bend_count = 0
    seen_bend_cells: set[tuple[int, int]] = set()
    for i in range(1, len(ordered) - 1):
        prev_cell = ordered[i - 1]
        cell = ordered[i]
        next_cell = ordered[i + 1]
        incoming = diff(cell, prev_cell)
        outgoing = diff(next_cell, cell)
        if incoming == outgoing:
            continue
        bend_count += 1
        seen_bend_cells.add(cell)
        control = add(cell, incoming)
        if not in_bounds(grid.shape, control[0], control[1]):
            return None
        color = int(grid[control])
        if color not in CONTROL_COLORS:
            return None
        if outgoing == left_of(incoming):
            turn = -1
        elif outgoing == right_of(incoming):
            turn = 1
        else:
            return None
        if color in mapping and mapping[color] != turn:
            return None
        mapping[color] = turn

    if bend_count == 0:
        return None
    if len(mapping) == 2 and mapping[PINK] == mapping[TEAL]:
        return None
    if len(mapping) == 1:
        only_color = next(iter(mapping))
        other_color = TEAL if only_color == PINK else PINK
        mapping[other_color] = opposite_turn(mapping[only_color])
    if len(mapping) != 2:
        return None

    green_set = set(ordered)
    for r, c in np.argwhere(np.isin(grid, CONTROL_COLORS)):
        control = (int(r), int(c))
        adjacent_greens = sum((neighbor in green_set) for neighbor in path_neighbors(control, green_set))
        if adjacent_greens > 1:
            return None
    return mapping


def infer_scene(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict[int, int]]:
    green_coords = [tuple(map(int, cell)) for cell in np.argwhere(grid == GREEN)]
    green_set = set(green_coords)
    assert len(green_set) >= 4, "expected a visible green prefix"

    degrees = {cell: len(path_neighbors(cell, green_set)) for cell in green_set}
    assert all(degree in (1, 2) for degree in degrees.values()), "green prefix must be a simple path"
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "green prefix must have exactly two endpoints"

    candidates: list[tuple[list[tuple[int, int]], dict[int, int]]] = []
    for endpoint in endpoints:
        ordered = order_path(green_set, endpoint)
        mapping = infer_order_mapping(grid, ordered)
        if mapping is not None:
            candidates.append((ordered, mapping))

    assert len(candidates) == 1, "prefix must uniquely identify the travelled direction"
    return candidates[0]


def continue_from_prefix(
    grid: np.ndarray,
    ordered: list[tuple[int, int]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    current = ordered[-1]
    direction = diff(ordered[-1], ordered[-2])
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (current[0], current[1], direction[0], direction[1])
        assert state not in seen_states, "ray must terminate instead of cycling"
        seen_states.add(state)

        nxt = add(current, direction)
        if not in_bounds(grid.shape, nxt[0], nxt[1]):
            return out

        value = int(grid[nxt])
        if value == BLACK:
            out[nxt] = GREEN
            current = nxt
            continue
        if value in CONTROL_COLORS:
            direction = turn_dir(direction, mapping[value])
            continue
        raise AssertionError("continuation may only traverse background and turn at controls")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the green path prefix by learning from its own demonstrated turn color semantics.

    Input structure:
    - A black-background grid contains one continuous green path prefix with two endpoints.
    - Pink and teal singleton control pixels remain visible; at least one already-passed control is encoded by a bend in the prefix.
    - The prefix has already travelled past one or more controls and stopped mid-route before the final exit.

    Rule:
    - Determine which endpoint is the current tip by finding the unique path ordering whose bends each have a control directly ahead in the incoming direction.
    - Each such bend demonstrates whether that control color means turn left or turn right; the other control color has the opposite meaning.
    - From the current tip, continue straight through black cells painting them green, turn in place when the next cell is a control, and stop when the next step would leave the grid.

    Key insight:
    - The color-to-turn mapping is not fixed across examples; it must be read from the already-drawn prefix geometry, not memorized from color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    allowed = {BLACK, GREEN, PINK, TEAL}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors in the scene"

    ordered, mapping = infer_scene(g)
    return continue_from_prefix(g, ordered, mapping)


def build_branch_turns(total_turns: int) -> tuple[list[int], int]:
    if total_turns == 2:
        branch = "single_demo"
    else:
        branch = random.choice(["single_demo", "both_demo"])
    prefix_turns = random.randint(1, total_turns - 1)

    if branch == "single_demo":
        demo_turn = random.choice([-1, 1])
        turns = [demo_turn] * prefix_turns
        future = [opposite_turn(demo_turn)]
        while len(turns) + len(future) < total_turns:
            future.append(random.choice([-1, 1]))
        random.shuffle(future)
        return turns + future, prefix_turns

    while True:
        prefix_turns = random.randint(2, total_turns - 1)
        first = [random.choice([-1, 1]) for _ in range(prefix_turns)]
        if -1 in first and 1 in first:
            rest = [random.choice([-1, 1]) for _ in range(total_turns - prefix_turns)]
            return first + rest, prefix_turns


def signs_to_colors(turns: Iterable[int], pink_is_left: bool) -> list[int]:
    colors = []
    for turn in turns:
        if pink_is_left:
            colors.append(PINK if turn < 0 else TEAL)
        else:
            colors.append(TEAL if turn < 0 else PINK)
    return colors


def edge_distance(shape: tuple[int, int], cell: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = cell
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return shape[0] - 1 - r
    if dc < 0:
        return c
    return shape[1] - 1 - c


def ray_cells(cell: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [add(cell, direction, step) for step in range(1, length + 1)]


def path_and_controls_clear(
    new_cells: Iterable[tuple[int, int]],
    control_cell: tuple[int, int] | None,
    path_set: set[tuple[int, int]],
    control_positions: set[tuple[int, int]],
) -> bool:
    for cell in new_cells:
        if cell in path_set or cell in control_positions:
            return False
    if control_cell is not None and (control_cell in path_set or control_cell in control_positions):
        return False
    return True


def search_full_path(
    shape: tuple[int, int],
    turns: list[int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]] | None:
    def rec(
        turn_index: int,
        current: tuple[int, int],
        current_direction: tuple[int, int],
        ordered: list[tuple[int, int]],
        path_set: set[tuple[int, int]],
        controls: list[tuple[int, int]],
        corner_indices: list[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]] | None:
        if turn_index == len(turns):
            final_len = edge_distance(shape, current, current_direction)
            if not (FINAL_SEGMENT_RANGE[0] <= final_len <= FINAL_SEGMENT_RANGE[1]):
                return None
            final_cells = ray_cells(current, current_direction, final_len)
            if not path_and_controls_clear(final_cells, None, path_set, set(controls)):
                return None
            total_cells = len(ordered) + len(final_cells)
            if not (TOTAL_PATH_CELLS_RANGE[0] <= total_cells <= TOTAL_PATH_CELLS_RANGE[1]):
                return None
            return ordered + final_cells, controls, corner_indices

        lengths = list(TURN_SEGMENT_LENGTHS)
        random.shuffle(lengths)
        control_positions = set(controls)
        for length in lengths:
            cells = ray_cells(current, current_direction, length)
            if not all(in_bounds(shape, r, c) for r, c in cells):
                continue
            control = add(current, current_direction, length + 1)
            if not in_bounds(shape, control[0], control[1]):
                continue
            if not path_and_controls_clear(cells, control, path_set, control_positions):
                continue
            next_direction = turn_dir(current_direction, turns[turn_index])
            if edge_distance(shape, cells[-1], next_direction) < 1:
                continue

            result = rec(
                turn_index + 1,
                cells[-1],
                next_direction,
                ordered + cells,
                path_set | set(cells),
                controls + [control],
                corner_indices + [len(ordered) + len(cells) - 1],
            )
            if result is not None:
                return result
        return None

    return rec(0, start, direction, [start], {start}, [], [])


def control_adjacency_is_clean(
    path: list[tuple[int, int]], controls: list[tuple[int, int]]
) -> bool:
    path_set = set(path)
    for control in controls:
        neighbors = sum((neighbor in path_set) for neighbor in path_neighbors(control, path_set))
        if neighbors != 1:
            return False
    return True


def render_grids(
    shape: tuple[int, int],
    full_path: list[tuple[int, int]],
    control_positions: list[tuple[int, int]],
    control_colors: list[int],
    prefix_end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    for cell, color in zip(control_positions, control_colors):
        inp[cell] = color
        out[cell] = color
    for cell in full_path:
        out[cell] = GREEN
    for cell in full_path[: prefix_end_index + 1]:
        inp[cell] = GREEN
    return inp, out


def simulate_with_mapping(
    grid: np.ndarray,
    ordered: list[tuple[int, int]],
    mapping: dict[int, int],
) -> np.ndarray | None:
    try:
        return continue_from_prefix(grid, ordered, mapping)
    except AssertionError:
        return None


def attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    total_turns = random.randint(*TOTAL_TURNS_RANGE)
    turns, prefix_turns = build_branch_turns(total_turns)
    pink_is_left = random.choice([False, True])
    control_colors = signs_to_colors(turns, pink_is_left)

    starts = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(starts)
    directions = DIRS[:]
    random.shuffle(directions)
    built = None
    for start in starts:
        for direction in directions:
            built = search_full_path((height, width), turns, start, direction)
            if built is not None:
                break
        if built is not None:
            break
    if built is None:
        return None

    full_path, control_positions, corner_indices = built
    if not control_adjacency_is_clean(full_path, control_positions):
        return None

    last_demo_corner = corner_indices[prefix_turns - 1]
    next_future_corner = corner_indices[prefix_turns]
    if next_future_corner - last_demo_corner < 1:
        return None
    prefix_end_index = random.randint(last_demo_corner + 1, next_future_corner)

    inp, out = render_grids(
        (height, width),
        full_path,
        control_positions,
        control_colors,
        prefix_end_index,
    )

    try:
        ordered, inferred_mapping = infer_scene(inp)
    except AssertionError:
        return None
    if not np.array_equal(main(inp), out):
        return None

    swapped = {PINK: opposite_turn(inferred_mapping[PINK]), TEAL: opposite_turn(inferred_mapping[TEAL])}
    swapped_out = simulate_with_mapping(inp, ordered, swapped)
    if swapped_out is not None and np.array_equal(swapped_out, out):
        return None

    pending_colors = {int(inp[cell]) for cell in control_positions[prefix_turns:]}
    seen_colors = {int(inp[cell]) for cell in control_positions[:prefix_turns]}
    if len(seen_colors) == 1 and pending_colors <= seen_colors:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = attempt_generate()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid prefix-demonstration ray input")
