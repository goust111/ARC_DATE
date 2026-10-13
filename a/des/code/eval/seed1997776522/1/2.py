from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
GRID_H_RANGE = (8, 15)
GRID_W_RANGE = (8, 15)


Coord = tuple[int, int]
Step = tuple[int, int]


def _add(a: Coord, b: Step) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Coord, b: Coord) -> Step:
    return a[0] - b[0], a[1] - b[1]


def _left_of(step: Step) -> Step:
    dr, dc = step
    return -dc, dr


def _right_of(step: Step) -> Step:
    dr, dc = step
    return dc, -dr


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _on_edge(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    h, w = shape
    return r in (0, h - 1) or c in (0, w - 1)


def _is_corner(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    h, w = shape
    return r in (0, h - 1) and c in (0, w - 1)


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _path_neighbors(path_cells: set[Coord], cell: Coord) -> list[Coord]:
    return [nbr for nbr in _neighbors4(cell) if nbr in path_cells]


def _select_start_endpoint(endpoints: list[Coord], shape: tuple[int, int]) -> Coord:
    return min(endpoints, key=lambda cell: (0 if _on_edge(shape, cell) else 1, cell[0], cell[1]))


def _ordered_path_from_grid(path_cells: set[Coord], shape: tuple[int, int]) -> list[Coord]:
    endpoints = [cell for cell in path_cells if len(_path_neighbors(path_cells, cell)) == 1]
    assert len(endpoints) == 2, "expected exactly two path endpoints"
    start = _select_start_endpoint(endpoints, shape)

    ordered = [start]
    prev: Coord | None = None
    current = start
    while True:
        options = [nbr for nbr in _path_neighbors(path_cells, current) if nbr != prev]
        if not options:
            break
        assert len(options) == 1, "path must not branch"
        nxt = options[0]
        ordered.append(nxt)
        prev, current = current, nxt

    assert len(ordered) == len(path_cells), "path traversal must cover all green cells"
    return ordered


def _path_from_grid(grid: np.ndarray) -> set[Coord]:
    g = np.asarray(grid, dtype=int)
    path_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == GREEN)}
    assert path_cells, "expected a green path"
    for cell in path_cells:
        degree = len(_path_neighbors(path_cells, cell))
        assert degree in (1, 2), "path must be a single-pixel-wide simple chain"
    return path_cells


def _marker_positions(start: Coord, initial_step: Step) -> tuple[Coord, Coord]:
    left_pos = _add(start, _left_of(initial_step))
    right_pos = _add(start, _right_of(initial_step))
    return left_pos, right_pos


def _turn_cross(in_step: Step, out_step: Step) -> int:
    return in_step[0] * out_step[1] - in_step[1] * out_step[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the green path from the selected endpoint and sow side markers at turns.

    The two start-adjacent red/blue pixels define which visible color means
    "left" and which means "right". At each 90-degree turn, the color follows
    the turn direction but the placement goes to the opposite outside side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    path_cells = _path_from_grid(g)
    ordered = _ordered_path_from_grid(path_cells, g.shape)
    assert len(ordered) >= 2, "path must have at least two cells"

    start = ordered[0]
    initial_step = _sub(ordered[1], ordered[0])
    left_pos, right_pos = _marker_positions(start, initial_step)
    assert _in_bounds(g.shape, left_pos) and _in_bounds(g.shape, right_pos)

    left_color = int(g[left_pos])
    right_color = int(g[right_pos])
    assert {left_color, right_color} == {RED, BLUE}, "start markers must be one red and one blue"

    out = g.copy()
    for i in range(1, len(ordered) - 1):
        prev_cell = ordered[i - 1]
        cell = ordered[i]
        next_cell = ordered[i + 1]
        in_step = _sub(cell, prev_cell)
        out_step = _sub(next_cell, cell)
        if in_step == out_step:
            continue

        cross = _turn_cross(in_step, out_step)
        assert cross in (-1, 1), "path turns must be 90 degrees"
        if cross > 0:
            target = _add(cell, _right_of(in_step))
            color = left_color
        else:
            target = _add(cell, _left_of(in_step))
            color = right_color

        if _in_bounds(g.shape, target) and int(out[target]) == BG:
            out[target] = color

    return out


def _can_append(shape: tuple[int, int], occupied: set[Coord], current: Coord, nxt: Coord, *, allow_edge: bool) -> bool:
    if not _in_bounds(shape, nxt) or nxt in occupied:
        return False
    if _on_edge(shape, nxt) and not allow_edge:
        return False
    return {nbr for nbr in _neighbors4(nxt) if nbr in occupied} == {current}


def _has_future_move(shape: tuple[int, int], occupied: set[Coord], current: Coord, direction: Step) -> bool:
    for step in (direction, _left_of(direction), _right_of(direction)):
        nxt = _add(current, step)
        if not _in_bounds(shape, nxt) or nxt in occupied:
            continue
        if _on_edge(shape, nxt):
            if not _is_corner(shape, nxt) and _can_append(shape, occupied, current, nxt, allow_edge=True):
                return True
            continue
        if _can_append(shape, occupied, current, nxt, allow_edge=False):
            return True
    return False


def _generate_base_path(height: int, width: int, want_blocked: bool) -> list[Coord] | None:
    shape = (height, width)
    start = (0, random.randint(1, width - 2))
    path = [start, (1, start[1])]
    occupied = set(path)
    direction = (1, 0)
    turns = 0

    min_turns = random.randint(4, 7) if want_blocked else random.randint(2, 5)
    min_len_lo = max(10, min_turns * 2 + 4)
    min_len_hi = max(min_len_lo, min(height * width // 2, height + width + 8))
    min_len = random.randint(min_len_lo, min_len_hi)
    max_steps = min(height * width // 2, max(min_len + 8, 2 * (height + width)))

    while len(path) < max_steps:
        current = path[-1]
        options: list[tuple[Step, Coord, bool]] = []
        for step in (direction, _left_of(direction), _right_of(direction)):
            nxt = _add(current, step)
            extra_turn = int(step != direction)
            if not _in_bounds(shape, nxt) or nxt in occupied:
                continue
            if _on_edge(shape, nxt):
                if (
                    not _is_corner(shape, nxt)
                    and len(path) + 1 >= min_len
                    and turns + extra_turn >= min_turns
                    and _can_append(shape, occupied, current, nxt, allow_edge=True)
                ):
                    options.append((step, nxt, True))
                continue
            if not _can_append(shape, occupied, current, nxt, allow_edge=False):
                continue
            next_occupied = occupied | {nxt}
            if _has_future_move(shape, next_occupied, nxt, step):
                options.append((step, nxt, False))

        edge_options = [item for item in options if item[2]]
        interior_options = [item for item in options if not item[2]]
        if edge_options and (not interior_options or (len(path) >= min_len and random.random() < 0.35)):
            step, nxt, _ = random.choice(edge_options)
            if step != direction:
                turns += 1
            path.append(nxt)
            return path
        if not interior_options:
            if edge_options:
                step, nxt, _ = random.choice(edge_options)
                if step != direction:
                    turns += 1
                path.append(nxt)
                return path
            return None

        weighted: list[tuple[Step, Coord, bool]] = []
        for item in interior_options:
            step, _nxt, _ = item
            weight = 4 if step == direction else 3
            if turns < min_turns and step == direction:
                weight = 1
            weighted.extend([item] * weight)

        step, nxt, _ = random.choice(weighted)
        if step != direction:
            turns += 1
        direction = step
        path.append(nxt)
        occupied.add(nxt)

    return None


def _transform_path(path: list[Coord], height: int, width: int) -> tuple[list[Coord], tuple[int, int]]:
    kind = random.randint(0, 7)
    if kind == 0:
        return path[:], (height, width)
    if kind == 1:
        return [(c, height - 1 - r) for r, c in path], (width, height)
    if kind == 2:
        return [(height - 1 - r, width - 1 - c) for r, c in path], (height, width)
    if kind == 3:
        return [(width - 1 - c, r) for r, c in path], (width, height)
    if kind == 4:
        return [(r, width - 1 - c) for r, c in path], (height, width)
    if kind == 5:
        return [(height - 1 - r, c) for r, c in path], (height, width)
    if kind == 6:
        return [(c, r) for r, c in path], (width, height)
    return [(width - 1 - c, height - 1 - r) for r, c in path], (width, height)


def _ordered_generated_path(path: list[Coord], shape: tuple[int, int]) -> list[Coord]:
    endpoints = [path[0], path[-1]]
    start = _select_start_endpoint(endpoints, shape)
    return path[:] if path[0] == start else list(reversed(path))


def _seed_stats(path: list[Coord], shape: tuple[int, int]) -> tuple[list[Coord], int, int, int, int, int]:
    ordered = _ordered_generated_path(path, shape)
    initial_step = _sub(ordered[1], ordered[0])
    left_pos, right_pos = _marker_positions(ordered[0], initial_step)
    if not (_in_bounds(shape, left_pos) and _in_bounds(shape, right_pos)):
        return ordered, -1, -1, -1, -1, -1

    occupied = set(ordered)
    occupied.add(left_pos)
    occupied.add(right_pos)

    blocked = 0
    placed_left = 0
    placed_right = 0
    left_turns = 0
    right_turns = 0
    for i in range(1, len(ordered) - 1):
        prev_cell = ordered[i - 1]
        cell = ordered[i]
        next_cell = ordered[i + 1]
        in_step = _sub(cell, prev_cell)
        out_step = _sub(next_cell, cell)
        if in_step == out_step:
            continue

        cross = _turn_cross(in_step, out_step)
        if cross > 0:
            left_turns += 1
            target = _add(cell, _right_of(in_step))
            place_left = True
        else:
            right_turns += 1
            target = _add(cell, _left_of(in_step))
            place_left = False

        if target in occupied:
            blocked += 1
            continue
        occupied.add(target)
        if place_left:
            placed_left += 1
        else:
            placed_right += 1

    return ordered, blocked, placed_left, placed_right, left_turns, right_turns


def _render_input(path: list[Coord], shape: tuple[int, int]) -> np.ndarray | None:
    ordered, blocked, placed_left, placed_right, left_turns, right_turns = _seed_stats(path, shape)
    if min(blocked, placed_left, placed_right, left_turns, right_turns) < 0:
        return None

    want_blocked = bool(random.getrandbits(1))
    if (blocked > 0) != want_blocked:
        return None
    if left_turns == 0 or right_turns == 0:
        return None
    if placed_left == 0 or placed_right == 0:
        return None

    grid = np.full(shape, BG, dtype=int)
    for r, c in ordered:
        grid[r, c] = GREEN

    initial_step = _sub(ordered[1], ordered[0])
    left_pos, right_pos = _marker_positions(ordered[0], initial_step)
    if random.random() < 0.5:
        grid[left_pos] = RED
        grid[right_pos] = BLUE
    else:
        grid[left_pos] = BLUE
        grid[right_pos] = RED

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        base_path = _generate_base_path(height, width, want_blocked=bool(random.getrandbits(1)))
        if base_path is None:
            continue
        path, shape = _transform_path(base_path, height, width)
        grid = _render_input(path, shape)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
