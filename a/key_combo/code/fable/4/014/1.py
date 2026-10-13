from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
TEAL = 8
OBSTACLE_COLORS = (YELLOW, TEAL)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS)}
TURN_DELTA = {"R": 1, "L": -1}

GEN_RETRIES = 300
PATH_RETRIES = 120
SEGMENT_MIN = 2
SEGMENT_MAX = 6
FINAL_SEGMENT_MAX = 7
GRID_MIN = 16
GRID_MAX = 26
TURN_COUNT_CHOICES = (4, 5, 6)
DECOY_MAX = 4


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _step(cell: tuple[int, int], dir_idx: int) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return cell[0] + dr, cell[1] + dc


def _is_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _rotate(dir_idx: int, turn: str) -> int:
    return (dir_idx + TURN_DELTA[turn]) % 4


def _turn_name(in_idx: int, out_idx: int) -> str:
    delta = (out_idx - in_idx) % 4
    if delta == 1:
        return "R"
    if delta == 3:
        return "L"
    raise ValueError("turn must be a ninety-degree rotation")


def _trace_green_path(grid: np.ndarray) -> list[tuple[int, int]]:
    green = grid == GREEN
    coords = [tuple(map(int, coord)) for coord in np.argwhere(green)]
    assert coords, "expected a green path prefix"

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for cell in coords:
        nbrs = [
            nb
            for nb in _neighbors4(cell)
            if _in_bounds(grid.shape, nb[0], nb[1]) and bool(green[nb])
        ]
        adjacency[cell] = nbrs
        degree = len(nbrs)
        if degree == 0:
            assert len(coords) == 1
            endpoints.append(cell)
        elif degree == 1:
            endpoints.append(cell)
        else:
            assert degree == 2, "green prefix must be a simple polyline"

    assert endpoints, "path prefix must have an endpoint"
    boundary_endpoints = [cell for cell in endpoints if _is_boundary(grid.shape, *cell)]
    if len(boundary_endpoints) == 1:
        start = boundary_endpoints[0]
    elif boundary_endpoints:
        start = boundary_endpoints[0]
    else:
        start = endpoints[0]

    ordered = [start]
    prev = None
    curr = start
    while True:
        next_cells = [cell for cell in adjacency[curr] if cell != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "green prefix must not branch"
        prev, curr = curr, next_cells[0]
        ordered.append(curr)

    assert len(ordered) == len(coords), "green prefix must be one connected path"
    return ordered


def _infer_turn_mapping(grid: np.ndarray, path: list[tuple[int, int]]) -> dict[int, str]:
    obstacle_colors = {
        int(color)
        for color in np.unique(grid)
        if int(color) not in (BLACK, RED, GREEN)
    }
    assert obstacle_colors == set(OBSTACLE_COLORS), "expected yellow and teal obstacle colors"

    mapping: dict[int, str] = {}
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        curr = path[i]
        nxt = path[i + 1]
        in_dir = (curr[0] - prev[0], curr[1] - prev[1])
        out_dir = (nxt[0] - curr[0], nxt[1] - curr[1])
        if in_dir == out_dir:
            continue

        in_idx = DIR_TO_IDX[in_dir]
        out_idx = DIR_TO_IDX[out_dir]
        ahead = curr[0] + in_dir[0], curr[1] + in_dir[1]
        assert _in_bounds(grid.shape, ahead[0], ahead[1])
        color = int(grid[ahead])
        assert color in obstacle_colors, "turn sample must point at an obstacle"
        turn = _turn_name(in_idx, out_idx)
        if color in mapping:
            assert mapping[color] == turn, "per-color turn code must be consistent"
        mapping[color] = turn

    assert set(mapping) == obstacle_colors, "prefix must demonstrate both obstacle colors"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the green path until the red stop line by decoding which obstacle color means left or right.

    Input structure:
    - A black background contains teal and yellow single-cell obstacles plus one short red terminal line.
    - A green 4-connected polyline starts on the boundary and already includes completed turns around both obstacle colors.
    - The green prefix is a simple non-branching path whose final direction is the continuation direction.

    Rule:
    - Recover the ordered green prefix from its boundary endpoint to its current endpoint.
    - At each completed turn, inspect the obstacle directly ahead of the turn cell to infer whether that color encodes a left or right turn in this grid.
    - From the current endpoint, keep moving straight; whenever the next cell is an obstacle, rotate ninety degrees according to that obstacle color and continue.
    - Stop just before entering the red line, and paint every newly visited continuation cell green while leaving all other cells unchanged.

    Key insight:
    - The visible green prefix is itself the legend for the per-instance turn code, so solving the task means reading past turns and then replaying the same deterministic motion law on the unfinished suffix.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    path = _trace_green_path(g)
    assert len(path) >= 2, "prefix must reveal a current direction"
    turn_by_color = _infer_turn_mapping(g, path)

    current = path[-1]
    heading = DIR_TO_IDX[(current[0] - path[-2][0], current[1] - path[-2][1])]

    for _ in range(g.size * 8):
        while True:
            nr, nc = _step(current, heading)
            assert _in_bounds(g.shape, nr, nc), "continuation left the grid"
            next_color = int(g[nr, nc])
            if next_color in turn_by_color:
                heading = _rotate(heading, turn_by_color[next_color])
                continue
            if next_color == RED:
                return out
            break

        current = nr, nc
        out[current] = GREEN

    raise RuntimeError("continuation did not terminate")


def _can_add_path_cell(
    cell: tuple[int, int],
    prev: tuple[int, int],
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c):
        return False
    if cell in path_set or cell in obstacles:
        return False
    if _is_boundary(shape, r, c):
        return False
    for nb in _neighbors4(cell):
        if nb in path_set and nb != prev:
            return False
    return True


def _extend_straight(
    current: tuple[int, int],
    heading: int,
    length: int,
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
) -> list[tuple[int, int]] | None:
    new_cells: list[tuple[int, int]] = []
    prev = current
    temp_path = set(path_set)
    for _ in range(length):
        nxt = _step(prev, heading)
        if not _can_add_path_cell(nxt, prev, shape, temp_path, obstacles):
            return None
        new_cells.append(nxt)
        temp_path.add(nxt)
        prev = nxt
    return new_cells


def _length_options(
    current: tuple[int, int],
    heading: int,
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
    max_len: int,
) -> list[int]:
    options = []
    for length in range(SEGMENT_MIN, max_len + 1):
        if _extend_straight(current, heading, length, shape, path_set, obstacles) is None:
            break
        options.append(length)
    random.shuffle(options)
    return options


def _choose_red_line(
    end: tuple[int, int],
    heading: int,
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
) -> list[tuple[int, int]] | None:
    base = _step(end, heading)
    if not _in_bounds(shape, base[0], base[1]):
        return None
    if base in path_set or base in obstacles:
        return None

    candidates: list[list[tuple[int, int]]] = []
    for turn in ("L", "R"):
        perp = _rotate(heading, turn)
        cells = [base]
        for length in range(2, 5):
            if length > 2:
                cells.append(_step(cells[-1], perp))
            if any(
                (not _in_bounds(shape, r, c))
                or (cell in path_set)
                or (cell in obstacles)
                for cell in cells
                for r, c in [cell]
            ):
                break
            ok = True
            for idx, cell in enumerate(cells):
                allowed_green = {end} if idx == 0 else set()
                green_neighbors = {nb for nb in _neighbors4(cell) if nb in path_set}
                if not green_neighbors <= allowed_green:
                    ok = False
                    break
            if ok:
                candidates.append(list(cells))
    if not candidates:
        return None
    return random.choice(candidates)


def _search_path(
    current: tuple[int, int],
    heading: int,
    turn_sequence: list[str],
    seg_idx: int,
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
    turn_records: list[tuple[int, int]],
    turn_to_color: dict[str, int],
) -> list[tuple[int, int]] | None:
    max_len = FINAL_SEGMENT_MAX if seg_idx == len(turn_sequence) else SEGMENT_MAX
    for length in _length_options(current, heading, shape, path_set, obstacles, max_len):
        new_cells = _extend_straight(current, heading, length, shape, path_set, obstacles)
        if new_cells is None:
            continue

        for cell in new_cells:
            path.append(cell)
            path_set.add(cell)
        end = path[-1]

        if seg_idx == len(turn_sequence):
            red_line = _choose_red_line(end, heading, shape, path_set, obstacles)
            if red_line is not None:
                return red_line
        else:
            turn = turn_sequence[seg_idx]
            next_heading = _rotate(heading, turn)
            obstacle_cell = _step(end, heading)
            first_after_turn = _step(end, next_heading)
            if (
                _in_bounds(shape, obstacle_cell[0], obstacle_cell[1])
                and obstacle_cell not in path_set
                and obstacle_cell not in obstacles
                and _can_add_path_cell(first_after_turn, end, shape, path_set, obstacles)
            ):
                obstacles[obstacle_cell] = turn_to_color[turn]
                turn_records.append((len(path) - 1, turn_to_color[turn]))
                red_line = _search_path(
                    end,
                    next_heading,
                    turn_sequence,
                    seg_idx + 1,
                    shape,
                    path,
                    path_set,
                    obstacles,
                    turn_records,
                    turn_to_color,
                )
                if red_line is not None:
                    return red_line
                turn_records.pop()
                del obstacles[obstacle_cell]

        for cell in reversed(new_cells):
            path_set.remove(cell)
            path.pop()
    return None


def _start_state(shape: tuple[int, int]) -> tuple[tuple[int, int], int]:
    h, w = shape
    side = random.randrange(4)
    if side == 0:
        return (0, random.randint(2, w - 3)), 2
    if side == 1:
        return (random.randint(2, h - 3), w - 1), 3
    if side == 2:
        return (h - 1, random.randint(2, w - 3)), 0
    return (random.randint(2, h - 3), 0), 1


def _turn_sequence(n_turns: int) -> list[str]:
    turns = ["L", "R"]
    random.shuffle(turns)
    turns.extend(random.choice(("L", "R")) for _ in range(n_turns - 2))
    return turns


def _place_decoys(
    grid: np.ndarray,
    path_cells: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
) -> None:
    candidates = []
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            cell = (r, c)
            if cell in path_cells or cell in red_cells or cell in obstacles:
                continue
            if any(nb in path_cells or nb in red_cells for nb in _neighbors4(cell)):
                continue
            candidates.append(cell)

    random.shuffle(candidates)
    for _ in range(random.randint(0, min(DECOY_MAX, len(candidates)))):
        if not candidates:
            break
        cell = candidates.pop()
        grid[cell] = random.choice(OBSTACLE_COLORS)


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    color_to_turn = {
        YELLOW: random.choice(("L", "R")),
        TEAL: None,
    }
    color_to_turn[TEAL] = "R" if color_to_turn[YELLOW] == "L" else "L"
    turn_to_color = {turn: color for color, turn in color_to_turn.items()}

    n_turns = random.choice(TURN_COUNT_CHOICES)
    turn_sequence = _turn_sequence(n_turns)

    for _ in range(PATH_RETRIES):
        start, heading = _start_state(shape)
        path = [start]
        path_set = {start}
        obstacles: dict[tuple[int, int], int] = {}
        turn_records: list[tuple[int, int]] = []

        red_line = _search_path(
            start,
            heading,
            turn_sequence,
            0,
            shape,
            path,
            path_set,
            obstacles,
            turn_records,
            turn_to_color,
        )
        if red_line is None:
            continue

        seen_colors: set[int] = set()
        min_prefix_end = None
        for turn_index, color in turn_records:
            seen_colors.add(color)
            if seen_colors == set(OBSTACLE_COLORS):
                min_prefix_end = turn_index + 1
                break
        if min_prefix_end is None:
            continue
        if min_prefix_end >= len(path) - 1:
            continue

        max_prefix_end = len(path) - 2
        if max_prefix_end - min_prefix_end >= 2:
            prefix_end = random.randint(min_prefix_end, max_prefix_end - 1)
        else:
            prefix_end = max_prefix_end

        inp = np.zeros(shape, dtype=int)
        out = np.zeros(shape, dtype=int)

        for cell, color in obstacles.items():
            inp[cell] = color
            out[cell] = color

        for cell in red_line:
            inp[cell] = RED
            out[cell] = RED

        for cell in path[: prefix_end + 1]:
            inp[cell] = GREEN
        for cell in path:
            out[cell] = GREEN

        _place_decoys(inp, set(path), set(red_line), obstacles)
        for cell in np.argwhere((inp != BLACK) & (out == BLACK)):
            r, c = map(int, cell)
            out[r, c] = inp[r, c]

        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, out):
            continue
        return inp, out

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        example = _build_example()
        if example is not None:
            return example[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
