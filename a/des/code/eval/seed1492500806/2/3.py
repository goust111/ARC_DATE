from __future__ import annotations

import random
from collections import deque

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

BODY_TO_HEAD = {
    BLUE: AZURE,
    RED: ORANGE,
    GREEN: YELLOW,
}
HEAD_TO_BODY = {head: body for body, head in BODY_TO_HEAD.items()}
SNAKE_COLORS = set(BODY_TO_HEAD) | set(HEAD_TO_BODY)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}

GEN_GRID_MIN = 15
GEN_GRID_MAX = 24
GEN_BODY_LEN_RANGE = (1, 3)
GEN_SEGMENT_LEN_RANGE = (1, 4)
GEN_RETRIES = 400
GEN_PER_SNAKE_RETRIES = 250

_GEN_COUNTER = 0
_SCENARIOS = [
    ["edge"],
    ["wall"],
    ["edge", "snake"],
    ["wall", "snake"],
    ["edge", "wall", "snake"],
    ["wall", "edge", "snake"],
]


def _cw(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIR_TO_INDEX[direction] + 1) % 4]


def _ccw(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIR_TO_INDEX[direction] - 1) % 4]


def _add(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * steps, cell[1] + direction[1] * steps


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _extract_snake_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    snakes: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) not in SNAKE_COLORS:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) in SNAKE_COLORS:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            head_cells = [cell for cell in cells if int(g[cell]) in HEAD_TO_BODY]
            assert len(head_cells) == 1, "each snake component must contain exactly one head"
            head = head_cells[0]
            head_color = int(g[head])
            body_color = HEAD_TO_BODY[head_color]
            assert all(int(g[cell]) in {head_color, body_color} for cell in cells), "snake colors must match the head/body pair"
            assert len(cells) >= 2, "each snake must have at least one body pixel"

            cell_set = set(cells)
            degrees = {cell: sum((nbr in cell_set) for nbr in _neighbors4(cell)) for cell in cells}
            assert degrees[head] == 1, "head must be at one end of the chain"
            tail_count = 0
            for cell in cells:
                degree = degrees[cell]
                if cell == head:
                    continue
                assert degree in {1, 2}, "snake body must form a simple path"
                if degree == 1:
                    tail_count += 1
            assert tail_count == 1, "snake must have exactly one tail"

            body_neighbors = [nbr for nbr in _neighbors4(head) if nbr in cell_set]
            assert len(body_neighbors) == 1 and int(g[body_neighbors[0]]) == body_color
            direction = (head[0] - body_neighbors[0][0], head[1] - body_neighbors[0][1])

            snakes.append(
                {
                    "head": head,
                    "head_color": head_color,
                    "body_color": body_color,
                    "direction": direction,
                }
            )

    return sorted(snakes, key=lambda snake: snake["head"])


def _simulate_snake(out: np.ndarray, snake: dict) -> None:
    head = snake["head"]
    direction = snake["direction"]
    head_color = snake["head_color"]
    body_color = snake["body_color"]

    while True:
        nr, nc = head[0] + direction[0], head[1] + direction[1]
        if not _in_bounds(out.shape, (nr, nc)):
            break
        cell = int(out[nr, nc])
        if cell not in {BLACK, GREY}:
            break

        turn = cell == GREY
        out[head] = body_color
        head = (nr, nc)
        out[head] = head_color
        if turn:
            direction = _cw(direction)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    snakes = _extract_snake_components(g)
    assert snakes, "expected at least one snake"

    out = g.copy()
    for snake in snakes:
        _simulate_snake(out, snake)
    return out


def _reverse_head(
    final_head: tuple[int, int],
    final_direction: tuple[int, int],
    segment_lengths: list[int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    pos = final_head
    direction = final_direction
    for index in range(len(segment_lengths) - 1, -1, -1):
        pos = _add(pos, (-direction[0], -direction[1]), segment_lengths[index])
        if index > 0:
            direction = _ccw(direction)
    return pos, direction


def _build_growth(
    head: tuple[int, int],
    direction: tuple[int, int],
    segment_lengths: list[int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    growth: list[tuple[int, int]] = []
    turn_cells: set[tuple[int, int]] = set()
    pos = head
    current = direction
    for index, length in enumerate(segment_lengths):
        for _ in range(length):
            pos = _add(pos, current)
            growth.append(pos)
        if index + 1 < len(segment_lengths):
            turn_cells.add(pos)
            current = _cw(current)
    return growth, turn_cells, pos


def _choose_final_head(
    shape: tuple[int, int],
    reason: str,
    blocker_cells: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None, tuple[int, int] | None] | None:
    h, w = shape
    if reason == "edge":
        direction = random.choice(DIRS)
        if direction == (-1, 0):
            return (0, random.randrange(w)), direction, None, None
        if direction == (1, 0):
            return (h - 1, random.randrange(w)), direction, None, None
        if direction == (0, -1):
            return (random.randrange(h), 0), direction, None, None
        return (random.randrange(h), w - 1), direction, None, None

    if reason == "wall":
        direction = random.choice(DIRS)
        if direction == (-1, 0):
            head = (random.randrange(1, h), random.randrange(w))
        elif direction == (1, 0):
            head = (random.randrange(0, h - 1), random.randrange(w))
        elif direction == (0, -1):
            head = (random.randrange(h), random.randrange(1, w))
        else:
            head = (random.randrange(h), random.randrange(0, w - 1))
        return head, direction, _add(head, direction), None

    if reason == "snake" and blocker_cells:
        blocker = random.choice(blocker_cells)
        options = []
        for direction in DIRS:
            head = _add(blocker, (-direction[0], -direction[1]))
            if _in_bounds(shape, head):
                options.append((head, direction, blocker))
        if options:
            head, direction, blocker = random.choice(options)
            return head, direction, None, blocker

    return None


def _has_invalid_external_adjacency(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    allow_pair: tuple[tuple[int, int], tuple[int, int]] | None,
) -> bool:
    allowed = set()
    if allow_pair is not None:
        a, b = allow_pair
        allowed = {(a, b), (b, a)}
    for cell in cells:
        for nbr in _neighbors4(cell):
            if nbr not in occupied or nbr in cells:
                continue
            if (cell, nbr) in allowed:
                continue
            return True
    return False


def _place_snake(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    body_blockers: list[tuple[int, int]],
    body_color: int,
    head_color: int,
    reason: str,
    turns: int,
) -> bool:
    occupied = {tuple(map(int, cell)) for cell in np.argwhere(grid != BLACK)}
    shape = grid.shape

    for _ in range(GEN_PER_SNAKE_RETRIES):
        body_len = random.randint(*GEN_BODY_LEN_RANGE)
        segment_lengths = [random.randint(*GEN_SEGMENT_LEN_RANGE) for _ in range(turns + 1)]

        target = _choose_final_head(shape, reason, body_blockers)
        if target is None:
            continue
        final_head, final_direction, wall_cell, blocker_cell = target

        reversed_plan = _reverse_head(final_head, final_direction, segment_lengths)
        input_head, start_direction = reversed_plan
        growth, turn_cells, check_final = _build_growth(input_head, start_direction, segment_lengths)
        if check_final != final_head:
            continue

        body_cells = [_add(input_head, (-start_direction[0], -start_direction[1]), step) for step in range(1, body_len + 1)]
        path_cells = set(growth) | {input_head} | set(body_cells)
        if wall_cell is not None:
            path_cells.add(wall_cell)

        if len(path_cells) != len(growth) + 1 + len(body_cells) + (1 if wall_cell is not None else 0):
            continue
        if not all(_in_bounds(shape, cell) for cell in path_cells):
            continue
        if path_cells & reserved:
            continue

        snake_cells = {input_head} | set(body_cells)
        if _has_invalid_external_adjacency(snake_cells, {cell for cell in occupied if int(grid[cell]) in SNAKE_COLORS}, None):
            continue

        allow_pair = (final_head, blocker_cell) if blocker_cell is not None else None
        if _has_invalid_external_adjacency(set(growth) | snake_cells, occupied, allow_pair):
            continue
        if wall_cell is not None and _has_invalid_external_adjacency({wall_cell}, occupied, None):
            continue

        if blocker_cell is not None:
            if blocker_cell not in occupied or int(grid[blocker_cell]) not in BODY_TO_HEAD:
                continue
            if _add(final_head, final_direction) != blocker_cell:
                continue
        elif wall_cell is not None:
            if wall_cell in occupied:
                continue
            if _add(final_head, final_direction) != wall_cell:
                continue
        else:
            if _in_bounds(shape, _add(final_head, final_direction)):
                continue

        trial = grid.copy()
        for cell in body_cells:
            trial[cell] = body_color
        trial[input_head] = head_color
        for cell in turn_cells:
            trial[cell] = GREY
        if wall_cell is not None:
            trial[wall_cell] = MAGENTA

        try:
            snakes = _extract_snake_components(trial)
        except AssertionError:
            continue
        if len(snakes) != len(_extract_snake_components(grid)) + 1:
            continue

        local_snake = None
        for snake in snakes:
            if snake["head"] == input_head:
                local_snake = snake
                break
        if local_snake is None:
            continue

        simulated = trial.copy()
        _simulate_snake(simulated, local_snake)
        if tuple(np.argwhere(simulated == head_color)[0]) != final_head:
            continue

        for cell in body_cells:
            grid[cell] = body_color
            body_blockers.append(cell)
        grid[input_head] = head_color
        for cell in turn_cells:
            grid[cell] = GREY
        if wall_cell is not None:
            grid[wall_cell] = MAGENTA
        reserved.update(set(growth) | snake_cells)
        if wall_cell is not None:
            reserved.add(wall_cell)
        return True

    return False


def _scenario_for_call() -> list[str]:
    global _GEN_COUNTER
    scenario = _SCENARIOS[_GEN_COUNTER % len(_SCENARIOS)]
    _GEN_COUNTER += 1
    return list(scenario)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        reasons = _scenario_for_call()
        h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
        w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
        grid = np.full((h, w), BLACK, dtype=int)
        reserved: set[tuple[int, int]] = set()
        body_blockers: list[tuple[int, int]] = []

        color_pairs = random.sample(list(BODY_TO_HEAD.items()), len(reasons))
        turns_seed = random.randrange(3)
        ok = True
        for index, (reason, (body_color, head_color)) in enumerate(zip(reasons, color_pairs)):
            turns = (turns_seed + index) % 3
            if reason == "snake" and not body_blockers:
                reason = "wall"
            if not _place_snake(grid, reserved, body_blockers, body_color, head_color, reason, turns):
                ok = False
                break

        if not ok:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        return grid

    raise RuntimeError("failed to generate a valid snake-growth input")
