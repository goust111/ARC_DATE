from __future__ import annotations

from collections import Counter
import random

import numpy as np


WHITE = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT_TURN = {DIRS[i]: DIRS[(i - 1) % 4] for i in range(4)}
RIGHT_TURN = {DIRS[i]: DIRS[(i + 1) % 4] for i in range(4)}
REVERSE_TURN = {d: (-d[0], -d[1]) for d in DIRS}

COLOR_TO_ACTION = {
    GREEN: "S",
    YELLOW: "L",
    ORANGE: "R",
    MAROON: "U",
}
ACTION_TO_COLOR = {action: color for color, action in COLOR_TO_ACTION.items()}
TURN_COLORS = tuple(COLOR_TO_ACTION)
SIGN_SIDES = ("top", "right", "bottom", "left")

# --------------- generator config ---------------

CELL_COUNT_RANGE = (4, 8)
MAZE_EXTRA_OPENINGS_MAX = 5
MAX_SEQUENCE_LEN = 8
MIN_SEQUENCE_LEN = 2
MIN_UNIQUE_ROUTE_CELLS = 10
ROUTE_ATTEMPTS_PER_MAZE = 160
GENERATE_ATTEMPTS = 400


def _neighbors4(mask: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    r, c = pos
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((int(nr), int(nc)))
    return out


def _direction(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    return (int(dst[0] - src[0]), int(dst[1] - src[1]))


def _apply_action(direction: tuple[int, int], action: str) -> tuple[int, int]:
    if action == "S":
        return direction
    if action == "L":
        return LEFT_TURN[direction]
    if action == "R":
        return RIGHT_TURN[direction]
    if action == "U":
        return REVERSE_TURN[direction]
    raise AssertionError(f"unknown action {action}")


def _passable_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, (WHITE, RED, BLUE))


def _unique_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (1, 2), f"expected one pixel of color {color}"
    return tuple(map(int, coords[0]))


def _read_sign_sequence(grid: np.ndarray) -> list[int]:
    coords = np.argwhere(np.isin(grid, TURN_COLORS))
    assert len(coords) > 0, "expected a route-sign strip"

    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})

    if len(rows) == 1:
        row = rows[0]
        assert row in (0, grid.shape[0] - 1), "horizontal signs must lie on the grid edge"
        ordered = coords[np.argsort(coords[:, 1])]
    else:
        assert len(cols) == 1, "signs must be collinear"
        col = cols[0]
        assert col in (0, grid.shape[1] - 1), "vertical signs must lie on the grid edge"
        ordered = coords[np.argsort(coords[:, 0])]

    return [int(grid[int(r), int(c)]) for r, c in ordered]


def _trace_route(
    passable: np.ndarray,
    start: tuple[int, int],
    exit_pos: tuple[int, int],
    actions: list[str],
) -> tuple[list[tuple[int, int]], str, int]:
    first_steps = _neighbors4(passable, start)
    assert len(first_steps) == 1, "entrance must have a unique way into the maze"

    prev = start
    pos = first_steps[0]
    direction = _direction(prev, pos)
    path = [start, pos]
    action_idx = 0

    step_cap = int(passable.size * max(4, len(actions) + 2))
    for _ in range(step_cap):
        if pos == exit_pos:
            return path, "exit", action_idx

        nbrs = _neighbors4(passable, pos)
        options = [nb for nb in nbrs if nb != prev]

        if len(options) >= 2:
            if action_idx >= len(actions):
                return path, "exhausted", action_idx
            next_dir = _apply_action(direction, actions[action_idx])
            action_idx += 1
            nxt = (int(pos[0] + next_dir[0]), int(pos[1] + next_dir[1]))
            if not (
                0 <= nxt[0] < passable.shape[0]
                and 0 <= nxt[1] < passable.shape[1]
                and bool(passable[nxt])
            ):
                return path, "invalid_turn", action_idx
        elif len(options) == 1:
            nxt = options[0]
            next_dir = _direction(pos, nxt)
        else:
            return path, "dead_end", action_idx

        prev, pos, direction = pos, nxt, next_dir
        path.append(pos)

    return path, "step_cap", action_idx


def _paint_output(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    exit_pos: tuple[int, int],
) -> np.ndarray:
    out = np.full_like(grid, WHITE)
    out[grid == GREY] = GREY
    for r, c in path:
        if (r, c) == start:
            out[r, c] = RED
        elif (r, c) == exit_pos:
            out[r, c] = BLUE
        else:
            out[r, c] = MAGENTA
    out[start] = RED
    out[exit_pos] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the maze route from the red entrance by reading an edge strip of relative-turn colors.

    Input structure:
    - White is the background, and Grey pixels form a single-pixel-wide channel maze.
    - One Red opening on the maze boundary marks the entrance, and one Blue opening marks the exit.
    - Outside the maze, along exactly one grid edge, a straight strip of Green, Yellow, Orange, and Maroon pixels gives an ordered route-sign sequence.

    Rule:
    - Interpret the route-sign colors as relative turns: Green = straight, Yellow = left, Orange = right, Maroon = U-turn.
    - Start at the Red opening and take the unique first step into the maze interior.
    - Keep walking through the corridor network; only when the current cell is a junction with at least two non-backtracking choices do you consume the next route sign and apply that relative turn.
    - Between junctions follow the only forward corridor cell automatically, and stop as soon as you reach the Blue exit or arrive at a junction after the route-sign sequence is exhausted.
    - In the output, keep Grey walls plus the Red and Blue markers, clear the outer sign strip back to White, color every traversed corridor cell Magenta, and leave all untouched corridor cells White.

    Key insight:
    - The colored edge strip is not a per-step direction code: it is an ordered list of relative decisions that is consumed only at junctions.
    """
    g = np.asarray(grid, dtype=int)
    signs = _read_sign_sequence(g)
    actions = [COLOR_TO_ACTION[color] for color in signs]

    start = _unique_coord(g, RED)
    exit_pos = _unique_coord(g, BLUE)
    path, _status, _used = _trace_route(_passable_mask(g), start, exit_pos, actions)
    return _paint_output(g, path, start, exit_pos)


def _carve_maze(cell_h: int, cell_w: int) -> np.ndarray:
    maze_h = 2 * cell_h + 1
    maze_w = 2 * cell_w + 1
    maze = np.full((maze_h, maze_w), GREY, dtype=int)
    visited = np.zeros((cell_h, cell_w), dtype=bool)

    def carve(cr: int, cc: int) -> None:
        visited[cr, cc] = True
        maze[2 * cr + 1, 2 * cc + 1] = WHITE
        dirs = DIRS[:]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and not bool(visited[nr, nc]):
                maze[2 * cr + 1 + dr, 2 * cc + 1 + dc] = WHITE
                carve(nr, nc)

    carve(random.randrange(cell_h), random.randrange(cell_w))

    extra_candidates: list[tuple[int, int]] = []
    for r in range(1, maze_h - 1):
        for c in range(1, maze_w - 1):
            if int(maze[r, c]) != GREY:
                continue
            if r % 2 == 0 and c % 2 == 1 and int(maze[r - 1, c]) == WHITE and int(maze[r + 1, c]) == WHITE:
                extra_candidates.append((r, c))
            if r % 2 == 1 and c % 2 == 0 and int(maze[r, c - 1]) == WHITE and int(maze[r, c + 1]) == WHITE:
                extra_candidates.append((r, c))

    random.shuffle(extra_candidates)
    extra_count = random.randint(0, min(MAZE_EXTRA_OPENINGS_MAX, len(extra_candidates)))
    for r, c in extra_candidates[:extra_count]:
        maze[r, c] = WHITE

    return maze


def _embed_maze(maze: np.ndarray, sign_side: str) -> tuple[np.ndarray, int, int]:
    maze_h, maze_w = map(int, maze.shape)
    full_h = maze_h + (1 if sign_side in ("top", "bottom") else 0)
    full_w = maze_w + (1 if sign_side in ("left", "right") else 0)

    grid = np.full((full_h, full_w), WHITE, dtype=int)
    row_offset = 1 if sign_side == "top" else 0
    col_offset = 1 if sign_side == "left" else 0
    grid[row_offset : row_offset + maze_h, col_offset : col_offset + maze_w] = maze
    return grid, row_offset, col_offset


def _boundary_openings(
    maze_h: int,
    maze_w: int,
    row_offset: int,
    col_offset: int,
    sign_side: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    openings: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if sign_side != "top":
        for c in range(1, maze_w, 2):
            openings.append(((row_offset, col_offset + c), (row_offset + 1, col_offset + c)))
    if sign_side != "bottom":
        for c in range(1, maze_w, 2):
            openings.append(
                ((row_offset + maze_h - 1, col_offset + c), (row_offset + maze_h - 2, col_offset + c))
            )
    if sign_side != "left":
        for r in range(1, maze_h, 2):
            openings.append(((row_offset + r, col_offset), (row_offset + r, col_offset + 1)))
    if sign_side != "right":
        for r in range(1, maze_h, 2):
            openings.append(
                ((row_offset + r, col_offset + maze_w - 1), (row_offset + r, col_offset + maze_w - 2))
            )

    return openings


def _available_actions(
    passable: np.ndarray,
    pos: tuple[int, int],
    prev: tuple[int, int],
    direction: tuple[int, int],
) -> list[str]:
    actions: list[str] = []
    for nb in _neighbors4(passable, pos):
        if nb == prev:
            continue
        next_dir = _direction(pos, nb)
        if next_dir == direction:
            actions.append("S")
        elif next_dir == LEFT_TURN[direction]:
            actions.append("L")
        elif next_dir == RIGHT_TURN[direction]:
            actions.append("R")
    actions.append("U")
    return actions


def _route_is_interesting(path: list[tuple[int, int]], actions: list[str]) -> bool:
    if not (MIN_SEQUENCE_LEN <= len(actions) <= MAX_SEQUENCE_LEN):
        return False
    if len(set(path)) < max(MIN_UNIQUE_ROUTE_CELLS, len(actions) * 2 + 2):
        return False
    distinct_actions = len(set(actions))
    need_distinct = 3 if random.random() < 0.35 else 2
    return distinct_actions >= need_distinct


def _sample_action_sequence(grid: np.ndarray, target_mode: str) -> list[str] | None:
    passable = _passable_mask(grid)
    start = _unique_coord(grid, RED)
    exit_pos = _unique_coord(grid, BLUE)
    first_steps = _neighbors4(passable, start)
    if len(first_steps) != 1:
        return None

    for _ in range(ROUTE_ATTEMPTS_PER_MAZE):
        target_len = random.randint(MIN_SEQUENCE_LEN, MAX_SEQUENCE_LEN)
        prev = start
        pos = first_steps[0]
        direction = _direction(prev, pos)
        path = [start, pos]
        actions: list[str] = []
        seen = Counter()

        while len(path) <= passable.size * 4 and len(actions) <= MAX_SEQUENCE_LEN:
            state = (pos, prev, direction, len(actions))
            seen[state] += 1
            if seen[state] > 2:
                break

            if pos == exit_pos:
                if target_mode == "exit" and _route_is_interesting(path, actions):
                    extra = 0
                    if len(actions) < MAX_SEQUENCE_LEN and random.random() < 0.5:
                        extra = random.randint(1, min(2, MAX_SEQUENCE_LEN - len(actions)))
                    return actions + [random.choice(tuple(ACTION_TO_COLOR)) for _ in range(extra)]
                break

            nbrs = _neighbors4(passable, pos)
            options = [nb for nb in nbrs if nb != prev]
            if len(options) >= 2:
                if target_mode == "exhausted" and len(actions) >= target_len:
                    if _route_is_interesting(path, actions):
                        return actions
                    break
                action = random.choice(_available_actions(passable, pos, prev, direction))
                next_dir = _apply_action(direction, action)
                nxt = (int(pos[0] + next_dir[0]), int(pos[1] + next_dir[1]))
                if not (
                    0 <= nxt[0] < passable.shape[0]
                    and 0 <= nxt[1] < passable.shape[1]
                    and bool(passable[nxt])
                ):
                    break
                actions.append(action)
            elif len(options) == 1:
                nxt = options[0]
                next_dir = _direction(pos, nxt)
            else:
                break

            prev, pos, direction = pos, nxt, next_dir
            path.append(pos)

    return None


def _place_signs(grid: np.ndarray, sign_side: str, sign_colors: list[int]) -> None:
    if sign_side in ("top", "bottom"):
        row = 0 if sign_side == "top" else grid.shape[0] - 1
        start = random.randint(0, grid.shape[1] - len(sign_colors))
        for idx, color in enumerate(sign_colors):
            grid[row, start + idx] = color
    else:
        col = 0 if sign_side == "left" else grid.shape[1] - 1
        start = random.randint(0, grid.shape[0] - len(sign_colors))
        for idx, color in enumerate(sign_colors):
            grid[start + idx, col] = color


def _gen_attempt() -> np.ndarray | None:
    target_mode = random.choice(("exit", "exhausted"))
    sign_side = random.choice(SIGN_SIDES)
    cell_h = random.randint(*CELL_COUNT_RANGE)
    cell_w = random.randint(*CELL_COUNT_RANGE)

    maze = _carve_maze(cell_h, cell_w)
    grid, row_offset, col_offset = _embed_maze(maze, sign_side)
    openings = _boundary_openings(maze.shape[0], maze.shape[1], row_offset, col_offset, sign_side)
    if len(openings) < 2:
        return None

    start_open, exit_open = random.sample(openings, 2)
    grid[start_open[0]] = RED
    grid[exit_open[0]] = BLUE

    actions = _sample_action_sequence(grid, target_mode)
    if actions is None:
        return None

    sign_colors = [ACTION_TO_COLOR[action] for action in actions]
    strip_len = grid.shape[1] if sign_side in ("top", "bottom") else grid.shape[0]
    if len(sign_colors) > strip_len:
        return None

    inp = grid.copy()
    _place_signs(inp, sign_side, sign_colors)

    start = _unique_coord(inp, RED)
    exit_pos = _unique_coord(inp, BLUE)
    route_path, route_status, _ = _trace_route(_passable_mask(inp), start, exit_pos, actions)
    if route_status != target_mode:
        return None

    expected = _paint_output(inp, route_path, start, exit_pos)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATE_ATTEMPTS} attempts")
