from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIR_ORDER = [UP, RIGHT, DOWN, LEFT]
ACTION_TO_COLOR = {"L": BLUE, "R": RED, "S": GREEN}
NODE_COLORS = tuple(ACTION_TO_COLOR.values())
PIPE_COLORS = {GREY, ORANGE, *NODE_COLORS}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_boundary(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _turn(direction: tuple[int, int], action: str) -> tuple[int, int]:
    idx = DIR_ORDER.index(direction)
    if action == "S":
        return direction
    if action == "R":
        return DIR_ORDER[(idx + 1) % 4]
    if action == "L":
        return DIR_ORDER[(idx - 1) % 4]
    raise AssertionError(f"unknown action {action!r}")


def _action_for(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> str | None:
    if out_dir == in_dir:
        return "S"
    if out_dir == _turn(in_dir, "R"):
        return "R"
    if out_dir == _turn(in_dir, "L"):
        return "L"
    return None


def _pipe_neighbors(g: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR_ORDER:
        rr, cc = r + dr, c + dc
        if _in_bounds(h, w, rr, cc) and int(g[rr, cc]) in PIPE_COLORS:
            out.append((rr, cc))
    return out


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    if abs(dr) + abs(dc) != 1:
        raise AssertionError("adjacent lattice nodes must differ by one step")
    return dr, dc


def _add_segment(cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((r0, c))
        return
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((r, c0))
        return
    raise AssertionError("segments must be axis-aligned")


def _node_positions(rows: list[int], cols: list[int]) -> list[tuple[int, int]]:
    return [(r, c) for r in rows for c in cols]


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the signal through the orthogonal pipe network and paint its visited route orange.

    Input structure:
    - The background is black.
    - Grey one-pixel-wide horizontal and vertical pipe segments form an orthogonal
      network whose intersections are colored routing nodes.
    - Each routing node is a single Red, Blue, or Green pixel placed at a pipe
      crossing, and exactly one boundary endpoint is marked Orange as the signal source.

    Rule:
    - Start at the unique Orange boundary endpoint and move in the only inward pipe direction.
    - Along Grey pipe cells the signal keeps moving straight.
    - At a Red node the signal turns right, at a Blue node it turns left, and at a
      Green node it keeps going straight, all relative to its current direction.
    - Stop when the signal reaches a different boundary endpoint, and recolor every
      visited pipe cell and visited routing node to Orange.

    Key insight:
    - The colored intersections are not destinations; they are local direction controls,
      so the output is obtained by simulating one continuous path from the source to
      the boundary exit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape

    sources = np.argwhere(g == ORANGE)
    assert sources.shape == (1, 2), "expected exactly one orange source"
    src = tuple(map(int, sources[0]))
    assert _is_boundary(src[0], src[1], h, w), "source must be on the boundary"

    src_neighbors = _pipe_neighbors(g, src[0], src[1])
    assert len(src_neighbors) == 1, "source must be a boundary endpoint"
    direction = (src_neighbors[0][0] - src[0], src_neighbors[0][1] - src[1])

    visited = [src]
    current = src
    seen_states: set[tuple[int, int, int, int]] = set()
    color_to_action = {BLUE: "L", RED: "R", GREEN: "S"}

    while True:
        state = (current[0], current[1], direction[0], direction[1])
        assert state not in seen_states, "signal path looped"
        seen_states.add(state)

        nr, nc = current[0] + direction[0], current[1] + direction[1]
        assert _in_bounds(h, w, nr, nc), "signal left the grid away from an endpoint"
        cell = int(g[nr, nc])
        assert cell in PIPE_COLORS, "signal must stay on pipe cells"

        current = (nr, nc)
        visited.append(current)
        neighbors = _pipe_neighbors(g, nr, nc)

        if current != src and _is_boundary(nr, nc, h, w):
            assert len(neighbors) == 1, "boundary stop must be a pipe endpoint"
            break

        if cell == GREY:
            assert len(neighbors) == 2, "non-node pipe cells must be straight segments"
            back = (-direction[0], -direction[1])
            expected = {(nr + back[0], nc + back[1]), (nr + direction[0], nc + direction[1])}
            assert set(neighbors) == expected, "grey pipe cells must continue straight"
            continue

        assert cell in color_to_action, "intersections must be red, blue, or green"
        assert len(neighbors) == 4, "routing nodes must be four-way crossings"
        direction = _turn(direction, color_to_action[cell])

    out = g.copy()
    for r, c in visited:
        out[r, c] = ORANGE
    return out


GRID_COUNT_MIN = 3
GRID_COUNT_MAX = 5
GRID_GAP_MIN = 2
GRID_GAP_MAX = 5
GRID_MARGIN_MIN = 1
GRID_MARGIN_MAX = 3
PATH_LEN_MIN = 4
PATH_LEN_MAX = 7
MAX_GENERATION_ATTEMPTS = 400
SIDES = ("left", "right", "top", "bottom")


def _sample_axis(count: int) -> tuple[int, list[int]]:
    start = random.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX)
    coords = [start]
    for _ in range(count - 1):
        coords.append(coords[-1] + random.randint(GRID_GAP_MIN, GRID_GAP_MAX))
    size = coords[-1] + random.randint(GRID_MARGIN_MIN, GRID_MARGIN_MAX) + 1
    assert size <= 30
    return size, coords


def _start_state(n_rows: int, n_cols: int, side: str, lane: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "left":
        return (lane, 0), RIGHT
    if side == "right":
        return (lane, n_cols - 1), LEFT
    if side == "top":
        return (0, lane), DOWN
    if side == "bottom":
        return (n_rows - 1, lane), UP
    raise AssertionError(f"unknown side {side!r}")


def _neighbor_nodes(node: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in DIR_ORDER:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n_rows and 0 <= cc < n_cols:
            out.append((rr, cc))
    random.shuffle(out)
    return out


def _exit_dirs(node: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    if r == 0:
        out.append(UP)
    if r == n_rows - 1:
        out.append(DOWN)
    if c == 0:
        out.append(LEFT)
    if c == n_cols - 1:
        out.append(RIGHT)
    random.shuffle(out)
    return out


def _derive_actions(
    path: list[tuple[int, int]],
    start_in_dir: tuple[int, int],
    exit_dir: tuple[int, int],
) -> list[str] | None:
    actions: list[str] = []
    incoming = start_in_dir
    for idx in range(len(path)):
        if idx + 1 < len(path):
            outgoing = _direction_between(path[idx], path[idx + 1])
        else:
            outgoing = exit_dir
        action = _action_for(incoming, outgoing)
        if action is None:
            return None
        actions.append(action)
        incoming = outgoing
    return actions


def _balanced_actions(actions: list[str]) -> bool:
    counts = [actions.count(label) for label in ("L", "R", "S")]
    return min(counts) >= 1 and max(counts) - min(counts) <= 1


def _search_path(
    n_rows: int,
    n_cols: int,
    side: str,
    lane: int,
    target_len: int,
) -> tuple[list[tuple[int, int]], list[str], tuple[int, int]] | None:
    start_node, start_dir = _start_state(n_rows, n_cols, side, lane)
    path = [start_node]
    visited = {start_node}
    solution: tuple[list[tuple[int, int]], list[str], tuple[int, int]] | None = None

    def dfs() -> bool:
        nonlocal solution
        if len(path) == target_len:
            for exit_dir in _exit_dirs(path[-1], n_rows, n_cols):
                actions = _derive_actions(path, start_dir, exit_dir)
                if actions is None or not _balanced_actions(actions):
                    continue
                solution = (list(path), actions, exit_dir)
                return True
            return False

        for nxt in _neighbor_nodes(path[-1], n_rows, n_cols):
            if nxt in visited:
                continue
            path.append(nxt)
            visited.add(nxt)
            if dfs():
                return True
            visited.remove(nxt)
            path.pop()
        return False

    if dfs():
        return solution
    return None


def _choose_path(n_rows: int, n_cols: int) -> tuple[str, int, list[tuple[int, int]], list[str], tuple[int, int]] | None:
    max_len = min(PATH_LEN_MAX, n_rows * n_cols - 1)
    if max_len < PATH_LEN_MIN:
        return None

    lengths = list(range(PATH_LEN_MIN, max_len + 1))
    random.shuffle(lengths)
    sides = list(SIDES)
    random.shuffle(sides)

    for target_len in lengths:
        for side in sides:
            lane_count = n_rows if side in {"left", "right"} else n_cols
            lanes = list(range(lane_count))
            random.shuffle(lanes)
            for lane in lanes:
                result = _search_path(n_rows, n_cols, side, lane, target_len)
                if result is not None:
                    path, actions, exit_dir = result
                    return side, lane, path, actions, exit_dir
    return None


def _node_to_pos(node: tuple[int, int], rows: list[int], cols: list[int]) -> tuple[int, int]:
    return rows[node[0]], cols[node[1]]


def _source_pos(side: str, lane: int, rows: list[int], cols: list[int], h: int, w: int) -> tuple[int, int]:
    if side == "left":
        return rows[lane], 0
    if side == "right":
        return rows[lane], w - 1
    if side == "top":
        return 0, cols[lane]
    if side == "bottom":
        return h - 1, cols[lane]
    raise AssertionError(f"unknown side {side!r}")


def _exit_pos(last_node: tuple[int, int], exit_dir: tuple[int, int], rows: list[int], cols: list[int], h: int, w: int) -> tuple[int, int]:
    r, c = _node_to_pos(last_node, rows, cols)
    if exit_dir == UP:
        return 0, c
    if exit_dir == DOWN:
        return h - 1, c
    if exit_dir == LEFT:
        return r, 0
    if exit_dir == RIGHT:
        return r, w - 1
    raise AssertionError("bad exit direction")


def _trace_constructed_path(
    h: int,
    w: int,
    rows: list[int],
    cols: list[int],
    side: str,
    lane: int,
    path: list[tuple[int, int]],
    exit_dir: tuple[int, int],
) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    source = _source_pos(side, lane, rows, cols, h, w)
    first = _node_to_pos(path[0], rows, cols)
    _add_segment(visited, source, first)

    for idx in range(len(path) - 1):
        a = _node_to_pos(path[idx], rows, cols)
        b = _node_to_pos(path[idx + 1], rows, cols)
        _add_segment(visited, a, b)

    last = _node_to_pos(path[-1], rows, cols)
    exit_cell = _exit_pos(path[-1], exit_dir, rows, cols, h, w)
    _add_segment(visited, last, exit_cell)
    return visited


def _build_grid(
    h: int,
    w: int,
    rows: list[int],
    cols: list[int],
    path: list[tuple[int, int]],
    actions: list[str],
    side: str,
    lane: int,
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r in rows:
        grid[r, :] = GREY
    for c in cols:
        grid[:, c] = GREY

    for r, c in _node_positions(rows, cols):
        grid[r, c] = random.choice(NODE_COLORS)

    for node, action in zip(path, actions):
        r, c = _node_to_pos(node, rows, cols)
        grid[r, c] = ACTION_TO_COLOR[action]

    sr, sc = _source_pos(side, lane, rows, cols, h, w)
    grid[sr, sc] = ORANGE
    return grid


def _expected_output(grid: np.ndarray, visited: set[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c in visited:
        out[r, c] = ORANGE
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(GRID_COUNT_MIN, GRID_COUNT_MAX)
    n_cols = random.randint(GRID_COUNT_MIN, GRID_COUNT_MAX)
    h, rows = _sample_axis(n_rows)
    w, cols = _sample_axis(n_cols)

    choice = _choose_path(n_rows, n_cols)
    if choice is None:
        return None
    side, lane, path, actions, exit_dir = choice
    assert len(set(path)) == len(path)

    visited = _trace_constructed_path(h, w, rows, cols, side, lane, path, exit_dir)
    grid = _build_grid(h, w, rows, cols, path, actions, side, lane)
    expected = _expected_output(grid, visited)

    if not _balanced_actions(actions):
        return None
    if len(np.argwhere(grid == ORANGE)) != 1:
        return None
    if max(grid.shape) > 30:
        return None
    if np.array_equal(grid, expected):
        return None

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
    raise RuntimeError("failed to generate a valid pipe-routing ARC input")
