import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

PIPE_COLORS = {GREY, RED, BLUE, GREEN}
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT = "L"
RIGHT = "R"


STEP = 4
MIN_LATTICE = 4
MAX_LATTICE = 6
MIN_INTERSECTIONS = 1
MAX_INTERSECTIONS = 6
BASE_PAD = MAX_INTERSECTIONS + 3
EXTRA_PAD = 3
START_END_CHOICES = (1, 2)
GEN_RETRIES = 400


def _rot_cw(dir_idx: int) -> int:
    return (dir_idx + 1) % 4


def _rot_ccw(dir_idx: int) -> int:
    return (dir_idx - 1) % 4


def _opposite(dir_idx: int) -> int:
    return (dir_idx + 2) % 4


def _apply_turn(dir_idx: int, turn: str) -> int:
    return _rot_cw(dir_idx) if turn == RIGHT else _rot_ccw(dir_idx)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_pipe(color: int) -> bool:
    return int(color) in PIPE_COLORS


def _pipe_neighbor_dirs(grid: np.ndarray, r: int, c: int) -> list[int]:
    result: list[int] = []
    for dir_idx, (dr, dc) in enumerate(DIRS):
        rr = r + dr
        cc = c + dc
        if _in_bounds(grid.shape, rr, cc) and _is_pipe(int(grid[rr, cc])):
            result.append(dir_idx)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the signal from the green pipe endpoint, turn at colored cross switches,
    recolor the traversed grey pipe to yellow, and append an orange marker whose
    length equals the number of visited intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    starts = np.argwhere(g == GREEN)
    assert starts.shape[0] == 1, "expected exactly one green start endpoint"
    start_r, start_c = map(int, starts[0])

    start_dirs = _pipe_neighbor_dirs(g, start_r, start_c)
    assert len(start_dirs) == 1, "green start must be a dead-end marker"
    travel_dir = int(start_dirs[0])

    out = g.copy()
    intersections = 0
    visited_states: set[tuple[int, int, int]] = set()
    cur_r, cur_c = start_r, start_c

    while True:
        state = (cur_r, cur_c, travel_dir)
        assert state not in visited_states, "signal path must not loop"
        visited_states.add(state)

        dr, dc = DIRS[travel_dir]
        next_r = cur_r + dr
        next_c = cur_c + dc
        assert _in_bounds(g.shape, next_r, next_c), "signal exited the grid"
        next_color = int(g[next_r, next_c])
        assert _is_pipe(next_color), "signal must stay on the pipe network"

        cur_r, cur_c = next_r, next_c
        color = int(g[cur_r, cur_c])

        if color == GREY:
            out[cur_r, cur_c] = YELLOW
            neighbor_dirs = _pipe_neighbor_dirs(g, cur_r, cur_c)
            if len(neighbor_dirs) == 1:
                end_dir = travel_dir
                break
            assert sorted(neighbor_dirs) == sorted([travel_dir, _opposite(travel_dir)]), (
                "grey pipe cells must be straight except at endpoints"
            )
            continue

        if color in (RED, BLUE):
            neighbor_dirs = _pipe_neighbor_dirs(g, cur_r, cur_c)
            assert len(neighbor_dirs) == 4, "switch cells must be full cross intersections"
            intersections += 1
            travel_dir = _rot_cw(travel_dir) if color == RED else _rot_ccw(travel_dir)
            assert travel_dir in neighbor_dirs
            continue

        raise AssertionError("green should only appear at the start endpoint")

    draw_r, draw_c = cur_r, cur_c
    end_dr, end_dc = DIRS[end_dir]
    for _ in range(intersections):
        draw_r += end_dr
        draw_c += end_dc
        assert _in_bounds(g.shape, draw_r, draw_c), "orange marker must fit inside the grid"
        assert int(g[draw_r, draw_c]) == BLACK, "orange marker must extend into empty space"
        out[draw_r, draw_c] = ORANGE

    return out


def _entry_candidates(rows: int, cols: int, incoming_dir: int) -> list[tuple[int, int]]:
    if incoming_dir == 1:
        return [(r, 0) for r in range(rows)]
    if incoming_dir == 3:
        return [(r, cols - 1) for r in range(rows)]
    if incoming_dir == 2:
        return [(0, c) for c in range(cols)]
    return [(rows - 1, c) for c in range(cols)]


def _can_exit(node: tuple[int, int], outgoing_dir: int, rows: int, cols: int) -> bool:
    r, c = node
    if outgoing_dir == 1:
        return c == cols - 1
    if outgoing_dir == 3:
        return c == 0
    if outgoing_dir == 2:
        return r == rows - 1
    return r == 0


def _build_turning_path(
    rows: int,
    cols: int,
    intersections: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, list[str]] | None:
    def dfs(
        current: tuple[int, int],
        incoming_dir: int,
        node_idx: int,
    ) -> list[str] | None:
        if node_idx == intersections - 1:
            turn_options = [LEFT, RIGHT]
            rng.shuffle(turn_options)
            for turn in turn_options:
                outgoing_dir = _apply_turn(incoming_dir, turn)
                if _can_exit(current, outgoing_dir, rows, cols):
                    return [turn]
            return None

        turn_options = [LEFT, RIGHT]
        rng.shuffle(turn_options)
        for turn in turn_options:
            outgoing_dir = _apply_turn(incoming_dir, turn)
            dr, dc = DIRS[outgoing_dir]
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            suffix = dfs(nxt, outgoing_dir, node_idx + 1)
            if suffix is not None:
                return [turn] + suffix
            visited.remove(nxt)
        return None

    for _ in range(300):
        start_incoming = rng.randrange(4)
        starts = _entry_candidates(rows, cols, start_incoming)
        rng.shuffle(starts)
        for start in starts:
            visited = {start}
            turns = dfs(start, start_incoming, 0)
            if turns is None:
                continue

            nodes = [start]
            cur = start
            incoming_dir = start_incoming
            for turn in turns[:-1]:
                outgoing_dir = _apply_turn(incoming_dir, turn)
                dr, dc = DIRS[outgoing_dir]
                cur = (cur[0] + dr, cur[1] + dc)
                nodes.append(cur)
                incoming_dir = outgoing_dir
            assert len(nodes) == intersections
            return nodes, start_incoming, turns
    return None


def _paint_line(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1, step):
            assert int(grid[r0, c]) == BLACK
            grid[r0, c] = color
        return
    assert c0 == c1
    step = 1 if r1 > r0 else -1
    for r in range(r0 + step, r1, step):
        assert int(grid[r, c0]) == BLACK
        grid[r, c0] = color


def _paint_stub(
    grid: np.ndarray,
    start: tuple[int, int],
    dir_idx: int,
    length: int,
    end_color: int,
) -> tuple[int, int]:
    r, c = start
    dr, dc = DIRS[dir_idx]
    endpoint = start
    for k in range(1, length + 1):
        rr = r + dr * k
        cc = c + dc * k
        assert int(grid[rr, cc]) == BLACK
        grid[rr, cc] = end_color if k == length else GREY
        endpoint = (rr, cc)
    return endpoint


def _to_actual(node: tuple[int, int], top_pad: int, left_pad: int) -> tuple[int, int]:
    return top_pad + node[0] * STEP, left_pad + node[1] * STEP


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    lattice_rows = rng.randint(MIN_LATTICE, MAX_LATTICE)
    lattice_cols = rng.randint(MIN_LATTICE, MAX_LATTICE)
    intersections = rng.randint(MIN_INTERSECTIONS, MAX_INTERSECTIONS)

    built = _build_turning_path(lattice_rows, lattice_cols, intersections, rng)
    if built is None:
        return None
    nodes, start_incoming, turns = built
    node_set = set(nodes)

    top_pad = BASE_PAD + rng.randint(0, EXTRA_PAD)
    bottom_pad = BASE_PAD + rng.randint(0, EXTRA_PAD)
    left_pad = BASE_PAD + rng.randint(0, EXTRA_PAD)
    right_pad = BASE_PAD + rng.randint(0, EXTRA_PAD)
    height = top_pad + bottom_pad + (lattice_rows - 1) * STEP + 1
    width = left_pad + right_pad + (lattice_cols - 1) * STEP + 1
    grid = np.full((height, width), BLACK, dtype=int)

    incoming_dirs: list[int] = []
    outgoing_dirs: list[int] = []
    incoming_dir = start_incoming
    for turn in turns:
        incoming_dirs.append(incoming_dir)
        outgoing_dir = _apply_turn(incoming_dir, turn)
        outgoing_dirs.append(outgoing_dir)
        incoming_dir = outgoing_dir

    try:
        for node, turn in zip(nodes, turns):
            r, c = _to_actual(node, top_pad, left_pad)
            assert int(grid[r, c]) == BLACK
            grid[r, c] = RED if turn == RIGHT else BLUE

        start_len = rng.choice(START_END_CHOICES)
        first_r, first_c = _to_actual(nodes[0], top_pad, left_pad)
        _paint_stub(grid, (first_r, first_c), _opposite(start_incoming), start_len, GREEN)

        for idx in range(len(nodes) - 1):
            start_xy = _to_actual(nodes[idx], top_pad, left_pad)
            end_xy = _to_actual(nodes[idx + 1], top_pad, left_pad)
            _paint_line(grid, start_xy, end_xy, GREY)

        end_len = rng.choice(START_END_CHOICES)
        last_r, last_c = _to_actual(nodes[-1], top_pad, left_pad)
        _paint_stub(grid, (last_r, last_c), outgoing_dirs[-1], end_len, GREY)

        for idx, node in enumerate(nodes):
            used_dirs = {_opposite(incoming_dirs[idx]), outgoing_dirs[idx]}
            for dir_idx in range(4):
                if dir_idx in used_dirs:
                    continue
                neighbor = (node[0] + DIRS[dir_idx][0], node[1] + DIRS[dir_idx][1])
                max_len = 1 if neighbor in node_set else 2
                branch_len = rng.randint(1, max_len)
                r, c = _to_actual(node, top_pad, left_pad)
                _paint_stub(grid, (r, c), dir_idx, branch_len, GREY)
    except AssertionError:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.count_nonzero(out == ORANGE)) != intersections:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network input")
