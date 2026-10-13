import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)

TURN_COLORS = (RED, BLUE, GREEN, YELLOW)

_GEN_ATTEMPTS = 300
_GEN_COUNTER = 0


def _turn(direction: tuple[int, int], color: int) -> tuple[int, int]:
    idx = DIRS.index(direction)
    if color == RED:
        return DIRS[(idx + 1) % 4]
    if color == BLUE:
        return DIRS[(idx - 1) % 4]
    if color == GREEN:
        return direction
    if color == YELLOW:
        return (-direction[0], -direction[1])
    raise AssertionError("unexpected turn color")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _edge_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for c in range(w):
        cells.append((0, c))
        if h > 1:
            cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        if w > 1:
            cells.append((r, w - 1))
    return cells


def _junction_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = np.zeros_like(g, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) not in TURN_COLORS:
                continue
            if (
                int(g[r - 1, c]) == GREY
                and int(g[r + 1, c]) == GREY
                and int(g[r, c - 1]) == GREY
                and int(g[r, c + 1]) == GREY
            ):
                mask[r, c] = True
    return mask


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    starts = [(r, c) for r, c in _edge_positions(g.shape) if int(g[r, c]) == ORANGE]
    assert len(starts) == 1, "expected exactly one orange edge start"
    return starts[0]


def _find_bulb_blocks(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(g[r, c])
            if color in (BLACK, GREY, ORANGE):
                continue
            block = g[r : r + 2, c : c + 2]
            if np.all(block == color):
                blocks.append((r, c, color))
    return blocks


def _adjacent_bulb_block(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if not _in_bounds((h, w), rr, cc):
            continue
        color = int(g[rr, cc])
        if color in (BLACK, GREY, ORANGE):
            continue
        found = None
        for r0 in (rr - 1, rr):
            for c0 in (cc - 1, cc):
                if not _in_bounds((h - 1, w - 1), r0, c0):
                    continue
                block = g[r0 : r0 + 2, c0 : c0 + 2]
                if np.all(block == color):
                    found = (r0, c0)
        if found is not None:
            return found
    return None


def _trace_signal(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    start = _find_start(g)
    junctions = _junction_mask(g)
    traversable = (g == GREY) | junctions
    neighbors: list[tuple[int, int, tuple[int, int]]] = []
    for dr, dc in DIRS:
        rr, cc = start[0] + dr, start[1] + dc
        if _in_bounds(g.shape, rr, cc) and bool(traversable[rr, cc]):
            neighbors.append((rr, cc, (dr, dc)))
    assert len(neighbors) == 1, "start must touch exactly one path endpoint"

    visited_turns: list[int] = []
    seen: set[tuple[int, int, tuple[int, int]]] = set()
    out[start] = ORANGE
    cur_r, cur_c, direction = neighbors[0]

    while True:
        state = (cur_r, cur_c, direction)
        assert state not in seen, "signal path must not cycle"
        seen.add(state)
        out[cur_r, cur_c] = ORANGE

        if bool(junctions[cur_r, cur_c]):
            color = int(g[cur_r, cur_c])
            visited_turns.append(color)
            direction = _turn(direction, color)

        nr, nc = cur_r + direction[0], cur_c + direction[1]
        if not _in_bounds(g.shape, nr, nc) or not bool(traversable[nr, nc]):
            bulb = _adjacent_bulb_block(g, cur_r, cur_c)
            assert bulb is not None, "path endpoint must touch a 2x2 bulb"
            br, bc = bulb
            out[br : br + 2, bc : bc + 2] = ORANGE
            return out, tuple(visited_turns), (cur_r, cur_c), bulb

        cur_r, cur_c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    return _trace_signal(np.asarray(grid, dtype=int))[0]


_GEN_BULB_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, CYAN, BROWN)


def _paint_bulb(grid: np.ndarray, endpoint: tuple[int, int], side: tuple[int, int], color: int) -> None:
    r, c = endpoint
    if side == UP:
        br, bc = r - 2, c + random.choice((-1, 0))
    elif side == DOWN:
        br, bc = r + 1, c + random.choice((-1, 0))
    elif side == LEFT:
        br, bc = r + random.choice((-1, 0)), c - 2
    elif side == RIGHT:
        br, bc = r + random.choice((-1, 0)), c + 1
    else:
        raise AssertionError("unexpected bulb side")
    grid[br : br + 2, bc : bc + 2] = int(color)


def _build_two_junction_grid(required_turn: int) -> np.ndarray:
    if required_turn == RED:
        start_side = LEFT
        top_color = RED
        bottom_color = random.choice((GREEN, RED, BLUE))
    elif required_turn == BLUE:
        start_side = RIGHT
        top_color = BLUE
        bottom_color = random.choice((GREEN, RED, BLUE))
    elif required_turn == GREEN:
        start_side = random.choice((LEFT, RIGHT))
        top_color = GREEN
        bottom_color = random.choice(TURN_COLORS)
    elif required_turn == YELLOW:
        start_side = random.choice((LEFT, RIGHT))
        top_color = RED if start_side == LEFT else BLUE
        bottom_color = YELLOW
    else:
        raise AssertionError("unexpected required turn")

    top_len = random.randint(2, 4)
    gap = random.randint(4, 6)
    bottom_left_len = random.randint(2, 4)
    bottom_right_len = random.randint(2, 4)
    bottom_down_len = random.randint(2, 4)
    side_exit_len = random.randint(2, 4)

    col = max(bottom_left_len + 2, 4)
    top_row = top_len + 2
    bottom_row = top_row + gap
    height = bottom_row + bottom_down_len + 3

    if start_side == LEFT:
        width = max(col + side_exit_len + 3, col + bottom_right_len + 3)
    else:
        width = max(col + bottom_right_len + 3, col + side_exit_len + 2)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[top_row : bottom_row + 1, col] = GREY
    grid[top_row - top_len : top_row, col] = GREY
    grid[bottom_row, col - bottom_left_len : col] = GREY
    grid[bottom_row, col + 1 : col + bottom_right_len + 1] = GREY
    grid[bottom_row + 1 : bottom_row + bottom_down_len + 1, col] = GREY

    if start_side == LEFT:
        right_endpoint = col + side_exit_len
        grid[top_row, 0] = ORANGE
        grid[top_row, 1 : right_endpoint + 1] = GREY
        side_bulb_endpoint = (top_row, right_endpoint)
        side_bulb_dir = RIGHT
    else:
        left_endpoint = col - side_exit_len
        grid[top_row, left_endpoint : width - 1] = GREY
        grid[top_row, width - 1] = ORANGE
        side_bulb_endpoint = (top_row, left_endpoint)
        side_bulb_dir = LEFT

    grid[top_row, col] = top_color
    grid[bottom_row, col] = bottom_color

    bulb_colors = random.sample(_GEN_BULB_COLORS, 5)
    bulbs = [
        ((top_row - top_len, col), UP),
        (side_bulb_endpoint, side_bulb_dir),
        ((bottom_row, col - bottom_left_len), LEFT),
        ((bottom_row, col + bottom_right_len), RIGHT),
        ((bottom_row + bottom_down_len, col), DOWN),
    ]
    for (endpoint, side), color in zip(bulbs, bulb_colors):
        _paint_bulb(grid, endpoint, side, color)
    return grid


def _gen_attempt(required_turn: int) -> np.ndarray | None:
    grid = _build_two_junction_grid(required_turn)
    grid = np.rot90(grid, random.randint(0, 3))

    try:
        out, visited_turns, _endpoint, lit_bulb = _trace_signal(grid)
    except AssertionError:
        return None

    if required_turn not in visited_turns:
        return None
    bulbs = _find_bulb_blocks(grid)
    if len(bulbs) != 5:
        return None
    lit_count = 0
    for br, bc, _color in bulbs:
        if np.all(out[br : br + 2, bc : bc + 2] == ORANGE):
            lit_count += 1
    if lit_count != 1:
        return None
    if all((br, bc) != lit_bulb for br, bc, _color in bulbs):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    required_turn = TURN_COLORS[_GEN_COUNTER % len(TURN_COLORS)]
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(required_turn)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid input for task 3")
