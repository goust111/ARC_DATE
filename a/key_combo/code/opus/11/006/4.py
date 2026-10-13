import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR_TO_STEP = {
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
    "right": (0, 1),
}

UNSET = -1
STATE_EMPTY = 0
STATE_LEFT = 1
STATE_RIGHT = 2
STATE_BOTH = 3

MAX_ATTEMPTS = 400


def _direction_from_marker(marker_col: int, total_width: int) -> str:
    quarter = total_width / 4.0
    if marker_col < quarter:
        return "left"
    if marker_col < 2 * quarter:
        return "up"
    if marker_col < 3 * quarter:
        return "down"
    return "right"


def _marker_choices(total_width: int, direction: str) -> list[int]:
    bins = {
        "left": [c for c in range(total_width) if c < total_width / 4.0],
        "up": [c for c in range(total_width) if total_width / 4.0 <= c < total_width / 2.0],
        "down": [c for c in range(total_width) if total_width / 2.0 <= c < 3.0 * total_width / 4.0],
        "right": [c for c in range(total_width) if c >= 3.0 * total_width / 4.0],
    }
    return bins[direction]


def _parse(grid: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 7 and w % 2 == 1, "expected control row plus two equal panels"

    control = g[0]
    red_cols = np.flatnonzero(control == RED)
    assert red_cols.size == 1, "expected exactly one red direction marker"
    assert np.all(np.isin(control, [GREY, RED])), "control row may only contain grey and red"
    direction = _direction_from_marker(int(red_cols[0]), w)

    body = g[1:]
    sep = w // 2
    assert np.all(body[:, sep] == GREY), "expected a solid grey separator column"

    left = body[:, :sep]
    right = body[:, sep + 1 :]
    assert left.shape == right.shape and left.size > 0
    assert np.all(np.isin(left, [BLACK, BLUE])), "left panel must use black/blue only"
    assert np.all(np.isin(right, [BLACK, YELLOW])), "right panel must use black/yellow only"
    return direction, left, right


def _seed_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left == BLACK) & (right == BLACK)


def _grow(out: np.ndarray, seeds: np.ndarray, direction: str) -> np.ndarray:
    dr, dc = DIR_TO_STEP[direction]
    for r, c in map(tuple, np.argwhere(seeds)):
        rr, cc = int(r) + dr, int(c) + dc
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == BLACK:
            out[rr, cc] = GREEN
            rr += dr
            cc += dc
    return out


def _transform(grid: np.ndarray) -> np.ndarray:
    direction, left, right = _parse(grid)
    common_empty = _seed_mask(left, right)

    out = np.concatenate([left.copy(), right.copy()], axis=1)
    panel_w = left.shape[1]
    out[:, :panel_w][common_empty] = GREEN
    out[:, panel_w:][common_empty] = GREEN

    seeds = out == GREEN
    return _grow(out, seeds, direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop off the control row and separator, then grow green from the panels' shared empty coordinates.

    Input structure:
    - The top row is a grey control band with exactly one red marker whose column falls into one of four quartiles.
    - Below it, a solid grey separator column splits the grid into equal-width left and right panels.
    - The left panel uses black background with blue foreground; the right panel uses black background with yellow foreground.

    Rule:
    - Decode the red marker's quartile as a growth direction: left, up, down, or right.
    - Remove the control row and separator, keep the two panels side by side, and preserve all blue/yellow foreground cells.
    - Wherever the two panels are both black at the same local coordinate, place a green seed in each retained panel position.
    - From every green seed, extend green step by step in the decoded direction through black cells only, stopping before any non-black cell or the grid edge.

    Key insight:
    - The comparison mask does not produce the final answer directly: shared empty coordinates only define the seed set, while the control-row marker supplies the direction that turns those seeds into longer green traces.
    """
    return _transform(grid)


def _iter_line(seed: tuple[int, int], direction: str, stopper: int) -> Iterable[tuple[int, int]]:
    r, c = seed
    if direction == "left":
        for cc in range(c - 1, stopper, -1):
            yield r, cc
    elif direction == "right":
        for cc in range(c + 1, stopper):
            yield r, cc
    elif direction == "up":
        for rr in range(r - 1, stopper, -1):
            yield rr, c
    else:
        for rr in range(r + 1, stopper):
            yield rr, c


def _stopper_coord(seed: tuple[int, int], direction: str, stopper: int) -> tuple[int, int]:
    r, c = seed
    if direction in ("left", "right"):
        return r, stopper
    return stopper, c


def _assign_if_free(state: np.ndarray, coord: tuple[int, int], value: int) -> bool:
    r, c = coord
    current = int(state[r, c])
    if current == UNSET or current == value:
        state[r, c] = value
        return True
    return False


def _fill_remaining(state: np.ndarray, rng: random.Random) -> None:
    base_values = [STATE_LEFT, STATE_RIGHT, STATE_BOTH]
    base_weights = [0.48, 0.48, 0.04]
    h, w = state.shape
    for r in range(h):
        for c in range(w):
            if state[r, c] != UNSET:
                continue
            choice = None
            neighbor_values = []
            if c > 0 and state[r, c - 1] in base_values:
                neighbor_values.append(int(state[r, c - 1]))
            if r > 0 and state[r - 1, c] in base_values:
                neighbor_values.append(int(state[r - 1, c]))
            if neighbor_values and rng.random() < 0.65:
                choice = rng.choice(neighbor_values)
                if choice == STATE_BOTH and rng.random() < 0.7:
                    choice = rng.choice([STATE_LEFT, STATE_RIGHT])
            if choice is None:
                choice = rng.choices(base_values, weights=base_weights, k=1)[0]
            state[r, c] = choice


def _state_to_panels(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros(state.shape, dtype=int)
    right = np.zeros(state.shape, dtype=int)
    left[np.isin(state, [STATE_LEFT, STATE_BOTH])] = BLUE
    right[np.isin(state, [STATE_RIGHT, STATE_BOTH])] = YELLOW
    return left, right


def _build_grid(direction: str, panel_h: int, panel_w: int, seed_count: int, rng: random.Random) -> np.ndarray | None:
    state = np.full((panel_h, panel_w), UNSET, dtype=int)
    horizontal = direction in ("left", "right")

    if horizontal:
        rows = rng.sample(range(panel_h), seed_count)
        seed_positions = []
        for row in rows:
            if direction == "left":
                col = rng.randrange(2, panel_w)
            else:
                col = rng.randrange(0, panel_w - 2)
            seed_positions.append((row, col))
    else:
        cols = rng.sample(range(panel_w), seed_count)
        seed_positions = []
        for col in cols:
            if direction == "up":
                row = rng.randrange(2, panel_h)
            else:
                row = rng.randrange(0, panel_h - 2)
            seed_positions.append((row, col))

    active_sides = ["left", "right"]
    while len(active_sides) < seed_count:
        active_sides.append(rng.choice(["left", "right"]))
    rng.shuffle(active_sides)

    grew_left = False
    grew_right = False
    for seed, active_side in zip(seed_positions, active_sides):
        if not _assign_if_free(state, seed, STATE_EMPTY):
            return None

        r, c = seed
        if direction == "left":
            stopper = rng.randrange(0, c - 1)
        elif direction == "right":
            stopper = rng.randrange(c + 2, panel_w)
        elif direction == "up":
            stopper = rng.randrange(0, r - 1)
        else:
            stopper = rng.randrange(r + 2, panel_h)

        path_state = STATE_RIGHT if active_side == "left" else STATE_LEFT
        stopper_state = STATE_LEFT if active_side == "left" else STATE_RIGHT
        for coord in _iter_line(seed, direction, stopper):
            if not _assign_if_free(state, coord, path_state):
                return None
        if not _assign_if_free(state, _stopper_coord(seed, direction, stopper), stopper_state):
            return None

        if active_side == "left":
            grew_left = True
        else:
            grew_right = True

    if not (grew_left and grew_right):
        return None

    _fill_remaining(state, rng)
    left, right = _state_to_panels(state)

    common_empty = _seed_mask(left, right)
    if int(common_empty.sum()) != seed_count:
        return None

    left_only = (left != BLACK) & (right == BLACK)
    right_only = (left == BLACK) & (right != BLACK)
    if not left_only.any() or not right_only.any():
        return None

    total_w = 2 * panel_w + 1
    marker_col = rng.choice(_marker_choices(total_w, direction))
    grid = np.zeros((panel_h + 1, total_w), dtype=int)
    grid[0, :] = GREY
    grid[0, marker_col] = RED
    grid[1:, :panel_w] = left
    grid[1:, panel_w] = GREY
    grid[1:, panel_w + 1 :] = right
    return grid


def _alternate_direction(direction: str) -> str:
    order = {
        "left": "up",
        "up": "right",
        "right": "down",
        "down": "left",
    }
    return order[direction]


def _move_marker(grid: np.ndarray, direction: str) -> np.ndarray:
    g = grid.copy()
    g[0, g[0] == RED] = GREY
    total_w = g.shape[1]
    g[0, _marker_choices(total_w, direction)[0]] = RED
    return g


def _semantic_check(grid: np.ndarray) -> bool:
    direction, left, right = _parse(grid)
    common_empty = _seed_mask(left, right)
    seed_count = int(common_empty.sum())
    if not (3 <= seed_count <= 8):
        return False

    if direction in ("left", "right"):
        if len(set(map(int, np.argwhere(common_empty)[:, 0]))) != seed_count:
            return False
    else:
        if len(set(map(int, np.argwhere(common_empty)[:, 1]))) != seed_count:
            return False

    left_only = (left != BLACK) & (right == BLACK)
    right_only = (left == BLACK) & (right != BLACK)
    if not left_only.any() or not right_only.any():
        return False

    out = _transform(grid)
    if out.shape != (grid.shape[0] - 1, grid.shape[1] - 1):
        return False

    min_seed_pixels = 2 * seed_count
    if int(np.count_nonzero(out == GREEN)) <= min_seed_pixels:
        return False

    alt_out = _transform(_move_marker(grid, _alternate_direction(direction)))
    if np.array_equal(out, alt_out):
        return False

    probe = grid.copy()
    r0, c0 = map(int, np.argwhere(common_empty)[0])
    probe[1 + r0, c0] = BLUE
    if np.array_equal(out, _transform(probe)):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        direction = rng.choice(list(DIR_TO_STEP))
        panel_h = rng.randint(8, 12)
        panel_w = rng.randint(7, 10)
        axis_cap = panel_h if direction in ("left", "right") else panel_w
        seed_count = rng.randint(3, min(8, axis_cap))

        grid = _build_grid(direction, panel_h, panel_w, seed_count, rng)
        if grid is None:
            continue
        if _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid double-panel growth input")
