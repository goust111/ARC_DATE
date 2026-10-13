import random
from collections import Counter

import numpy as np


BG = 0
SIGNAL = 4
CORRIDOR = 5
PATH_COLORS = (1, 2, 3, 6, 7, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)
TURN_STRAIGHT = "straight"
TURN_RIGHT = "right"
TURN_UTURN = "uturn"
TURN_LEFT = "left"
ALL_TURNS = (TURN_LEFT, TURN_RIGHT, TURN_STRAIGHT, TURN_UTURN)
TURN_SHIFT = {
    TURN_STRAIGHT: 0,
    TURN_RIGHT: 1,
    TURN_UTURN: 2,
    TURN_LEFT: -1,
}
SIGNAL_TO_TURN = {
    (-1, -1): TURN_LEFT,
    (-1, 1): TURN_RIGHT,
    (1, -1): TURN_STRAIGHT,
    (1, 1): TURN_UTURN,
}
TURN_TO_SIGNAL = {turn: diag for diag, turn in SIGNAL_TO_TURN.items()}

MIN_ARM = 2
MAX_ARM = 5
MIN_HALF_SPAN = 3
MAX_HALF_SPAN = 6
GEN_MAX_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(pos: tuple[int, int], traversable: np.ndarray) -> list[tuple[int, int]]:
    r, c = pos
    result: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _in_bounds(traversable.shape, nr, nc) and bool(traversable[nr, nc]):
            result.append((nr, nc))
    return result


def _turn_direction(direction: tuple[int, int], turn: str) -> tuple[int, int]:
    idx = DIRS.index(direction)
    return DIRS[(idx + TURN_SHIFT[turn]) % 4]


def _collect_intersections(grid: np.ndarray, traversable: np.ndarray) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    intersections: dict[tuple[int, int], str] = {}
    signal_usage: Counter[tuple[int, int]] = Counter()

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != CORRIDOR:
                continue
            if not all(
                _in_bounds(grid.shape, r + dr, c + dc) and bool(traversable[r + dr, c + dc])
                for dr, dc in DIRS
            ):
                continue

            diags = []
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = r + dr, c + dc
                    if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == SIGNAL:
                        diags.append((dr, dc))
            assert len(diags) == 1, "each intersection must have exactly one yellow diagonal signal"

            signal_pos = (r + diags[0][0], c + diags[0][1])
            signal_usage[signal_pos] += 1
            intersections[(r, c)] = SIGNAL_TO_TURN[diags[0]]

    signal_cells = {tuple(map(int, pos)) for pos in np.argwhere(grid == SIGNAL)}
    assert intersections, "expected at least one routable intersection"
    assert set(signal_usage) == signal_cells, "every yellow pixel must encode one intersection"
    assert all(count == 1 for count in signal_usage.values()), "signals may not be shared by intersections"
    return intersections


def _endpoint_positions(grid: np.ndarray, traversable: np.ndarray) -> list[tuple[int, int]]:
    endpoints = [tuple(map(int, pos)) for pos in np.argwhere((grid != BG) & (grid != SIGNAL) & (grid != CORRIDOR))]
    assert endpoints, "expected at least one colored endpoint"
    assert len({int(grid[pos]) for pos in endpoints}) == len(endpoints), "endpoint colors must be unique"
    for pos in endpoints:
        assert len(_neighbors(pos, traversable)) == 1, "colored markers must be dead-end endpoints"
    return endpoints


def _validate_corridor_degrees(
    grid: np.ndarray,
    traversable: np.ndarray,
    intersections: dict[tuple[int, int], str],
) -> None:
    for pos in np.argwhere(grid == CORRIDOR):
        cell = tuple(map(int, pos))
        degree = len(_neighbors(cell, traversable))
        if cell in intersections:
            assert degree == 4, "intersections must be true four-way crossings"
        else:
            assert degree == 2, "non-intersection corridor cells must lie on a single path"


def _trace_path(
    start: tuple[int, int],
    grid: np.ndarray,
    traversable: np.ndarray,
    intersections: dict[tuple[int, int], str],
    endpoint_set: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[str], tuple[int, int]]:
    first_steps = _neighbors(start, traversable)
    assert len(first_steps) == 1, "each endpoint must have a single outgoing corridor"

    prev = start
    cur = first_steps[0]
    seen_states = {(prev, cur)}
    corridor_cells: list[tuple[int, int]] = []
    turns_used: list[str] = []

    while True:
        if cur in endpoint_set:
            return corridor_cells, turns_used, cur

        assert int(grid[cur]) == CORRIDOR, "only grey cells may appear inside a traced corridor"
        corridor_cells.append(cur)

        if cur in intersections:
            turn = intersections[cur]
            turns_used.append(turn)
            incoming = (cur[0] - prev[0], cur[1] - prev[1])
            outgoing = _turn_direction(incoming, turn)
            nxt = (cur[0] + outgoing[0], cur[1] + outgoing[1])
            assert _in_bounds(grid.shape, nxt[0], nxt[1]) and bool(traversable[nxt]), "turn signal must lead to a corridor arm"
        else:
            candidates = [pos for pos in _neighbors(cur, traversable) if pos != prev]
            assert len(candidates) == 1, "ordinary corridor cells must have one forward continuation"
            nxt = candidates[0]

        prev, cur = cur, nxt
        assert (prev, cur) not in seen_states, "routing produced a closed loop without reaching an endpoint"
        seen_states.add((prev, cur))


def _solve(grid: np.ndarray, *, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    traversable = (g != BG) & (g != SIGNAL)
    intersections = _collect_intersections(g, traversable)
    endpoints = _endpoint_positions(g, traversable)
    _validate_corridor_degrees(g, traversable, intersections)

    endpoint_order = sorted(endpoints, key=lambda pos: (pos[1], pos[0]))
    endpoint_set = set(endpoints)
    out = g.copy()

    path_sets: list[set[tuple[int, int]]] = []
    turns_seen: set[str] = set()
    destinations: dict[tuple[int, int], tuple[int, int]] = {}

    for start in endpoint_order:
        cells, turns_used, end = _trace_path(start, g, traversable, intersections, endpoint_set)
        destinations[start] = end
        turns_seen.update(turns_used)
        unique_cells = set(cells)
        path_sets.append(unique_cells)

        color = int(g[start])
        for cell in cells:
            out[cell] = color

    if not with_meta:
        return out

    usage = Counter()
    for cell_set in path_sets:
        for cell in cell_set:
            usage[cell] += 1

    return out, {
        "endpoint_order": endpoint_order,
        "destinations": destinations,
        "turns_seen": turns_seen,
        "max_overlap": max(usage.values(), default=1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each route in a grey traffic network by following the hidden turn signal at every crossing.

    Input structure:
    - The background is black.
    - Grey pixels form a one-cell-wide orthogonal corridor network whose true
      crossings are four-way intersections.
    - Distinct colored dead-end markers sit at the network endpoints, and each
      intersection has exactly one yellow pixel on one diagonal corner.

    Rule:
    - Read the yellow diagonal at each intersection as a turn instruction:
      upper-left means turn left, upper-right means turn right, lower-left
      means go straight, and lower-right means make a U-turn, all relative to
      the direction of travel.
    - Trace every endpoint path in left-to-right, then top-to-bottom order,
      following the unique corridor continuation between intersections and the
      signal-selected arm at intersections until a dead-end endpoint is reached.
    - Recolor every grey corridor pixel visited by that path to the start
      marker's color; if later paths reuse a corridor segment, the later color
      overwrites the earlier one. Yellow signal pixels stay unchanged.

    Key insight:
    - The yellow pixel is not part of the corridor: its diagonal placement is a
      codebook for how the path should leave the crossing, so solving the task
      requires decoding those local routing instructions before tracing the
      global paths.
    """
    return _solve(grid)


def _build_candidate() -> np.ndarray:
    top_arm = random.randint(MIN_ARM, MAX_ARM)
    bottom_arm = random.randint(MIN_ARM, MAX_ARM)
    left_arm = random.randint(MIN_ARM, MAX_ARM)
    right_arm = random.randint(MIN_ARM, MAX_ARM)
    upper_half = random.randint(MIN_HALF_SPAN, MAX_HALF_SPAN)
    lower_half = random.randint(MIN_HALF_SPAN, MAX_HALF_SPAN)
    left_half = random.randint(MIN_HALF_SPAN, MAX_HALF_SPAN)
    right_half = random.randint(MIN_HALF_SPAN, MAX_HALF_SPAN)

    height = top_arm + upper_half + lower_half + bottom_arm + 1
    width = left_arm + left_half + right_half + right_arm + 1
    grid = np.full((height, width), BG, dtype=int)

    r_top = top_arm
    r_mid = top_arm + upper_half
    r_bot = top_arm + upper_half + lower_half
    c_left = left_arm
    c_mid = left_arm + left_half
    c_right = left_arm + left_half + right_half

    grid[:, c_mid] = CORRIDOR
    grid[r_mid, :] = CORRIDOR
    grid[r_top, c_left : c_right + 1] = CORRIDOR
    grid[r_bot, c_left : c_right + 1] = CORRIDOR
    grid[r_top : r_bot + 1, c_left] = CORRIDOR
    grid[r_top : r_bot + 1, c_right] = CORRIDOR

    endpoints = [
        (0, c_mid),
        (r_mid, width - 1),
        (height - 1, c_mid),
        (r_mid, 0),
    ]
    for pos, color in zip(endpoints, random.sample(PATH_COLORS, len(endpoints))):
        grid[pos] = color

    intersections = [
        (r_top, c_mid),
        (r_mid, c_right),
        (r_bot, c_mid),
        (r_mid, c_left),
        (r_mid, c_mid),
    ]
    turns = list(ALL_TURNS) + [random.choice(ALL_TURNS)]
    random.shuffle(turns)
    for pos, turn in zip(intersections, turns):
        dr, dc = TURN_TO_SIGNAL[turn]
        signal_pos = (pos[0] + dr, pos[1] + dc)
        assert int(grid[signal_pos]) == BG
        grid[signal_pos] = SIGNAL

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate()
        try:
            out, meta = _solve(grid, with_meta=True)
        except AssertionError:
            continue

        if meta["turns_seen"] != set(ALL_TURNS):
            continue
        if meta["max_overlap"] < 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if not np.array_equal(main(grid), out):
            continue
        return grid

    raise RuntimeError("failed to generate a valid traffic-routing ARC input")
