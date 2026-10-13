from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    GREEN = 3
    TEAL = 8
    CYAN = 8


Coord = tuple[int, int]
Direction = tuple[int, int]
Segment = tuple[Coord, Direction, int, int | None]

DIRS: tuple[Direction, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

GEN_HEIGHT_RANGE = (10, 19)
GEN_WIDTH_RANGE = (10, 19)
GEN_TURN_COUNT_RANGE = (2, 4)
GEN_MAX_TURN_SEG_LEN = 8
GEN_MIN_PAINTED_CELLS = 8
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _is_border(cell: Coord, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _dir_index(direction: Direction) -> int:
    return DIRS.index(direction)


def _turn_right(direction: Direction) -> Direction:
    return DIRS[(_dir_index(direction) + 1) % 4]


def _turn_left(direction: Direction) -> Direction:
    return DIRS[(_dir_index(direction) - 1) % 4]


def _turn(direction: Direction, router_color: int) -> Direction:
    if int(router_color) == Color.RED:
        return _turn_right(direction)
    assert int(router_color) == Color.TEAL
    return _turn_left(direction)


def _seed_direction(seed: Coord, shape: tuple[int, int]) -> Direction:
    r, c = seed
    h, w = shape
    choices: list[Direction] = []
    if r == 0:
        choices.append((1, 0))
    if r == h - 1:
        choices.append((-1, 0))
    if c == 0:
        choices.append((0, 1))
    if c == w - 1:
        choices.append((0, -1))
    assert len(choices) == 1, "seed must lie on exactly one edge"
    return choices[0]


def _step(cell: Coord, direction: Direction, amount: int = 1) -> Coord:
    return cell[0] + direction[0] * amount, cell[1] + direction[1] * amount


def _distance_to_exit(cell: Coord, direction: Direction, shape: tuple[int, int]) -> int:
    r, c = cell
    dr, dc = direction
    h, w = shape
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    assert dc == 1
    return w - 1 - c


def _segment_cells(start: Coord, direction: Direction, length: int) -> list[Coord]:
    return [_step(start, direction, step) for step in range(1, int(length) + 1)]


def _router_position(segment: Segment) -> Coord:
    start, direction, length, router_color = segment
    assert router_color is not None
    return _step(start, direction, int(length) + 1)


def _find_seed(grid: np.ndarray) -> Coord:
    seeds = np.argwhere(grid == Color.GREEN)
    assert seeds.shape[0] == 1, "expected exactly one green seed"
    return int(seeds[0, 0]), int(seeds[0, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a green border ray through red and teal single-pixel routers.

    Input structure
    - The grid is a black background with exactly one green seed pixel on a border cell away from corners.
    - Zero or more internal single-pixel routers use red or teal (cyan).
    - No other colors participate in the puzzle.

    Rule
    - Launch the ray from the seed perpendicular to the border edge and advance one cell at a time, painting each traversed background cell green.
    - If the next cell straight ahead is red or teal, do not enter it; instead turn 90 degrees in place, with red meaning clockwise and teal meaning counterclockwise.
    - Stop when the next step would leave the grid, and keep the seed plus every router in their original colors.

    Key insight
    - The interior dots are not paint targets or endpoints; they are one-cell routers whose colors encode the bends of one deterministic orthogonal polyline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    allowed = {Color.BLACK, Color.RED, Color.GREEN, Color.TEAL}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors"

    seed = _find_seed(g)
    direction = _seed_direction(seed, g.shape)

    routers = np.argwhere((g == Color.RED) | (g == Color.TEAL))
    for rr, cc in routers:
        assert not _is_border((int(rr), int(cc)), g.shape), "routers must be internal"

    out = g.copy()
    r, c = seed
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (int(r), int(c), int(direction[0]), int(direction[1]))
        assert state not in seen_states, "ray entered a loop"
        seen_states.add(state)

        nr, nc = r + direction[0], c + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            break

        value = int(g[nr, nc])
        if value == Color.RED:
            direction = _turn_right(direction)
            continue
        if value == Color.TEAL:
            direction = _turn_left(direction)
            continue

        assert value in {Color.BLACK, Color.GREEN}, "only routers may block the ray"
        r, c = nr, nc
        out[r, c] = Color.GREEN

    return out


def _sample_seed(shape: tuple[int, int]) -> Coord:
    h, w = shape
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge == "top":
        return 0, random.randint(1, w - 2)
    if edge == "bottom":
        return h - 1, random.randint(1, w - 2)
    if edge == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _sample_turn_colors(turn_count: int) -> list[int]:
    colors = [Color.RED, Color.TEAL]
    while len(colors) < turn_count:
        colors.append(random.choice((Color.RED, Color.TEAL)))
    random.shuffle(colors)
    return colors


def _plan_segments(
    shape: tuple[int, int],
    current: Coord,
    direction: Direction,
    turn_colors: list[int],
    occupied_path: set[Coord],
    routers: set[Coord],
) -> list[Segment] | None:
    if not turn_colors:
        final_length = _distance_to_exit(current, direction, shape)
        if final_length < 1:
            return None
        final_cells = _segment_cells(current, direction, final_length)
        if any(cell in occupied_path or cell in routers for cell in final_cells):
            return None
        return [(current, direction, final_length, None)]

    max_turn_length = min(GEN_MAX_TURN_SEG_LEN, _distance_to_exit(current, direction, shape) - 2)
    if max_turn_length < 1:
        return None

    candidate_lengths = list(range(1, max_turn_length + 1))
    random.shuffle(candidate_lengths)
    router_color = int(turn_colors[0])

    for length in candidate_lengths:
        path_cells = _segment_cells(current, direction, length)
        if any(cell in occupied_path or cell in routers for cell in path_cells):
            continue

        router = _step(current, direction, length + 1)
        if router in occupied_path or router in routers or _is_border(router, shape):
            continue

        endpoint = path_cells[-1]
        new_direction = _turn(direction, router_color)

        remaining_turns = len(turn_colors) - 1
        needed_clearance = 3 if remaining_turns > 0 else 1
        if _distance_to_exit(endpoint, new_direction, shape) < needed_clearance:
            continue

        first_after_turn = _step(endpoint, new_direction)
        if first_after_turn in occupied_path or first_after_turn in routers:
            continue

        rest = _plan_segments(
            shape,
            endpoint,
            new_direction,
            turn_colors[1:],
            occupied_path | set(path_cells),
            routers | {router},
        )
        if rest is not None:
            return [(current, direction, length, router_color)] + rest

    return None


def _render_input(shape: tuple[int, int], seed: Coord, segments: list[Segment]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    grid[seed] = Color.GREEN
    for segment in segments:
        if segment[3] is None:
            continue
        rr, cc = _router_position(segment)
        grid[rr, cc] = int(segment[3])
    return grid


def _render_expected(inp: np.ndarray, segments: list[Segment]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for start, direction, length, _router_color in segments:
        for rr, cc in _segment_cells(start, direction, length):
            out[rr, cc] = Color.GREEN
    return out


def _validate_segments(
    shape: tuple[int, int],
    seed: Coord,
    start_direction: Direction,
    segments: list[Segment],
) -> bool:
    current = seed
    direction = start_direction
    path_seen: set[Coord] = {seed}
    router_seen: set[Coord] = set()

    for index, segment in enumerate(segments):
        seg_start, seg_direction, length, router_color = segment
        if seg_start != current or seg_direction != direction or length < 1:
            return False

        cells = _segment_cells(seg_start, seg_direction, length)
        if any(not _in_bounds(shape, rr, cc) for rr, cc in cells):
            return False
        if any(cell in path_seen or cell in router_seen for cell in cells):
            return False
        path_seen.update(cells)

        current = cells[-1]
        if router_color is None:
            return index == len(segments) - 1 and not _in_bounds(shape, *_step(current, direction))

        router = _router_position(segment)
        if (
            router in path_seen
            or router in router_seen
            or not _in_bounds(shape, router[0], router[1])
            or _is_border(router, shape)
        ):
            return False
        router_seen.add(router)
        direction = _turn(direction, int(router_color))

        if index == len(segments) - 1:
            return False
        if not _in_bounds(shape, *_step(current, direction)):
            return False

    return False


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GEN_HEIGHT_RANGE),
        random.randint(*GEN_WIDTH_RANGE),
    )
    seed = _sample_seed(shape)
    start_direction = _seed_direction(seed, shape)

    turn_count = random.randint(*GEN_TURN_COUNT_RANGE)
    turn_colors = _sample_turn_colors(turn_count)
    segments = _plan_segments(shape, seed, start_direction, turn_colors, {seed}, set())
    if segments is None:
        return None
    if not _validate_segments(shape, seed, start_direction, segments):
        return None

    inp = _render_input(shape, seed, segments)
    expected = _render_expected(inp, segments)

    if int(np.sum(expected != inp)) < GEN_MIN_PAINTED_CELLS:
        return None
    if not ({Color.RED, Color.TEAL} <= set(turn_colors)):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routing-ray input")
