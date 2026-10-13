from __future__ import annotations

import hashlib
import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
PINK = 6
TEAL = 8

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left

_GEN_CALL_COUNT = 0
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _array_sig(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=int)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _turn(direction: int, instruction: int) -> int:
    if instruction == PINK:
        return (direction + 1) % 4
    if instruction == TEAL:
        return (direction - 1) % 4
    raise AssertionError("unknown instruction color")


def _start_and_direction(arena: np.ndarray) -> tuple[tuple[int, int], int]:
    starts = np.argwhere(arena == GREEN)
    assert starts.shape[0] == 1, "arena must contain exactly one green start pixel"
    r, c = map(int, starts[0])

    sides: list[int] = []
    if r == 0:
        sides.append(2)
    if r == arena.shape[0] - 1:
        sides.append(0)
    if c == 0:
        sides.append(1)
    if c == arena.shape[1] - 1:
        sides.append(3)
    assert len(sides) == 1, "start must lie on exactly one arena edge"
    return (r, c), sides[0]


def _parse_grid(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 5 <= h <= 30 and 5 <= w <= 30
    assert w >= 5, "need control column, separator, and arena"
    assert np.all(g[:, 1] == GREY), "second column must be the grey separator"

    instructions: list[int] = []
    row = 0
    while row < h and int(g[row, 0]) in (PINK, TEAL):
        instructions.append(int(g[row, 0]))
        row += 1
    assert instructions, "control column must contain at least one instruction"
    assert np.all(g[row:, 0] == BLACK), "control column tail must be black"

    arena = g[:, 2:].copy()
    allowed = {BLACK, GREEN, GREY}
    assert set(map(int, np.unique(arena))) <= allowed, "arena colors must be black/green/grey only"

    obstacle_coords = np.argwhere(arena == GREY)
    assert obstacle_coords.shape[0] >= 2, "need internal grey obstacles"
    for r, c in obstacle_coords:
        rr = int(r)
        cc = int(c)
        assert 0 < rr < arena.shape[0] - 1 and 0 < cc < arena.shape[1] - 1, "obstacles must be internal"

    _start_and_direction(arena)
    return instructions, arena


def _simulate_arena(
    arena: np.ndarray,
    instructions: list[int],
    *,
    allow_multi_turn: bool = True,
    stop_after_last_turn: bool = False,
    ignore_obstacles: bool = False,
) -> tuple[np.ndarray, dict]:
    out = arena.copy()
    (r, c), direction = _start_and_direction(arena)
    pointer = 0
    post_exhaustion_moves = 0
    turn_event_sizes: list[int] = []
    current_turns = 0
    collision_checks = 0

    while True:
        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        blocked = not _in_bounds(arena.shape, nr, nc)
        if not blocked and not ignore_obstacles:
            blocked = int(arena[nr, nc]) == GREY

        if blocked:
            collision_checks += 1
            if pointer >= len(instructions):
                if current_turns:
                    turn_event_sizes.append(current_turns)
                break

            direction = _turn(direction, instructions[pointer])
            pointer += 1
            current_turns += 1
            if stop_after_last_turn and pointer == len(instructions):
                turn_event_sizes.append(current_turns)
                break

            if not allow_multi_turn:
                dr2, dc2 = DIRS[direction]
                nr2, nc2 = r + dr2, c + dc2
                blocked2 = not _in_bounds(arena.shape, nr2, nc2)
                if not blocked2 and not ignore_obstacles:
                    blocked2 = int(arena[nr2, nc2]) == GREY
                if blocked2:
                    turn_event_sizes.append(current_turns)
                    break
            continue

        if current_turns:
            turn_event_sizes.append(current_turns)
            current_turns = 0

        r, c = nr, nc
        out[r, c] = GREEN
        if pointer >= len(instructions):
            post_exhaustion_moves += 1

    return out, {
        "instructions_used": pointer,
        "post_exhaustion_moves": post_exhaustion_moves,
        "turn_event_sizes": turn_event_sizes,
        "multi_turn_events": sum(size > 1 for size in turn_event_sizes),
        "collision_checks": collision_checks,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the green ray through the arena, consuming the left-column turn tape whenever a blocked direction is encountered.

    Input structure:
    - The leftmost column is a control tape: pink and teal pixels appear contiguously from the top, and the rest of the column is black.
    - The second column is a full-height grey separator.
    - To the right is a black arena containing internal grey obstacle cells and exactly one green start pixel on the arena boundary.

    Rule:
    - Crop away the control and separator columns only after simulating the ray inside the arena.
    - The ray starts at the green boundary pixel and moves orthogonally inward, painting every traversed passable cell green.
    - Whenever the next step would hit an obstacle or leave the arena, consume the next control instruction: pink turns the ray right and teal turns it left; if the new facing is still blocked, immediately consume another instruction and turn again.
    - After all instructions are exhausted, the ray no longer turns; it keeps going straight until the next blocked step and then stops in place.

    Key insight:
    - The control column is a time-ordered instruction tape: turns are consumed by the chronological sequence of collision checks, not by spatial proximity to any obstacle.
    """
    instructions, arena = _parse_grid(grid)
    out, _ = _simulate_arena(arena, instructions)
    return out


def _build_canonical_arena(
    height: int,
    width: int,
    start_row: int,
    collision_col: int,
    trap_upper: bool,
    ending_type: int,
) -> tuple[np.ndarray, list[int]]:
    arena = np.full((height, width), BLACK, dtype=int)
    arena[start_row, 0] = GREEN
    arena[start_row, collision_col + 1] = GREY
    arena[start_row - 1 if trap_upper else start_row + 1, collision_col] = GREY

    instructions: list[int]
    if trap_upper:
        instructions = [TEAL, TEAL, PINK]
        if ending_type == 0:
            instructions.append(TEAL)
        elif ending_type == 1:
            instructions.append(PINK)
        else:
            instructions.extend([PINK, PINK])
    else:
        instructions = [PINK, PINK, TEAL]
        if ending_type == 0:
            instructions.append(PINK)
        elif ending_type == 1:
            instructions.append(TEAL)
        else:
            instructions.extend([TEAL, TEAL])

    return arena, instructions


def _compose_input(arena: np.ndarray, instructions: list[int]) -> np.ndarray:
    h, w = arena.shape
    assert len(instructions) < h
    grid = np.full((h, w + 2), BLACK, dtype=int)
    grid[:, 1] = GREY
    for r, color in enumerate(instructions):
        grid[r, 0] = color
    grid[:, 2:] = arena
    return grid


def _generate_candidate(call_index: int) -> tuple[np.ndarray, int]:
    rotation = call_index % 4
    ending_type = call_index % 3
    trap_upper = ((call_index // 2) % 2 == 0)

    height = random.randint(9, 16)
    width = random.randint(10, 17)
    start_row = random.randint(2, height - 3)
    collision_col = random.randint(3, width - 5)

    arena, instructions = _build_canonical_arena(
        height=height,
        width=width,
        start_row=start_row,
        collision_col=collision_col,
        trap_upper=trap_upper,
        ending_type=ending_type,
    )
    arena = np.rot90(arena, rotation)
    return _compose_input(arena, instructions), ending_type


def _validate_candidate(grid: np.ndarray, ending_type: int) -> tuple[np.ndarray, str, str] | None:
    instructions, arena = _parse_grid(grid)
    expected, meta = _simulate_arena(arena, instructions)

    if meta["instructions_used"] != len(instructions):
        return None
    if not meta["turn_event_sizes"] or meta["turn_event_sizes"][0] != 2:
        return None
    if meta["multi_turn_events"] < 1:
        return None
    if ending_type == 0:
        if meta["post_exhaustion_moves"] != 0:
            return None
    else:
        if meta["post_exhaustion_moves"] <= 0:
            return None

    if int(np.count_nonzero(expected == GREEN)) < 8:
        return None

    reversed_out, _ = _simulate_arena(arena, list(reversed(instructions)))
    if np.array_equal(reversed_out, expected):
        return None

    single_turn_out, _ = _simulate_arena(arena, instructions, allow_multi_turn=False)
    if np.array_equal(single_turn_out, expected):
        return None

    no_obstacle_out, _ = _simulate_arena(arena, instructions, ignore_obstacles=True)
    if np.array_equal(no_obstacle_out, expected):
        return None

    if ending_type != 0:
        early_stop_out, _ = _simulate_arena(arena, instructions, stop_after_last_turn=True)
        if np.array_equal(early_stop_out, expected):
            return None

    if not np.array_equal(main(grid), expected):
        return None

    in_sig = _array_sig(grid)
    out_sig = _array_sig(expected)
    if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return None
    return grid, in_sig, out_sig


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(500):
        grid, ending_type = _generate_candidate(_GEN_CALL_COUNT)
        checked = _validate_candidate(grid, ending_type)
        if checked is None:
            continue

        good_grid, in_sig, out_sig = checked
        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        _GEN_CALL_COUNT += 1
        return good_grid

    raise RuntimeError("generate_input: failed to build a valid non-degenerate sample")
