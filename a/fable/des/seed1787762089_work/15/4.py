from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AZURE = 8

INSTRUCTION_COLORS = (AZURE, RED)
START_SIDES = ("left", "right", "bottom")
STOP_NO_INSTRUCTIONS = "no_instructions"
STOP_BLOCKED_TURN = "blocked_turn"

GEN_RETRIES = 600
BLANK_RETRIES = 500
OBSTACLE_LAYOUT_RETRIES = 300
RECT_RETRIES = 200


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _is_boundary_cell(r: int, c: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _parse_instructions(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    assert h >= 4 and w >= 4
    tape: list[int] = []
    c = 0
    while c < w and int(grid[0, c]) in INSTRUCTION_COLORS:
        tape.append(int(grid[0, c]))
        c += 1
    assert 2 <= len(tape) <= 5, "expected a top-row instruction tape of length 2..5"
    assert np.all(grid[1, :] == GREY), "second row must be the full grey separator wall"
    return tape


def _find_start(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(grid == GREEN)
    assert coords.shape == (1, 2), "expected exactly one green start pixel"
    r, c = map(int, coords[0])
    h, w = grid.shape
    assert not ((r == 0 or r == h - 1) and (c == 0 or c == w - 1)), "start cannot be in a corner"
    if c == 0:
        return (r, c), (0, 1)
    if c == w - 1:
        return (r, c), (0, -1)
    if r == h - 1:
        return (r, c), (-1, 0)
    raise AssertionError("start must lie on the left, right, or bottom boundary")


def _simulate_from_state(
    grid: np.ndarray,
    instructions: list[int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, object]:
    h, w = grid.shape
    out = np.array(grid, copy=True)
    r, c = start
    dr, dc = direction
    used = 0
    path_cells: set[tuple[int, int]] = set()
    checked_cells: set[tuple[int, int]] = set()
    obstacle_hits = 0
    collisions = 0

    for _ in range(h * w * (len(instructions) + 2) * 4):
        nr = r + dr
        nc = c + dc
        in_bounds = 0 <= nr < h and 0 <= nc < w
        if in_bounds:
            checked_cells.add((nr, nc))

        blocked = (not in_bounds) or _is_boundary_cell(nr, nc, grid.shape) or int(grid[nr, nc]) == GREY
        if not blocked:
            if int(out[nr, nc]) == BLACK:
                out[nr, nc] = YELLOW
            path_cells.add((nr, nc))
            r, c = nr, nc
            continue

        collisions += 1
        if in_bounds and int(grid[nr, nc]) == GREY and nr != 1:
            obstacle_hits += 1

        if used >= len(instructions):
            stop_mode = STOP_NO_INSTRUCTIONS
            break

        turn_color = instructions[used]
        used += 1
        if turn_color == AZURE:
            ndr, ndc = _turn_left((dr, dc))
        else:
            assert turn_color == RED
            ndr, ndc = _turn_right((dr, dc))

        tr = r + ndr
        tc = c + ndc
        turn_in_bounds = 0 <= tr < h and 0 <= tc < w
        if turn_in_bounds:
            checked_cells.add((tr, tc))

        turn_blocked = (not turn_in_bounds) or _is_boundary_cell(tr, tc, grid.shape) or int(grid[tr, tc]) == GREY
        if turn_blocked:
            stop_mode = STOP_BLOCKED_TURN
            break

        dr, dc = ndr, ndc
    else:
        raise AssertionError("particle simulation did not terminate")

    return {
        "output": out,
        "checked_cells": checked_cells,
        "path_cells": path_cells,
        "final_pos": (r, c),
        "instructions_used": used,
        "collisions": collisions,
        "stop_mode": stop_mode,
        "interior_stop": 0 < r < h - 1 and 0 < c < w - 1,
        "obstacle_hits": obstacle_hits,
    }


def _simulate(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    instructions = _parse_instructions(g)
    start, direction = _find_start(g)
    return _simulate_from_state(g, instructions, start, direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate the green-edge particle that paints yellow trails and turns left/right
    on each collision according to the top instruction tape.
    """
    result = _simulate(grid)
    return np.asarray(result["output"], dtype=int)


def _base_grid(
    h: int,
    w: int,
    instructions: Iterable[int],
    side: str,
    start_index: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    grid = np.full((h, w), BLACK, dtype=int)
    tape = list(instructions)
    grid[0, : len(tape)] = tape
    grid[1, :] = GREY
    if side == "left":
        start = (start_index, 0)
        direction = (0, 1)
    elif side == "right":
        start = (start_index, w - 1)
        direction = (0, -1)
    else:
        assert side == "bottom"
        start = (h - 1, start_index)
        direction = (-1, 0)
    grid[start] = GREEN
    return grid, start, direction


def _choose_blank_layout(
    rng: random.Random,
    desired_stop: str,
) -> tuple[np.ndarray, list[int], tuple[int, int], tuple[int, int], dict[str, object]]:
    for _ in range(BLANK_RETRIES):
        h = rng.randint(14, 20)
        w = rng.randint(14, 20)
        instr_len = rng.randint(2, 5)
        instructions = [rng.choice(INSTRUCTION_COLORS) for _ in range(instr_len)]
        side = rng.choice(START_SIDES)
        if side in ("left", "right"):
            start_index = rng.randint(2, h - 2)
        else:
            start_index = rng.randint(1, w - 2)
        grid, start, direction = _base_grid(h, w, instructions, side, start_index)
        result = _simulate_from_state(grid, instructions, start, direction)
        if result["stop_mode"] != desired_stop:
            continue
        if int(result["instructions_used"]) != instr_len:
            continue
        if not bool(result["interior_stop"]):
            continue
        if len(result["path_cells"]) < max(6, instr_len + 2):
            continue
        return grid, instructions, start, direction, result
    raise RuntimeError("failed to find a valid blank layout")


def _rect_touches(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    ar0, c0, ar1, c1 = rect_a
    br0, d0, br1, d1 = rect_b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or c1 + 1 < d0 or d1 + 1 < c0)


def _can_place_rect(
    rect: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
    existing: list[tuple[int, int, int, int]],
) -> bool:
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in forbidden:
                return False
    for other in existing:
        if _rect_touches(rect, other):
            return False
    return True


def _place_obstacles(
    rng: random.Random,
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    count: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = grid.shape
    if h < 7 or w < 7:
        return None

    for _ in range(OBSTACLE_LAYOUT_RETRIES):
        placed: list[tuple[int, int, int, int]] = []
        ok = True
        for _ in range(count):
            success = False
            for _ in range(RECT_RETRIES):
                rect_h = rng.choice([2, 2, 3, 3, 4])
                rect_w = rng.choice([2, 2, 3, 3, 4])
                max_r0 = h - rect_h - 2
                max_c0 = w - rect_w - 2
                if max_r0 < 3 or max_c0 < 1:
                    break
                r0 = rng.randint(3, max_r0)
                c0 = rng.randint(1, max_c0)
                rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                if not _can_place_rect(rect, forbidden, placed):
                    continue
                placed.append(rect)
                success = True
                break
            if not success:
                ok = False
                break
        if not ok:
            continue

        out = np.array(grid, copy=True)
        for r0, c0, r1, c1 in placed:
            out[r0 : r1 + 1, c0 : c1 + 1] = GREY
        return out, placed
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        desired_stop = rng.choice((STOP_NO_INSTRUCTIONS, STOP_BLOCKED_TURN))
        obstacle_count = rng.randint(2, 4)
        try:
            grid, instructions, start, direction, blank_result = _choose_blank_layout(rng, desired_stop)
        except RuntimeError:
            continue

        forbidden = set(blank_result["checked_cells"])
        forbidden.add(start)
        placed = _place_obstacles(rng, grid, forbidden, obstacle_count)
        if placed is None:
            continue
        candidate, rects = placed

        result = _simulate_from_state(candidate, instructions, start, direction)
        if result["stop_mode"] != desired_stop:
            continue
        if int(result["instructions_used"]) != len(instructions):
            continue
        if not bool(result["interior_stop"]):
            continue
        if len(rects) != obstacle_count:
            continue

        return candidate
    raise RuntimeError("generate_input: failed to produce a valid puzzle instance")
