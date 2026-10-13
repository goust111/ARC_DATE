from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7
CYAN = 8

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # clockwise: up, right, down, left
TURN_RULE = {CYAN: 1, ORANGE: -1}  # +1 = right turn, -1 = left turn
REVERSED_TURN_RULE = {CYAN: -1, ORANGE: 1}

GRID_MIN, GRID_MAX = 12, 26
GARAGE_LEN_RANGE = (2, 4)
RED_LEN_RANGE = (2, 5)
TURN_COUNT_RANGE = (2, 5)
STRAIGHT_BEFORE_BEAD_RANGE = (1, 4)
STRAIGHT_TO_STOP_RANGE = (1, 4)
DISTRACTOR_COUNT_RANGE = (1, 4)
MAX_ATTEMPTS = 400

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _step(pos: Coord, direction: Coord, k: int = 1) -> Coord:
    return pos[0] + direction[0] * k, pos[1] + direction[1] * k


def _turn(direction: Coord, amount: int) -> Coord:
    idx = DIRS.index(direction)
    return DIRS[(idx + amount) % 4]


def _neighbors4(pos: Coord) -> Iterable[Coord]:
    r, c = pos
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _line_info(coords: list[Coord]) -> tuple[str, list[Coord]]:
    assert coords, "expected a non-empty line"
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1)), "green/red line must be contiguous"
        return "h", [(r, cols[0]), (r, cols[-1])]
    if len(cols) == 1:
        c = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1)), "green/red line must be contiguous"
        return "v", [(rows[0], c), (rows[-1], c)]
    raise AssertionError("expected a straight axis-aligned line")


def _garage_candidates(grid: np.ndarray) -> list[tuple[Coord, Coord]]:
    green_coords = [tuple(map(int, p)) for p in np.argwhere(grid == GREEN)]
    axis, endpoints = _line_info(green_coords)
    if axis == "h":
        left, right = endpoints
        return [(left, (0, -1)), (right, (0, 1))]
    top, bottom = endpoints
    return [(top, (-1, 0)), (bottom, (1, 0))]


def _simulate(grid: np.ndarray, start: Coord, direction: Coord, turn_rule: dict[int, int]) -> dict | None:
    current = start
    heading = direction
    consumed: list[Coord] = []
    consumed_set: set[Coord] = set()
    seen_states = {(current, heading)}
    max_steps = int(grid.shape[0] * grid.shape[1] * 4)

    for _ in range(max_steps):
        nxt = _step(current, heading)
        if not _in_bounds(grid.shape, *nxt):
            return None
        color = int(grid[nxt])
        if color == BLACK:
            current = nxt
        elif color in turn_rule:
            if nxt in consumed_set:
                return None
            current = nxt
            consumed.append(nxt)
            consumed_set.add(nxt)
            heading = _turn(heading, int(turn_rule[color]))
        elif color == RED:
            return {"start": start, "direction": direction, "consumed": tuple(consumed), "stop": nxt}
        else:
            return None

        state = (current, heading)
        if state in seen_states:
            return None
        seen_states.add(state)
    return None


def _successful_runs(grid: np.ndarray, turn_rule: dict[int, int]) -> list[dict]:
    return [
        run
        for start, direction in _garage_candidates(grid)
        if (run := _simulate(grid, start, direction, turn_rule)) is not None
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the invisible car route from the green garage and erase only the beads it actually consumes.

    Input structure:
    - The grid has a black background, one green straight line segment, and one red straight line segment.
    - Cyan and orange single-pixel beads are scattered on black cells; some lie on the car's hidden route and some are distractors.
    - The green segment is the garage: one endpoint is the car's front and its axis gives the initial travel direction.

    Rule:
    - Try the two outward directions implied by the green line's endpoints and keep the unique run that reaches the red line.
    - The car moves straight through black cells; when the next cell is cyan it enters that cell, consumes it, and turns right, and when the next cell is orange it consumes it and turns left.
    - When the next cell is red, the car stops on that red cell. Output the input with the whole green garage erased, every consumed bead erased, and the stopping red cell recolored green; untouched beads and the rest of the red line stay unchanged.

    Key insight:
    - The path itself never appears in either grid, so the only visible evidence of the route is which beads vanish and which red cell becomes the final green stop marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    runs = _successful_runs(g, TURN_RULE)
    assert len(runs) == 1, "expected exactly one valid garage endpoint"
    run = runs[0]

    out = g.copy()
    out[g == GREEN] = BLACK
    for bead in run["consumed"]:
        out[bead] = BLACK
    out[run["stop"]] = GREEN
    return out


def _bbox(points: Iterable[Coord]) -> tuple[int, int, int, int]:
    pts = list(points)
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _translate_points(points: Iterable[Coord], dr: int, dc: int) -> list[Coord]:
    return [(r + dr, c + dc) for r, c in points]


def _build_abstract_route(rng: random.Random) -> dict | None:
    garage_len = rng.randint(*GARAGE_LEN_RANGE)
    n_turns = rng.randint(*TURN_COUNT_RANGE)
    turns = [1, -1] + [rng.choice([1, -1]) for _ in range(n_turns - 2)]
    rng.shuffle(turns)

    start_dir = rng.choice(DIRS)
    garage = [_step((0, 0), start_dir, -k) for k in range(garage_len)]
    occupied = set(garage)
    current = (0, 0)
    heading = start_dir
    path_cells: list[Coord] = []
    consumed: list[tuple[Coord, int]] = []

    for turn_sign in turns:
        n_black = rng.randint(*STRAIGHT_BEFORE_BEAD_RANGE)
        for _ in range(n_black):
            current = _step(current, heading)
            if current in occupied:
                return None
            occupied.add(current)
            path_cells.append(current)

        bead = _step(current, heading)
        if bead in occupied:
            return None
        current = bead
        occupied.add(current)
        path_cells.append(current)
        consumed.append((current, CYAN if turn_sign == 1 else ORANGE))
        heading = _turn(heading, turn_sign)

    n_final = rng.randint(*STRAIGHT_TO_STOP_RANGE)
    for _ in range(n_final):
        current = _step(current, heading)
        if current in occupied:
            return None
        occupied.add(current)
        path_cells.append(current)

    pre_stop = current
    stop = _step(pre_stop, heading)
    if stop in occupied:
        return None

    if rng.random() < 0.5:
        red_axis = heading
        red_len = rng.randint(*RED_LEN_RANGE)
        red_line = [_step(stop, red_axis, k) for k in range(red_len)]
    else:
        red_axis = _turn(heading, rng.choice([1, -1]))
        red_len = rng.randint(*RED_LEN_RANGE)
        neg = rng.randint(0, red_len - 1)
        pos = red_len - 1 - neg
        red_line = [_step(stop, red_axis, k) for k in range(-neg, pos + 1)]

    if any(cell in occupied for cell in red_line):
        return None

    all_points = set(garage) | set(path_cells) | set(red_line)
    r0, c0, r1, c1 = _bbox(all_points)
    if max(r1 - r0 + 1, c1 - c0 + 1) > 24:
        return None

    return {
        "garage": garage,
        "path_cells": path_cells,
        "consumed": consumed,
        "stop": stop,
        "red_line": red_line,
        "all_points": all_points,
    }


def _embed_route(rng: random.Random, route: dict) -> dict | None:
    r0, c0, r1, c1 = _bbox(route["all_points"])
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1

    min_h = max(GRID_MIN, obj_h + 4)
    min_w = max(GRID_MIN, obj_w + 4)
    max_h = min(GRID_MAX, obj_h + 10)
    max_w = min(GRID_MAX, obj_w + 10)
    if min_h > max_h or min_w > max_w:
        return None

    H = rng.randint(min_h, max_h)
    W = rng.randint(min_w, max_w)

    dr_lo, dr_hi = 1 - r0, H - 2 - r1
    dc_lo, dc_hi = 1 - c0, W - 2 - c1
    if dr_lo > dr_hi or dc_lo > dc_hi:
        return None

    dr = rng.randint(dr_lo, dr_hi)
    dc = rng.randint(dc_lo, dc_hi)

    garage = _translate_points(route["garage"], dr, dc)
    path_cells = _translate_points(route["path_cells"], dr, dc)
    consumed = [((pos[0] + dr, pos[1] + dc), color) for pos, color in route["consumed"]]
    stop = (route["stop"][0] + dr, route["stop"][1] + dc)
    red_line = _translate_points(route["red_line"], dr, dc)

    return {
        "shape": (H, W),
        "garage": garage,
        "path_cells": path_cells,
        "consumed": consumed,
        "stop": stop,
        "red_line": red_line,
    }


def _candidate_distractors(shape: tuple[int, int], blocked: set[Coord], visible: set[Coord]) -> list[Coord]:
    H, W = shape
    out: list[Coord] = []
    for r in range(H):
        for c in range(W):
            p = (r, c)
            if p in blocked:
                continue
            if any(nb in visible for nb in _neighbors4(p)):
                continue
            out.append(p)
    return out


def _render_case(rng: random.Random, embedded: dict) -> np.ndarray | None:
    H, W = embedded["shape"]
    grid = np.full((H, W), BLACK, dtype=int)

    for r, c in embedded["garage"]:
        grid[r, c] = GREEN
    for r, c in embedded["red_line"]:
        grid[r, c] = RED
    for (r, c), color in embedded["consumed"]:
        grid[r, c] = color

    blocked = set(embedded["garage"]) | set(embedded["path_cells"]) | set(embedded["red_line"])
    visible = set(embedded["garage"]) | {p for p, _ in embedded["consumed"]} | set(embedded["red_line"])
    candidates = _candidate_distractors((H, W), blocked, visible)
    if not candidates:
        return None

    n_distractors = min(rng.randint(*DISTRACTOR_COUNT_RANGE), len(candidates))
    if n_distractors < 1:
        return None

    rng.shuffle(candidates)
    used: set[Coord] = set()
    distractors: list[tuple[Coord, int]] = []
    for p in candidates:
        if p in used:
            continue
        if any(nb in used for nb in _neighbors4(p)):
            continue
        color = rng.choice([CYAN, ORANGE])
        distractors.append((p, color))
        used.add(p)
        if len(distractors) == n_distractors:
            break
    if not distractors:
        return None

    for (r, c), color in distractors:
        grid[r, c] = color

    expected = grid.copy()
    for r, c in embedded["garage"]:
        expected[r, c] = BLACK
    for (r, c), _ in embedded["consumed"]:
        expected[r, c] = BLACK
    expected[embedded["stop"]] = GREEN

    runs = _successful_runs(grid, TURN_RULE)
    if len(runs) != 1:
        return None
    run = runs[0]
    expected_consumed = tuple(pos for pos, _ in embedded["consumed"])
    if tuple(run["consumed"]) != expected_consumed:
        return None
    if tuple(run["stop"]) != tuple(embedded["stop"]):
        return None
    if _successful_runs(grid, REVERSED_TURN_RULE):
        return None
    if len(expected_consumed) < 2 or {grid[p] for p in expected_consumed} != {CYAN, ORANGE}:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    route = _build_abstract_route(rng)
    if route is None:
        return None
    embedded = _embed_route(rng, route)
    if embedded is None:
        return None
    return _render_case(rng, embedded)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid ghost-car input after {MAX_ATTEMPTS} attempts")
