from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIR_ORDER = (UP, RIGHT, DOWN, LEFT)

# Canonical 3-pixel arrow with the tip at the origin.
ARROW_OFFSETS = {
    UP: ((0, 0), (1, 0), (1, 1)),
    RIGHT: ((0, 0), (0, -1), (1, -1)),
    DOWN: ((0, 0), (-1, 0), (-1, -1)),
    LEFT: ((0, 0), (0, 1), (-1, 1)),
}

PATTERN_TO_DIRECTION = {
    frozenset({(0, 0), (1, 0), (1, 1)}): UP,
    frozenset({(0, 0), (0, 1), (1, 0)}): RIGHT,
    frozenset({(0, 0), (0, 1), (1, 1)}): DOWN,
    frozenset({(0, 1), (1, 0), (1, 1)}): LEFT,
}

GEN_ATTEMPTS = 400
ARROW_PLACEMENT_ATTEMPTS = 240
GRID_SIDE_RANGE = (18, 26)
ARROW_COUNT_RANGE = (2, 4)
MAX_PRISM_SEGMENT = 8


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _scale(d: tuple[int, int], k: int) -> tuple[int, int]:
    return d[0] * k, d[1] * k


def _rot_cw(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return dc, -dr


def _rot_ccw(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return -dc, dr


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = tip
    return [(tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS[direction]]


def _steps_to_boundary(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    steps = 0
    while 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]:
        r += dr
        c += dc
        steps += 1
    return steps


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], dist: int) -> list[tuple[int, int]]:
    return [_add(start, _scale(direction, step)) for step in range(1, dist + 1)]


def _turn_kind_for_prism(cell: tuple[int, int]) -> str:
    r, c = cell
    return "cw" if (r % 2) == (c % 2) else "ccw"


def _decode_arrow(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    assert coords.shape == (3, 2)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert r1 == r0 + 1 and c1 == c0 + 1, "arrow must occupy a 2x2 bbox"

    pattern = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    direction = PATTERN_TO_DIRECTION.get(pattern)
    assert direction is not None, "foreground objects must be canonical 3-pixel arrows"

    offsets = ARROW_OFFSETS[direction]
    tip_rel = offsets[0]
    tip = (r0 + tip_rel[0], c0 + tip_rel[1])
    return tip, direction


def _trace_ray(
    grid: np.ndarray,
    out: np.ndarray,
    color: int,
    start_tip: tuple[int, int],
    start_dir: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    shape = grid.shape
    pos = start_tip
    direction = start_dir
    last_painted: tuple[int, int] | None = None
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (pos[0], pos[1], direction[0], direction[1])
        assert state not in seen, "ray path must not cycle"
        seen.add(state)

        nxt = _add(pos, direction)
        if not _in_bounds(shape, nxt):
            break

        value = int(grid[nxt])
        if value == BG:
            assert int(out[nxt]) in (BG, color)
            out[nxt] = color
            pos = nxt
            last_painted = nxt
            continue

        if value == GREY:
            pos = nxt
            direction = _rot_cw(direction) if _turn_kind_for_prism(nxt) == "cw" else _rot_ccw(direction)
            continue

        break

    assert last_painted is not None, "each arrow must paint at least one ray cell"
    return last_painted, direction


def _draw_arrow_copy(out: np.ndarray, color: int, tip: tuple[int, int], direction: tuple[int, int]) -> None:
    shape = out.shape
    for cell in _arrow_cells(tip, direction):
        assert _in_bounds(shape, cell), "copied arrow must fit inside the grid"
        current = int(out[cell])
        assert current in (BG, color), "copied arrow must not overwrite other objects"
        out[cell] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Fire each colored L-arrow through grey prisms and stamp a matching arrow copy at the ray endpoint.

    Input structure:
    - The background is black.
    - Several distinct non-grey colors each form exactly one 3-pixel L-shaped arrow inside a 2x2 box.
    - Grey singleton pixels act as prisms; all other non-background pixels belong to arrows.

    Rule:
    - Decode every arrow's pointing direction from its canonical 3-pixel L orientation.
    - Starting from the arrow tip, advance one cell at a time in that direction: paint black cells with the arrow color, but keep grey prism cells grey.
    - When the ray reaches a grey prism, turn 90 degrees clockwise if the prism row and column have the same parity, otherwise turn 90 degrees counterclockwise.
    - The ray stops before leaving the grid or before entering any non-black, non-grey cell; then place a same-color arrow copy whose tip is the last painted ray cell and whose direction is the ray's final direction.

    Key insight:
    - The arrow sprite simultaneously encodes the launch direction and the endpoint copy template, while the hidden turn rule depends on each prism's row/column parity rather than on its color alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, GREY)]
    arrows: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in sorted(colors):
        coords = np.argwhere(g == color)
        assert coords.shape == (3, 2), "each arrow color must occupy exactly three pixels"
        tip, direction = _decode_arrow(coords)
        arrows.append((color, tip, direction))

    out = g.copy()
    for color, tip, direction in arrows:
        end_tip, end_direction = _trace_ray(g, out, color, tip, direction)
        _draw_arrow_copy(out, color, end_tip, end_direction)

    return out


def _extend_route(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    turns: list[str],
    turn_idx: int,
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    if turn_idx == len(turns):
        end_dist = _steps_to_boundary(shape, pos, direction)
        if end_dist < 2:
            return None
        black_cells = _segment_cells(pos, direction, end_dist)
        if any(cell in blocked for cell in black_cells):
            return None

        final_tip = black_cells[-1]
        copy_cells = _arrow_cells(final_tip, direction)
        if not all(_in_bounds(shape, cell) for cell in copy_cells):
            return None
        if any(cell in blocked for cell in copy_cells if cell not in black_cells):
            return None

        return {
            "path_cells": black_cells,
            "prisms": [],
            "final_tip": final_tip,
            "final_direction": direction,
            "copy_cells": copy_cells,
        }

    max_dist = min(_steps_to_boundary(shape, pos, direction), MAX_PRISM_SEGMENT)
    turn_kind = turns[turn_idx]
    dists = [
        dist
        for dist in range(2, max_dist + 1)
        if _turn_kind_for_prism(_add(pos, _scale(direction, dist))) == turn_kind
    ]
    random.shuffle(dists)

    for dist in dists:
        prism = _add(pos, _scale(direction, dist))
        black_cells = _segment_cells(pos, direction, dist - 1)
        if any(cell in blocked for cell in black_cells):
            continue
        if prism in blocked:
            continue

        next_dir = _rot_cw(direction) if turn_kind == "cw" else _rot_ccw(direction)
        tail = _extend_route(shape, prism, next_dir, turns, turn_idx + 1, blocked | set(black_cells) | {prism})
        if tail is None:
            continue

        return {
            "path_cells": black_cells + tail["path_cells"],
            "prisms": [prism] + tail["prisms"],
            "final_tip": tail["final_tip"],
            "final_direction": tail["final_direction"],
            "copy_cells": tail["copy_cells"],
        }

    return None


def _random_turn_lists(count: int) -> list[list[str]]:
    plans: list[list[str]] = [["cw"], ["ccw"]]
    options = [[], ["cw"], ["ccw"], ["cw", "ccw"], ["ccw", "cw"]]
    while len(plans) < count:
        plans.append(random.choice(options))
    random.shuffle(plans)
    return plans


def _try_place_arrow(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    turns: list[str],
) -> dict[str, object] | None:
    for _ in range(ARROW_PLACEMENT_ATTEMPTS):
        direction = random.choice(DIR_ORDER)
        tip = (random.randrange(shape[0]), random.randrange(shape[1]))
        arrow_cells = _arrow_cells(tip, direction)
        if not all(_in_bounds(shape, cell) for cell in arrow_cells):
            continue
        if any(cell in reserved for cell in arrow_cells):
            continue

        route = _extend_route(shape, tip, direction, turns, 0, reserved | set(arrow_cells))
        if route is None:
            continue

        return {
            "tip": tip,
            "direction": direction,
            "arrow_cells": arrow_cells,
            "path_cells": route["path_cells"],
            "prisms": route["prisms"],
            "copy_cells": route["copy_cells"],
            "final_tip": route["final_tip"],
            "final_direction": route["final_direction"],
            "turns": turns,
        }
    return None


def _build_case() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    shape = (h, w)

    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, arrow_count)
    plans = _random_turn_lists(arrow_count)

    placements: list[dict[str, object]] = []
    reserved: set[tuple[int, int]] = set()
    for turns in plans:
        placement = _try_place_arrow(shape, reserved, turns)
        if placement is None:
            return None
        placements.append(placement)
        reserved |= set(placement["arrow_cells"])
        reserved |= set(placement["path_cells"])
        reserved |= set(placement["prisms"])
        reserved |= set(placement["copy_cells"])

    grid = np.full(shape, BG, dtype=int)
    for placement, color in zip(placements, colors):
        for cell in placement["prisms"]:
            assert int(grid[cell]) == BG
            grid[cell] = GREY
        for cell in placement["arrow_cells"]:
            assert int(grid[cell]) == BG
            grid[cell] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(grid == GREY)) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-arrow ARC input")
