from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9

CLOCK_SIZE = 7
INNER_START = 1
INNER_STOP = 6
CENTER = (3, 3)

DIRECTION_ORDER = ("up", "right", "down", "left")
DIRECTION_STEPS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
MIDPOINTS = {
    "up": (1, 3),
    "right": (3, 5),
    "down": (5, 3),
    "left": (3, 1),
}

TICK_COLORS = (1, 2, 3, 4, 6, 8)
GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 80


def _border_is_grey(patch: np.ndarray) -> bool:
    return (
        np.all(patch[0, :] == GREY)
        and np.all(patch[-1, :] == GREY)
        and np.all(patch[:, 0] == GREY)
        and np.all(patch[:, -1] == GREY)
    )


def _pointer_direction(patch: np.ndarray) -> str | None:
    found: list[str] = []
    cr, cc = CENTER
    for name in DIRECTION_ORDER:
        dr, dc = DIRECTION_STEPS[name]
        if int(patch[cr + dr, cc + dc]) == MAROON:
            found.append(name)
    if len(found) != 1:
        return None
    return found[0]


def _is_clock_patch(patch: np.ndarray) -> bool:
    if patch.shape != (CLOCK_SIZE, CLOCK_SIZE):
        return False
    if not _border_is_grey(patch):
        return False
    if int(patch[CENTER]) != ORANGE:
        return False

    tick_values = [int(patch[MIDPOINTS[name]]) for name in DIRECTION_ORDER]
    if any(color in {BLACK, GREY, ORANGE, MAROON} for color in tick_values):
        return False
    if len(set(tick_values)) != 4:
        return False

    direction = _pointer_direction(patch)
    if direction is None:
        return False

    allowed = {CENTER, *MIDPOINTS.values()}
    dr, dc = DIRECTION_STEPS[direction]
    allowed.add((CENTER[0] + dr, CENTER[1] + dc))

    for r in range(INNER_START, INNER_STOP):
        for c in range(INNER_START, INNER_STOP):
            if (r, c) in allowed:
                continue
            if int(patch[r, c]) != BLACK:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each clock-face interior with the color of the tick selected by its pointer.

    Input structure:
    - The background is black.
    - The grid contains 2-4 separate square clock faces, each with a 1-pixel grey border.
    - Inside each border is a 5x5 interior with four colored midpoint ticks, one orange center pivot, and a short maroon pointer.

    Rule:
    - Detect every 7x7 clock face by its grey border and interior layout.
    - Read the maroon pointer direction from the orange center toward one of the four midpoint ticks.
    - Recolor the full 5x5 interior of that clock face with the tick color in the pointed direction, while leaving the grey border and background unchanged.

    Key insight:
    - The pointer does not provide the output color directly; it only indexes which visible midpoint tick color should be expanded to the whole interior.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = g.shape

    for r0 in range(height - CLOCK_SIZE + 1):
        for c0 in range(width - CLOCK_SIZE + 1):
            patch = g[r0 : r0 + CLOCK_SIZE, c0 : c0 + CLOCK_SIZE]
            if not _is_clock_patch(patch):
                continue
            direction = _pointer_direction(patch)
            assert direction is not None
            fill_color = int(patch[MIDPOINTS[direction]])
            out[r0 + INNER_START : r0 + INNER_STOP, c0 + INNER_START : c0 + INNER_STOP] = fill_color

    return out


def _paint_clock(
    grid: np.ndarray,
    r0: int,
    c0: int,
    tick_map: dict[str, int],
    direction: str,
) -> None:
    grid[r0, c0 : c0 + CLOCK_SIZE] = GREY
    grid[r0 + CLOCK_SIZE - 1, c0 : c0 + CLOCK_SIZE] = GREY
    grid[r0 : r0 + CLOCK_SIZE, c0] = GREY
    grid[r0 : r0 + CLOCK_SIZE, c0 + CLOCK_SIZE - 1] = GREY

    for name, (rr, cc) in MIDPOINTS.items():
        grid[r0 + rr, c0 + cc] = tick_map[name]

    cr, cc = CENTER
    grid[r0 + cr, c0 + cc] = ORANGE
    dr, dc = DIRECTION_STEPS[direction]
    grid[r0 + cr + dr, c0 + cc + dc] = MAROON


def _sample_positions(
    height: int,
    width: int,
    n_clocks: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(height - CLOCK_SIZE + 1) for c in range(width - CLOCK_SIZE + 1)]

    for _ in range(PLACEMENT_ATTEMPTS):
        rng.shuffle(candidates)
        occupied = np.zeros((height, width), dtype=bool)
        positions: list[tuple[int, int]] = []
        for r0, c0 in candidates:
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + CLOCK_SIZE + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + CLOCK_SIZE + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            positions.append((r0, c0))
            if len(positions) == n_clocks:
                return positions
    return None


def _expected_output(
    grid: np.ndarray,
    clocks: list[tuple[int, int, str]],
    tick_map: dict[str, int],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, direction in clocks:
        out[r0 + INNER_START : r0 + INNER_STOP, c0 + INNER_START : c0 + INNER_STOP] = tick_map[direction]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_clocks = rng.choice((2, 3, 4))
    height = rng.randint(15, 30)
    width = rng.randint(15, 30)
    positions = _sample_positions(height, width, n_clocks, rng)
    if positions is None:
        return None

    tick_values = rng.sample(TICK_COLORS, 4)
    tick_map = dict(zip(DIRECTION_ORDER, tick_values))
    directions = rng.sample(DIRECTION_ORDER, n_clocks)
    rng.shuffle(positions)

    grid = np.full((height, width), BLACK, dtype=int)
    clocks: list[tuple[int, int, str]] = []
    for (r0, c0), direction in zip(positions, directions):
        _paint_clock(grid, r0, c0, tick_map, direction)
        clocks.append((r0, c0, direction))

    expected = _expected_output(grid, clocks, tick_map)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock-face input")
