from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TRACK_COLORS = {GREY, RED, BLUE, GREEN, YELLOW}
SIGNAL_COLORS = {RED, BLUE, GREEN, YELLOW}
MAX_GENERATION_ATTEMPTS = 400


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _track_neighbor_dirs(grid: np.ndarray, r: int, c: int) -> list[int]:
    dirs: list[int] = []
    for idx, (dr, dc) in enumerate(DIRS):
        rr, cc = r + dr, c + dc
        if _inside(grid.shape, rr, cc) and int(grid[rr, cc]) in TRACK_COLORS:
            dirs.append(idx)
    return dirs


def _turn(direction: int, signal: int) -> int:
    if signal == RED:
        return (direction + 1) % 4
    if signal == BLUE:
        return (direction - 1) % 4
    return direction


def _simulate_slider(grid: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    start_dirs = _track_neighbor_dirs(grid, r, c)
    if len(start_dirs) != 1:
        return start

    direction = start_dirs[0]
    seen: set[tuple[int, int, int]] = set()

    while True:
        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        if not _inside(grid.shape, nr, nc):
            return r, c

        color = int(grid[nr, nc])
        if color == BLACK or color == MAROON:
            return r, c

        r, c = nr, nc
        if color == ORANGE or color == YELLOW:
            return r, c
        if color in (RED, BLUE):
            direction = _turn(direction, color)
        elif color not in (GREY, GREEN):
            return r, c

        state = (r, c, direction)
        if state in seen:
            return r, c
        seen.add(state)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate maroon sliders moving through grey tracks and color-coded crossings.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    starts = [tuple(map(int, pt)) for pt in np.argwhere(g == MAROON)]
    out[out == MAROON] = BLACK

    for start in starts:
        rr, cc = _simulate_slider(g, start)
        out[rr, cc] = MAROON

    return out


def _signal_spec(signal: int) -> dict:
    start_dir = random.randrange(4)
    arm_lengths = [random.randint(1, 4) for _ in range(4)]

    incoming_arm = (start_dir + 2) % 4
    arm_lengths[incoming_arm] = random.randint(2, 5)

    if signal == GREEN:
        arm_lengths[start_dir] = random.randint(2, 5)
    elif signal in (RED, BLUE):
        exit_dir = _turn(start_dir, signal)
        arm_lengths[exit_dir] = random.randint(2, 5)

    for idx in range(4):
        if idx not in (incoming_arm, start_dir):
            arm_lengths[idx] = random.randint(1, 3)

    up, right, down, left = arm_lengths
    return {
        "signal": signal,
        "start_dir": start_dir,
        "incoming_arm": incoming_arm,
        "arm_lengths": arm_lengths,
        "height": up + down + 3,
        "width": left + right + 3,
    }


def _paint_cross(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    up, right, down, left_len = spec["arm_lengths"]
    center_r = top + up + 1
    center_c = left + left_len + 1
    grid[center_r, center_c] = int(spec["signal"])

    for direction, length in enumerate(spec["arm_lengths"]):
        dr, dc = DIRS[direction]
        for step in range(1, length + 1):
            rr = center_r + dr * step
            cc = center_c + dc * step
            grid[rr, cc] = GREY
        end_r = center_r + dr * (length + 1)
        end_c = center_c + dc * (length + 1)
        grid[end_r, end_c] = MAROON if direction == int(spec["incoming_arm"]) else ORANGE


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + height + 1)
    c1 = min(occupied.shape[1], left + width + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + height + 1)
    c1 = min(occupied.shape[1], left + width + 1)
    occupied[r0:r1, c0:c1] = True


def _is_valid_input(grid: np.ndarray) -> bool:
    starts = [tuple(map(int, pt)) for pt in np.argwhere(grid == MAROON)]
    if len(starts) < 4:
        return False
    if not SIGNAL_COLORS.issubset(set(map(int, np.unique(grid)))):
        return False
    for r, c in starts:
        if len(_track_neighbor_dirs(grid, r, c)) != 1:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if int(np.sum(out == MAROON)) != len(starts):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    extra_count = random.randint(0, 2)
    signals = [RED, BLUE, GREEN, YELLOW] + [random.choice([RED, BLUE, GREEN, YELLOW]) for _ in range(extra_count)]
    random.shuffle(signals)
    specs = [_signal_spec(signal) for signal in signals]

    min_side = 18 + 2 * len(specs)
    max_side = min_side + 8
    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)

    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int, dict]] = []

    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        placed = False
        for _ in range(200):
            top = random.randint(0, height - int(spec["height"]))
            left = random.randint(0, width - int(spec["width"]))
            if not _can_place(occupied, top, left, int(spec["height"]), int(spec["width"])):
                continue
            _mark_occupied(occupied, top, left, int(spec["height"]), int(spec["width"]))
            placements.append((top, left, spec))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    for top, left, spec in placements:
        _paint_cross(grid, top, left, spec)

    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
