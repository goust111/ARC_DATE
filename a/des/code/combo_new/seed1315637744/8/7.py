import math
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

SATELLITE_COLORS = (GREEN, YELLOW, MAGENTA, ORANGE)
SATELLITE_OFFSETS = [(-2, 0), (0, 2), (2, 0), (0, -2)]  # top, right, bottom, left
POINTER_OFFSETS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}

PATCH_SIZE = 5
BOX_GAP = 1
COUNT_RANGE = (4, 6)
MAX_SIDE = 30
GEN_RETRIES = 300


def _rotate_satellites(values: list[int], steps: int) -> list[int]:
    return [int(values[(i - steps) % 4]) for i in range(4)]


def _clock_steps(grid: np.ndarray, r: int, c: int) -> int | None:
    if int(grid[r, c]) != AZURE:
        return None

    pointer_steps = None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            val = int(grid[r + dr, c + dc])
            if (dr, dc) in POINTER_OFFSETS.values():
                if val == RED:
                    if pointer_steps is not None:
                        return None
                    for steps, offset in POINTER_OFFSETS.items():
                        if offset == (dr, dc):
                            pointer_steps = steps
                            break
                elif val != GREY:
                    return None
            elif val != GREY:
                return None

    if pointer_steps is None:
        return None

    satellites = [int(grid[r + dr, c + dc]) for dr, dc in SATELLITE_OFFSETS]
    if len(set(satellites)) != 4:
        return None
    if any(color in (BG, RED, GREY, AZURE) for color in satellites):
        return None
    return int(pointer_steps)


def _find_clocks(grid: np.ndarray) -> list[tuple[int, int, int]]:
    rows, cols = map(int, grid.shape)
    clocks: list[tuple[int, int, int]] = []
    for r in range(2, rows - 2):
        for c in range(2, cols - 2):
            steps = _clock_steps(grid, r, c)
            if steps is not None:
                clocks.append((r, c, steps))
    return clocks


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each clock's four satellite pixels by the amount encoded in its pointer direction.

    Input structure:
    - The background is black.
    - Each clock is a 3x3 grey face whose center is azure and whose single red orthogonal neighbor is the pointer.
    - Four distinct satellite pixels sit immediately outside that 3x3 face at the top, right, bottom, and left positions.

    Rule:
    - Read the red pointer direction on each clock independently.
    - Up means rotate the four satellite colors by 0 steps, right by 1 clockwise step, down by 2 steps, and left by 1 counterclockwise step.
    - Keep the 3x3 clock face unchanged and only rewrite the four satellite positions.

    Key insight:
    - The pointer is not a movement arrow for a single pixel; it encodes how many quarter-turns to apply to the four-pixel satellite cycle around that clock.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    clocks = _find_clocks(g)
    assert clocks, "expected at least one valid clock"

    for r, c, steps in clocks:
        old_values = [int(g[r + dr, c + dc]) for dr, dc in SATELLITE_OFFSETS]
        new_values = _rotate_satellites(old_values, steps)
        for (dr, dc), value in zip(SATELLITE_OFFSETS, new_values):
            out[r + dr, c + dc] = int(value)

    return out


def _build_clock_patch(steps: int, satellites: list[int]) -> np.ndarray:
    patch = np.full((PATCH_SIZE, PATCH_SIZE), BG, dtype=int)
    patch[1:4, 1:4] = GREY
    patch[2, 2] = AZURE
    dr, dc = POINTER_OFFSETS[int(steps)]
    patch[2 + dr, 2 + dc] = RED
    for (off_r, off_c), color in zip(SATELLITE_OFFSETS, satellites):
        patch[2 + off_r, 2 + off_c] = int(color)
    return patch


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = BOX_GAP,
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _sample_layout(count: int) -> tuple[int, int]:
    layouts = []
    for across in range(1, count + 1):
        down = math.ceil(count / across)
        rows_min = PATCH_SIZE * down + BOX_GAP * (down - 1)
        cols_min = PATCH_SIZE * across + BOX_GAP * (across - 1)
        if rows_min <= MAX_SIDE and cols_min <= MAX_SIDE:
            layouts.append((rows_min, cols_min))
    rows_min, cols_min = random.choice(layouts)
    return random.randint(rows_min, MAX_SIDE), random.randint(cols_min, MAX_SIDE)


def _sample_anchors(rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    anchors = [(r, c) for r in range(rows - PATCH_SIZE + 1) for c in range(cols - PATCH_SIZE + 1)]
    random.shuffle(anchors)

    chosen: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for r, c in anchors:
        box = (r, c, r + PATCH_SIZE - 1, c + PATCH_SIZE - 1)
        if all(_boxes_separated(box, prev) for prev in boxes):
            chosen.append((r, c))
            boxes.append(box)
            if len(chosen) == count:
                return chosen
    return None


def _direction_list(count: int) -> list[int]:
    directions = [0, 1, 2, 3]
    directions.extend(random.randrange(4) for _ in range(count - 4))
    random.shuffle(directions)
    return directions


def _manual_expected(
    grid: np.ndarray,
    specs: list[tuple[int, int, int, list[int]]],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, steps, satellites in specs:
        rotated = _rotate_satellites(satellites, steps)
        patch = _build_clock_patch(steps, rotated)
        out[r0 : r0 + PATCH_SIZE, c0 : c0 + PATCH_SIZE] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*COUNT_RANGE)
    rows, cols = _sample_layout(count)
    anchors = _sample_anchors(rows, cols, count)
    if anchors is None:
        return None

    grid = np.full((rows, cols), BG, dtype=int)
    directions = _direction_list(count)
    specs: list[tuple[int, int, int, list[int]]] = []

    for (r0, c0), steps in zip(anchors, directions):
        satellites = random.sample(SATELLITE_COLORS, 4)
        patch = _build_clock_patch(steps, satellites)
        grid[r0 : r0 + PATCH_SIZE, c0 : c0 + PATCH_SIZE] = patch
        specs.append((r0, c0, steps, satellites))

    if set(directions) != {0, 1, 2, 3}:
        return None

    expected = _manual_expected(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock-rotation input")
