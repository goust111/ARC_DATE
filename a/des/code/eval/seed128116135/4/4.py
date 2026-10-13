import math
import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
USABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
CLOCK_SIZE = 5
MIDPOINTS = [(0, 2), (2, 4), (4, 2), (2, 0)]  # top, right, bottom, left
QUADRANT_CELLS = [(1, 1), (1, 3), (3, 1), (3, 3)]
CROSS_CELLS = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
OUTER_RING = [
    (r, c)
    for r in range(CLOCK_SIZE)
    for c in range(CLOCK_SIZE)
    if r in (0, CLOCK_SIZE - 1) or c in (0, CLOCK_SIZE - 1)
]
NON_MID_BORDER = [pos for pos in OUTER_RING if pos not in MIDPOINTS]
MAX_RETRIES = 300


def _mode_color(values: list[int]) -> int:
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def _surrounding_is_background(grid: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    for r in range(max(0, top - 1), min(h, top + CLOCK_SIZE + 1)):
        for c in range(max(0, left - 1), min(w, left + CLOCK_SIZE + 1)):
            inside = top <= r < top + CLOCK_SIZE and left <= c < left + CLOCK_SIZE
            if not inside and int(grid[r, c]) != BG:
                return False
    return True


def _parse_clock(grid: np.ndarray, top: int, left: int) -> np.ndarray | None:
    h, w = grid.shape
    if top + CLOCK_SIZE > h or left + CLOCK_SIZE > w:
        return None
    if not _surrounding_is_background(grid, top, left):
        return None

    window = grid[top : top + CLOCK_SIZE, left : left + CLOCK_SIZE]

    for r, c in NON_MID_BORDER:
        if int(window[r, c]) != GREY:
            return None
    for r, c in CROSS_CELLS:
        if int(window[r, c]) != GREY:
            return None

    pointer_dirs = []
    for idx, (r, c) in enumerate(MIDPOINTS):
        value = int(window[r, c])
        if value == GREY:
            continue
        if value == BG:
            return None
        pointer_dirs.append(idx)
    if len(pointer_dirs) != 1:
        return None

    observed = np.zeros((2, 2), dtype=int)
    for idx, (r, c) in enumerate(QUADRANT_CELLS):
        value = int(window[r, c])
        if value in (BG, GREY):
            return None
        observed[idx // 2, idx % 2] = value

    correction = pointer_dirs[0]
    return np.rot90(observed, correction)


def _find_clocks(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    clocks = []
    for top in range(g.shape[0] - CLOCK_SIZE + 1):
        for left in range(g.shape[1] - CLOCK_SIZE + 1):
            parsed = _parse_clock(g, top, left)
            if parsed is not None:
                clocks.append(parsed)
    return clocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Correct each 5x5 clock's orientation from its border pointer, then output
    the per-quadrant majority colors across all clocks as a 2x2 summary block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    clocks = _find_clocks(g)
    assert clocks, "no valid clock found"

    out = np.zeros((2, 2), dtype=int)
    for r in range(2):
        for c in range(2):
            out[r, c] = _mode_color([int(clock[r, c]) for clock in clocks])
    return out


def _target_quadrants(rng: random.Random, palette: list[int]) -> np.ndarray:
    while True:
        values = [rng.choice(palette) for _ in range(4)]
        if len(set(values)) >= 2:
            return np.array(values, dtype=int).reshape(2, 2)


def _quadrant_assignments(
    rng: random.Random,
    num_clocks: int,
    target: int,
    palette: list[int],
) -> list[int]:
    majority = rng.randint(num_clocks // 2 + 1, num_clocks)
    if num_clocks > 1 and majority == num_clocks and rng.random() < 0.75:
        majority -= 1

    values = [target] * num_clocks
    indices = list(range(num_clocks))
    rng.shuffle(indices)
    alt_choices = [color for color in palette if color != target]
    for idx in indices[majority:]:
        values[idx] = rng.choice(alt_choices)
    return values


def _build_clock(corrected: np.ndarray, pointer_color: int, direction: int) -> np.ndarray:
    observed = np.rot90(corrected, -direction)
    clock = np.full((CLOCK_SIZE, CLOCK_SIZE), BG, dtype=int)

    for r, c in OUTER_RING:
        clock[r, c] = GREY
    for r, c in CROSS_CELLS:
        clock[r, c] = GREY
    for idx, (r, c) in enumerate(QUADRANT_CELLS):
        clock[r, c] = int(observed[idx // 2, idx % 2])

    pr, pc = MIDPOINTS[direction]
    clock[pr, pc] = pointer_color
    return clock


def _random_layout(rng: random.Random, num_clocks: int) -> tuple[int, int, list[tuple[int, int]]]:
    while True:
        rows = rng.randint(2, min(4, num_clocks))
        cols = rng.randint(2, 4)
        if rows * cols >= num_clocks:
            break

    row_sizes = [rng.randint(6, 9) for _ in range(rows)]
    col_sizes = [rng.randint(6, 9) for _ in range(cols)]
    margin_top = rng.randint(0, 3)
    margin_left = rng.randint(0, 3)
    margin_bottom = rng.randint(0, 3)
    margin_right = rng.randint(0, 3)

    y_starts = [margin_top]
    for size in row_sizes[:-1]:
        y_starts.append(y_starts[-1] + size)
    x_starts = [margin_left]
    for size in col_sizes[:-1]:
        x_starts.append(x_starts[-1] + size)

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    chosen_slots = slots[:num_clocks]

    positions = []
    for slot_r, slot_c in chosen_slots:
        dy = rng.randint(0, row_sizes[slot_r] - CLOCK_SIZE - 1)
        dx = rng.randint(0, col_sizes[slot_c] - CLOCK_SIZE - 1)
        positions.append((y_starts[slot_r] + dy, x_starts[slot_c] + dx))

    height = margin_top + sum(row_sizes) + margin_bottom
    width = margin_left + sum(col_sizes) + margin_right
    return height, width, positions


def _build_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    num_clocks = rng.randint(4, 8)
    pointer_color = rng.choice(USABLE_COLORS)
    palette = [color for color in USABLE_COLORS if color != pointer_color]
    target = _target_quadrants(rng, palette)

    quadrant_values = [
        _quadrant_assignments(rng, num_clocks, int(target[r, c]), palette)
        for r in range(2)
        for c in range(2)
    ]

    corrected_clocks = []
    for idx in range(num_clocks):
        corrected_clocks.append(
            np.array(
                [
                    [quadrant_values[0][idx], quadrant_values[1][idx]],
                    [quadrant_values[2][idx], quadrant_values[3][idx]],
                ],
                dtype=int,
            )
        )

    directions = [0, 1, 2, 3]
    while len(directions) < num_clocks:
        directions.append(rng.randint(0, 3))
    rng.shuffle(directions)

    height, width, positions = _random_layout(rng, num_clocks)
    grid = np.full((height, width), BG, dtype=int)

    for corrected, direction, (top, left) in zip(corrected_clocks, directions, positions):
        clock = _build_clock(corrected, pointer_color, direction)
        grid[top : top + CLOCK_SIZE, left : left + CLOCK_SIZE] = clock

    return grid, target


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid, expected = _build_instance(rng)
        if len(_find_clocks(grid)) == 0:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    raise RuntimeError(f"failed to generate input after {MAX_RETRIES} attempts")
