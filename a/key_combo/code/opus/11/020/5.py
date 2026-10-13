from __future__ import annotations

from functools import lru_cache
from itertools import product
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIR_ORDER = ("up", "down", "left", "right")
DIRECTION_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
RANK_COLORS = (BLUE, RED, GREEN, YELLOW)
GRID_SIDES = (9, 11, 13)
MAX_GENERATION_ATTEMPTS = 400


def _ray_length(grid: np.ndarray, r: int, c: int, direction: str) -> int:
    dr, dc = DIRECTION_STEPS[direction]
    length = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
        if int(grid[nr, nc]) == GREY:
            break
        length += 1
        nr += dr
        nc += dc
    return length


def _ray_lengths(grid: np.ndarray, r: int, c: int) -> dict[str, int]:
    return {direction: _ray_length(grid, r, c, direction) for direction in DIR_ORDER}


def _direction_colors(lengths: dict[str, int]) -> dict[str, int]:
    ordered = sorted(DIR_ORDER, key=lambda direction: lengths[direction], reverse=True)
    assert len({lengths[direction] for direction in ordered}) == 4, "ray lengths must be pairwise distinct"
    return {direction: color for direction, color in zip(ordered, RANK_COLORS)}


def _quadrant_color(lengths: dict[str, int], direction_colors: dict[str, int], d1: str, d2: str) -> int:
    winner = d1 if lengths[d1] > lengths[d2] else d2
    return direction_colors[winner]


def _fill_black(region: np.ndarray, color: int) -> None:
    mask = region == BLACK
    region[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Color each quadrant around the orange marker according to the longer adjacent ray.

    Input structure:
    - The grid has a black background, exactly one orange marker pixel, and zero or more grey obstacle pixels.
    - The marker is away from the border; obstacles may appear on the marker row/column or elsewhere.
    - Looking from the marker, each of the four axis directions has a unique stopping distance to the nearest grey obstacle or to the grid boundary.

    Rule:
    - From the orange marker, measure the four ray lengths up, down, left, and right, stopping before the first grey obstacle or the boundary.
    - Rank those four lengths from longest to shortest and assign colors blue, red, green, and yellow respectively.
    - Split the grid by the marker row and marker column into four quadrants and fill every black cell in each quadrant with the color of the longer of its two adjacent rays, while leaving the marker, grey obstacles, and the full marker row and column unchanged.

    Key insight:
    - The output colors are not tied to fixed quadrants; they are induced by the relative ordering of the four ray lengths, then each quadrant inherits the color of whichever adjacent ray is longer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_coords = np.argwhere(g == ORANGE)
    assert marker_coords.shape == (1, 2), "expected exactly one orange marker"
    marker_r, marker_c = map(int, marker_coords[0])

    lengths = _ray_lengths(g, marker_r, marker_c)
    direction_colors = _direction_colors(lengths)

    out = g.copy()
    _fill_black(out[:marker_r, :marker_c], _quadrant_color(lengths, direction_colors, "up", "left"))
    _fill_black(out[:marker_r, marker_c + 1 :], _quadrant_color(lengths, direction_colors, "up", "right"))
    _fill_black(out[marker_r + 1 :, :marker_c], _quadrant_color(lengths, direction_colors, "down", "left"))
    _fill_black(out[marker_r + 1 :, marker_c + 1 :], _quadrant_color(lengths, direction_colors, "down", "right"))
    return out


@lru_cache(maxsize=None)
def _valid_length_tuples(spaces: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], ...]:
    candidates = []
    for lengths in product(*(range(space + 1) for space in spaces)):
        if len(set(lengths)) == 4:
            candidates.append(tuple(map(int, lengths)))
    return tuple(candidates)


def _choose_marker(shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = shape
    center = (rows // 2, cols // 2)
    while True:
        marker = (random.randint(1, rows - 2), random.randint(1, cols - 2))
        if marker != center:
            return marker


def _spaces_from_marker(shape: tuple[int, int], marker: tuple[int, int]) -> dict[str, int]:
    rows, cols = shape
    r, c = marker
    return {
        "up": r,
        "down": rows - 1 - r,
        "left": c,
        "right": cols - 1 - c,
    }


def _choose_lengths(spaces: dict[str, int]) -> dict[str, int]:
    space_tuple = tuple(spaces[direction] for direction in DIR_ORDER)
    candidates = list(_valid_length_tuples(space_tuple))
    if not candidates:
        raise ValueError("no valid distinct ray lengths for this marker position")

    desired_boundary_count = random.choices([0, 1, 2, 3, 4], weights=[1, 3, 4, 3, 1], k=1)[0]
    filtered = [
        lengths
        for lengths in candidates
        if sum(length == space for length, space in zip(lengths, space_tuple)) == desired_boundary_count
    ]
    if filtered:
        candidates = filtered

    if random.random() < 0.3:
        zero_candidates = [lengths for lengths in candidates if 0 in lengths]
        if zero_candidates:
            candidates = zero_candidates

    if random.random() < 0.3:
        interior_candidates = [
            lengths
            for lengths in candidates
            if any(0 < length < space for length, space in zip(lengths, space_tuple))
        ]
        if interior_candidates:
            candidates = interior_candidates

    return {direction: value for direction, value in zip(DIR_ORDER, random.choice(candidates))}


def _place_ray_obstacles(grid: np.ndarray, marker: tuple[int, int], lengths: dict[str, int], spaces: dict[str, int]) -> None:
    marker_r, marker_c = marker
    for direction in DIR_ORDER:
        ray_length = lengths[direction]
        max_length = spaces[direction]
        if ray_length >= max_length:
            continue

        dr, dc = DIRECTION_STEPS[direction]
        stop_distance = ray_length + 1
        stop_r = marker_r + dr * stop_distance
        stop_c = marker_c + dc * stop_distance
        grid[stop_r, stop_c] = GREY

        if stop_distance < max_length and random.random() < 0.35:
            extra_distance = random.randint(stop_distance + 1, max_length)
            extra_r = marker_r + dr * extra_distance
            extra_c = marker_c + dc * extra_distance
            grid[extra_r, extra_c] = GREY


def _quadrant_cells(grid: np.ndarray, marker: tuple[int, int]) -> list[tuple[int, int]]:
    marker_r, marker_c = marker
    cells = []
    for r in range(grid.shape[0]):
        if r == marker_r:
            continue
        for c in range(grid.shape[1]):
            if c == marker_c:
                continue
            if int(grid[r, c]) == BLACK:
                cells.append((r, c))
    return cells


def _scatter_off_axis_obstacles(grid: np.ndarray, marker: tuple[int, int]) -> None:
    marker_r, marker_c = marker
    quadrant_ranges = [
        (range(0, marker_r), range(0, marker_c)),
        (range(0, marker_r), range(marker_c + 1, grid.shape[1])),
        (range(marker_r + 1, grid.shape[0]), range(0, marker_c)),
        (range(marker_r + 1, grid.shape[0]), range(marker_c + 1, grid.shape[1])),
    ]

    added = 0
    for row_range, col_range in quadrant_ranges:
        available = [(r, c) for r in row_range for c in col_range if int(grid[r, c]) == BLACK]
        if not available:
            continue
        count = min(random.randint(0, 2), len(available))
        for r, c in random.sample(available, count):
            grid[r, c] = GREY
            added += 1

    if added >= 2:
        return

    available = _quadrant_cells(grid, marker)
    extra_needed = min(2 - added, len(available))
    for r, c in random.sample(available, extra_needed):
        grid[r, c] = GREY


def _generate_candidate() -> np.ndarray:
    shape = (random.choice(GRID_SIDES), random.choice(GRID_SIDES))
    marker = _choose_marker(shape)
    spaces = _spaces_from_marker(shape, marker)
    lengths = _choose_lengths(spaces)

    grid = np.full(shape, BLACK, dtype=int)
    grid[marker] = ORANGE
    _place_ray_obstacles(grid, marker, lengths, spaces)
    _scatter_off_axis_obstacles(grid, marker)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        marker_r, marker_c = map(int, np.argwhere(grid == ORANGE)[0])
        lengths = _ray_lengths(grid, marker_r, marker_c)
        if len(set(lengths.values())) != 4:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid ray-ranking quadrant puzzle input")
