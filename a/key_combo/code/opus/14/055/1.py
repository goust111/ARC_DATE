from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SIDE_DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRY_LIMIT = 400


def _solid_color(cells: np.ndarray) -> int:
    values = np.unique(cells)
    assert len(values) == 1
    return int(values[0])


def _frame_direction_map(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    top = _solid_color(grid[0, 1:-1])
    bottom = _solid_color(grid[-1, 1:-1])
    left = _solid_color(grid[1:-1, 0])
    right = _solid_color(grid[1:-1, -1])
    return {
        top: SIDE_DIRECTIONS["top"],
        bottom: SIDE_DIRECTIONS["bottom"],
        left: SIDE_DIRECTIONS["left"],
        right: SIDE_DIRECTIONS["right"],
    }


def _distance_to_target_side(r: int, c: int, dr: int, dc: int, height: int, width: int) -> int:
    if dr == -1:
        return r
    if dr == 1:
        return height - 1 - r
    if dc == -1:
        return c
    return width - 1 - c


def _inside_interior(r: int, c: int, height: int, width: int) -> bool:
    return 1 <= r < height - 1 and 1 <= c < width - 1


def _slide_pixels(
    grid: np.ndarray,
    *,
    tie_mode: str = "col_row",
    blockers_include_pixels: bool = True,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, int | str | tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    directions = _frame_direction_map(g)

    moving: list[tuple[int, int, int, int, int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            color = int(g[r, c])
            if color not in directions:
                continue
            dr, dc = directions[color]
            distance = _distance_to_target_side(r, c, dr, dc, height, width)
            moving.append((r, c, color, dr, dc, distance))

    if tie_mode == "col_row":
        moving.sort(key=lambda item: (item[5], item[1], item[0]))
    elif tie_mode == "row_col":
        moving.sort(key=lambda item: (item[5], item[0], item[1]))
    else:
        raise ValueError(f"unknown tie mode: {tie_mode}")

    out = g.copy()
    for r, c, _color, _dr, _dc, _distance in moving:
        out[r, c] = BLACK

    details: list[dict[str, int | str | tuple[int, int]]] = []
    for r, c, color, dr, dc, distance in moving:
        cur_r, cur_c = r, c
        while True:
            next_r = cur_r + dr
            next_c = cur_c + dc
            if not _inside_interior(next_r, next_c, height, width):
                stop_reason = "border"
                break
            next_value = int(out[next_r, next_c])
            if next_value == BLACK:
                cur_r, cur_c = next_r, next_c
                continue
            if next_value == GREY:
                stop_reason = "obstacle"
                break
            if blockers_include_pixels:
                stop_reason = "pixel"
                break
            cur_r, cur_c = next_r, next_c
        out[cur_r, cur_c] = color
        details.append(
            {
                "source": (r, c),
                "target": (cur_r, cur_c),
                "color": color,
                "distance": distance,
                "stop": stop_reason,
            }
        )

    if return_details:
        return out, details
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each interior pixel all the way toward the frame side of the same color, with grey blockers and ordered stacking.

    Input structure:
    - The grid has a one-cell-thick outer frame whose corners are black and whose four sides are solid colors: top, bottom, left, and right.
    - Inside the frame are isolated single-pixel movers whose colors match one of the frame sides, plus 2-3 grey single-pixel obstacles on a black background.

    Rule:
    - Grey obstacle pixels never move.
    - Each non-black, non-grey interior pixel moves straight toward the frame side of the same color.
    - Process movers by increasing Manhattan distance to their own target side; break ties by position from left to right, then from top to bottom.
    - A mover keeps sliding until the next step would hit the frame, a grey obstacle, or a previously settled mover, then stops in the previous cell.
    - The outer frame remains unchanged.

    Key insight:
    - The side colors define four simultaneous gravity directions, while obstacle blocking and the exact processing order together determine the final stacks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert 5 <= height <= 30 and 5 <= width <= 30
    assert {int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])} == {BLACK}
    return _slide_pixels(g, tie_mode="col_row", blockers_include_pixels=True, return_details=False)


def _paint_frame(height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, 1:-1] = RED
    grid[-1, 1:-1] = BLUE
    grid[1:-1, 0] = YELLOW
    grid[1:-1, -1] = GREEN
    return grid


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, *, pad: int = 1) -> None:
    height, width = reserved.shape
    rr0 = max(1, r0 - pad)
    cc0 = max(1, c0 - pad)
    rr1 = min(height - 2, r1 + pad)
    cc1 = min(width - 2, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _chebyshev_far_enough(cells: Iterable[tuple[int, int]], candidate: tuple[int, int], gap: int = 1) -> bool:
    r, c = candidate
    for rr, cc in cells:
        if max(abs(r - rr), abs(c - cc)) <= gap:
            return False
    return True


def _can_place_single(
    candidate: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: np.ndarray,
    height: int,
    width: int,
) -> bool:
    r, c = candidate
    if not _inside_interior(r, c, height, width):
        return False
    if reserved[r, c]:
        return False
    if candidate in occupied:
        return False
    return _chebyshev_far_enough(occupied, candidate, gap=1)


def _try_place_optional_blocker(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: np.ndarray,
    rng: random.Random,
) -> bool:
    height, width = grid.shape
    options = [
        (RED, (2, 0)),
        (BLUE, (-2, 0)),
        (YELLOW, (0, 2)),
        (GREEN, (0, -2)),
    ]
    rng.shuffle(options)
    for color, source_delta in options:
        for _ in range(30):
            obstacle = (rng.randint(2, height - 3), rng.randint(2, width - 3))
            source = (obstacle[0] + source_delta[0], obstacle[1] + source_delta[1])
            if not _can_place_single(obstacle, occupied, reserved, height, width):
                continue
            if not _can_place_single(source, occupied, reserved, height, width):
                continue
            grid[obstacle] = GREY
            grid[source] = color
            occupied.add(obstacle)
            occupied.add(source)
            _reserve_box(
                reserved,
                min(obstacle[0], source[0]),
                min(obstacle[1], source[1]),
                max(obstacle[0], source[0]),
                max(obstacle[1], source[1]),
                pad=1,
            )
            return True
    return False


def _fill_random_pixels(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: np.ndarray,
    rng: random.Random,
    target_count: int,
) -> None:
    height, width = grid.shape
    palette = [RED, BLUE, YELLOW, GREEN]

    current = int(np.count_nonzero((grid[1:-1, 1:-1] != BLACK) & (grid[1:-1, 1:-1] != GREY)))
    attempts = 0
    while current < target_count and attempts < 300:
        attempts += 1
        candidate = (rng.randint(1, height - 2), rng.randint(1, width - 2))
        if not _can_place_single(candidate, occupied, reserved, height, width):
            continue
        color = rng.choice(palette)
        grid[candidate] = color
        occupied.add(candidate)
        current += 1


def _frame_ok(grid: np.ndarray) -> bool:
    return (
        np.all(grid[0, 1:-1] == RED)
        and np.all(grid[-1, 1:-1] == BLUE)
        and np.all(grid[1:-1, 0] == YELLOW)
        and np.all(grid[1:-1, -1] == GREEN)
        and {int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])} == {BLACK}
    )


def _semantic_checks(grid: np.ndarray) -> bool:
    if not _frame_ok(grid):
        return False

    interior = grid[1:-1, 1:-1]
    moving_count = int(np.count_nonzero((interior != BLACK) & (interior != GREY)))
    obstacle_count = int(np.count_nonzero(interior == GREY))
    if moving_count < 6 or moving_count > 10:
        return False
    if obstacle_count not in {2, 3}:
        return False
    colors = set(map(int, np.unique(interior))) - {BLACK, GREY}
    if colors != {RED, BLUE, GREEN, YELLOW}:
        return False

    out, details = _slide_pixels(grid, tie_mode="col_row", blockers_include_pixels=True, return_details=True)
    if not np.array_equal(out, main(grid)):
        return False
    if not any(detail["stop"] == "obstacle" for detail in details):
        return False
    if not any(detail["stop"] == "pixel" for detail in details):
        return False

    wrong_tie = _slide_pixels(grid, tie_mode="row_col", blockers_include_pixels=True, return_details=False)
    if np.array_equal(out, wrong_tie):
        return False

    no_pixel_block = _slide_pixels(grid, tie_mode="col_row", blockers_include_pixels=False, return_details=False)
    if np.array_equal(out, no_pixel_block):
        return False

    no_obstacles = grid.copy()
    no_obstacles[no_obstacles == GREY] = BLACK
    no_obstacle_out = _slide_pixels(no_obstacles, tie_mode="col_row", blockers_include_pixels=True, return_details=False)
    if np.array_equal(out, no_obstacle_out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(12, 15)
    width = rng.randint(12, 15)
    grid = _paint_frame(height, width)
    reserved = np.zeros((height, width), dtype=bool)
    occupied: set[tuple[int, int]] = set()

    k_max = min(4, height // 2 - 2, width // 2 - 2)
    if k_max < 2:
        return None
    k = rng.randint(2, k_max)

    tie_obstacle = (k, k)
    tie_red = (k + 2, k)
    tie_yellow = (k + 1, k + 2)
    for cell, color in ((tie_obstacle, GREY), (tie_red, RED), (tie_yellow, YELLOW)):
        grid[cell] = color
        occupied.add(cell)
    _reserve_box(reserved, 1, 1, k + 2, k + 2, pad=1)

    blue_col_min = max(width // 2, k + 4)
    blue_col_max = width - 6
    if blue_col_min > blue_col_max:
        return None
    blue_col = rng.randint(blue_col_min, blue_col_max)
    blue_obstacle_row_min = max(7, k + 6)
    blue_obstacle_row_max = height - 3
    if blue_obstacle_row_min > blue_obstacle_row_max:
        return None
    blue_obstacle = (rng.randint(blue_obstacle_row_min, blue_obstacle_row_max), blue_col)
    blue_low = (blue_obstacle[0] - 2, blue_col)
    blue_high = (blue_obstacle[0] - 4, blue_col)
    for cell, color in ((blue_obstacle, GREY), (blue_low, BLUE), (blue_high, BLUE)):
        grid[cell] = color
        occupied.add(cell)
    _reserve_box(reserved, blue_high[0], blue_col - 1, blue_obstacle[0], blue_col + 1, pad=1)

    green_row_choices = [
        r
        for r in range(2, height - 2)
        if r not in {k, k + 1, k + 2, blue_high[0], blue_low[0], blue_obstacle[0]}
    ]
    if not green_row_choices:
        return None
    green_row = rng.choice(green_row_choices)
    green_near = width - 4
    green_far = green_near - rng.choice([2, 3])
    if green_far <= 1:
        return None
    green_cells = [(green_row, green_far), (green_row, green_near)]
    for cell in green_cells:
        if cell in occupied:
            return None
    for cell in green_cells:
        grid[cell] = GREEN
        occupied.add(cell)
    _reserve_box(reserved, green_row - 1, green_far, green_row + 1, width - 2, pad=1)

    obstacle_total = 2
    if rng.random() < 0.6:
        if _try_place_optional_blocker(grid, occupied, reserved, rng):
            obstacle_total += 1

    target_pixels = rng.randint(6 + (obstacle_total - 2), 10)
    _fill_random_pixels(grid, occupied, reserved, rng, target_pixels)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
