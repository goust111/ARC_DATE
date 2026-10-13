from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500


def _find_start(grid: np.ndarray) -> tuple[int, int, int]:
    interior = np.argwhere(grid[:, 1:-1] != BLACK)
    assert interior.shape[0] == 1, "expected exactly one interior start pixel"
    r, c = map(int, interior[0])
    c += 1
    return r, c, int(grid[r, c])


def _initial_horizontal_direction(grid: np.ndarray, start_color: int) -> int:
    left_has = bool(np.any(grid[:, 0] == start_color))
    right_has = bool(np.any(grid[:, -1] == start_color))
    assert left_has ^ right_has, "start color must appear on exactly one side wall"
    return -1 if left_has else 1


def _trace_geometry(
    height: int,
    width: int,
    start_r: int,
    start_c: int,
    dc: int,
    *,
    gap_bounce_index: int | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], str]:
    visited: list[tuple[int, int]] = []
    bounces: list[tuple[int, int, int]] = []
    r, c = int(start_r), int(start_c)
    step_dc = int(dc)

    while True:
        nr = r - 1
        if nr < 0:
            return visited, bounces, "top"

        nc = c + step_dc
        if nc == 0 or nc == width - 1:
            bounce_index = len(bounces)
            bounces.append((nr, nc, step_dc))
            if gap_bounce_index is not None and bounce_index == gap_bounce_index:
                return visited, bounces, "gap"
            step_dc = -step_dc
            nc = c + step_dc

        assert 1 <= nc <= width - 2, "path must stay inside the wall columns"
        visited.append((nr, nc))
        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a diagonally bouncing path whose starting side and repaint colors are controlled by the two colored side walls.

    Input structure:
    - The leftmost and rightmost columns are vertical side walls made of colored segments with occasional black gaps.
    - The interior is otherwise black except for one colored start pixel.
    - The start pixel's color appears on exactly one of the two side walls, and there is no top wall.

    Rule:
    - Launch from the start pixel diagonally upward toward the wall that contains the same color, with the current path color initialized to the start color.
    - Each upward step paints the visited interior cell with the current color.
    - If the next step would enter a side-wall cell, inspect that wall cell instead: a colored wall cell reflects the path horizontally and replaces the current path color with the wall color, while a black gap ends the trajectory immediately.

    Key insight:
    - The side walls simultaneously encode the initial direction, act as bounce surfaces, recolor the path at each collision, and decide whether the trajectory exits through a gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 4, "grid must have side walls and interior"

    start_r, start_c, current_color = _find_start(g)
    dc = _initial_horizontal_direction(g, current_color)

    out = g.copy()
    r, c = start_r, start_c
    while True:
        nr = r - 1
        if nr < 0:
            break

        nc = c + dc
        if nc == 0 or nc == width - 1:
            wall_color = int(g[nr, nc])
            if wall_color == BLACK:
                break
            current_color = wall_color
            dc = -dc
            nc = c + dc

        assert 1 <= nc <= width - 2, "reflected step must land inside the corridor"
        assert int(g[nr, nc]) == BLACK, "only empty interior cells may be painted"
        out[nr, nc] = current_color
        r, c = nr, nc

    return out


def _random_segmented_wall(height: int, palette: list[int]) -> np.ndarray:
    wall = np.empty(height, dtype=int)
    row = 0
    prev = None
    while row < height:
        run = random.randint(1, 4)
        choices = [color for color in palette if color != prev]
        color = random.choice(choices)
        wall[row : min(height, row + run)] = color
        prev = color
        row += run
    return wall


def _choose_gap_index(bounce_count: int) -> int:
    if bounce_count == 1:
        return 0
    if random.random() < 0.2:
        return 0
    return random.randint(1, bounce_count - 1)


def _choose_bounce_colors(start_color: int, count: int) -> list[int]:
    colors: list[int] = []
    prev = start_color
    for _ in range(count):
        choices = [color for color in COLORS if color != start_color and color != prev]
        if not choices:
            choices = [color for color in COLORS if color != start_color]
        color = random.choice(choices)
        colors.append(color)
        prev = color
    return colors


def _protected_wall_rows(
    all_bounces: list[tuple[int, int, int]],
    termination: str,
    gap_index: int | None,
) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    for i, (row, wall_col, _dc) in enumerate(all_bounces):
        protected.add((row, wall_col))
        if termination == "gap" and i == gap_index:
            break
    return protected


def _apply_extra_gaps(
    left_wall: np.ndarray,
    right_wall: np.ndarray,
    protected: set[tuple[int, int]],
    start_r: int,
    width: int,
) -> None:
    available: list[tuple[int, int]] = []
    height = left_wall.shape[0]
    for row in range(min(height, start_r + 1)):
        for wall_col in (0, width - 1):
            if (row, wall_col) not in protected:
                available.append((row, wall_col))

    gap_count = random.randint(1, min(2, len(available))) if available else 0
    for row, wall_col in random.sample(available, gap_count):
        if wall_col == 0:
            left_wall[row] = BLACK
        else:
            right_wall[row] = BLACK


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 24)
    width = random.randint(6, 12)

    start_r = random.randint(max(5, height // 2), height - 2)
    start_c = random.randint(1, width - 2)
    start_color = random.choice(COLORS)
    initial_side = random.choice(("left", "right"))
    dc = -1 if initial_side == "left" else 1

    visited_full, all_bounces, _ = _trace_geometry(height, width, start_r, start_c, dc)
    bounce_count = len(all_bounces)
    if bounce_count < 1 or bounce_count > 5 or len(visited_full) < 4:
        return None

    termination = random.choice(("top", "gap"))
    gap_index = None
    if termination == "gap":
        gap_index = _choose_gap_index(bounce_count)
        visited, used_bounces, end_mode = _trace_geometry(
            height,
            width,
            start_r,
            start_c,
            dc,
            gap_bounce_index=gap_index,
        )
        assert end_mode == "gap"
        colored_bounce_count = gap_index
    else:
        visited = visited_full
        used_bounces = all_bounces
        colored_bounce_count = bounce_count

    if len(visited) < 3:
        return None

    bounce_colors = _choose_bounce_colors(start_color, colored_bounce_count)
    palette = sorted(set(bounce_colors))
    target_palette_size = random.randint(3, 5)
    while len(palette) < target_palette_size:
        color = random.choice([c for c in COLORS if c != start_color])
        if color not in palette:
            palette.append(color)
    palette = sorted(palette)

    left_wall = _random_segmented_wall(height, palette)
    right_wall = _random_segmented_wall(height, palette)

    protected = _protected_wall_rows(all_bounces, termination, gap_index)

    indicator_rows = [row for row in range(start_r + 1, height) if (row, 0) not in protected and (row, width - 1) not in protected]
    if not indicator_rows:
        return None
    indicator_row = random.choice(indicator_rows)
    if initial_side == "left":
        left_wall[indicator_row] = start_color
        right_wall[right_wall == start_color] = random.choice(palette)
    else:
        right_wall[indicator_row] = start_color
        left_wall[left_wall == start_color] = random.choice(palette)

    for index, (row, wall_col, _old_dc) in enumerate(used_bounces):
        if termination == "gap" and index == gap_index:
            if wall_col == 0:
                left_wall[row] = BLACK
            else:
                right_wall[row] = BLACK
            break

        wall_color = bounce_colors[index]
        if wall_col == 0:
            left_wall[row] = wall_color
        else:
            right_wall[row] = wall_color

    _apply_extra_gaps(left_wall, right_wall, protected, start_r, width)

    if initial_side == "left" and np.any(right_wall == start_color):
        return None
    if initial_side == "right" and np.any(left_wall == start_color):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, 0] = left_wall
    grid[:, -1] = right_wall
    grid[start_r, start_c] = start_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    painted = int(np.count_nonzero((out != grid) & (grid == BLACK)))
    if painted < 3:
        return None

    if not (np.any(grid[:, 0] == BLACK) or np.any(grid[:, -1] == BLACK)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
