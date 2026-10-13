from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_vertical_rails(grid: np.ndarray) -> list[tuple[int, int, int]]:
    rails: list[tuple[int, int, int]] = []
    for comp in _components_of_color(grid, GREY):
        rows = [r for r, _ in comp]
        cols = {c for _, c in comp}
        if len(cols) != 1:
            continue
        col = next(iter(cols))
        r0 = min(rows)
        r1 = max(rows)
        if len(comp) != r1 - r0 + 1:
            continue
        rails.append((col, r0, r1))
    rails.sort()
    if len(rails) != 2:
        raise ValueError(f"expected exactly 2 vertical rails, found {len(rails)}")
    return rails


def _simulate_path(shape: tuple[int, int], start: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    h, w = shape
    r, c = start
    dr, dc = -1, 1
    reflections = 0
    path = [(r, c)]
    while True:
        nr = r + dr
        nc = c + dc
        if nc < 0 or nc >= w:
            reflections += 1
            dc = -dc
            nc = c + dc
        if nr < 0 or nr >= h:
            break
        r, c = nr, nc
        path.append((r, c))
    return path, reflections


def _border_color(reflections: int) -> int:
    if reflections == 2:
        return RED
    if reflections == 3:
        return GREEN
    if reflections == 4:
        return YELLOW
    return PINK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the region bounded by gray rails and the bouncing blue ray, then frame it by reflection count.

    Input structure:
    - A black background contains exactly one blue start pixel on the left edge.
    - Two gray vertical line segments act as left/right guide rails, with teal decorative pixels scattered between them.
    - The blue pixel seeds an upward-right 45-degree ray that reflects only on the grid's left/right borders.

    Rule:
    - Detect the two gray vertical rails and sort them by column.
    - Simulate the blue ray from the start pixel, reversing horizontal direction whenever it hits a left/right border and stopping when it exits through the top or bottom.
    - Record the ray's minimum/maximum row and the number of side-wall reflections, crop rows to that ray span and columns to the rail columns, then add a 1-cell border colored by the reflection count.

    Key insight:
    - The crop's columns come from static rail detection, while the crop's rows and the final frame color come from dynamic ray simulation.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D grid")

    starts = np.argwhere(g == BLUE)
    if starts.shape[0] != 1:
        raise ValueError(f"expected exactly 1 blue start pixel, found {starts.shape[0]}")
    start = tuple(map(int, starts[0]))

    rails = _find_vertical_rails(g)
    c_left, _, _ = rails[0]
    c_right, _, _ = rails[1]

    path, reflections = _simulate_path(g.shape, start)
    rows = [r for r, _ in path]
    r_min = min(rows)
    r_max = max(rows)

    painted = g.copy()
    for r, c in path:
        painted[r, c] = BLUE

    crop = painted[r_min : r_max + 1, c_left : c_right + 1]
    frame = np.full((crop.shape[0] + 2, crop.shape[1] + 2), _border_color(reflections), dtype=int)
    frame[1:-1, 1:-1] = crop
    return frame


def _choose_width_and_start(target_reflections: int) -> tuple[int, int]:
    max_width = min(12, 1 + (26 // target_reflections))
    widths = list(range(5, max_width + 1))
    random.shuffle(widths)
    for width in widths:
        step = width - 1
        low = target_reflections * step + 1
        high = min((target_reflections + 1) * step, 27)
        if low <= high:
            return width, random.randint(low, high)
    raise RuntimeError("no valid width/start pair for reflection target")


def _sample_rail_spans(height: int, start_row: int) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(400):
        len_left = random.randint(8, 12)
        len_right = random.randint(8, 12)
        left_start_min = max(0, start_row - len_left + 1)
        right_start_min = max(0, start_row - len_right + 1)
        left_start = random.randint(left_start_min, start_row)
        right_start = random.randint(right_start_min, start_row)
        left_end = left_start + len_left - 1
        right_end = right_start + len_right - 1
        if left_end >= height or right_end >= height:
            continue
        overlap_top = max(left_start, right_start)
        overlap_bottom = min(left_end, right_end)
        overlap_len = overlap_bottom - overlap_top + 1
        if overlap_len <= 0:
            continue
        if overlap_len >= min(len_left, len_right):
            continue
        if not (overlap_top <= start_row <= overlap_bottom):
            continue
        return (left_start, left_end), (right_start, right_end)
    raise RuntimeError("failed to sample rail spans")


def _rail_columns(width: int) -> tuple[int, int]:
    c_left_choices = [c for c in range(1, max(2, width // 2)) if c < width - 3]
    if not c_left_choices:
        c_left_choices = [1]
    c_left = random.choice(c_left_choices)
    c_right_min = max(c_left + 2, width // 2)
    c_right_choices = list(range(c_right_min, width - 1))
    if not c_right_choices:
        c_right_choices = [width - 2]
    c_right = random.choice(c_right_choices)
    return c_left, c_right


def _scatter_teal(
    grid: np.ndarray,
    c_left: int,
    c_right: int,
    path_set: set[tuple[int, int]],
    start_row: int,
) -> None:
    h, _ = grid.shape
    candidates_keep = [
        (r, c)
        for r in range(0, start_row + 1)
        for c in range(c_left + 1, c_right)
        if (r, c) not in path_set and int(grid[r, c]) == BLACK
    ]
    candidates_drop = [
        (r, c)
        for r in range(start_row + 1, h)
        for c in range(c_left + 1, c_right)
        if (r, c) not in path_set and int(grid[r, c]) == BLACK
    ]
    if len(candidates_keep) < 2:
        raise RuntimeError("not enough interior cells for teal decoration")

    random.shuffle(candidates_keep)
    random.shuffle(candidates_drop)

    keep_n = random.randint(2, min(10, len(candidates_keep)))
    for r, c in candidates_keep[:keep_n]:
        grid[r, c] = TEAL

    if candidates_drop:
        drop_n = random.randint(1, min(4, len(candidates_drop)))
        for r, c in candidates_drop[:drop_n]:
            grid[r, c] = TEAL


def _count_vertical_rails(grid: np.ndarray) -> int:
    try:
        return len(_find_vertical_rails(grid))
    except ValueError:
        return 0


def _generate_attempt() -> np.ndarray | None:
    target_reflections = random.choices([2, 3, 4, 5], weights=[3, 3, 3, 2], k=1)[0]
    width, start_row = _choose_width_and_start(target_reflections)
    min_height = max(12, start_row + 1)
    max_height = min(30, start_row + 3)
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[start_row, 0] = BLUE

    try:
        left_span, right_span = _sample_rail_spans(height, start_row)
    except RuntimeError:
        return None
    c_left, c_right = _rail_columns(width)

    for r in range(left_span[0], left_span[1] + 1):
        grid[r, c_left] = GREY
    for r in range(right_span[0], right_span[1] + 1):
        grid[r, c_right] = GREY

    path, reflections = _simulate_path(grid.shape, (start_row, 0))
    if reflections != target_reflections or reflections < 2:
        return None
    path_set = set(path)

    try:
        _scatter_teal(grid, c_left, c_right, path_set, start_row)
    except RuntimeError:
        return None

    rail_overlap_top = max(left_span[0], right_span[0])
    rail_overlap_bottom = min(left_span[1], right_span[1])
    if not (rail_overlap_top <= start_row <= rail_overlap_bottom):
        return None
    if _count_vertical_rails(grid) != 2:
        return None

    output = main(grid)
    if output.ndim != 2 or max(output.shape) > 30:
        return None

    expected_border = _border_color(reflections)
    border_cells = np.concatenate((output[0], output[-1], output[1:-1, 0], output[1:-1, -1]))
    if not np.all(border_cells == expected_border):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
