from __future__ import annotations

from collections import deque
import random

import numpy as np


# ARC palette aliases.
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

# The task's "mix" rule is an explicit visible color precedence, not a numeric comparison.
COLOR_PRIORITY = {
    BLUE: 0,
    RED: 1,
    GREEN: 2,
    YELLOW: 3,
    GREY: 4,
    MAGENTA: 5,
    ORANGE: 6,
    AZURE: 7,
    MAROON: 8,
}
SEED_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE, MAROON)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mix_colors(color_a: int, color_b: int) -> int:
    return min((int(color_a), int(color_b)), key=COLOR_PRIORITY.__getitem__)


def _collect_enclosed_regions(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    region_map = np.full((h, w), -1, dtype=int)
    seed_colors: list[int] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == GREY:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            seed_set: set[int] = set()

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                value = int(grid[cr, cc])
                if value not in (BLACK, GREY):
                    seed_set.add(value)

                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != GREY:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if touches_edge:
                continue

            assert len(seed_set) == 1, "each enclosed region must contain exactly one seed color"
            region_id = len(seed_colors)
            seed_color = next(iter(seed_set))
            seed_colors.append(seed_color)
            for rr, cc in coords:
                region_map[rr, cc] = region_id

    assert seed_colors, "expected at least one enclosed region"
    return region_map, seed_colors


def _paint_shared_walls(grid: np.ndarray, region_map: np.ndarray, seed_colors: list[int]) -> np.ndarray:
    h, w = grid.shape
    out = np.array(grid, dtype=int, copy=True)
    out[(out != BLACK) & (out != GREY)] = BLACK

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY:
                continue

            up = int(region_map[r - 1, c]) if r > 0 else -1
            down = int(region_map[r + 1, c]) if r + 1 < h else -1
            left = int(region_map[r, c - 1]) if c > 0 else -1
            right = int(region_map[r, c + 1]) if c + 1 < w else -1

            horizontal_pair = up >= 0 and down >= 0 and up != down
            vertical_pair = left >= 0 and right >= 0 and left != right

            if horizontal_pair == vertical_pair:
                continue

            a, b = (up, down) if horizontal_pair else (left, right)
            out[r, c] = _mix_colors(seed_colors[a], seed_colors[b])

    # Junction pixels are ambiguous unless all incident colored wall segments agree.
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY or int(out[r, c]) != GREY:
                continue

            adjacent_wall_colors = []
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY and int(out[nr, nc]) != GREY:
                    adjacent_wall_colors.append(int(out[nr, nc]))
            if len(adjacent_wall_colors) >= 2 and len(set(adjacent_wall_colors)) == 1:
                out[r, c] = adjacent_wall_colors[0]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor internal grey wall segments by the lower-priority seed color of the two regions they separate,
    keep outer walls grey, and erase the seeds.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    region_map, seed_colors = _collect_enclosed_regions(g)
    return _paint_shared_walls(g, region_map, seed_colors)


# --------------- generator ---------------

# Canvas and partition geometry.
GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 26)
MARGIN_RANGE = (0, 3)
MIN_REGION_SPAN = 3
MAX_REGIONS = len(SEED_COLORS)
TARGET_REGION_COUNTS = (2, 3, 4, 5, 6, 7, 8)
TARGET_REGION_WEIGHTS = (1, 2, 3, 3, 2, 1, 1)

# Retry budget.
GEN_RETRY_LIMIT = 500


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _sample_root_box() -> tuple[int, int, int, int, int, int] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    outer_top = top_margin
    outer_bottom = h - 1 - bottom_margin
    outer_left = left_margin
    outer_right = w - 1 - right_margin
    inner_h = outer_bottom - outer_top - 1
    inner_w = outer_right - outer_left - 1

    if inner_h < MIN_REGION_SPAN or inner_w < MIN_REGION_SPAN:
        return None
    return h, w, outer_top, outer_bottom, outer_left, outer_right


def _split_options(box: tuple[int, int, int, int]) -> list[str]:
    r0, r1, c0, c1 = box
    options = []
    if c1 - c0 + 1 >= 2 * MIN_REGION_SPAN + 1:
        options.append("v")
    if r1 - r0 + 1 >= 2 * MIN_REGION_SPAN + 1:
        options.append("h")
    return options


def _split_box(box: tuple[int, int, int, int], orient: str) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[str, int, int, int],
]:
    r0, r1, c0, c1 = box
    if orient == "v":
        wall_col = random.randint(c0 + MIN_REGION_SPAN, c1 - MIN_REGION_SPAN)
        left_box = (r0, r1, c0, wall_col - 1)
        right_box = (r0, r1, wall_col + 1, c1)
        return left_box, right_box, ("v", wall_col, r0 - 1, r1 + 1)

    wall_row = random.randint(r0 + MIN_REGION_SPAN, r1 - MIN_REGION_SPAN)
    top_box = (r0, wall_row - 1, c0, c1)
    bottom_box = (wall_row + 1, r1, c0, c1)
    return top_box, bottom_box, ("h", wall_row, c0 - 1, c1 + 1)


def _draw_outer_border(grid: np.ndarray, outer_top: int, outer_bottom: int, outer_left: int, outer_right: int) -> None:
    grid[outer_top, outer_left:outer_right + 1] = GREY
    grid[outer_bottom, outer_left:outer_right + 1] = GREY
    grid[outer_top:outer_bottom + 1, outer_left] = GREY
    grid[outer_top:outer_bottom + 1, outer_right] = GREY


def _draw_wall_line(grid: np.ndarray, wall: tuple[str, int, int, int]) -> None:
    orient, fixed, start, end = wall
    if orient == "v":
        grid[start:end + 1, fixed] = GREY
    else:
        grid[fixed, start:end + 1] = GREY


def _build_partition() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    sampled = _sample_root_box()
    if sampled is None:
        return None
    h, w, outer_top, outer_bottom, outer_left, outer_right = sampled

    root = (outer_top + 1, outer_bottom - 1, outer_left + 1, outer_right - 1)
    root_h = root[1] - root[0] + 1
    root_w = root[3] - root[2] + 1
    if root_h < MIN_REGION_SPAN or root_w < MIN_REGION_SPAN:
        return None

    max_by_area = min(MAX_REGIONS, max(1, (root_h * root_w) // (MIN_REGION_SPAN * MIN_REGION_SPAN)))
    choices = [n for n in TARGET_REGION_COUNTS if n <= max_by_area]
    if not choices:
        return None
    weights = [TARGET_REGION_WEIGHTS[TARGET_REGION_COUNTS.index(n)] for n in choices]
    target_regions = random.choices(choices, weights=weights)[0]

    boxes = [root]
    walls: list[tuple[str, int, int, int]] = []

    while len(boxes) < target_regions:
        candidates = []
        for idx, box in enumerate(boxes):
            for orient in _split_options(box):
                candidates.append((idx, orient, _box_area(box)))
        if not candidates:
            return None

        # Larger boxes are split more often, which keeps layouts readable.
        areas = [area for _idx, _orient, area in candidates]
        chosen_idx = random.choices(range(len(candidates)), weights=areas)[0]
        box_index, orient, _area = candidates[chosen_idx]
        box = boxes.pop(box_index)
        a, b, wall = _split_box(box, orient)
        boxes.extend([a, b])
        walls.append(wall)

    grid = np.full((h, w), BLACK, dtype=int)
    _draw_outer_border(grid, outer_top, outer_bottom, outer_left, outer_right)
    for wall in walls:
        _draw_wall_line(grid, wall)
    return grid, boxes


def _place_seeds(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, list[int], np.ndarray]:
    region_map = np.full(grid.shape, -1, dtype=int)
    seed_colors = random.sample(SEED_COLORS, len(boxes))
    random.shuffle(boxes)

    for region_id, (box, color) in enumerate(zip(boxes, seed_colors)):
        r0, r1, c0, c1 = box
        region_map[r0:r1 + 1, c0:c1 + 1] = region_id
        seed_r = random.randint(r0, r1)
        seed_c = random.randint(c0, c1)
        grid[seed_r, seed_c] = color

    return grid, seed_colors, region_map


def _count_internal_wall_orientations(grid: np.ndarray, region_map: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    horizontal = 0
    vertical = 0
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY:
                continue
            up = int(region_map[r - 1, c]) if r > 0 else -1
            down = int(region_map[r + 1, c]) if r + 1 < h else -1
            left = int(region_map[r, c - 1]) if c > 0 else -1
            right = int(region_map[r, c + 1]) if c + 1 < w else -1
            if up >= 0 and down >= 0 and up != down:
                horizontal += 1
            if left >= 0 and right >= 0 and left != right:
                vertical += 1
    return horizontal, vertical


def _oracle_from_layout(grid: np.ndarray, region_map: np.ndarray, seed_colors: list[int]) -> np.ndarray:
    return _paint_shared_walls(grid, region_map, seed_colors)


def _is_interesting_sample(grid: np.ndarray, expected: np.ndarray, boxes: list[tuple[int, int, int, int]], region_map: np.ndarray) -> bool:
    colored_wall_mask = (grid == GREY) & (expected != GREY)
    if int(colored_wall_mask.sum()) < 3:
        return False

    distinct_wall_colors = {int(v) for v in np.unique(expected[colored_wall_mask])}
    if len(boxes) >= 4 and len(distinct_wall_colors) < 2:
        return False

    horizontal, vertical = _count_internal_wall_orientations(grid, region_map)
    if len(boxes) >= 5 and (horizontal == 0 or vertical == 0):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    built = _build_partition()
    if built is None:
        return None
    grid, boxes = built

    grid, seed_colors, region_map = _place_seeds(grid, boxes)
    expected = _oracle_from_layout(grid, region_map, seed_colors)

    if not _is_interesting_sample(grid, expected, boxes, region_map):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
