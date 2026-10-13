import random
from typing import Iterable

import numpy as np


TOP = 1
RIGHT = 2
BOTTOM = 3
LEFT = 4
GREY = 5
BLACK = 0

MOVE_PRIORITY = {TOP: 0, RIGHT: 1, BOTTOM: 2, LEFT: 3}
FRAME_COLORS = {TOP, RIGHT, BOTTOM, LEFT}
ALL_INPUT_COLORS = {BLACK, TOP, RIGHT, BOTTOM, LEFT, GREY}

GRID_MIN = 13
GRID_MAX = 18
FRAME_MIN = 6
FRAME_MAX = 10
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    if coords.size == 0:
        raise ValueError("expected a non-empty grid")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_completed_frame(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0] = LEFT
    out[r0 : r1 + 1, c1] = RIGHT
    out[r0, c0 : c1 + 1] = TOP
    out[r1, c0 : c1 + 1] = BOTTOM


def _target_for(color: int, r: int, c: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if color == TOP:
        return r0 + 1, c
    if color == RIGHT:
        return r, c1 - 1
    if color == BOTTOM:
        return r1 - 1, c
    if color == LEFT:
        return r, c0 + 1
    raise ValueError(f"unsupported movable color: {color}")


def _sort_key(color: int, r: int, c: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if color == TOP:
        return r - (r0 + 1), c
    if color == RIGHT:
        return (c1 - 1) - c, r
    if color == BOTTOM:
        return (r1 - 1) - r, c
    if color == LEFT:
        return c - (c0 + 1), r
    raise ValueError(f"unsupported movable color: {color}")


def _step(color: int) -> tuple[int, int]:
    if color == TOP:
        return -1, 0
    if color == RIGHT:
        return 0, 1
    if color == BOTTOM:
        return 1, 0
    if color == LEFT:
        return 0, -1
    raise ValueError(f"unsupported movable color: {color}")


def _interior_pixels(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    pixels: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color != BLACK:
                pixels.append((r, c, color))
    return pixels


def _solve_pixels(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int]]]]:
    bbox = _frame_bbox(grid)
    movable: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int]]] = []
    for r, c, color in _interior_pixels(grid, bbox):
        if color == GREY:
            continue
        if color not in FRAME_COLORS:
            raise ValueError(f"unexpected interior color: {color}")
        movable.append(((r, c), color, _sort_key(color, r, c, bbox), _target_for(color, r, c, bbox)))

    movable.sort(key=lambda item: (item[2][0], item[2][1], MOVE_PRIORITY[item[1]]))

    occupied: set[tuple[int, int]] = set()
    solved: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int]]] = []
    for start, color, key, target in movable:
        dr, dc = _step(color)
        r, c = start
        while (r, c) != target:
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        solved.append((start, color, key, (r, c)))
    return bbox, solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the four-color rectangular frame, then slide each interior pixel toward the matching side.

    Input structure:
    - A black background contains one incomplete rectangular frame whose top, right, bottom, and left sides
      use blue, red, green, and yellow respectively.
    - The four frame corners belong to the horizontal sides, so the top corners are blue and the bottom
      corners are green.
    - The frame interior contains isolated single pixels colored blue, red, green, yellow, or grey.

    Rule:
    - Reconstruct the full rectangle border with the fixed side colors, keeping horizontal colors on the corners.
    - Delete every grey interior pixel.
    - For each remaining interior pixel, map its color to a side and slide it straight toward that side's
      inner border cell.
    - Process pixels by increasing distance to their target side; ties are broken by the coordinate
      perpendicular to the motion direction, and a pixel stops one cell before any earlier settled pixel.

    Key insight:
    - The incomplete border first establishes the rectangle and its four directional color anchors; only then
      can the interior singletons be slid inward with the specified ordering and collision blocking.
    """
    g = np.asarray(grid, dtype=int)
    bbox, solved = _solve_pixels(g)
    out = np.zeros_like(g)
    _draw_completed_frame(out, bbox)
    for _start, color, _key, end in solved:
        out[end] = color
    return out


def _neighbors8(coord: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = coord
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _can_place(coord: tuple[int, int], occupied: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = coord
    if not (r0 + 1 <= r <= r1 - 1 and c0 + 1 <= c <= c1 - 1):
        return False
    if coord in occupied:
        return False
    return all(neighbor not in occupied for neighbor in _neighbors8(coord))


def _append_pixel(
    pixels: list[tuple[int, int, int]],
    occupied: set[tuple[int, int]],
    coord: tuple[int, int],
    color: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    if not _can_place(coord, occupied, bbox):
        return False
    pixels.append((coord[0], coord[1], color))
    occupied.add(coord)
    return True


def _place_stack_pair(
    color: int,
    pixels: list[tuple[int, int, int]],
    occupied: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    for _ in range(PLACEMENT_ATTEMPTS):
        if color == TOP:
            c = random.randint(c0 + 1, c1 - 1)
            near_r = random.randint(r0 + 2, r1 - 2)
            far_min = near_r + 2
            if far_min > r1 - 1:
                continue
            coords = ((near_r, c), (random.randint(far_min, r1 - 1), c))
        elif color == RIGHT:
            r = random.randint(r0 + 1, r1 - 1)
            near_c = random.randint(c0 + 2, c1 - 2)
            far_max = near_c - 2
            if far_max < c0 + 1:
                continue
            coords = ((r, near_c), (r, random.randint(c0 + 1, far_max)))
        elif color == BOTTOM:
            c = random.randint(c0 + 1, c1 - 1)
            near_r = random.randint(r0 + 2, r1 - 2)
            far_max = near_r - 2
            if far_max < r0 + 1:
                continue
            coords = ((near_r, c), (random.randint(r0 + 1, far_max), c))
        else:
            r = random.randint(r0 + 1, r1 - 1)
            near_c = random.randint(c0 + 2, c1 - 2)
            far_min = near_c + 2
            if far_min > c1 - 1:
                continue
            coords = ((r, near_c), (r, random.randint(far_min, c1 - 1)))

        if all(_can_place(coord, occupied, bbox) for coord in coords):
            for coord in coords:
                _append_pixel(pixels, occupied, coord, color, bbox)
            return True
    return False


def _place_cross_collision_pair(
    pixels: list[tuple[int, int, int]],
    occupied: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    modes = ["blue_red", "blue_yellow", "green_red", "green_yellow"]
    random.shuffle(modes)
    for mode in modes:
        for _ in range(PLACEMENT_ATTEMPTS):
            if mode == "blue_red":
                mover = (random.randint(r0 + 4, r1 - 1), c1 - 1)
                blocker = (random.randint(r0 + 1, mover[0] - 1), c1 - 3)
                payload = ((mover, TOP), (blocker, RIGHT))
            elif mode == "blue_yellow":
                mover = (random.randint(r0 + 4, r1 - 1), c0 + 1)
                blocker = (random.randint(r0 + 1, mover[0] - 1), c0 + 3)
                payload = ((mover, TOP), (blocker, LEFT))
            elif mode == "green_red":
                mover = (random.randint(r0 + 1, r1 - 4), c1 - 1)
                blocker = (random.randint(mover[0] + 1, r1 - 1), c1 - 3)
                payload = ((mover, BOTTOM), (blocker, RIGHT))
            else:
                mover = (random.randint(r0 + 1, r1 - 4), c0 + 1)
                blocker = (random.randint(mover[0] + 1, r1 - 1), c0 + 3)
                payload = ((mover, BOTTOM), (blocker, LEFT))

            coords = [coord for coord, _color in payload]
            if all(_can_place(coord, occupied, bbox) for coord in coords):
                for coord, color in payload:
                    _append_pixel(pixels, occupied, coord, color, bbox)
                return True
    return False


def _random_isolated_coord(
    occupied: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    for _ in range(PLACEMENT_ATTEMPTS):
        coord = (random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1))
        if _can_place(coord, occupied, bbox):
            return coord
    return None


def _side_gap_positions(side_len: int, allow_corners: bool) -> list[int]:
    if allow_corners:
        return list(range(1, side_len - 1))
    return list(range(side_len))


def _apply_frame_gaps(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    width = c1 - c0 + 1
    height = r1 - r0 + 1

    top_gaps = random.sample(_side_gap_positions(width, allow_corners=True), random.randint(1, 2))
    bottom_gaps = random.sample(_side_gap_positions(width, allow_corners=True), random.randint(1, 2))
    left_candidates = list(range(r0 + 1, r1))
    right_candidates = list(range(r0 + 1, r1))
    left_gaps = random.sample(left_candidates, random.randint(1, 2))
    right_gaps = random.sample(right_candidates, random.randint(1, 2))

    for offset in top_gaps:
        grid[r0, c0 + offset] = BLACK
    for offset in bottom_gaps:
        grid[r1, c0 + offset] = BLACK
    for r in left_gaps:
        grid[r, c0] = BLACK
    for r in right_gaps:
        grid[r, c1] = BLACK

    assert int(np.count_nonzero(grid[r0, c0 : c1 + 1] == TOP)) >= 2
    assert int(np.count_nonzero(grid[r1, c0 : c1 + 1] == BOTTOM)) >= 2
    assert int(np.count_nonzero(grid[r0 + 1 : r1, c0] == LEFT)) >= 2
    assert int(np.count_nonzero(grid[r0 + 1 : r1, c1] == RIGHT)) >= 2


def _side_missing_counts(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    top_missing = sum(int(grid[r0, c] != TOP) for c in range(c0 + 1, c1))
    right_missing = sum(int(grid[r, c1] != RIGHT) for r in range(r0 + 1, r1))
    bottom_missing = sum(int(grid[r1, c] != BOTTOM) for c in range(c0 + 1, c1))
    left_missing = sum(int(grid[r, c0] != LEFT) for r in range(r0 + 1, r1))
    return top_missing, right_missing, bottom_missing, left_missing


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        if grid.ndim != 2 or max(grid.shape) > 30:
            return False
        if not set(map(int, np.unique(grid))).issubset(ALL_INPUT_COLORS):
            return False

        bbox = _frame_bbox(grid)
        movable = []
        color_counts = {TOP: 0, RIGHT: 0, BOTTOM: 0, LEFT: 0, GREY: 0}
        occupied: list[tuple[int, int]] = []
        for r, c, color in _interior_pixels(grid, bbox):
            color_counts[color] += 1
            occupied.append((r, c))
            if color in FRAME_COLORS:
                movable.append((r, c, color))

        colored_total = sum(color_counts[color] for color in FRAME_COLORS)
        grey_total = color_counts[GREY]
        if not (4 <= colored_total <= 8 and 0 <= grey_total <= 2):
            return False
        if any(color_counts[color] == 0 for color in FRAME_COLORS):
            return False

        for i, (r1, c1) in enumerate(occupied):
            for r2, c2 in occupied[i + 1 :]:
                if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                    return False

        keys = [_sort_key(color, r, c, bbox) for r, c, color in movable]
        if len(set(keys)) != len(keys):
            return False

        missing = _side_missing_counts(grid, bbox)
        if any(count not in (1, 2) for count in missing):
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if np.any(out == GREY):
            return False
        blocked = 0
        _, solved = _solve_pixels(grid)
        for start, color, _key, end in solved:
            if end != _target_for(color, start[0], start[1], bbox):
                blocked += 1
        if blocked < 2:
            return False

        completed = np.zeros_like(grid)
        _draw_completed_frame(completed, bbox)
        r0, c0, r1, c1 = bbox
        if not np.array_equal(out[r0 : r1 + 1, c0], completed[r0 : r1 + 1, c0]):
            return False
        if not np.array_equal(out[r0 : r1 + 1, c1], completed[r0 : r1 + 1, c1]):
            return False
        if not np.array_equal(out[r0, c0 : c1 + 1], completed[r0, c0 : c1 + 1]):
            return False
        if not np.array_equal(out[r1, c0 : c1 + 1], completed[r1, c0 : c1 + 1]):
            return False
        return True
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    frame_h = random.randint(FRAME_MIN, min(FRAME_MAX, height - 2))
    frame_w = random.randint(FRAME_MIN, min(FRAME_MAX, width - 2))
    r0 = random.randint(1, height - frame_h - 1)
    c0 = random.randint(1, width - frame_w - 1)
    bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    grid = np.zeros((height, width), dtype=int)
    _draw_completed_frame(grid, bbox)
    _apply_frame_gaps(grid, bbox)

    pixels: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()

    stack_color = random.choice((TOP, RIGHT, BOTTOM, LEFT))
    if not _place_stack_pair(stack_color, pixels, occupied, bbox):
        return None
    if not _place_cross_collision_pair(pixels, occupied, bbox):
        return None

    represented = {color for _r, _c, color in pixels}
    for color in (TOP, RIGHT, BOTTOM, LEFT):
        if color in represented:
            continue
        coord = _random_isolated_coord(occupied, bbox)
        if coord is None or not _append_pixel(pixels, occupied, coord, color, bbox):
            return None

    colored_count = len([1 for _r, _c, color in pixels if color in FRAME_COLORS])
    extra_colored = random.randint(0, max(0, 8 - colored_count))
    for _ in range(extra_colored):
        coord = _random_isolated_coord(occupied, bbox)
        if coord is None:
            break
        color = random.choice((TOP, RIGHT, BOTTOM, LEFT))
        trial_pixels = pixels + [(coord[0], coord[1], color)]
        trial_keys = [_sort_key(pc, pr, pc2, bbox) for pr, pc2, pc in trial_pixels if pc in FRAME_COLORS]
        if len(set(trial_keys)) != len(trial_keys):
            continue
        _append_pixel(pixels, occupied, coord, color, bbox)

    grey_count = random.randint(0, 2)
    for _ in range(grey_count):
        coord = _random_isolated_coord(occupied, bbox)
        if coord is None:
            break
        _append_pixel(pixels, occupied, coord, GREY, bbox)

    for r, c, color in pixels:
        grid[r, c] = color

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
