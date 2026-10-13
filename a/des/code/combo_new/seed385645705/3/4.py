from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_ROWS_RANGE = (12, 24)
GRID_COLS_RANGE = (12, 24)
OBSTACLE_COUNT_RANGE = (1, 5)
OBSTACLE_SIDE_RANGE = (2, 5)
PAIR_COUNT_RANGE = (2, 6)
MIN_CLEAR_LENGTH = 4
MAX_CLEAR_LENGTH = 14
GENERATION_ATTEMPTS = 600
OBSTACLE_PLACEMENT_ATTEMPTS = 120
PAIR_PLACEMENT_ATTEMPTS = 600

Rect = tuple[int, int, int, int]
Cell = tuple[int, int]


def _ordered_line(a: Cell, b: Cell) -> list[Cell]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    assert c0 == c1, "paired pixels must share a row or a column"
    lo, hi = sorted((r0, r1))
    return [(r, c0) for r in range(lo, hi + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of same-colored pixels with a solid line unless the path crosses grey obstacles, in which case draw a dashed line.

    Input structure:
    - Black is the background.
    - Grey pixels form separated solid rectangular obstacles that stay fixed.
    - Every non-black, non-grey color appears exactly twice, and the two pixels
      of that color lie on one shared row or one shared column.

    Rule:
    - For each color pair, consider the straight row or column segment between
      the two endpoints, including the endpoints.
    - If the segment contains no grey obstacle pixels, recolor the whole segment
      with the pair color.
    - If the segment contains at least one grey obstacle pixel, keep grey pixels
      grey and recolor every other non-grey cell of the segment, starting from
      the endpoint closer to the upper-left corner.
    - Grey obstacle pixels are never overwritten.

    Key insight:
    - The endpoints only identify the line direction and color; whether the
      intervening segment touches a grey obstacle switches the drawing style
      from continuous to dashed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    planned: dict[Cell, int] = {}
    colors = [int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)]
    assert colors, "expected at least one colored pair"

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each drawing color must occur exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        assert (a[0] == b[0]) ^ (a[1] == b[1]), "paired pixels must be horizontal or vertical"

        path = _ordered_line(a, b)
        endpoints = {path[0], path[-1]}
        has_obstacle = any(int(g[cell]) == GREY for cell in path)

        if has_obstacle:
            visible_index = 0
            for cell in path:
                value = int(g[cell])
                assert value in (BLACK, GREY, color), "line paths must not cross other color markers"
                if value == GREY:
                    continue
                if visible_index % 2 == 0 or cell in endpoints:
                    previous = planned.get(cell)
                    assert previous in (None, color), "line drawings must not conflict"
                    planned[cell] = color
                visible_index += 1
        else:
            for cell in path:
                value = int(g[cell])
                assert value in (BLACK, color), "clear line paths must contain only background and endpoints"
                previous = planned.get(cell)
                assert previous in (None, color), "line drawings must not conflict"
                planned[cell] = color

    out = g.copy()
    for (r, c), color in planned.items():
        if int(g[r, c]) != GREY:
            out[r, c] = color
    out[g == GREY] = GREY
    return out


def _line_is_available(grid: np.ndarray, reserved: np.ndarray, path: list[Cell], *, needs_grey: bool) -> bool:
    has_grey = False
    for r, c in path:
        if reserved[r, c]:
            return False
        value = int(grid[r, c])
        if value == GREY:
            has_grey = True
        elif value != BLACK:
            return False
    return has_grey if needs_grey else not has_grey


def _odd_visible_count(grid: np.ndarray, path: list[Cell]) -> bool:
    return sum(int(grid[r, c]) != GREY for r, c in path) % 2 == 1


def _commit_pair(grid: np.ndarray, reserved: np.ndarray, color: int, path: list[Cell]) -> None:
    for r, c in path:
        reserved[r, c] = True
    r0, c0 = path[0]
    r1, c1 = path[-1]
    grid[r0, c0] = color
    grid[r1, c1] = color


def _place_obstacles(grid: np.ndarray) -> list[Rect] | None:
    h, w = grid.shape
    rects: list[Rect] = []
    occupied = np.zeros_like(grid, dtype=bool)
    target_count = random.randint(*OBSTACLE_COUNT_RANGE)

    for _ in range(target_count):
        placed = False
        for _attempt in range(OBSTACLE_PLACEMENT_ATTEMPTS):
            rh = random.randint(*OBSTACLE_SIDE_RANGE)
            rw = random.randint(*OBSTACLE_SIDE_RANGE)
            if rh > h - 4 or rw > w - 4:
                continue
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            if np.any(occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
                continue
            grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    return rects


def _try_place_blocked_pair(grid: np.ndarray, reserved: np.ndarray, rects: list[Rect], color: int) -> bool:
    h, w = grid.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        r0, c0, r1, c1 = random.choice(rects)
        horizontal = random.random() < 0.5

        if horizontal:
            if c0 == 0 or c1 == w - 1:
                continue
            r = random.randint(r0, r1)
            left = list(range(0, c0))
            right = list(range(c1 + 1, w))
            if not left or not right:
                continue
            random.shuffle(left)
            random.shuffle(right)
            for lc in left[:8]:
                for rc in right[:8]:
                    path = [(r, c) for c in range(lc, rc + 1)]
                    if len(path) < MIN_CLEAR_LENGTH:
                        continue
                    if _line_is_available(grid, reserved, path, needs_grey=True) and _odd_visible_count(grid, path):
                        _commit_pair(grid, reserved, color, path)
                        return True
        else:
            if r0 == 0 or r1 == h - 1:
                continue
            c = random.randint(c0, c1)
            above = list(range(0, r0))
            below = list(range(r1 + 1, h))
            if not above or not below:
                continue
            random.shuffle(above)
            random.shuffle(below)
            for tr in above[:8]:
                for br in below[:8]:
                    path = [(r, c) for r in range(tr, br + 1)]
                    if len(path) < MIN_CLEAR_LENGTH:
                        continue
                    if _line_is_available(grid, reserved, path, needs_grey=True) and _odd_visible_count(grid, path):
                        _commit_pair(grid, reserved, color, path)
                        return True

    return False


def _try_place_clear_pair(grid: np.ndarray, reserved: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        horizontal = random.random() < 0.5
        if horizontal:
            r = random.randrange(h)
            length = random.randint(MIN_CLEAR_LENGTH, min(MAX_CLEAR_LENGTH, w))
            c0 = random.randint(0, w - length)
            c1 = c0 + length - 1
            path = [(r, c) for c in range(c0, c1 + 1)]
        else:
            c = random.randrange(w)
            length = random.randint(MIN_CLEAR_LENGTH, min(MAX_CLEAR_LENGTH, h))
            r0 = random.randint(0, h - length)
            r1 = r0 + length - 1
            path = [(r, c) for r in range(r0, r1 + 1)]

        if _line_is_available(grid, reserved, path, needs_grey=False):
            _commit_pair(grid, reserved, color, path)
            return True

    return False


def _valid_generated_grid(grid: np.ndarray, pair_count: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid == GREY) == 0:
        return False
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BLACK, GREY)]
    if len(colors) != pair_count:
        return False
    if any(np.count_nonzero(grid == color) != 2 for color in colors):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return bool(out.shape == grid.shape and not np.array_equal(out, grid))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    rects = _place_obstacles(grid)
    if not rects:
        return None

    max_pairs = min(PAIR_COUNT_RANGE[1], len(PAIR_COLORS))
    pair_count = random.randint(PAIR_COUNT_RANGE[0], max_pairs)
    blocked_count = pair_count // 2
    if pair_count % 2 and random.random() < 0.5:
        blocked_count += 1
    blocked_count = max(1, min(pair_count - 1, blocked_count))
    clear_count = pair_count - blocked_count

    colors = random.sample(PAIR_COLORS, pair_count)
    random.shuffle(colors)
    blocked_colors = colors[:blocked_count]
    clear_colors = colors[blocked_count:]
    random.shuffle(blocked_colors)
    random.shuffle(clear_colors)

    reserved = np.zeros_like(grid, dtype=bool)
    for color in blocked_colors:
        if not _try_place_blocked_pair(grid, reserved, rects, color):
            return None
    for color in clear_colors:
        if not _try_place_clear_pair(grid, reserved, color):
            return None

    if clear_count == 0 or blocked_count == 0:
        return None
    if not _valid_generated_grid(grid, pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
