import random

import numpy as np


BG = 0
WALL = 5
ARROW = 2


def _arrow_direction(red: set[tuple[int, int]]) -> tuple[int, int] | None:
    for r, c in red:
        patterns = [
            ({(r, c), (r, c + 1), (r, c + 2), (r - 1, c + 2)}, (0, 1)),
            ({(r, c), (r, c + 1), (r, c + 2), (r + 1, c)}, (0, -1)),
            ({(r, c), (r + 1, c), (r + 2, c), (r + 2, c + 1)}, (1, 0)),
            ({(r, c), (r + 1, c), (r + 2, c), (r, c - 1)}, (-1, 0)),
        ]
        for cells, direction in patterns:
            if cells == red:
                return direction
    return None


def main(grid: np.ndarray) -> np.ndarray:
    wall = np.argwhere(grid == WALL)
    if wall.size == 0:
        return grid.copy()
    top, left = wall.min(axis=0)
    bottom, right = wall.max(axis=0)
    red = {(int(r), int(c)) for r, c in np.argwhere(grid == ARROW)}
    direction = _arrow_direction(red)
    if direction is None:
        return (grid == WALL).astype(int) * WALL
    dr, dc = direction
    n = 3
    h, w = grid.shape
    if dc == 1:
        right = min(w - 1, right + n)
    elif dc == -1:
        left = max(0, left - n)
    elif dr == 1:
        bottom = min(h - 1, bottom + n)
    else:
        top = max(0, top - n)
    out = np.zeros_like(grid)
    out[top, left : right + 1] = WALL
    out[bottom, left : right + 1] = WALL
    out[top : bottom + 1, left] = WALL
    out[top : bottom + 1, right] = WALL
    return out


def _draw_room(grid: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL


def _draw_arrow(grid: np.ndarray, direction: tuple[int, int], r: int, c: int) -> None:
    if direction == (0, 1):
        cells = [(r, c), (r, c + 1), (r, c + 2), (r - 1, c + 2)]
    elif direction == (0, -1):
        cells = [(r, c), (r, c + 1), (r, c + 2), (r + 1, c)]
    elif direction == (1, 0):
        cells = [(r, c), (r + 1, c), (r + 2, c), (r + 2, c + 1)]
    else:
        cells = [(r, c), (r + 1, c), (r + 2, c), (r, c - 1)]
    for rr, cc in cells:
        grid[rr, cc] = ARROW


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        h = rng.randint(13, 24)
        w = rng.randint(13, 24)
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        room_h = rng.randint(8, 11)
        room_w = rng.randint(8, 11)
        margin = 4
        top_min = margin if direction == (-1, 0) else 1
        left_min = margin if direction == (0, -1) else 1
        top_max = h - room_h - (margin if direction == (1, 0) else 1)
        left_max = w - room_w - (margin if direction == (0, 1) else 1)
        if top_min > top_max or left_min > left_max:
            continue
        top = rng.randint(top_min, top_max)
        left = rng.randint(left_min, left_max)
        bottom = top + room_h - 1
        right = left + room_w - 1
        grid = np.zeros((h, w), dtype=int)
        _draw_room(grid, top, left, bottom, right)
        if direction in ((0, 1), (0, -1)):
            ar = rng.randint(top + 2, bottom - 2)
            ac = rng.randint(left + 2, right - 4)
        else:
            ar = rng.randint(top + 2, bottom - 4)
            ac = rng.randint(left + 2, right - 2)
        _draw_arrow(grid, direction, ar, ac)
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("generate_input failed")
