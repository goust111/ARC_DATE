from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIRS = ("up", "right", "down", "left")
MAX_GENERATION_ATTEMPTS = 300
ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
CELL_SIZE_CHOICES = (3, 5, 7)


def _separator_positions(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_seps = [r for r in range(1, h - 1) if np.all(grid[r, 1:-1] == GREY)]
    col_seps = [c for c in range(1, w - 1) if np.all(grid[1:-1, c] == GREY)]
    assert len(row_seps) >= 2 and len(col_seps) >= 2
    assert row_seps[0] == 1 and row_seps[-1] == h - 2
    assert col_seps[0] == 1 and col_seps[-1] == w - 2
    return row_seps, col_seps


def _cell_spans(seps: list[int]) -> list[tuple[int, int]]:
    spans = [(seps[i] + 1, seps[i + 1] - 1) for i in range(len(seps) - 1)]
    lengths = {end - start + 1 for start, end in spans}
    assert spans and min(lengths) >= 3 and len(lengths) == 1
    only = next(iter(lengths))
    assert only % 2 == 1
    return spans


def _pointer_direction(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> str:
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    assert int(grid[cr, cc]) == GREY

    scores = {
        "up": int(np.count_nonzero(grid[r0:cr, cc] == ORANGE)),
        "right": int(np.count_nonzero(grid[cr, cc + 1 : c1 + 1] == ORANGE)),
        "down": int(np.count_nonzero(grid[cr + 1 : r1 + 1, cc] == ORANGE)),
        "left": int(np.count_nonzero(grid[cr, c0:cc] == ORANGE)),
    }
    best = max(scores.values())
    assert best > 0 and sum(score == best for score in scores.values()) == 1
    return max(scores, key=scores.get)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the pointer direction in each dial and fill the dial with the color
    stored on the matching side of the outer border ring.
    """
    g = np.asarray(grid, dtype=int)
    row_seps, col_seps = _separator_positions(g)
    row_spans = _cell_spans(row_seps)
    col_spans = _cell_spans(col_seps)

    side_colors = {
        "up": int(g[0, 0]),
        "right": int(g[1, -1]),
        "down": int(g[-1, 0]),
        "left": int(g[1, 0]),
    }

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            direction = _pointer_direction(g, r0, c0, r1, c1)
            out[r0 : r1 + 1, c0 : c1 + 1] = side_colors[direction]
    return out


def _sample_directions(rng: random.Random, count: int) -> list[str]:
    directions = list(DIRS)
    directions.extend(rng.choice(DIRS) for _ in range(count - len(DIRS)))
    rng.shuffle(directions)
    return directions


def _dial_layout(row_count: int, col_count: int, cell_h: int, cell_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = [1 + i * (cell_h + 1) for i in range(row_count + 1)]
    col_seps = [1 + i * (cell_w + 1) for i in range(col_count + 1)]
    return _cell_spans(row_seps), _cell_spans(col_seps)


def _paint_pointer(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, direction: str) -> None:
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    grid[cr, cc] = GREY
    if direction == "up":
        grid[r0:cr, cc] = ORANGE
    elif direction == "right":
        grid[cr, cc + 1 : c1 + 1] = ORANGE
    elif direction == "down":
        grid[cr + 1 : r1 + 1, cc] = ORANGE
    else:
        grid[cr, c0:cc] = ORANGE


def _expected_output(grid: np.ndarray, directions: list[list[str]], row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    side_colors = {
        "up": int(grid[0, 0]),
        "right": int(grid[1, -1]),
        "down": int(grid[-1, 0]),
        "left": int(grid[1, 0]),
    }
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = side_colors[directions[r][c]]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.randint(*ROW_COUNT_RANGE)
    col_count = rng.randint(*COL_COUNT_RANGE)
    cell_h = rng.choice(CELL_SIZE_CHOICES)
    cell_w = rng.choice(CELL_SIZE_CHOICES)

    h = 2 + (row_count + 1) + row_count * cell_h
    w = 2 + (col_count + 1) + col_count * cell_w
    grid = np.full((h, w), BLACK, dtype=int)

    grid[0, :] = RED
    grid[-1, :] = GREEN
    grid[1:-1, 0] = YELLOW
    grid[1:-1, -1] = BLUE

    row_seps = [1 + i * (cell_h + 1) for i in range(row_count + 1)]
    col_seps = [1 + i * (cell_w + 1) for i in range(col_count + 1)]
    for r in row_seps:
        grid[r, 1:-1] = GREY
    for c in col_seps:
        grid[1:-1, c] = GREY

    row_spans, col_spans = _dial_layout(row_count, col_count, cell_h, cell_w)
    flat_dirs = _sample_directions(rng, row_count * col_count)
    directions = [
        flat_dirs[r * col_count : (r + 1) * col_count]
        for r in range(row_count)
    ]

    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            _paint_pointer(grid, r0, c0, r1, c1, directions[r][c])

    expected = _expected_output(grid, directions, row_spans, col_spans)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
