from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
LEGEND_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


PATTERNS_BY_SIZE: dict[int, dict[int, tuple[tuple[tuple[int, int], ...], ...]]] = {
    9: {
        2: (
            ((4, 6), (5, 4)),
            ((1, 5), (3, 4)),
            ((4, 2), (5, 0)),
            ((3, 5), (4, 3)),
            ((3, 2), (4, 4)),
            ((2, 4), (4, 3)),
            ((3, 3), (4, 1)),
            ((4, 4), (6, 5)),
        ),
        3: (
            ((2, 4), (4, 3), (4, 5)),
            ((3, 5), (4, 3), (4, 7)),
            ((5, 5), (6, 3), (7, 6)),
            ((3, 2), (4, 4), (6, 5)),
            ((5, 5), (6, 3), (6, 7)),
            ((2, 3), (3, 5), (5, 6)),
            ((4, 4), (5, 2), (6, 5)),
            ((0, 1), (2, 2), (4, 3)),
            ((3, 5), (4, 1), (5, 6)),
            ((2, 6), (3, 4), (4, 7)),
            ((4, 6), (5, 4), (5, 8)),
        ),
        4: (
            ((3, 7), (4, 5), (5, 3), (6, 1)),
            ((2, 6), (3, 4), (3, 8), (4, 6)),
            ((3, 5), (4, 2), (5, 0), (5, 6)),
            ((2, 8), (3, 6), (4, 4), (5, 2)),
        ),
    },
    11: {
        2: (
            ((5, 5), (7, 6)),
            ((5, 7), (6, 5)),
            ((3, 5), (5, 4)),
            ((5, 3), (6, 1)),
            ((4, 6), (5, 4)),
            ((4, 4), (5, 2)),
            ((2, 6), (4, 5)),
            ((4, 3), (5, 5)),
        ),
        3: (
            ((5, 7), (6, 5), (6, 9)),
            ((3, 5), (4, 7), (5, 4)),
            ((6, 6), (7, 4), (7, 8)),
            ((5, 5), (6, 3), (6, 7)),
            ((4, 6), (5, 4), (6, 7)),
            ((4, 3), (5, 5), (7, 6)),
        ),
        4: (
            ((1, 2), (2, 6), (3, 3), (4, 5)),
        ),
    },
}


def _legend_sequence(grid: np.ndarray) -> list[int]:
    positions = [int(c) for c in np.flatnonzero(grid[-1] != BLACK)]
    return [int(grid[-1, c]) for c in positions]


def _simulate_spiral(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    grid = np.asarray(grid, dtype=int)
    n, m = grid.shape
    if n != m:
        raise ValueError("grid must be square")

    legend = _legend_sequence(grid)
    if not (3 <= len(legend) <= 4):
        raise ValueError("legend row must contain 3 or 4 colored pixels")

    obstacle_set = {tuple(map(int, rc)) for rc in np.argwhere(grid[:-1] == GREY)}
    out = grid.copy()
    painted = np.zeros((n - 1, n), dtype=bool)

    r, c = 0, 0
    direction = 0
    color_index = 0
    colors_used = {legend[color_index]}
    obstacle_hits: dict[tuple[int, int], int] = {cell: 0 for cell in obstacle_set}
    turn_counts = {"obstacle": 0, "boundary": 0, "legend": 0, "self": 0}

    while True:
        out[r, c] = legend[color_index]
        painted[r, c] = True

        moved = False
        for _ in range(4):
            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc

            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                turn_counts["boundary"] += 1
                direction = (direction + 1) % 4
                continue
            if nr == n - 1:
                turn_counts["legend"] += 1
                direction = (direction + 1) % 4
                continue
            if (nr, nc) in obstacle_set:
                turn_counts["obstacle"] += 1
                obstacle_hits[(nr, nc)] += 1
                color_index = (color_index + 1) % len(legend)
                colors_used.add(legend[color_index])
                direction = (direction + 1) % 4
                continue
            if painted[nr, nc]:
                turn_counts["self"] += 1
                direction = (direction + 1) % 4
                continue

            r, c = nr, nc
            moved = True
            break

        if not moved:
            break

    meta: dict[str, object] = {
        "legend": legend,
        "obstacles": obstacle_set,
        "obstacle_hits": obstacle_hits,
        "colors_used": colors_used,
        "painted_count": int(painted.sum()),
    }
    meta.update(turn_counts)
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a clockwise spiral whose color changes only when an obstacle blocks the next step.

    Input structure:
    - A square black grid of size 9x9 or 11x11.
    - The bottom row is a legend row containing 3 or 4 separated colored pixels whose left-to-right order defines a color cycle.
    - The rows above the legend contain isolated grey obstacle pixels; all other cells are black.

    Rule:
    - Start at the top-left cell, moving right, with the first legend color as the active drawing color.
    - Paint the current cell, then inspect the next cell in the current direction.
    - If the next cell is inside the drawing area, unpainted, and not grey, move there and continue.
    - If the next cell is a grey obstacle, advance to the next legend color, turn clockwise, and re-check.
    - If the next cell is outside the grid, already painted, or in the legend row, turn clockwise without changing color; stop only if all four directions fail.

    Key insight:
    - The spiral geometry comes from ordinary clockwise turning, but only obstacle-triggered turns advance the legend cycle, so obstacle positions determine the color segmentation.
    """
    out, _ = _simulate_spiral(grid)
    return out


def _build_legend_row(size: int, rng: random.Random, count: int) -> np.ndarray:
    while True:
        colors = rng.sample(LEGEND_PALETTE, count)
        gaps = [rng.randint(1, 2) for _ in range(count - 1)]
        span = count + sum(gaps)
        if span > size:
            continue

        row = np.zeros(size, dtype=int)
        col = rng.randint(0, size - span)
        row[col] = colors[0]
        for idx, gap in enumerate(gaps, start=1):
            col += gap + 1
            row[col] = colors[idx]
        return row


def _obstacles_are_isolated(obstacles: Iterable[tuple[int, int]]) -> bool:
    cells = list(obstacles)
    for i, (r0, c0) in enumerate(cells):
        for r1, c1 in cells[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _is_good_example(grid: np.ndarray) -> bool:
    n = int(grid.shape[0])
    if grid.shape != (n, n) or n not in (9, 11):
        return False

    legend_positions = [int(c) for c in np.flatnonzero(grid[-1] != BLACK)]
    legend = [int(grid[-1, c]) for c in legend_positions]
    if len(legend) not in (3, 4) or any(color == GREY for color in legend):
        return False
    if len(set(legend)) != len(legend):
        return False
    if any((b - a) not in (2, 3) for a, b in zip(legend_positions, legend_positions[1:])):
        return False

    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(grid[:-1] == GREY)}
    if len(obstacles) not in (2, 3, 4):
        return False
    if (0, 0) in obstacles or (0, n - 1) in obstacles:
        return False
    if not _obstacles_are_isolated(obstacles):
        return False

    out, meta = _simulate_spiral(grid)
    open_cells = (n - 1) * n - len(obstacles)
    colors_used = set(meta["colors_used"])
    obstacle_hits: dict[tuple[int, int], int] = meta["obstacle_hits"]  # type: ignore[assignment]

    if meta["painted_count"] != open_cells:
        return False
    if any(hit == 0 for hit in obstacle_hits.values()):
        return False
    if colors_used != set(legend):
        return False
    if meta["obstacle"] <= 0 or meta["boundary"] <= 0 or meta["legend"] <= 0 or meta["self"] <= 0:
        return False
    if meta["boundary"] + meta["legend"] + meta["self"] <= meta["obstacle"]:
        return False
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[-1], grid[-1]):
        return False
    if not np.array_equal(out[:-1] == GREY, grid[:-1] == GREY):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray:
    size = rng.choice((9, 11))
    obstacle_count = rng.choice((2, 3, 4))
    patterns_for_size = PATTERNS_BY_SIZE[size]
    if obstacle_count not in patterns_for_size:
        obstacle_count = rng.choice(tuple(patterns_for_size))
    obstacle_pattern = rng.choice(patterns_for_size[obstacle_count])

    legend_count = rng.choice((3, 4))
    grid = np.zeros((size, size), dtype=int)
    grid[-1] = _build_legend_row(size, rng, legend_count)
    for r, c in obstacle_pattern:
        grid[r, c] = GREY
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if _is_good_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid obstacle-triggered spiral example")
