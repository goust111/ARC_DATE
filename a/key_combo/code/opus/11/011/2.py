from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (18, 24)
SEED_COUNT_CHOICES = (2, 3)
TOTAL_OBSTACLE_RANGE = (3, 5)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full grey separator column"
    return int(cols[0])


def _seed_rows(grid: np.ndarray) -> list[int]:
    rows = [int(r) for r in np.flatnonzero(grid[:, 0] == RED)]
    assert 2 <= len(rows) <= 3, "expected 2-3 red seeds in the leftmost column"
    return sorted(rows)


def _first_obstacle_on_row(grid: np.ndarray, row: int) -> tuple[int, int]:
    sep = _separator_col(grid)
    for c in range(1, grid.shape[1]):
        if c == sep:
            continue
        if int(grid[row, c]) == ORANGE:
            side_color = RED if c < sep else GREEN
            return c, side_color
    raise AssertionError("each seed row must contain a reachable obstacle")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_col(g)
    seeds = _seed_rows(g)
    out = g.copy()
    wing_candidates: list[tuple[int, int, int]] = []

    for row in seeds:
        current = RED
        hit_col = None
        hit_color = None
        for c in range(1, g.shape[1]):
            if c == sep:
                current = GREEN
                continue
            cell = int(g[row, c])
            if cell == ORANGE:
                hit_col = c
                hit_color = current
                break
            if cell == BACKGROUND:
                out[row, c] = current
        assert hit_col is not None and hit_color is not None
        wing_candidates.append((row - 1, hit_col, hit_color))
        wing_candidates.append((row + 1, hit_col, hit_color))

    for r, c, color in wing_candidates:
        if not _in_bounds(out.shape, r, c):
            continue
        if int(g[r, c]) != BACKGROUND:
            continue
        if int(out[r, c]) != BACKGROUND:
            continue
        out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed's horizontal ray, flip its color after crossing the grey separator, and add side wakes at obstacle collisions.

    Input structure:
    - The background is black, with one full grey vertical separator column splitting the grid into left and right regions.
    - The leftmost column contains 2-3 red seed pixels, and the grid also contains several orange single-pixel obstacles away from the separator.

    Rule:
    - Each red seed emits a horizontal ray to the right through background cells only.
    - The ray is red on the separator's left side; it skips the grey separator column itself and continues as green on the right side.
    - The ray stops at the first orange obstacle on its row, keeps that obstacle unchanged, and attempts to paint one-cell side wakes directly above and below the obstacle using the ray's current color.
    - A side wake is drawn only if its target cell is still background; seeds, obstacles, separator cells, and already painted cells block it.

    Key insight:
    - The grey column is not a mirror axis but a color-change boundary: the same ray continues straight through it while switching from red to green, and obstacle side wakes inherit the post-crossing or pre-crossing color.
    """
    return _solve(grid)


def _sample_seed_rows(height: int, count: int) -> list[int] | None:
    rows = list(range(1, height - 1))
    random.shuffle(rows)
    chosen: list[int] = []
    for row in rows:
        if all(abs(row - other) >= 2 for other in chosen):
            chosen.append(row)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _sample_side_pattern(seed_count: int) -> list[str]:
    if seed_count == 2:
        pattern = ["left", "right"]
        random.shuffle(pattern)
        return pattern
    patterns = [
        ["left", "left", "right"],
        ["left", "right", "left"],
        ["left", "right", "right"],
        ["right", "left", "left"],
        ["right", "left", "right"],
        ["right", "right", "left"],
    ]
    return list(random.choice(patterns))


def _random_obstacle_col(side: str, sep: int, width: int) -> int:
    if side == "left":
        return random.randint(2, sep - 2)
    return random.randint(sep + 2, width - 2)


def _obstacle_side(col: int, sep: int) -> str:
    return "left" if col < sep else "right"


def _analyze_instance(grid: np.ndarray) -> dict[str, object]:
    sep = _separator_col(grid)
    seeds = _seed_rows(grid)
    obstacles = [(int(r), int(c)) for r, c in np.argwhere(grid == ORANGE)]
    obstacle_sides = {_obstacle_side(c, sep) for _, c in obstacles}
    hits = {}
    for row in seeds:
        hit_col, hit_color = _first_obstacle_on_row(grid, row)
        hits[row] = (hit_col, hit_color)
    return {
        "sep": sep,
        "seeds": seeds,
        "obstacles": obstacles,
        "obstacle_sides": obstacle_sides,
        "hits": hits,
    }


def _valid_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze_instance(grid)
        out = _solve(grid)
    except AssertionError:
        return False

    seeds = info["seeds"]
    obstacles = info["obstacles"]
    sep = info["sep"]
    hits = info["hits"]
    if not (2 <= len(seeds) <= 3):
        return False
    if not (3 <= len(obstacles) <= 5):
        return False
    if info["obstacle_sides"] != {"left", "right"}:
        return False

    seen_row_side: set[tuple[int, str]] = set()
    for r, c in obstacles:
        if c == sep:
            return False
        key = (r, _obstacle_side(c, sep))
        if key in seen_row_side:
            return False
        seen_row_side.add(key)

    hit_colors = {int(color) for _, color in hits.values()}
    if hit_colors != {RED, GREEN}:
        return False

    left_wakes = 0
    right_wakes = 0
    for row, (hit_col, hit_color) in hits.items():
        for rr in (row - 1, row + 1):
            if _in_bounds(grid.shape, rr, hit_col) and int(out[rr, hit_col]) == hit_color:
                if hit_color == RED:
                    left_wakes += 1
                else:
                    right_wakes += 1
    if left_wakes == 0 or right_wakes == 0:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    sep = random.randint(6, width - 7)
    seed_count = random.choice(SEED_COUNT_CHOICES)
    total_obstacles = random.randint(max(seed_count, TOTAL_OBSTACLE_RANGE[0]), TOTAL_OBSTACLE_RANGE[1])

    seed_rows = _sample_seed_rows(height, seed_count)
    if seed_rows is None:
        return None
    side_pattern = _sample_side_pattern(seed_count)

    grid = np.zeros((height, width), dtype=int)
    grid[:, sep] = GREY
    for row in seed_rows:
        grid[row, 0] = RED

    used_rows = set(seed_rows)
    obstacles: list[tuple[int, int]] = []
    hit_obstacles: list[tuple[int, int, str]] = []

    for row, side in zip(seed_rows, side_pattern):
        col = _random_obstacle_col(side, sep, width)
        grid[row, col] = ORANGE
        obstacles.append((row, col))
        hit_obstacles.append((row, col, side))

    remaining = total_obstacles - len(obstacles)

    blocker_added = False
    if remaining > 0 and random.random() < 0.7:
        candidates = []
        for row, col, _side in hit_obstacles:
            for rr in (row - 1, row + 1):
                if 0 < rr < height - 1 and rr not in used_rows:
                    candidates.append((rr, col))
        random.shuffle(candidates)
        for rr, cc in candidates:
            if grid[rr, cc] == BACKGROUND:
                grid[rr, cc] = ORANGE
                obstacles.append((rr, cc))
                used_rows.add(rr)
                remaining -= 1
                blocker_added = True
                break

    available_rows = [r for r in range(1, height - 1) if r not in used_rows]
    random.shuffle(available_rows)
    for row in available_rows:
        if remaining == 0:
            break
        side = random.choice(("left", "right"))
        col = _random_obstacle_col(side, sep, width)
        if grid[row, col] != BACKGROUND:
            continue
        grid[row, col] = ORANGE
        obstacles.append((row, col))
        remaining -= 1

    if remaining != 0:
        return None
    if not blocker_added and total_obstacles > seed_count and random.random() < 0.3:
        return None
    if not _valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-ray puzzle input")
