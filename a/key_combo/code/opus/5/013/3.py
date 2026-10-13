from __future__ import annotations

import random

import numpy as np


BG = 0
SEED = 1
WALL = 5
OBSTACLE_COLORS = (2, 3, 4, 6, 7)
STRIPE_WIDTH = 3
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _wall_info(grid: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(grid == WALL)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("wall must be a U-shape with non-empty cavity")
    return r0, c0, r1, c1


def _obstacles(grid: np.ndarray, wall_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, _c0, r1, c1 = wall_bbox
    open_rows = set(range(r0 + 1, r1))
    coords = np.argwhere((grid != BG) & (grid != SEED) & (grid != WALL))
    obs = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in coords
        if int(r) in open_rows and int(c) >= int(c1)
    ]
    if not obs:
        raise ValueError("expected at least one obstacle")
    return obs


def _fill_reachable(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    wall_bbox = _wall_info(grid)
    obstacles = _obstacles(grid, wall_bbox)

    shadow_block = np.zeros_like(grid, dtype=bool)
    for r, c, _color in obstacles:
        if c + 1 < grid.shape[1]:
            shadow_block[r, c + 1 :] = True

    passable = ((grid == BG) | (grid == SEED)) & ~shadow_block
    seen = np.zeros_like(passable, dtype=bool)
    queue = [(int(r), int(c)) for r, c in np.argwhere(grid == SEED)]
    for r, c in queue:
        if passable[r, c]:
            seen[r, c] = True

    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        if not seen[r, c]:
            continue
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))

    return seen, obstacles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill through the U-wall opening, measure each obstacle's rightward shadow, and output the obstacle colors as sorted stripes.

    Input structure:
    - The background is black (0); a grey (5) U-shaped wall sits on the left with its opening facing right.
    - A blue (1) seed pixel lies inside the wall cavity.
    - Three to five single-pixel obstacles of distinct colors appear to the right of the opening, each on a different opening-row.

    Rule:
    - Start a 4-connected blue flood-fill from the seed over background cells, but for each obstacle treat every background cell strictly to its right on the same row as blocked shadow.
    - Measure each obstacle's shadow area as the number of unfilled background cells to its right on that row.
    - Sort the obstacle colors by shadow area from largest to smallest and output one 1x3 solid color stripe per obstacle in that order.

    Key insight:
    - The obstacle nearest to the wall opening blocks the longest rightward run, so flood-fill shadow length, not row position or color id, determines the output order.
    """
    g = np.asarray(grid, dtype=int)
    filled, obstacles = _fill_reachable(g)

    ranked: list[tuple[int, int, int, int]] = []
    for r, c, color in obstacles:
        shadow = int(np.count_nonzero((g[r, c + 1 :] == BG) & (~filled[r, c + 1 :])))
        ranked.append((shadow, r, c, color))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    colors = np.array([color for _shadow, _r, _c, color in ranked], dtype=int)
    return np.repeat(colors[:, None], STRIPE_WIDTH, axis=1)


def _expected_output(width: int, obstacles: list[tuple[int, int, int]]) -> np.ndarray:
    ranked = sorted(
        ((width - c - 1, r, c, color) for r, c, color in obstacles),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    colors = np.array([color for _shadow, _r, _c, color in ranked], dtype=int)
    return np.repeat(colors[:, None], STRIPE_WIDTH, axis=1)


def _shortcut_orders(obstacles: list[tuple[int, int, int]]) -> list[list[int]]:
    return [
        [color for r, c, color in sorted(obstacles, key=lambda x: (x[0], x[1]))],
        [color for r, c, color in sorted(obstacles, key=lambda x: (-x[0], x[1]))],
        [color for r, c, color in sorted(obstacles, key=lambda x: (x[2], x[0], x[1]))],
        [color for r, c, color in sorted(obstacles, key=lambda x: (-x[2], x[0], x[1]))],
    ]


def _build_case() -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    n_obs = random.randint(3, 5)
    inner_h = random.randint(n_obs, 7)
    arm_len = random.randint(4, 7)

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_space = random.randint(n_obs + 2, 12)

    r0 = top_margin
    r1 = r0 + inner_h + 1
    c0 = left_margin
    c1 = c0 + arm_len

    height = r1 + bottom_margin + 1
    width = c1 + right_space + 1
    if height > 30 or width > 30:
        return None

    open_rows = list(range(r0 + 1, r1))
    obstacle_rows = random.sample(open_rows, n_obs)
    obstacle_cols = random.sample(list(range(c1, width - 1)), n_obs)
    obstacle_colors = random.sample(list(OBSTACLE_COLORS), n_obs)
    obstacles = [(int(r), int(c), int(color)) for r, c, color in zip(obstacle_rows, obstacle_cols, obstacle_colors)]

    shadows = [width - c - 1 for _r, c, _color in obstacles]
    if len(set(shadows)) != n_obs:
        return None
    if len({r for r, _c, _color in obstacles}) != n_obs or len({c for _r, c, _color in obstacles}) != n_obs:
        return None

    expected = _expected_output(width, obstacles)
    expected_order = expected[:, 0].tolist()
    if any(order == expected_order for order in _shortcut_orders(obstacles)):
        return None

    seed_r = random.choice(open_rows)
    seed_c = random.randint(c0 + 1, c1 - 1)

    grid = np.zeros((height, width), dtype=int)
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[seed_r, seed_c] = SEED
    for r, c, color in obstacles:
        grid[r, c] = color

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if actual.shape != (n_obs, STRIPE_WIDTH):
        return None

    return grid, obstacles


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            grid, _obstacles = built
            return grid
    raise RuntimeError("failed to generate a valid obstacle-shadow sorting input")
