from __future__ import annotations

import random

import numpy as np


AZURE = 8
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GRID_H_RANGE = (10, 24)
GRID_W_RANGE = (10, 24)
PAIR_COUNT_RANGE = (3, 7)
MARGIN = 2
MIN_ENDPOINT_GAP = 2
MAX_UNBLOCKED_SPAN = 11
MAX_BLOCKED_SIDE_SPAN = 9
PLACE_RETRIES = 300
GENERATE_RETRIES = 500

Coord = tuple[int, int]


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("paired pixels must share a row or a column")


def _pair_infos(grid: np.ndarray) -> list[tuple[int, Coord, Coord, list[Coord], bool]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != AZURE]
    assert colors, "expected at least one foreground color"

    infos: list[tuple[int, Coord, Coord, list[Coord], bool]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        assert a[0] == b[0] or a[1] == b[1], "same-color pixels must be orthogonally aligned"

        cells = _segment_cells(a, b)
        interior = cells[1:-1]
        blocked = any(int(grid[r, c]) != AZURE for r, c in interior)
        infos.append((color, a, b, interior, blocked))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect unobstructed same-color pixel pairs and erase pairs whose straight channel is blocked.

    Input structure:
    - The background is Azure.
    - Each foreground color appears as exactly two single pixels.
    - The two pixels of every foreground color lie in one row or one column, and the cells between them may contain endpoint pixels of other colors as obstacles.

    Rule:
    - For each foreground color, inspect the open horizontal or vertical segment between its two pixels in the original input.
    - If every intervening cell is Azure, recolor those intervening Azure cells with the pair color to form a straight connector.
    - If any intervening cell is a non-Azure foreground pixel, do not draw the connector and recolor the two endpoints of that blocked pair to Azure.
    - Obstacle pixels of other colors are otherwise left unchanged, except when their own color pair is also blocked by this same rule.

    Key insight:
    - The visible same-color pairing is simple; the important decision is whether the original channel is passable, with blocked pairs being removed instead of merely left unconnected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert AZURE in set(map(int, np.unique(g))), "expected Azure background"

    infos = _pair_infos(g)
    out = g.copy()

    for _color, a, b, _interior, blocked in infos:
        if blocked:
            out[a] = AZURE
            out[b] = AZURE

    for color, _a, _b, interior, blocked in infos:
        if not blocked:
            for r, c in interior:
                assert int(g[r, c]) == AZURE, "unblocked channels must pass only through Azure"
                out[r, c] = color

    return out


# --------------- generator ---------------


def _place_unblocked_pair(
    grid: np.ndarray,
    color: int,
    occupied: set[Coord],
    protected: set[Coord],
    rng: random.Random,
) -> tuple[Coord, Coord] | None:
    height, width = grid.shape
    for _ in range(PLACE_RETRIES):
        horizontal = rng.choice((True, False))
        if horizontal:
            max_gap = min(MAX_UNBLOCKED_SPAN, width - 2 * MARGIN - 1)
            if max_gap < MIN_ENDPOINT_GAP:
                continue
            gap = rng.randint(MIN_ENDPOINT_GAP, max_gap)
            row = rng.randint(MARGIN, height - MARGIN - 1)
            col0 = rng.randint(MARGIN, width - MARGIN - 1 - gap)
            a, b = (row, col0), (row, col0 + gap)
        else:
            max_gap = min(MAX_UNBLOCKED_SPAN, height - 2 * MARGIN - 1)
            if max_gap < MIN_ENDPOINT_GAP:
                continue
            gap = rng.randint(MIN_ENDPOINT_GAP, max_gap)
            row0 = rng.randint(MARGIN, height - MARGIN - 1 - gap)
            col = rng.randint(MARGIN, width - MARGIN - 1)
            a, b = (row0, col), (row0 + gap, col)

        cells = _segment_cells(a, b)
        interior = set(cells[1:-1])
        endpoints = {a, b}
        if endpoints & occupied or endpoints & protected:
            continue
        if interior & occupied or interior & protected:
            continue

        grid[a] = color
        grid[b] = color
        occupied.update(endpoints)
        protected.update(interior)
        return a, b
    return None


def _place_blocked_pair(
    grid: np.ndarray,
    color: int,
    obstacles: list[Coord],
    obstacle_use: dict[Coord, int],
    occupied: set[Coord],
    protected: set[Coord],
    rng: random.Random,
) -> tuple[Coord, Coord] | None:
    height, width = grid.shape
    shuffled_obstacles = list(obstacles)
    rng.shuffle(shuffled_obstacles)

    for obstacle in shuffled_obstacles:
        if obstacle_use.get(obstacle, 0) >= 1:
            continue
        orow, ocol = obstacle
        for _ in range(PLACE_RETRIES):
            horizontal_options = []
            if ocol >= MIN_ENDPOINT_GAP and width - 1 - ocol >= MIN_ENDPOINT_GAP:
                horizontal_options.append(True)
            if orow >= MIN_ENDPOINT_GAP and height - 1 - orow >= MIN_ENDPOINT_GAP:
                horizontal_options.append(False)
            if not horizontal_options:
                break

            horizontal = rng.choice(horizontal_options)
            if horizontal:
                left = rng.randint(MIN_ENDPOINT_GAP, min(ocol, MAX_BLOCKED_SIDE_SPAN))
                right = rng.randint(MIN_ENDPOINT_GAP, min(width - 1 - ocol, MAX_BLOCKED_SIDE_SPAN))
                a, b = (orow, ocol - left), (orow, ocol + right)
            else:
                up = rng.randint(MIN_ENDPOINT_GAP, min(orow, MAX_BLOCKED_SIDE_SPAN))
                down = rng.randint(MIN_ENDPOINT_GAP, min(height - 1 - orow, MAX_BLOCKED_SIDE_SPAN))
                a, b = (orow - up, ocol), (orow + down, ocol)

            cells = _segment_cells(a, b)
            interior = set(cells[1:-1])
            endpoints = {a, b}
            if obstacle not in interior:
                continue
            if endpoints & occupied or endpoints & protected:
                continue
            if (interior - {obstacle}) & protected:
                continue
            if (interior & occupied) != {obstacle}:
                continue

            grid[a] = color
            grid[b] = color
            occupied.update(endpoints)
            obstacle_use[obstacle] = obstacle_use.get(obstacle, 0) + 1
            return a, b
    return None


def _has_balanced_branches(grid: np.ndarray) -> bool:
    try:
        infos = _pair_infos(grid)
        out = main(grid)
    except Exception:
        return False

    blocked_by_color = {color: blocked for color, _a, _b, _interior, blocked in infos}
    blocked_count = sum(1 for blocked in blocked_by_color.values() if blocked)
    connected_count = len(infos) - blocked_count
    if blocked_count == 0 or connected_count == 0:
        return False
    if np.array_equal(out, grid):
        return False

    for color, _a, _b, interior, blocked in infos:
        if not blocked:
            if not interior:
                return False
            continue
        obstacle_colors = {int(grid[r, c]) for r, c in interior if int(grid[r, c]) != AZURE}
        if not obstacle_colors:
            return False
        if not any(not blocked_by_color.get(obstacle_color, True) for obstacle_color in obstacle_colors):
            return False
        assert blocked_by_color[color]
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    n_pairs = rng.randint(*PAIR_COUNT_RANGE)
    blocked_count = n_pairs // 2
    unblocked_count = n_pairs - blocked_count
    colors = rng.sample(FOREGROUND_COLORS, n_pairs)

    grid = np.full((height, width), AZURE, dtype=int)
    occupied: set[Coord] = set()
    protected: set[Coord] = set()
    unblocked_endpoints: list[Coord] = []

    for color in colors[:unblocked_count]:
        pair = _place_unblocked_pair(grid, color, occupied, protected, rng)
        if pair is None:
            return None
        unblocked_endpoints.extend(pair)

    obstacle_use: dict[Coord, int] = {}
    for color in colors[unblocked_count:]:
        pair = _place_blocked_pair(
            grid,
            color,
            unblocked_endpoints,
            obstacle_use,
            occupied,
            protected,
            rng,
        )
        if pair is None:
            return None

    if not _has_balanced_branches(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
