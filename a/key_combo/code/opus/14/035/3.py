import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
ORANGE = 7

DIAGONAL_DIRS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
GRID_ROWS_RANGE = (13, 19)
GRID_COLS_RANGE = (13, 19)
DIAGONAL_STEPS_RANGE = (2, 6)
VISIBLE_STRAIGHT_STEPS_RANGE = (1, 5)
MAX_GENERATION_ATTEMPTS = 500


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _path_segments(blue: tuple[int, int], red: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    blue_r, blue_c = blue
    red_r, red_c = red
    step_r = _sign(red_r - blue_r)
    step_c = _sign(red_c - blue_c)
    row_dist = abs(red_r - blue_r)
    col_dist = abs(red_c - blue_c)

    assert row_dist > 0 and col_dist > 0, "blue and red must differ in both row and column"
    assert row_dist != col_dist, "the route must include both diagonal and straight motion"

    diag_steps = min(row_dist, col_dist)
    straight_steps = abs(row_dist - col_dist) - 1

    diag_cells: list[tuple[int, int]] = []
    straight_cells: list[tuple[int, int]] = []
    r, c = blue

    for _ in range(diag_steps):
        r += step_r
        c += step_c
        diag_cells.append((r, c))

    if row_dist > col_dist:
        for _ in range(straight_steps):
            r += step_r
            straight_cells.append((r, c))
    else:
        for _ in range(straight_steps):
            c += step_c
            straight_cells.append((r, c))

    full_path = diag_cells + straight_cells
    assert len(full_path) == max(row_dist, col_dist) - 1
    assert full_path[-1] in {
        (red_r - step_r, red_c),
        (red_r, red_c - step_c),
    }, "path must end one step before the red endpoint"
    return diag_cells, straight_cells


def _trace_ray(
    shape: tuple[int, int],
    blockers: dict[tuple[int, int], int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict:
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []

    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return {
                "direction": direction,
                "cells": cells,
                "stop": "edge",
                "blocker": None,
                "blocker_color": None,
            }
        if (nr, nc) in blockers:
            return {
                "direction": direction,
                "cells": cells,
                "stop": "blocked",
                "blocker": (nr, nc),
                "blocker_color": blockers[(nr, nc)],
            }
        cells.append((nr, nc))
        r, c = nr, nc


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN}, "unexpected input colors"
    assert int(np.count_nonzero(g)) == 3, "expected exactly three non-background pixels"

    blue = _single_coord(g, BLUE)
    red = _single_coord(g, RED)
    green = _single_coord(g, GREEN)

    assert len({blue[0], red[0], green[0]}) == 3, "the three markers must lie on distinct rows"
    assert len({blue[1], red[1], green[1]}) == 3, "the three markers must lie on distinct columns"

    diag_cells, straight_cells = _path_segments(blue, red)
    path_cells = diag_cells + straight_cells
    assert green not in path_cells, "the green emitter must not lie on the future path"

    out = g.copy()
    for r, c in path_cells:
        assert int(out[r, c]) == BLACK, "the gray path may only occupy background cells"
        out[r, c] = GRAY

    blockers = {blue: BLUE, red: RED, green: GREEN}
    blockers.update({cell: GRAY for cell in path_cells})

    rays = [_trace_ray(out.shape, blockers, green, direction) for direction in DIAGONAL_DIRS]
    for ray in rays:
        for r, c in ray["cells"]:
            assert int(out[r, c]) == BLACK, "orange rays may only occupy background cells"
            out[r, c] = ORANGE

    return out, diag_cells, straight_cells, rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the blue-red pair with a Chebyshev path, then let that path block diagonal rays from the green emitter.

    Input structure:
    - The grid is black background with exactly one blue pixel, one red pixel, and one green pixel.
    - The blue and red pixels define a unique unequal-axis king-move route, and all three markers occupy different rows and columns.

    Rule:
    - Preserve the three input markers.
    - Draw the blue-to-red path in gray by taking all diagonal steps toward the red pixel first, then the remaining straight steps along the longer axis, excluding both endpoints.
    - From the green pixel, extend orange rays in the four diagonal directions, stopping before the first non-black cell or the grid edge.

    Key insight:
    - The same gray structure has two jobs at once: it is the connector between blue and red, and its exact bend geometry is also the obstacle that truncates the diagonal radiation pattern.
    """
    return _solve_details(grid)[0]


def _line_side(blue: tuple[int, int], red: tuple[int, int], point: tuple[int, int]) -> int:
    return (red[0] - blue[0]) * (point[1] - blue[1]) - (red[1] - blue[1]) * (point[0] - blue[0])


def _candidate_green_positions(
    shape: tuple[int, int],
    blue: tuple[int, int],
    red: tuple[int, int],
    diag_cells: list[tuple[int, int]],
    straight_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    path_set = set(diag_cells) | set(straight_cells)
    candidates: list[tuple[int, int]] = []

    for r in range(shape[0]):
        for c in range(shape[1]):
            point = (r, c)
            if point in path_set or point == blue or point == red:
                continue
            if r in {blue[0], red[0]} or c in {blue[1], red[1]}:
                continue
            if _line_side(blue, red, point) == 0:
                continue
            candidates.append(point)
    return candidates


def _semantic_checks(grid: np.ndarray) -> None:
    out, diag_cells, straight_cells, rays = _solve_details(grid)
    blue = _single_coord(grid, BLUE)
    red = _single_coord(grid, RED)
    green = _single_coord(grid, GREEN)

    assert diag_cells, "generated tasks must show a visible diagonal path segment"
    assert straight_cells, "generated tasks must show a visible straight path segment"
    assert int(out[blue]) == BLUE and int(out[red]) == RED and int(out[green]) == GREEN
    assert int(np.count_nonzero(out == GRAY)) == len(diag_cells) + len(straight_cells)
    assert int(np.count_nonzero(out == ORANGE)) == sum(len(ray["cells"]) for ray in rays)

    diag_set = set(diag_cells)
    straight_set = set(straight_cells)
    blocked_by_path = [
        ray
        for ray in rays
        if ray["stop"] == "blocked" and ray["blocker_color"] == GRAY
    ]
    edge_rays = [ray for ray in rays if ray["stop"] == "edge" and len(ray["cells"]) > 0]

    assert len(blocked_by_path) >= 2, "at least two diagonal rays must be truncated by the gray path"
    assert edge_rays, "at least one diagonal ray must reach the grid edge"
    assert any(len(ray["cells"]) >= 2 for ray in edge_rays), "an unblocked ray should be visibly long"
    assert any(ray["blocker"] in diag_set for ray in blocked_by_path), "the diagonal path segment must matter"
    assert any(ray["blocker"] in straight_set for ray in blocked_by_path), "the straight path segment must matter"
    assert sum(len(ray["cells"]) for ray in rays) >= 5, "the output should contain a visible ray pattern"


def _sample_endpoints() -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)

    longer_axis = random.choice(("row", "col"))
    diag_len = random.randint(*DIAGONAL_STEPS_RANGE)
    visible_straight = random.randint(*VISIBLE_STRAIGHT_STEPS_RANGE)
    row_dist = diag_len + visible_straight + 1 if longer_axis == "row" else diag_len
    col_dist = diag_len if longer_axis == "row" else diag_len + visible_straight + 1

    if row_dist >= h - 1 or col_dist >= w - 1:
        return None

    row_sign = random.choice((-1, 1))
    col_sign = random.choice((-1, 1))

    blue_r_min = 1 if row_sign > 0 else row_dist + 1
    blue_r_max = h - row_dist - 2 if row_sign > 0 else h - 2
    blue_c_min = 1 if col_sign > 0 else col_dist + 1
    blue_c_max = w - col_dist - 2 if col_sign > 0 else w - 2
    if blue_r_min > blue_r_max or blue_c_min > blue_c_max:
        return None

    blue = (
        random.randint(blue_r_min, blue_r_max),
        random.randint(blue_c_min, blue_c_max),
    )
    red = (blue[0] + row_sign * row_dist, blue[1] + col_sign * col_dist)
    return h, w, blue, red


def _pick_green(
    h: int,
    w: int,
    blue: tuple[int, int],
    red: tuple[int, int],
    diag_cells: list[tuple[int, int]],
    straight_cells: list[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = _candidate_green_positions((h, w), blue, red, diag_cells, straight_cells)
    random.shuffle(candidates)

    scored: list[tuple[int, int]] = []
    for green in candidates:
        grid = np.zeros((h, w), dtype=int)
        grid[blue] = BLUE
        grid[red] = RED
        grid[green] = GREEN

        try:
            out, diag_cells2, straight_cells2, rays = _solve_details(grid)
        except AssertionError:
            continue

        diag_set = set(diag_cells2)
        straight_set = set(straight_cells2)
        blocked_by_path = [
            ray
            for ray in rays
            if ray["stop"] == "blocked" and ray["blocker_color"] == GRAY
        ]
        if len(blocked_by_path) < 2:
            continue
        if not any(ray["blocker"] in diag_set for ray in blocked_by_path):
            continue
        if not any(ray["blocker"] in straight_set for ray in blocked_by_path):
            continue

        edge_lengths = [len(ray["cells"]) for ray in rays if ray["stop"] == "edge"]
        if not edge_lengths or max(edge_lengths) < 2:
            continue

        total_orange = int(np.count_nonzero(out == ORANGE))
        if total_orange < 5:
            continue

        score = (
            len(blocked_by_path),
            max(edge_lengths),
            total_orange,
            random.random(),
        )
        scored.append((score, green))

    if not scored:
        return None
    scored.sort(reverse=True)
    top_choices = [green for _score, green in scored[: min(8, len(scored))]]
    return random.choice(top_choices)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_endpoints()
        if sampled is None:
            continue
        h, w, blue, red = sampled

        diag_cells, straight_cells = _path_segments(blue, red)
        green = _pick_green(h, w, blue, red, diag_cells, straight_cells)
        if green is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        grid[blue] = BLUE
        grid[red] = RED
        grid[green] = GREEN

        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid chebyshev-path radiation task")
