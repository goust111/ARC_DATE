from __future__ import annotations

import random
from collections import Counter

import numpy as np


RED = 2
GREY = 5
PINK = 6
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
BACKGROUND_CHOICES = [0, 1, 3, 4, 7, 8, 9]
BRANCH_TYPES = ["row", "col", "corner"]
GEN_RETRIES = 400


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _find_marker(g: np.ndarray) -> Coord:
    coords = np.argwhere(g == RED)
    assert coords.shape[0] == 1, "expected exactly one red marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _extract_tail(g: np.ndarray, marker: Coord) -> tuple[Coord, list[Coord]]:
    mr, mc = marker
    adjacent: list[Coord] = []
    for dr, dc in DIAGONALS:
        nr, nc = mr + dr, mc + dc
        if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == PINK:
            adjacent.append((dr, dc))
    assert len(adjacent) == 1, "expected one diagonal pink tail attached to the marker"

    dr, dc = adjacent[0]
    tail: list[Coord] = []
    r, c = mr + dr, mc + dc
    while _in_bounds(g.shape, r, c) and int(g[r, c]) == PINK:
        tail.append((r, c))
        r += dr
        c += dc

    pink_coords = {tuple(map(int, coord)) for coord in np.argwhere(g == PINK)}
    assert pink_coords == set(tail), "pink cells must form one straight diagonal tail"
    return (dr, dc), tail


def _reflected_step(pos: Coord, direction: Coord, shape: tuple[int, int]) -> tuple[Coord, Coord, bool, bool]:
    h, w = map(int, shape)
    r, c = pos
    dr, dc = direction
    nr, nc = r + dr, c + dc

    bounce_row = nr < 0 or nr >= h
    bounce_col = nc < 0 or nc >= w
    if bounce_row:
        dr = -dr
    if bounce_col:
        dc = -dc

    nr, nc = r + dr, c + dc
    assert _in_bounds(shape, nr, nc), "reflected step must land inside the grid"
    return (nr, nc), (dr, dc), bounce_row, bounce_col


def _trace_candidates(
    shape: tuple[int, int],
    marker: Coord,
    ray_dir: Coord,
    blocked: set[Coord],
) -> list[dict[str, object]]:
    r, c = marker
    dr, dc = ray_dir
    seen_states = {(r, c, dr, dc)}
    visited_path: set[Coord] = set()
    first_bounce: str | None = None
    bounce_count = 0
    records: list[dict[str, object]] = []

    for _ in range(shape[0] * shape[1] * 8):
        (nr, nc), (dr, dc), bounce_row, bounce_col = _reflected_step((r, c), (dr, dc), shape)
        if bounce_row or bounce_col:
            bounce_count += 1
            bounce_kind = "corner" if (bounce_row and bounce_col) else ("row" if bounce_row else "col")
            if first_bounce is None:
                first_bounce = bounce_kind

        cell = (nr, nc)
        if cell in blocked or cell in visited_path:
            break

        records.append(
            {
                "pos": cell,
                "first_bounce": first_bounce,
                "bounce_count": bounce_count,
            }
        )
        visited_path.add(cell)
        r, c = cell
        state = (r, c, dr, dc)
        if state in seen_states:
            break
        seen_states.add(state)

    return records


def _apply_ray(grid: np.ndarray, marker: Coord, ray_dir: Coord, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    pos = marker
    direction = ray_dir

    for _ in range(out.shape[0] * out.shape[1] * 8):
        nxt, direction, _bounce_row, _bounce_col = _reflected_step(pos, direction, out.shape)
        nr, nc = nxt
        if int(out[nr, nc]) != int(bg):
            out[nr, nc] = RED
            return out
        out[nr, nc] = PINK
        pos = nxt

    raise AssertionError("ray never hit a non-background cell")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a pink diagonal ray opposite the tail, reflecting off borders until the first obstacle turns red.

    Input structure:
    - A single red marker pixel sits next to a straight diagonal pink tail that indicates the incoming direction.
    - Grey obstacle pixels are scattered on a uniform background.
    - The ray must travel opposite the tail and reaches an obstacle only after at least one border reflection.

    Rule:
    - Read the diagonal direction from the pink tail cell adjacent to the red marker.
    - Launch a pink diagonal ray from the marker in the opposite direction, reflecting on grid borders like a mirror;
      if both coordinates would leave the grid at once, the ray reverses along the same diagonal.
    - Paint each traversed background cell pink, and stop when the next cell is any non-background pixel; recolor that hit cell red.

    Key insight:
    - The tail gives the ray direction, but the target is found only by simulating diagonal mirror reflections on the canvas boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    marker = _find_marker(g)
    tail_dir, _tail = _extract_tail(g, marker)
    ray_dir = (-tail_dir[0], -tail_dir[1])
    return _apply_ray(g, marker, ray_dir, bg)


def _distance_to_boundary(index: int, limit: int, delta: int) -> int:
    return int(index) if int(delta) < 0 else int(limit) - 1 - int(index)


def _marker_options(height: int, width: int, ray_dir: Coord, branch: str) -> list[Coord]:
    dr, dc = ray_dir
    options: list[Coord] = []
    for r in range(height):
        for c in range(width):
            row_dist = _distance_to_boundary(r, height, dr)
            col_dist = _distance_to_boundary(c, width, dc)
            if branch == "row" and row_dist >= col_dist:
                continue
            if branch == "col" and col_dist >= row_dist:
                continue
            if branch == "corner" and row_dist != col_dist:
                continue

            tail_dr, tail_dc = -dr, -dc
            tail_room = min(
                _distance_to_boundary(r, height, tail_dr),
                _distance_to_boundary(c, width, tail_dc),
            )
            if tail_room >= 1:
                options.append((r, c))
    return options


def _can_place_far(cell: Coord, others: set[Coord], min_chebyshev: int) -> bool:
    r, c = cell
    for rr, cc in others:
        if max(abs(r - rr), abs(c - cc)) < min_chebyshev:
            return False
    return True


def _build_input(
    shape: tuple[int, int],
    bg: int,
    marker: Coord,
    tail_cells: list[Coord],
    stop_cell: Coord,
    blocked_prefix: set[Coord],
) -> np.ndarray | None:
    grid = np.full(shape, bg, dtype=int)
    grid[marker] = RED
    for r, c in tail_cells:
        grid[r, c] = PINK
    grid[stop_cell] = GREY

    occupied = {marker, stop_cell, *tail_cells}
    forbidden = occupied | blocked_prefix
    extra_obstacles = random.randint(0, 4)
    for _ in range(extra_obstacles):
        candidates = [
            (r, c)
            for r in range(shape[0])
            for c in range(shape[1])
            if (r, c) not in forbidden and _can_place_far((r, c), occupied, 2)
        ]
        if not candidates:
            break
        cell = random.choice(candidates)
        grid[cell] = GREY
        occupied.add(cell)
        forbidden.add(cell)

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(8, 18)
    width = random.randint(8, 18)
    bg = random.choice(BACKGROUND_CHOICES)
    branch = random.choice(BRANCH_TYPES)
    ray_dir = random.choice(DIAGONALS)

    markers = _marker_options(height, width, ray_dir, branch)
    if not markers:
        return None
    marker = random.choice(markers)

    tail_dr, tail_dc = -ray_dir[0], -ray_dir[1]
    tail_room = min(
        _distance_to_boundary(marker[0], height, tail_dr),
        _distance_to_boundary(marker[1], width, tail_dc),
    )
    tail_len = random.randint(1, min(4, tail_room))
    tail_cells = [(marker[0] + tail_dr * step, marker[1] + tail_dc * step) for step in range(1, tail_len + 1)]

    blocked = {marker, *tail_cells}
    records = _trace_candidates((height, width), marker, ray_dir, blocked)
    candidates = [
        (index, rec)
        for index, rec in enumerate(records)
        if rec["bounce_count"] >= 1 and rec["first_bounce"] == branch
    ]
    if not candidates:
        return None

    later = [item for item in candidates if item[0] >= 3]
    stop_index, stop_record = random.choice(later if later else candidates)
    stop_cell = stop_record["pos"]
    blocked_prefix = {rec["pos"] for rec in records[:stop_index]}

    grid = _build_input((height, width), bg, marker, tail_cells, stop_cell, blocked_prefix)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
