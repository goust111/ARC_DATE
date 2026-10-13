from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
OBSTACLE_COLORS = (1, 2, 4)
COL_LINE = 3
ROW_LINE = 5
START = 9

DIRS = {
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
    "U": (-1, 0),
}
LEFT_TURN = {"R": "U", "U": "L", "L": "D", "D": "R"}
RIGHT_TURN = {"R": "D", "D": "L", "L": "U", "U": "R"}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] != BG)]
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] != BG)]
    return rows, cols


def _line_mask(shape: tuple[int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True
    return mask


def _start_and_direction(grid: np.ndarray, rows: list[int], cols: list[int]) -> tuple[tuple[int, int], str]:
    starts = np.argwhere(grid == START)
    if len(starts) != 1:
        raise ValueError("expected exactly one start marker")
    r, c = map(int, starts[0])
    h, w = grid.shape
    row_set = set(rows)
    col_set = set(cols)

    if r == 0 and c in col_set:
        return (r, c), "D"
    if r == h - 1 and c in col_set:
        return (r, c), "U"
    if c == 0 and r in row_set:
        return (r, c), "R"
    if c == w - 1 and r in row_set:
        return (r, c), "L"
    raise ValueError("start marker is not on a lattice endpoint")


def _obstacles(grid: np.ndarray, rows: list[int], cols: list[int]) -> dict[tuple[int, int], int]:
    row_set = set(rows)
    col_set = set(cols)
    out: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(np.isin(grid, OBSTACLE_COLORS)):
        rr = int(r)
        cc = int(c)
        if rr in row_set and cc in col_set:
            out[(rr, cc)] = int(grid[rr, cc])
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            cells.append((r0, c))
        return cells
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            cells.append((r, c0))
        return cells
    raise ValueError("segment must be axis-aligned")


def _ray_to_boundary(point: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = point
    dr, dc = DIRS[direction]
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return out
        out.append((r, c))
        if r in (0, shape[0] - 1) or c in (0, shape[1] - 1):
            return out


def _intersections_ahead(
    point: tuple[int, int], direction: str, rows: list[int], cols: list[int]
) -> list[tuple[int, int]]:
    r, c = point
    if direction == "R":
        return [(r, cc) for cc in cols if cc > c]
    if direction == "L":
        return [(r, cc) for cc in reversed(cols) if cc < c]
    if direction == "D":
        return [(rr, c) for rr in rows if rr > r]
    return [(rr, c) for rr in reversed(rows) if rr < r]


def _turn(direction: str, collision_index: int, mode: str) -> str:
    if mode == "left":
        return LEFT_TURN[direction]
    if mode == "right":
        return RIGHT_TURN[direction]
    return LEFT_TURN[direction] if collision_index % 2 == 1 else RIGHT_TURN[direction]


def _simulate(
    grid: np.ndarray,
    *,
    turn_mode: str = "alternate",
    inherit_colors: bool = True,
    stop_on_trail: bool = True,
    treat_all_intersections_as_obstacles: bool = False,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    rows, cols = _rows_cols(g)
    row_set = set(rows)
    col_set = set(cols)
    line_mask = _line_mask(g.shape, rows, cols)
    obstacle_map = _obstacles(g, rows, cols)
    (r, c), direction = _start_and_direction(g, rows, cols)
    current_color = START
    out = g.copy()
    visited = {(r, c)}
    hit_colors: list[int] = []
    hit_positions: list[tuple[int, int]] = []
    green_crossings = 0
    stop_reason = "stuck"

    for _ in range(g.size + 5):
        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            stop_reason = "no_line"
            break
        if stop_on_trail and (nr, nc) in visited:
            stop_reason = "trail"
            break
        if not line_mask[nr, nc]:
            stop_reason = "no_line"
            break

        r, c = nr, nc
        is_intersection = r in row_set and c in col_set
        obstacle_color = obstacle_map.get((r, c))
        if obstacle_color is not None or (treat_all_intersections_as_obstacles and is_intersection):
            if obstacle_color is None:
                obstacle_color = int(g[r, c])
            if inherit_colors:
                current_color = int(obstacle_color)
            out[r, c] = int(current_color)
            visited.add((r, c))
            hit_colors.append(int(obstacle_color))
            hit_positions.append((r, c))
            direction = _turn(direction, len(hit_positions), turn_mode)
            tr = r + DIRS[direction][0]
            tc = c + DIRS[direction][1]
            if not _in_bounds(g.shape, tr, tc) or not line_mask[tr, tc]:
                stop_reason = "no_line"
                break
            if stop_on_trail and (tr, tc) in visited:
                stop_reason = "trail"
                break
        else:
            if is_intersection:
                green_crossings += 1
            out[r, c] = int(current_color)
            visited.add((r, c))
            if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1):
                stop_reason = "boundary"
                break
    else:
        stop_reason = "loop_guard"

    return out, {
        "rows": rows,
        "cols": cols,
        "visited": visited,
        "hit_colors": hit_colors,
        "hit_positions": hit_positions,
        "collisions": len(hit_positions),
        "green_crossings": green_crossings,
        "stop_reason": stop_reason,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate a lattice-bound particle that alternates left/right turns and inherits obstacle colors.

    Input structure:
    - Black background with full gray separator rows and full green separator columns that form a lattice.
    - Some row/column intersections are replaced by single-pixel colored obstacles.
    - Exactly one maroon start marker sits at a boundary endpoint of one lattice line.

    Rule:
    - Launch the particle from the start marker toward the grid interior along that lattice line, initially colored maroon.
    - Each step paints the entered lattice cell with the particle's current color.
    - When the particle enters an obstacle intersection, it adopts that obstacle's color and turns 90 degrees:
      odd-numbered collisions turn left, even-numbered collisions turn right.
    - Stop when the particle reaches a boundary cell, when the turned direction is not connected to a lattice cell,
      or when the next step would re-enter its own painted trail.

    Key insight:
    - The separator rows/columns define the only legal motion graph, while collision parity controls the turn choice
      and obstacle colors encode the encounter sequence directly onto the painted trajectory.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _sample_spaced_positions(limit: int, count_range: tuple[int, int], min_gap: int) -> list[int] | None:
    values = list(range(2, limit - 2))
    random.shuffle(values)
    for count in range(random.randint(*count_range), count_range[0] - 1, -1):
        for _ in range(80):
            chosen = sorted(random.sample(values, count))
            if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
                return chosen
    return None


def _choose_start(rows: list[int], cols: list[int], h: int, w: int) -> tuple[tuple[int, int], str]:
    options = []
    for r in rows:
        options.append(((r, 0), "R"))
        options.append(((r, w - 1), "L"))
    for c in cols:
        options.append(((0, c), "D"))
        options.append(((h - 1, c), "U"))
    return random.choice(options)


def _boundary_tail(
    point: tuple[int, int], direction: str, shape: tuple[int, int], visited: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    tail = _ray_to_boundary(point, direction, shape)
    if not tail:
        return None
    if any(cell in visited for cell in tail):
        return None
    return tail


def _trail_tail(
    point: tuple[int, int], direction: str, shape: tuple[int, int], visited: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    tail = _ray_to_boundary(point, direction, shape)
    walked: list[tuple[int, int]] = []
    for cell in tail:
        if cell in visited:
            return walked
        walked.append(cell)
    return None


def _can_stop_no_line(point: tuple[int, int], direction: str, shape: tuple[int, int]) -> bool:
    dr, dc = DIRS[direction]
    nr = point[0] + dr
    nc = point[1] + dc
    return not _in_bounds(shape, nr, nc)


def _plan_path(
    rows: list[int],
    cols: list[int],
    start: tuple[int, int],
    initial_direction: str,
    shape: tuple[int, int],
    n_collisions: int,
    stop_mode: str,
) -> dict | None:
    visited: set[tuple[int, int]] = {start}
    plan: list[tuple[int, int]] = []

    def dfs(
        point: tuple[int, int],
        direction: str,
        collision_index: int,
        crossed_green: int,
    ) -> dict | None:
        if collision_index == n_collisions:
            if stop_mode == "boundary":
                tail = _boundary_tail(point, direction, shape, visited)
                if tail is None:
                    return None
            elif stop_mode == "trail":
                tail = _trail_tail(point, direction, shape, visited)
                if tail is None:
                    return None
            else:
                if not _can_stop_no_line(point, direction, shape):
                    return None
                tail = []
            return {
                "obstacles": plan.copy(),
                "tail": tail,
                "green_crossings": crossed_green,
            }

        candidates = _intersections_ahead(point, direction, rows, cols)
        random.shuffle(candidates)
        candidates.sort(key=lambda p: len(_segment_cells(point, p)), reverse=True)

        for target in candidates:
            if (
                stop_mode == "no_line"
                and collision_index + 1 < n_collisions - 1
                and (target[0] in (0, shape[0] - 1) or target[1] in (0, shape[1] - 1))
            ):
                continue
            segment = _segment_cells(point, target)
            min_len = 1 if (
                stop_mode == "no_line"
                and collision_index + 1 == n_collisions
                and (target[0] in (0, shape[0] - 1) or target[1] in (0, shape[1] - 1))
            ) else 2
            if len(segment) < min_len:
                continue
            if any(cell in visited for cell in segment):
                continue

            intersection_crossings = 0
            if point[0] == target[0]:
                lo, hi = sorted((point[1], target[1]))
                intersection_crossings = sum(lo < c < hi for c in cols)
            else:
                lo, hi = sorted((point[0], target[0]))
                intersection_crossings = sum(lo < r < hi for r in rows)

            for cell in segment:
                visited.add(cell)
            plan.append(target)
            next_direction = _turn(direction, collision_index + 1, "alternate")

            if collision_index + 1 < n_collisions:
                next_targets = _intersections_ahead(target, next_direction, rows, cols)
                if next_targets:
                    found = dfs(target, next_direction, collision_index + 1, crossed_green + intersection_crossings)
                    if found is not None:
                        return found
            else:
                found = dfs(target, next_direction, collision_index + 1, crossed_green + intersection_crossings)
                if found is not None:
                    return found

            plan.pop()
            for cell in segment:
                visited.remove(cell)

        return None

    return dfs(start, initial_direction, 0, 0)


def _assign_obstacle_colors(n_collisions: int) -> list[int]:
    while True:
        colors = [random.choice(OBSTACLE_COLORS)]
        for _ in range(1, n_collisions):
            if random.random() < 0.7:
                choices = [c for c in OBSTACLE_COLORS if c != colors[-1]]
                colors.append(random.choice(choices))
            else:
                colors.append(random.choice(OBSTACLE_COLORS))
        if len(set(colors)) >= 2:
            return colors


def _render_input(
    h: int,
    w: int,
    rows: list[int],
    cols: list[int],
    start: tuple[int, int],
    obstacles: list[tuple[int, int]],
    obstacle_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[rows, :] = ROW_LINE
    grid[:, cols] = COL_LINE
    for (r, c), color in zip(obstacles, obstacle_colors):
        grid[r, c] = int(color)
    grid[start] = START
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    out, meta = _simulate(grid)
    assert meta["collisions"] >= 3
    assert len(meta["hit_positions"]) == len(set(meta["hit_positions"]))
    assert meta["stop_reason"] in {"boundary", "no_line"}
    assert meta["green_crossings"] >= 1
    assert len(set(meta["hit_colors"])) >= 2
    assert len(meta["visited"]) >= 10
    assert not np.array_equal(out, grid)

    assert not np.array_equal(out, _simulate(grid, turn_mode="left")[0])
    assert not np.array_equal(out, _simulate(grid, turn_mode="right")[0])
    assert not np.array_equal(out, _simulate(grid, inherit_colors=False)[0])
    assert not np.array_equal(out, _simulate(grid, treat_all_intersections_as_obstacles=True)[0])

    rows, cols = _rows_cols(grid)
    row_set = set(rows)
    col_set = set(cols)
    for r, c in np.argwhere(np.isin(grid, OBSTACLE_COLORS)):
        assert int(r) in row_set and int(c) in col_set


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(13, 19)
    w = random.randint(13, 19)
    rows = _sample_spaced_positions(h, (3, 5), min_gap=2)
    cols = _sample_spaced_positions(w, (3, 5), min_gap=2)
    if rows is None or cols is None:
        return None

    start, initial_direction = _choose_start(rows, cols, h, w)
    stop_mode = random.choices(["boundary", "no_line"], weights=[3, 2])[0]
    if stop_mode == "no_line":
        if initial_direction == "R":
            rows = sorted(set(rows) | {0})
            base = random.randint(2, max(2, w - 3))
            cols = sorted(set(cols) | {base, min(w - 1, base + 1)})
        elif initial_direction == "L":
            rows = sorted(set(rows) | {h - 1})
            base = random.randint(1, max(1, w - 4))
            cols = sorted(set(cols) | {base, base + 1})
        elif initial_direction == "D":
            cols = sorted(set(cols) | {w - 1})
            base = random.randint(2, max(2, h - 3))
            rows = sorted(set(rows) | {base, min(h - 1, base + 1)})
        else:
            cols = sorted(set(cols) | {0})
            base = random.randint(1, max(1, h - 4))
            rows = sorted(set(rows) | {base, base + 1})
        n_collisions = 5
    else:
        n_collisions = random.randint(3, 6)
    plan = _plan_path(rows, cols, start, initial_direction, (h, w), n_collisions, stop_mode)
    if plan is None:
        return None
    if int(plan["green_crossings"]) < 1:
        return None

    obstacle_colors = _assign_obstacle_colors(n_collisions)
    grid = _render_input(h, w, rows, cols, start, plan["obstacles"], obstacle_colors)

    out, meta = _simulate(grid)
    if meta["collisions"] != n_collisions:
        return None
    if len(meta["hit_positions"]) != len(plan["obstacles"]):
        return None
    if Counter(meta["hit_positions"]) != Counter(plan["obstacles"]):
        return None

    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        made = _generate_attempt()
        if made is not None:
            return made
    raise RuntimeError("failed to generate a valid lattice-path instance")
