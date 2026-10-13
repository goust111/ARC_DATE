from __future__ import annotations

import random
from typing import Iterable

import numpy as np

BG = 7
MAGENTA = 6
RED = 2
YELLOW = 4
PASSABLE = {BG, MAGENTA, RED}
WALL_PALETTE = (1, 3, 5, 8, 9)

RIGHT = (0, 1)
LEFT = (0, -1)
UP = (-1, 0)
DOWN = (1, 0)
PERPENDICULAR = {
    RIGHT: (UP, DOWN),
    LEFT: (UP, DOWN),
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
}

MAX_GEN_ATTEMPTS = 400
HEIGHT_OPTIONS = (13, 15, 17, 19)
WIDTH_RANGE = (15, 20)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_wall(value: int) -> bool:
    return int(value) not in PASSABLE


def _turn_candidate(grid: np.ndarray, r: int, c: int, direction: tuple[int, int]) -> bool:
    """Return True iff the candidate cell is a bg corridor entrance for `direction`."""
    h, w = grid.shape
    if not _in_bounds((h, w), r, c) or int(grid[r, c]) != BG:
        return False
    if direction in (LEFT, RIGHT):
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and _is_wall(a)


def _trace_path(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Trace one particle using the puzzle's local deterministic movement rule."""
    h, w = grid.shape
    r, c = start
    direction = RIGHT
    path = [(r, c)]
    seen_states = {(r, c, direction)}
    limit = h * w * 8

    for _ in range(limit):
        dr, dc = direction
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            break
        if int(grid[nr, nc]) in PASSABLE:
            r, c = nr, nc
            path.append((r, c))
            state = (r, c, direction)
            if state in seen_states:
                raise AssertionError("particle entered a movement loop")
            seen_states.add(state)
            continue

        options: list[tuple[int, int, tuple[int, int]]] = []
        for turn_dir in PERPENDICULAR[direction]:
            tr, tc = r + turn_dir[0], c + turn_dir[1]
            if _turn_candidate(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn"
        if not options:
            break
        r, c, direction = options[0]
        path.append((r, c))
        state = (r, c, direction)
        if state in seen_states:
            raise AssertionError("particle entered a turn loop")
        seen_states.add(state)

    return path


def _path_index(path: Iterable[tuple[int, int]]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for i, cell in enumerate(path):
        if cell in out:
            raise AssertionError("intended paths must not self-intersect")
        out[cell] = i
    return out


def _crossings(path_a: list[tuple[int, int]], path_b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen_b = set(path_b)
    return [cell for cell in path_a if cell in seen_b]


def main(grid: np.ndarray) -> np.ndarray:
    """Trace two corridor particles, then paint a yellow ring around their unique path crossing.

    Input structure:
    - Background is orange (7). Exactly one magenta particle (6) and one red particle (2) start in column 0 on two different odd rows.
    - Even rows contain mostly colored wall bars with carved gaps; odd rows are mostly orange corridor rows, with a few odd-row blocker cells that force turns.
    - Valid turn entrances are orange cells whose perpendicular flanking cells are same-color wall bars, and the eventual shared crossing lies away from the border.

    Rule:
    - Trace each particle independently from its start while moving right through passable cells; when the next cell is a wall, turn 90 degrees only into the unique perpendicular orange cell that is flanked by same-color walls.
    - Paint every visited orange cell with that particle's color, then locate all cells visited by both traces.
    - For each crossing cell, compare the inclusive walked path length from each start to that crossing; the crossing cell keeps the longer-path particle color.
    - Around each crossing, recolor every non-wall Chebyshev-1 neighbor that is orange or a particle-trail color to yellow (4), while wall-bar cells stay unchanged.

    Key insight:
    - The center color depends on walked corridor length, not on geometric proximity, so the solver must follow both full traces before deciding the crossing marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    starts = {
        MAGENTA: [tuple(map(int, rc)) for rc in np.argwhere(g == MAGENTA)],
        RED: [tuple(map(int, rc)) for rc in np.argwhere(g == RED)],
    }
    assert len(starts[MAGENTA]) == 1 and len(starts[RED]) == 1

    magenta_path = _trace_path(g, starts[MAGENTA][0])
    red_path = _trace_path(g, starts[RED][0])
    magenta_idx = _path_index(magenta_path)
    red_idx = _path_index(red_path)

    out = g.copy()
    for color, path in ((MAGENTA, magenta_path), (RED, red_path)):
        for r, c in path:
            if int(g[r, c]) == BG:
                out[r, c] = color

    crossings = _crossings(magenta_path, red_path)
    for r, c in crossings:
        len_magenta = magenta_idx[(r, c)] + 1
        len_red = red_idx[(r, c)] + 1
        assert len_magenta != len_red, "crossing path lengths must be unequal"
        out[r, c] = MAGENTA if len_magenta > len_red else RED

    for r, c in crossings:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) == (r, c) or not _in_bounds(g.shape, nr, nc):
                    continue
                if int(out[nr, nc]) in {BG, MAGENTA, RED}:
                    out[nr, nc] = YELLOW

    return out


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = _sign(c1 - c0)
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = _sign(r1 - r0)
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be orthogonal")


def _path_from_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = [points[0]]
    for a, b in zip(points, points[1:]):
        seg = _cells_between(a, b)
        cells.extend(seg[1:])
    return cells


def _horizontal_rows(points: list[tuple[int, int]]) -> set[int]:
    rows = set()
    for a, b in zip(points, points[1:]):
        if a[0] == b[0]:
            rows.add(a[0])
    return rows


def _used_even_groups(used_horizontal_rows: set[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    rows = sorted(used_horizontal_rows)
    if not rows:
        return groups
    start = rows[0]
    prev = rows[0]
    for row in rows[1:]:
        if row == prev + 2:
            prev = row
            continue
        groups.append((start - 1, prev + 1))
        start = prev = row
    groups.append((start - 1, prev + 1))
    return groups


def _assign_even_row_colors(h: int, used_horizontal_rows: set[int]) -> dict[int, int]:
    groups = _used_even_groups(used_horizontal_rows)
    locked: dict[int, int] = {}
    for e0, e1 in groups:
        color = random.choice(WALL_PALETTE)
        for even_row in range(e0, e1 + 1, 2):
            locked[even_row] = color

    colors: dict[int, int] = {}
    prev: int | None = None
    for even_row in range(0, h, 2):
        if even_row in locked:
            colors[even_row] = locked[even_row]
            prev = colors[even_row]
            continue
        choices = list(WALL_PALETTE)
        if prev in choices and len(choices) > 1:
            choices.remove(prev)
        next_locked = locked.get(even_row + 2)
        if next_locked in choices and len(choices) > 1:
            choices.remove(next_locked)
        colors[even_row] = random.choice(choices)
        prev = colors[even_row]
    return colors


def _wall_color_for_row(even_colors: dict[int, int], row: int, h: int) -> int:
    if row > 0 and row - 1 in even_colors:
        return even_colors[row - 1]
    if row + 1 in even_colors:
        return even_colors[row + 1]
    return even_colors[0 if row < h // 2 else h - 1]


def _build_grid(
    h: int,
    w: int,
    bar_start: int,
    path_points: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    used_rows = set()
    for points in path_points.values():
        used_rows |= _horizontal_rows(points)

    even_colors = _assign_even_row_colors(h, used_rows)
    g = np.full((h, w), BG, dtype=int)
    for even_row, color in even_colors.items():
        g[even_row, bar_start:] = color

    for color, points in path_points.items():
        for a, b in zip(points, points[1:]):
            if a[1] == b[1]:
                col = a[1]
                step = _sign(b[0] - a[0])
                for r in range(a[0] + step, b[0] + step, step):
                    g[r, col] = BG

        for prev_pt, cur_pt, next_pt in zip(points, points[1:], points[2:]):
            prev_dir = (_sign(cur_pt[0] - prev_pt[0]), _sign(cur_pt[1] - prev_pt[1]))
            next_dir = (_sign(next_pt[0] - cur_pt[0]), _sign(next_pt[1] - cur_pt[1]))
            r, c = cur_pt

            if prev_dir in (LEFT, RIGHT) and next_dir in (UP, DOWN):
                blocker_c = c + prev_dir[1]
                g[r, blocker_c] = _wall_color_for_row(even_colors, r, h)
                g[r + next_dir[0], c] = BG
            elif prev_dir in (UP, DOWN) and next_dir in (LEFT, RIGHT):
                disambiguate_c = c - next_dir[1]
                g[r, disambiguate_c] = _wall_color_for_row(even_colors, r, h)
            else:
                raise AssertionError("points must alternate H/V segments")

        start_r, start_c = points[0]
        assert start_c == 0
        g[start_r, start_c] = color

    return g


def _odd_rows(lo: int, hi: int) -> list[int]:
    return [r for r in range(lo, hi + 1) if r % 2 == 1]


def _sample_family_magenta_long(h: int, w: int, bar_start: int) -> dict | None:
    top_exit = random.random() < 0.5
    cross_choices = _odd_rows(5, h - 6)
    if not cross_choices:
        return None
    cross_r = random.choice(cross_choices)

    if top_exit:
        magenta_start = cross_r + 2
        far_red_rows = _odd_rows(1, cross_r - 4)
        magenta_mid_rows = _odd_rows(cross_r + 4, h - 2)
        if not far_red_rows or not magenta_mid_rows:
            return None
        red_start = random.choice(far_red_rows)
        magenta_mid = random.choice(magenta_mid_rows)
        boundary_r = 0
    else:
        magenta_start = cross_r - 2
        far_red_rows = _odd_rows(cross_r + 4, h - 2)
        magenta_mid_rows = _odd_rows(1, cross_r - 4)
        if not far_red_rows or not magenta_mid_rows:
            return None
        red_start = random.choice(far_red_rows)
        magenta_mid = random.choice(magenta_mid_rows)
        boundary_r = h - 1

    turn_red_choices = list(range(bar_start + 2, w - 8))
    if not turn_red_choices:
        return None
    red_turn = random.choice(turn_red_choices)
    magenta_turn_choices = list(range(red_turn + 2, w - 6))
    if not magenta_turn_choices:
        return None
    magenta_turn = random.choice(magenta_turn_choices)
    cross_c_choices = list(range(magenta_turn + 2, w - 2))
    if not cross_c_choices:
        return None
    cross_c = random.choice(cross_c_choices)

    magenta_points = [
        (magenta_start, 0),
        (magenta_start, magenta_turn),
        (magenta_mid, magenta_turn),
        (magenta_mid, cross_c),
        (boundary_r, cross_c),
    ]
    red_points = [
        (red_start, 0),
        (red_start, red_turn),
        (cross_r, red_turn),
        (cross_r, w - 1),
    ]
    return {
        "winner": MAGENTA,
        "closer": MAGENTA,
        "cross": (cross_r, cross_c),
        "points": {MAGENTA: magenta_points, RED: red_points},
    }


def _sample_family_red_long(h: int, w: int, bar_start: int) -> dict | None:
    top_exit = random.random() < 0.5
    cross_choices = _odd_rows(5, h - 6)
    if not cross_choices:
        return None
    cross_r = random.choice(cross_choices)

    if top_exit:
        red_start = cross_r + 2
        far_magenta_rows = _odd_rows(1, cross_r - 4)
        red_mid_rows = _odd_rows(cross_r + 4, h - 2)
        if not far_magenta_rows or not red_mid_rows:
            return None
        magenta_start = random.choice(far_magenta_rows)
        red_mid = random.choice(red_mid_rows)
        boundary_r = 0
    else:
        red_start = cross_r - 2
        far_magenta_rows = _odd_rows(cross_r + 4, h - 2)
        red_mid_rows = _odd_rows(1, cross_r - 4)
        if not far_magenta_rows or not red_mid_rows:
            return None
        magenta_start = random.choice(far_magenta_rows)
        red_mid = random.choice(red_mid_rows)
        boundary_r = h - 1

    turn_magenta_choices = list(range(bar_start + 2, w - 8))
    if not turn_magenta_choices:
        return None
    magenta_turn = random.choice(turn_magenta_choices)
    red_turn_choices = list(range(magenta_turn + 2, w - 6))
    if not red_turn_choices:
        return None
    red_turn = random.choice(red_turn_choices)
    cross_c_choices = list(range(red_turn + 2, w - 2))
    if not cross_c_choices:
        return None
    cross_c = random.choice(cross_c_choices)

    magenta_points = [
        (magenta_start, 0),
        (magenta_start, magenta_turn),
        (cross_r, magenta_turn),
        (cross_r, w - 1),
    ]
    red_points = [
        (red_start, 0),
        (red_start, red_turn),
        (red_mid, red_turn),
        (red_mid, cross_c),
        (boundary_r, cross_c),
    ]
    return {
        "winner": RED,
        "closer": RED,
        "cross": (cross_r, cross_c),
        "points": {MAGENTA: magenta_points, RED: red_points},
    }


def _sample_plan() -> tuple[np.ndarray, dict] | None:
    h = random.choice(HEIGHT_OPTIONS)
    w = random.randint(*WIDTH_RANGE)
    bar_start = random.choice((2, 3))

    plan = (
        _sample_family_magenta_long(h, w, bar_start)
        if random.random() < 0.5
        else _sample_family_red_long(h, w, bar_start)
    )
    if plan is None:
        return None

    grid = _build_grid(h, w, bar_start, plan["points"])
    return grid, plan


def _validate_plan(grid: np.ndarray, plan: dict) -> bool:
    expected_magenta = _path_from_points(plan["points"][MAGENTA])
    expected_red = _path_from_points(plan["points"][RED])
    traced_magenta = _trace_path(grid, plan["points"][MAGENTA][0])
    traced_red = _trace_path(grid, plan["points"][RED][0])
    if traced_magenta != expected_magenta or traced_red != expected_red:
        return False

    crossings = _crossings(traced_magenta, traced_red)
    if crossings != [plan["cross"]]:
        return False
    cross_r, cross_c = plan["cross"]
    if not (1 <= cross_r < grid.shape[0] - 1 and 1 <= cross_c < grid.shape[1] - 1):
        return False

    magenta_idx = _path_index(traced_magenta)
    red_idx = _path_index(traced_red)
    magenta_len = magenta_idx[plan["cross"]] + 1
    red_len = red_idx[plan["cross"]] + 1
    if magenta_len == red_len:
        return False
    winner = MAGENTA if magenta_len > red_len else RED
    if winner != plan["winner"]:
        return False

    magenta_start = plan["points"][MAGENTA][0]
    red_start = plan["points"][RED][0]
    magenta_geom = abs(magenta_start[0] - cross_r)
    red_geom = abs(red_start[0] - cross_r)
    closer = MAGENTA if magenta_geom < red_geom else RED if red_geom < magenta_geom else None
    if closer != plan["closer"]:
        return False
    if winner != closer:
        return False

    out = main(grid)
    if int(out[cross_r, cross_c]) != winner:
        return False
    ring = out[cross_r - 1 : cross_r + 2, cross_c - 1 : cross_c + 2]
    if np.count_nonzero(ring == YELLOW) < 4:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        sample = _sample_plan()
        if sample is None:
            continue
        grid, plan = sample
        if _validate_plan(grid, plan):
            return grid
    raise RuntimeError("failed to generate a valid corridor-crossing ARC input")
