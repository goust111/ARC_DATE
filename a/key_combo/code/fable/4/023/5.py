import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8

DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GENERATION_ATTEMPTS = 400
PRIMARY_LAYOUT_ATTEMPTS = 250
PRIMARY_BOUNCE_STEPS = range(2, 7)
RECT_EXTRA_RANGE = range(3)
EXTRA_EMITTER_SAMPLE_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _emitter_cells(head: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = head
    dr, dc = direction
    return [(r - 2 * dr, c - 2 * dc), (r - dr, c - dc), (r, c)]


def _simulate_ray(
    obstacles: np.ndarray,
    head: tuple[int, int],
    direction: tuple[int, int],
    blocked_cells: Iterable[tuple[int, int]] = (),
):
    h, w = obstacles.shape
    blocked = set(blocked_cells)
    r, c = head
    dr, dc = direction
    visited_states = {(r, c, dr, dc)}
    path = [(r, c)]
    bounces = 0

    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            return {"exit": (r, c), "bounces": bounces, "path": tuple(path)}

        if obstacles[nr, nc]:
            block_r = _in_bounds((h, w), r + dr, c) and bool(obstacles[r + dr, c])
            block_c = _in_bounds((h, w), r, c + dc) and bool(obstacles[r, c + dc])
            if block_r == block_c:
                return None
            if block_r:
                dr = -dr
            else:
                dc = -dc
            bounces += 1
            state = (r, c, dr, dc)
            if state in visited_states:
                return None
            visited_states.add(state)
            continue

        r, c = nr, nc
        if (r, c) in blocked:
            return None
        path.append((r, c))
        state = (r, c, dr, dc)
        if state in visited_states:
            return None
        visited_states.add(state)


def _find_emitters(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    emitters = []
    for r, c in np.argwhere(g == YELLOW):
        r = int(r)
        c = int(c)
        matches = []
        for dr, dc in DIAGONALS:
            tail1 = (r - dr, c - dc)
            tail2 = (r - 2 * dr, c - 2 * dc)
            if not (_in_bounds(g.shape, *tail1) and _in_bounds(g.shape, *tail2)):
                continue
            if g[tail1] != TEAL or g[tail2] != TEAL:
                continue
            tail3 = (r - 3 * dr, c - 3 * dc)
            if _in_bounds(g.shape, *tail3) and g[tail3] in (TEAL, YELLOW):
                continue
            matches.append((dr, dc))
        if len(matches) == 1:
            emitters.append(((r, c), matches[0]))
    emitters.sort()
    return emitters


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    obstacles = g == RED
    emitters = _find_emitters(g)
    assert emitters, "expected at least one emitter"

    for head, direction in emitters:
        result = _simulate_ray(obstacles, head, direction)
        assert result is not None, "invalid ray geometry"
        exit_r, exit_c = result["exit"]
        out[exit_r, exit_c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each hidden diagonal ray through red reflectors and mark only its exit cell.

    Input structure:
    - Black background with one or more emitters, each made of two teal tail cells
      followed by one yellow head cell on a diagonal.
    - One or more solid red rectangular obstacles.
    - No visible ray path is drawn in the input.

    Rule:
    - For each emitter, launch a ray from the yellow head in the diagonal direction
      encoded by the teal-to-yellow order.
    - When the next diagonal step would enter red, flip the blocked direction
      component and continue; each collision flips exactly one component.
    - When the ray leaves the grid, color the last in-bounds cell it visited yellow,
      while preserving the original emitters and red obstacles.

    Key insight:
    - The only new output pixels are the hidden rays' exit traces, so the solver must
      reconstruct each full bounce path from emitter orientation plus rectangle geometry.
    """
    return _solve(grid)


def _rect_touches(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return not (ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1)


def _rect_ok(rect: tuple[int, int, int, int], rects: list[tuple[int, int, int, int]]) -> bool:
    return all(not _rect_touches(rect, other) for other in rects)


def _render_obstacles(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _near_cells(shape: tuple[int, int], cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _rect_near_cells(rect: tuple[int, int, int, int], cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for rr in range(r0 - 1, r1 + 2):
        for cc in range(c0 - 1, c1 + 2):
            if (rr, cc) in cells:
                return True
    return False


def _bounce_rect_options(
    shape: tuple[int, int],
    point: tuple[int, int],
    direction: tuple[int, int],
    flip_row: bool,
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    r, c = point
    dr, dc = direction
    options = []

    if flip_row:
        rr = r + dr
        base_c0, base_c1 = sorted((c, c + dc))
        if not (0 <= rr < h):
            return []
        for left in RECT_EXTRA_RANGE:
            for right in RECT_EXTRA_RANGE:
                c0 = base_c0 - left
                c1 = base_c1 + right
                if 0 <= c0 <= c1 < w:
                    options.append((rr, c0, rr, c1))
    else:
        cc = c + dc
        base_r0, base_r1 = sorted((r, r + dr))
        if not (0 <= cc < w):
            return []
        for up in RECT_EXTRA_RANGE:
            for down in RECT_EXTRA_RANGE:
                r0 = base_r0 - up
                r1 = base_r1 + down
                if 0 <= r0 <= r1 < h:
                    options.append((r0, cc, r1, cc))
    return options


def _segment_clear(
    obstacles: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
) -> bool:
    r, c = start
    dr, dc = direction
    for step in range(1, steps + 1):
        rr = r + step * dr
        cc = c + step * dc
        if not _in_bounds(obstacles.shape, rr, cc) or obstacles[rr, cc]:
            return False
    return True


def _construct_primary_layout(rng: random.Random):
    for _ in range(PRIMARY_LAYOUT_ATTEMPTS):
        shape = (rng.randint(18, 28), rng.randint(18, 28))
        direction = rng.choice(DIAGONALS)
        head = (rng.randrange(shape[0]), rng.randrange(shape[1]))
        emitter_cells = _emitter_cells(head, direction)
        if not all(_in_bounds(shape, r, c) for r, c in emitter_cells):
            continue

        rects: list[tuple[int, int, int, int]] = []
        current = head
        dr, dc = direction
        bounce_goal = rng.randint(2, 4)
        ok = True

        for _bounce_index in range(bounce_goal):
            obstacles = _render_obstacles(shape, rects)
            options = []
            for steps in PRIMARY_BOUNCE_STEPS:
                if not _segment_clear(obstacles, current, (dr, dc), steps):
                    continue
                point = (current[0] + steps * dr, current[1] + steps * dc)
                for flip_row in (False, True):
                    new_direction = (-dr, dc) if flip_row else (dr, -dc)
                    next_cell = (point[0] + new_direction[0], point[1] + new_direction[1])
                    if not _in_bounds(shape, *next_cell) or obstacles[next_cell]:
                        continue
                    for rect in _bounce_rect_options(shape, point, (dr, dc), flip_row):
                        if not _rect_ok(rect, rects):
                            continue
                        options.append((point, new_direction, rect))

            if not options:
                ok = False
                break

            point, (dr, dc), rect = rng.choice(options)
            rects.append(rect)
            current = point

        if not ok:
            continue

        emitter_neighborhood = _near_cells(shape, emitter_cells)
        if any(_rect_near_cells(rect, emitter_neighborhood) for rect in rects):
            continue

        obstacles = _render_obstacles(shape, rects)
        result = _simulate_ray(obstacles, head, direction, blocked_cells=emitter_cells)
        if result is None or result["bounces"] != bounce_goal or result["bounces"] < 2:
            continue
        if result["exit"] in set(emitter_cells):
            continue

        return shape, rects, (head, direction)

    return None


def _candidate_emitter_ok(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    obstacles: np.ndarray,
    occupied_emitters: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> bool:
    h, w = shape
    emitter_neighbors = _near_cells(shape, occupied_emitters)

    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if obstacles[r, c] or (r, c) in occupied_emitters or (r, c) in forbidden_cells:
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if obstacles[r0:r1, c0:c1].any():
            return False
        if (r, c) in emitter_neighbors:
            return False
    return True


def _sample_extra_emitter(
    shape: tuple[int, int],
    obstacles: np.ndarray,
    occupied_emitters: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
    used_exits: set[tuple[int, int]],
    rng: random.Random,
):
    h, w = shape
    for _ in range(EXTRA_EMITTER_SAMPLE_ATTEMPTS):
        head = (rng.randrange(h), rng.randrange(w))
        direction = rng.choice(DIAGONALS)
        cells = _emitter_cells(head, direction)
        if not _candidate_emitter_ok(shape, cells, obstacles, occupied_emitters, forbidden_cells):
            continue
        result = _simulate_ray(obstacles, head, direction, blocked_cells=occupied_emitters | set(cells))
        if result is None:
            continue
        if result["exit"] == head or result["exit"] in used_exits:
            continue
        if result["exit"] in occupied_emitters or result["exit"] in set(cells):
            continue
        return head, direction, result
    return None


def _render_input(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    emitters: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = RED
    for head, direction in emitters:
        tail2, tail1, (hr, hc) = _emitter_cells(head, direction)
        grid[tail2] = TEAL
        grid[tail1] = TEAL
        grid[hr, hc] = YELLOW
    return grid


def _scene_is_valid(
    inp: np.ndarray,
    emitters: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    if max(inp.shape) > 30:
        return False
    if len(_find_emitters(inp)) != len(emitters):
        return False

    obstacles = inp == RED
    all_emitter_cells = set()
    for head, direction in emitters:
        all_emitter_cells.update(_emitter_cells(head, direction))

    expected = inp.copy()
    exits = set()
    for head, direction in emitters:
        result = _simulate_ray(obstacles, head, direction, blocked_cells=all_emitter_cells)
        if result is None or result["exit"] in all_emitter_cells:
            return False
        exits.add(result["exit"])
        er, ec = result["exit"]
        expected[er, ec] = YELLOW

    actual = main(inp)
    return bool(np.array_equal(actual, expected))


def _generate_scene(rng: random.Random):
    primary = _construct_primary_layout(rng)
    if primary is None:
        return None

    shape, rects, primary_emitter = primary
    obstacles = _render_obstacles(shape, rects)
    emitters = [primary_emitter]
    occupied_emitters = set(_emitter_cells(*primary_emitter))

    primary_result = _simulate_ray(obstacles, primary_emitter[0], primary_emitter[1], blocked_cells=occupied_emitters)
    if primary_result is None:
        return None
    used_exits = {primary_result["exit"]}
    forbidden_cells = set(primary_result["path"])

    target_count = rng.randint(2, 3)
    while len(emitters) < target_count:
        sampled = _sample_extra_emitter(
            shape,
            obstacles,
            occupied_emitters,
            forbidden_cells | used_exits,
            used_exits,
            rng,
        )
        if sampled is None:
            return None
        head, direction, result = sampled
        emitters.append((head, direction))
        occupied_emitters.update(_emitter_cells(head, direction))
        forbidden_cells.update(result["path"])
        used_exits.add(result["exit"])

    inp = _render_input(shape, rects, emitters)
    if not _scene_is_valid(inp, emitters):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for hidden-bounce exits")
