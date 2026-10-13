import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SEED_COLORS = (1, 2, 3, 6, 7, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)
RIGHT_TURN = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
LEFT_TURN = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}

GRID_H_RANGE = (13, 22)
GRID_W_RANGE = (13, 22)
SEED_COUNT_CHOICES = (1, 2, 3)
SEED_COUNT_WEIGHTS = (2, 3, 2)
TURN_COUNT_CHOICES = (0, 1, 2, 3)
TURN_COUNT_WEIGHTS = (1, 3, 3, 2)
TURN_MODE_CHOICES = ("left", "right", "tie")
MIRROR_LENGTH_RANGE = (2, 4)
STRAIGHT_RUN_RANGE = (1, 5)
MIN_TOTAL_PATH_CELLS = 4
GEN_ATTEMPTS = 300
RAY_ATTEMPTS = 200
DECOY_MIRROR_RANGE = (0, 2)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_orientation_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirrors = _connected_components(grid == GREY)
    orientation: dict[tuple[int, int], str] = {}
    for comp in mirrors:
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert 2 <= len(comp) <= 4, "mirror length must be 2-4"
        if len(rows) == 1:
            orient = "horizontal"
        elif len(cols) == 1:
            orient = "vertical"
        else:
            raise AssertionError("grey mirrors must be straight")
        for cell in comp:
            orientation[cell] = orient
    return orientation


def _find_seeds(grid: np.ndarray) -> list[tuple[int, int, int, tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    seeds: list[tuple[int, int, int, tuple[int, int], tuple[int, int]]] = []
    yellow_owner: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, YELLOW, GREY):
                continue
            yellow_neighbors = []
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == YELLOW:
                    yellow_neighbors.append((nr, nc))
            assert len(yellow_neighbors) == 1, "each seed must have exactly one yellow neighbor"
            marker = yellow_neighbors[0]
            direction = (int(marker[0] - r), int(marker[1] - c))
            assert direction in DIRS
            seeds.append((r, c, color, marker, direction))
            assert marker not in yellow_owner, "each yellow marker must belong to one seed"
            yellow_owner[marker] = (r, c)
    assert seeds, "expected at least one seed"

    for r, c in np.argwhere(grid == YELLOW):
        owner_count = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BLACK, YELLOW, GREY):
                owner_count += 1
        assert owner_count == 1, "each yellow must indicate exactly one seed"
    return seeds


def _space_count(
    grid: np.ndarray,
    nonblack: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    count = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if (r, c) in nonblack:
            break
        if int(grid[r, c]) != BLACK:
            break
        count += 1
    return count


def _reflected_direction(
    grid: np.ndarray,
    nonblack: set[tuple[int, int]],
    pos: tuple[int, int],
    direction: tuple[int, int],
    mirror_orientation: str,
) -> tuple[int, int] | None:
    if direction[0] == 0:
        if mirror_orientation != "vertical":
            return None
    else:
        if mirror_orientation != "horizontal":
            return None

    left_dir = LEFT_TURN[direction]
    right_dir = RIGHT_TURN[direction]
    left_space = _space_count(grid, nonblack, pos, left_dir)
    right_space = _space_count(grid, nonblack, pos, right_dir)
    if left_space > right_space:
        return left_dir
    return right_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored rays from seed pixels. A yellow neighbor gives the initial
    direction. Grey horizontal/vertical segments act as 90-degree mirrors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mirror_map = _mirror_orientation_map(g)
    seeds = _find_seeds(g)

    out = g.copy()
    out[out == YELLOW] = BLACK
    nonblack = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != GREY))
    }

    for seed_r, seed_c, color, marker, direction in sorted(seeds):
        visited_cells = {(seed_r, seed_c)}
        seen_states: set[tuple[int, int, int, int]] = set()
        r, c = seed_r, seed_c
        dr, dc = direction

        while True:
            state = (r, c, dr, dc)
            if state in seen_states:
                break
            seen_states.add(state)

            nr = r + dr
            nc = c + dc
            if not _in_bounds(g.shape, nr, nc):
                break

            if int(g[nr, nc]) == GREY:
                new_dir = _reflected_direction(g, nonblack, (r, c), (dr, dc), mirror_map[(nr, nc)])
                if new_dir is None:
                    break
                dr, dc = new_dir
                continue

            if (nr, nc) in visited_cells:
                break

            cell = int(g[nr, nc])
            if cell == BLACK or ((nr, nc) == marker and cell == YELLOW):
                visited_cells.add((nr, nc))
                out[nr, nc] = color
                r, c = nr, nc
                continue

            break

    return out


def _cells_along(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * i, c + dc * i) for i in range(1, steps + 1)]


def _mirror_segments(
    shape: tuple[int, int],
    hit: tuple[int, int],
    incoming: tuple[int, int],
    blocked_nonblack: set[tuple[int, int]],
    blocked_path: set[tuple[int, int]],
    existing_grey: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    if incoming[0] == 0:
        orientation = "vertical"
    else:
        orientation = "horizontal"

    hr, hc = hit
    segments: list[list[tuple[int, int]]] = []
    for length in range(MIRROR_LENGTH_RANGE[0], MIRROR_LENGTH_RANGE[1] + 1):
        for offset in range(length):
            cells = []
            ok = True
            for i in range(length):
                if orientation == "vertical":
                    cell = (hr - offset + i, hc)
                else:
                    cell = (hr, hc - offset + i)
                if not _in_bounds(shape, cell[0], cell[1]):
                    ok = False
                    break
                cells.append(cell)
            if not ok:
                continue
            cell_set = set(cells)
            if cell_set & blocked_nonblack:
                continue
            if cell_set & blocked_path:
                continue
            touches_other = False
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if (nr, nc) in existing_grey and (nr, nc) not in cell_set:
                        touches_other = True
                        break
                if touches_other:
                    break
            if not touches_other:
                segments.append(cells)
    return segments


def _count_open_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    blocked_nonblack: set[tuple[int, int]],
) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    count = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if (r, c) in blocked_nonblack:
            break
        count += 1
    return count


def _line_to_boundary(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    cells = set()
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.add((r, c))
    return cells


def _has_foreign_neighbor(
    cell: tuple[int, int],
    allowed: set[tuple[int, int]],
    occupied_nonblack: set[tuple[int, int]],
) -> bool:
    r, c = cell
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) in occupied_nonblack and (nr, nc) not in allowed:
            return True
    return False


def _enumerate_turn_options(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    local_nonblack: set[tuple[int, int]],
    global_nonblack: set[tuple[int, int]],
    global_grey: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
    reserved_nonblack: set[tuple[int, int]],
    preferred_mode: str | None,
) -> list[dict]:
    options: list[dict] = []
    all_nonblack = local_nonblack | global_nonblack
    local_grey = {cell for cell in local_nonblack if cell not in local_path}
    blocked_path = all_nonblack | local_path | global_path
    blocked_nonblack = blocked_path | reserved_nonblack
    left_dir = LEFT_TURN[direction]
    right_dir = RIGHT_TURN[direction]

    for advance in range(STRAIGHT_RUN_RANGE[0], STRAIGHT_RUN_RANGE[1] + 1):
        forward_cells = _cells_along(current, direction, advance)
        if not forward_cells:
            continue
        if any(not _in_bounds(shape, r, c) for r, c in forward_cells):
            continue
        if any((r, c) in blocked_path for r, c in forward_cells):
            continue

        pivot = forward_cells[-1]
        hit = (pivot[0] + direction[0], pivot[1] + direction[1])
        if not _in_bounds(shape, hit[0], hit[1]):
            continue
        if hit in blocked_nonblack:
            continue

        left_space = _count_open_cells(shape, pivot, left_dir, all_nonblack | {hit})
        right_space = _count_open_cells(shape, pivot, right_dir, all_nonblack | {hit})
        if left_space > right_space:
            mode = "left"
            new_dir = left_dir
        elif right_space > left_space:
            mode = "right"
            new_dir = right_dir
        else:
            mode = "tie"
            new_dir = right_dir
        if preferred_mode is not None and mode != preferred_mode:
            continue

        first_after_turn = (pivot[0] + new_dir[0], pivot[1] + new_dir[1])
        if not _in_bounds(shape, first_after_turn[0], first_after_turn[1]):
            continue
        if first_after_turn in blocked_path:
            continue

        mirror_candidates = _mirror_segments(
            shape,
            hit,
            direction,
            blocked_nonblack,
            blocked_path,
            local_grey | global_grey,
        )
        if not mirror_candidates:
            continue

        sight = _line_to_boundary(shape, pivot, left_dir) | _line_to_boundary(shape, pivot, right_dir)
        options.append(
            {
                "advance_cells": forward_cells,
                "pivot": pivot,
                "hit": hit,
                "new_dir": new_dir,
                "mode": mode,
                "mirror_cells": random.choice(mirror_candidates),
                "sight": sight,
            }
        )

    return options


def _build_ray(
    shape: tuple[int, int],
    color: int,
    global_nonblack: set[tuple[int, int]],
    global_grey: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
    reserved_nonblack: set[tuple[int, int]],
    target_turns: int,
) -> dict | None:
    h, w = shape
    for _ in range(RAY_ATTEMPTS):
        direction = random.choice(DIRS)
        seed = (random.randrange(h), random.randrange(w))
        marker = (seed[0] + direction[0], seed[1] + direction[1])
        if not _in_bounds(shape, marker[0], marker[1]):
            continue
        if seed in global_nonblack or seed in global_path or seed in reserved_nonblack:
            continue
        if marker in global_nonblack or marker in global_path or marker in reserved_nonblack:
            continue
        if _has_foreign_neighbor(seed, {marker}, global_nonblack):
            continue
        if _has_foreign_neighbor(marker, {seed}, global_nonblack):
            continue

        local_nonblack = {seed, marker}
        local_path = {marker}
        local_sight: set[tuple[int, int]] = set()
        mirrors: list[tuple[int, int]] = []
        current = marker
        current_dir = direction
        turn_modes = [random.choice(TURN_MODE_CHOICES) for _ in range(target_turns)]
        success = True

        for preferred_mode in turn_modes:
            options = _enumerate_turn_options(
                shape,
                current,
                current_dir,
                local_nonblack,
                global_nonblack,
                global_grey,
                local_path,
                global_path,
                reserved_nonblack,
                preferred_mode,
            )
            if not options:
                success = False
                break
            option = random.choice(options)
            for cell in option["advance_cells"]:
                local_path.add(cell)
            for cell in option["mirror_cells"]:
                local_nonblack.add(cell)
                mirrors.append(cell)
            local_sight |= option["sight"]
            current = option["pivot"]
            current_dir = option["new_dir"]

        if not success:
            continue

        tail = []
        r, c = current
        dr, dc = current_dir
        while True:
            r += dr
            c += dc
            if not _in_bounds(shape, r, c):
                break
            if (r, c) in local_nonblack or (r, c) in global_nonblack or (r, c) in global_path:
                success = False
                break
            tail.append((r, c))
        if not success:
            continue
        if not tail:
            continue
        for cell in tail:
            local_path.add(cell)

        if len(local_path) < MIN_TOTAL_PATH_CELLS:
            continue

        return {
            "color": color,
            "seed": seed,
            "marker": marker,
            "direction": direction,
            "mirrors": mirrors,
            "grey": set(mirrors),
            "path": local_path,
            "nonblack": local_nonblack,
            "sight": local_sight,
        }
    return None


def _add_decoy_mirror(
    shape: tuple[int, int],
    occupied_nonblack: set[tuple[int, int]],
    occupied_grey: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
    reserved_nonblack: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    for _ in range(80):
        orientation = random.choice(("horizontal", "vertical"))
        length = random.randint(MIRROR_LENGTH_RANGE[0], MIRROR_LENGTH_RANGE[1])
        if orientation == "horizontal":
            r = random.randrange(h)
            c0 = random.randrange(w - length + 1)
            cells = [(r, c0 + i) for i in range(length)]
        else:
            c = random.randrange(w)
            r0 = random.randrange(h - length + 1)
            cells = [(r0 + i, c) for i in range(length)]

        cell_set = set(cells)
        if cell_set & occupied_nonblack:
            continue
        if cell_set & reserved_path:
            continue
        if cell_set & reserved_nonblack:
            continue
        ok = True
        for rr, cc in cells:
            for nr, nc in _neighbors4(rr, cc):
                if (nr, nc) in occupied_grey and (nr, nc) not in cell_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return cells
    return None


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert np.any(grid == YELLOW)
    _mirror_orientation_map(grid)
    _find_seeds(grid)
    main(grid)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)
    seed_count = random.choices(SEED_COUNT_CHOICES, weights=SEED_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(SEED_COLORS, seed_count)

    occupied_nonblack: set[tuple[int, int]] = set()
    occupied_grey: set[tuple[int, int]] = set()
    reserved_path: set[tuple[int, int]] = set()
    reserved_nonblack: set[tuple[int, int]] = set()
    plans = []

    for color in colors:
        turn_count = random.choices(TURN_COUNT_CHOICES, weights=TURN_COUNT_WEIGHTS, k=1)[0]
        plan = _build_ray(shape, color, occupied_nonblack, occupied_grey, reserved_path, reserved_nonblack, turn_count)
        if plan is None:
            return None
        plans.append(plan)
        occupied_nonblack |= plan["nonblack"]
        occupied_grey |= plan["grey"]
        reserved_path |= plan["path"]
        reserved_nonblack |= plan["path"] | plan["sight"]

    decoy_count = random.randint(*DECOY_MIRROR_RANGE)
    for _ in range(decoy_count):
        mirror = _add_decoy_mirror(shape, occupied_nonblack, occupied_grey, reserved_path, reserved_nonblack)
        if mirror is not None:
            occupied_nonblack |= set(mirror)
            occupied_grey |= set(mirror)

    grid = np.full(shape, BLACK, dtype=int)
    for plan in plans:
        sr, sc = plan["seed"]
        mr, mc = plan["marker"]
        grid[sr, sc] = plan["color"]
        grid[mr, mc] = YELLOW
        for rr, cc in plan["mirrors"]:
            grid[rr, cc] = GREY

    extra_grey = occupied_nonblack - {
        (int(r), int(c))
        for r, c in np.argwhere((grid != BLACK) & (grid != GREY))
    }
    for rr, cc in extra_grey:
        if grid[rr, cc] == BLACK:
            grid[rr, cc] = GREY

    try:
        _validate_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-mirror input")
