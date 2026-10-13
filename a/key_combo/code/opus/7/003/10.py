from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds, neighbors4

BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
YELLOW = Color.YELLOW
GREY = Color.GREY
ORANGE = Color.ORANGE
MAROON = Color.MAROON

SPRITE_SIZES = ((2, 2), (2, 3), (3, 2))
GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
GEN_RETRIES = 400
PATH_PLACE_RETRIES = 120


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _rect_cells(top: tuple[int, int], size: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top
    h, w = size
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _rect_in_bounds(shape: tuple[int, int], top: tuple[int, int], size: tuple[int, int]) -> bool:
    r0, c0 = top
    h, w = size
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def _is_solid_yellow_rect(comp: dict) -> bool:
    if int(comp["color"]) != int(YELLOW):
        return False
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return int(comp["area"]) == h * w and (h, w) in SPRITE_SIZES


def _find_sprite(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    comps = find_components(grid, bg=BG, colors=[YELLOW], connectivity=4)
    sprites = [comp for comp in comps if _is_solid_yellow_rect(comp)]
    assert len(sprites) == 1, "expected one solid yellow sprite rectangle"
    comp = sprites[0]
    r0, c0, r1, c1 = comp["bbox"]
    top = (int(r0), int(c0))
    size = (int(r1 - r0 + 1), int(c1 - c0 + 1))
    return top, size, set(map(tuple, comp["coords"]))


def _find_indicator(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    gray_comps = find_components(grid, bg=BG, colors=[GREY], connectivity=4)
    assert len(gray_comps) == 1, "expected one gray path component"
    gray_coords = set(map(tuple, gray_comps[0]["coords"]))
    assert 3 <= len(gray_coords) <= 6, "gray path length must be 3-6"

    endpoints: dict[int, tuple[int, int]] = {}
    for color in (RED, BLUE):
        comps = find_components(grid, bg=BG, colors=[color], connectivity=4)
        assert len(comps) == 1 and int(comps[0]["area"]) == 1, "expected singleton red/blue endpoints"
        pt = tuple(map(int, comps[0]["coords"][0]))
        adjacent_gray = sum((nr, nc) in gray_coords for nr, nc in neighbors4(*pt))
        assert adjacent_gray == 1, "endpoint must touch the gray path at exactly one cell"
        endpoints[int(color)] = pt

    graph = gray_coords | {endpoints[int(RED)], endpoints[int(BLUE)]}
    degrees = {}
    for cell in graph:
        degrees[cell] = sum(nei in graph for nei in neighbors4(*cell))
    assert degrees[endpoints[int(RED)]] == 1 and degrees[endpoints[int(BLUE)]] == 1
    assert sum(deg == 1 for deg in degrees.values()) == 2, "path graph must have exactly two endpoints"
    return endpoints[int(RED)], endpoints[int(BLUE)], gray_coords


def _secondary_direction(current: int, target: int) -> int:
    if current < target:
        return 1
    if current > target:
        return -1
    return -1


def _move(top: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return int(top[0] + delta[0]), int(top[1] + delta[1])


def _obstacle_overlap(obstacle_mask: np.ndarray, top: tuple[int, int], size: tuple[int, int]) -> bool:
    r0, c0 = top
    h, w = size
    return bool(obstacle_mask[r0 : r0 + h, c0 : c0 + w].any())


def _simulate_positions(
    shape: tuple[int, int],
    start: tuple[int, int],
    target: tuple[int, int],
    size: tuple[int, int],
    obstacle_mask: np.ndarray,
    main_axis: int,
) -> list[tuple[int, int]]:
    positions = [start]
    cur = start
    max_steps = shape[0] * shape[1] * 4
    main_dir = _sign(target[main_axis] - start[main_axis])

    if main_axis == 0:
        main_step = (main_dir, 0)
    else:
        main_step = (0, main_dir)

    for _ in range(max_steps):
        if cur == target:
            return positions

        if cur[main_axis] == target[main_axis]:
            raise AssertionError("secondary offset remained after main displacement finished")

        proposed = _move(cur, main_step)
        if not _rect_in_bounds(shape, proposed, size):
            raise AssertionError("main step left the grid")

        if not _obstacle_overlap(obstacle_mask, proposed, size):
            cur = proposed
            positions.append(cur)
            continue

        sec_axis = 1 - main_axis
        sec_dir = _secondary_direction(cur[sec_axis], target[sec_axis])
        delta = (sec_dir, 0) if sec_axis == 0 else (0, sec_dir)
        detour = _move(cur, delta)
        if not _rect_in_bounds(shape, detour, size):
            raise AssertionError("detour left the grid")
        if _obstacle_overlap(obstacle_mask, detour, size):
            raise AssertionError("detour hit an obstacle")
        cur = detour
        positions.append(cur)

    raise AssertionError("simulation did not reach target")


def _render_output(
    grid: np.ndarray,
    positions: list[tuple[int, int]],
    size: tuple[int, int],
    sprite_cells: set[tuple[int, int]],
    indicator_cells: set[tuple[int, int]],
) -> np.ndarray:
    base = np.array(grid, dtype=int).copy()
    for r, c in sprite_cells | indicator_cells:
        base[r, c] = BG

    trail: set[tuple[int, int]] = set()
    for cur, nxt in zip(positions, positions[1:]):
        cur_cells = _rect_cells(cur, size)
        nxt_cells = _rect_cells(nxt, size)
        for cell in cur_cells - nxt_cells:
            if int(base[cell]) == int(BG):
                trail.add(cell)

    out = base
    for r, c in trail:
        out[r, c] = MAROON
    for r, c in _rect_cells(positions[-1], size):
        out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the yellow sprite toward the displacement encoded by the red-blue gray path, detouring around orange obstacles and leaving a maroon trail.

    Input structure:
    - Black background, one small solid yellow rectangle, one gray 4-connected path with a red pixel on one end and a blue pixel on the other, and scattered orange obstacle pixels.
    - The gray path may bend, but it is a single component with exactly the red and blue pixels as its two graph endpoints.
    - The yellow sprite is the only solid yellow 2x2 / 2x3 / 3x2 rectangle.

    Rule:
    - Compute the displacement vector from the red endpoint to the blue endpoint and add it to the sprite's top-left corner to get the target top-left.
    - If the row displacement magnitude is at least the column displacement magnitude, the main motion axis is vertical; otherwise it is horizontal.
    - Repeatedly try to move the whole sprite one cell along the main axis toward the target. If that next main-axis position would overlap any orange obstacle cell, move one cell along the secondary axis instead, choosing the direction that gets closer to the target on that axis, with ties going to the smaller row/column index.
    - Each time the sprite vacates cells, paint those vacated black cells maroon. Stop once the sprite reaches the target. Erase the gray path and the red/blue endpoints, but keep the orange obstacles.

    Key insight:
    - The gray path's shape is irrelevant except for identifying the red and blue endpoint coordinates, while the obstacle layout determines when the encoded displacement must be realized as sideways detours.
    """
    g = np.asarray(grid, dtype=int)
    start, size, sprite_cells = _find_sprite(g)
    red, blue, gray_cells = _find_indicator(g)
    dr = int(blue[0] - red[0])
    dc = int(blue[1] - red[1])
    target = (int(start[0] + dr), int(start[1] + dc))
    assert _rect_in_bounds(g.shape, target, size), "target sprite position must be inside the grid"

    obstacle_mask = g == ORANGE
    main_axis = 0 if abs(dr) >= abs(dc) else 1
    positions = _simulate_positions(g.shape, start, target, size, obstacle_mask, main_axis)
    indicator_cells = set(gray_cells) | {red, blue}
    return _render_output(g, positions, size, sprite_cells, indicator_cells)


def _schedule_positions(
    start: tuple[int, int],
    main_step: tuple[int, int],
    side_step: tuple[int, int],
    main_count: int,
    detour_slots: set[int],
) -> list[tuple[int, int]]:
    pos = start
    positions = [pos]
    for slot in range(main_count):
        if slot in detour_slots:
            pos = _move(pos, side_step)
            positions.append(pos)
        pos = _move(pos, main_step)
        positions.append(pos)
    return positions


def _relative_path_edges(dr: int, dc: int) -> list[tuple[int, int]]:
    vr = [_sign(dr)] * abs(dr)
    hc = [_sign(dc)] * abs(dc)
    if not vr:
        order = [("c", step) for step in hc]
    elif not hc:
        order = [("r", step) for step in vr]
    elif random.random() < 0.5 or abs(dr) == 1 or abs(dc) == 1:
        if random.random() < 0.5:
            order = [("r", step) for step in vr] + [("c", step) for step in hc]
        else:
            order = [("c", step) for step in hc] + [("r", step) for step in vr]
    else:
        if abs(dr) >= abs(dc):
            split = random.randint(1, abs(dr) - 1)
            order = (
                [("r", vr[0])] * split
                + [("c", step) for step in hc]
                + [("r", vr[0])] * (abs(dr) - split)
            )
        else:
            split = random.randint(1, abs(dc) - 1)
            order = (
                [("c", hc[0])] * split
                + [("r", step) for step in vr]
                + [("c", hc[0])] * (abs(dc) - split)
            )

    cells = [(0, 0)]
    r = c = 0
    for axis, step in order:
        if axis == "r":
            r += step
        else:
            c += step
        cells.append((r, c))
    assert cells[-1] == (dr, dc)
    return cells


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if in_bounds(shape, rr, cc):
                    out.add((int(rr), int(cc)))
    return out


def _pick_displacement() -> tuple[int, int, int]:
    want_detour = random.random() < 0.78
    main_axis = random.randint(0, 1)
    options = []
    for main_mag in range(3, 7):
        for minor_mag in range(0, main_mag + 1):
            if want_detour and minor_mag == 0:
                continue
            if not want_detour and minor_mag != 0:
                continue
            if 4 <= main_mag + minor_mag <= 7:
                options.append((main_mag, minor_mag))
    main_mag, minor_mag = random.choice(options)
    s_main = random.choice((-1, 1))
    s_minor = random.choice((-1, 1)) if minor_mag else 0
    if main_axis == 0:
        return s_main * main_mag, s_minor * minor_mag, main_axis
    return s_minor * minor_mag, s_main * main_mag, main_axis


def _blocking_cells_for_schedule(
    start: tuple[int, int],
    detour_slots: set[int],
    main_count: int,
    size: tuple[int, int],
    main_step: tuple[int, int],
    side_step: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = size
    obstacles = set()
    current = start
    for slot in range(main_count):
        if slot in detour_slots:
            proposed = _move(current, main_step)
            rr, cc = proposed
            if main_step[0] != 0:
                if side_step[1] > 0:
                    block_col = cc
                else:
                    block_col = cc + w - 1
                block_row = rr + random.randrange(h)
            else:
                if side_step[0] > 0:
                    block_row = rr
                else:
                    block_row = rr + h - 1
                block_col = cc + random.randrange(w)
            obstacles.add((int(block_row), int(block_col)))
            current = _move(current, side_step)
        current = _move(current, main_step)
    return obstacles


def _alt_output_signatures(
    shape: tuple[int, int],
    start: tuple[int, int],
    size: tuple[int, int],
    dr: int,
    dc: int,
    main_axis: int,
    sprite_cells: set[tuple[int, int]],
    indicator_cells: set[tuple[int, int]],
) -> dict[tuple[int, ...], set[int]]:
    main_delta = dr if main_axis == 0 else dc
    minor_delta = dc if main_axis == 0 else dr
    main_step = (_sign(main_delta), 0) if main_axis == 0 else (0, _sign(main_delta))
    side_step = (0, _sign(minor_delta)) if main_axis == 0 else (_sign(minor_delta), 0)
    main_count = abs(main_delta)
    minor_count = abs(minor_delta)
    target = (start[0] + dr, start[1] + dc)
    base = np.zeros(shape, dtype=int)
    signatures: dict[tuple[int, ...], set[int]] = {}

    for detour_slots in itertools.combinations(range(main_count), minor_count):
        positions = _schedule_positions(start, main_step, side_step, main_count, set(detour_slots))
        if positions[-1] != target:
            continue
        out = _render_output(base, positions, size, sprite_cells, indicator_cells)
        sig = tuple(out.flatten().tolist()) + shape
        signatures.setdefault(sig, set()).add(hash(detour_slots))
    return signatures


def _place_path(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]] | None:
    rel = _relative_path_edges(dr, dc)
    rel_set = set(rel)
    rmin = min(r for r, _ in rel)
    rmax = max(r for r, _ in rel)
    cmin = min(c for _, c in rel)
    cmax = max(c for _, c in rel)

    low_r = 1 - rmin
    high_r = shape[0] - 2 - rmax
    low_c = 1 - cmin
    high_c = shape[1] - 2 - cmax
    if low_r > high_r or low_c > high_c:
        return None

    forbidden = _halo(occupied, shape)
    candidates = [
        (r0, c0)
        for r0 in range(low_r, high_r + 1)
        for c0 in range(low_c, high_c + 1)
    ]
    random.shuffle(candidates)

    for base_r, base_c in candidates[:PATH_PLACE_RETRIES]:
        abs_cells = {(base_r + r, base_c + c) for r, c in rel_set}
        if abs_cells & forbidden:
            continue
        red = (base_r, base_c)
        blue = (base_r + dr, base_c + dc)
        gray = abs_cells - {red, blue}
        if len(gray) < 3 or len(gray) > 6:
            continue
        return red, blue, gray
    return None


def _render_input(
    shape: tuple[int, int],
    start: tuple[int, int],
    size: tuple[int, int],
    obstacles: set[tuple[int, int]],
    red: tuple[int, int],
    blue: tuple[int, int],
    gray: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in _rect_cells(start, size):
        grid[r, c] = YELLOW
    for r, c in obstacles:
        grid[r, c] = ORANGE
    for r, c in gray:
        grid[r, c] = GREY
    grid[red] = RED
    grid[blue] = BLUE
    return grid


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    shape = (H, W)
    size = random.choice(SPRITE_SIZES)
    dr, dc, main_axis = _pick_displacement()

    main_delta = dr if main_axis == 0 else dc
    minor_delta = dc if main_axis == 0 else dr
    main_count = abs(main_delta)
    minor_count = abs(minor_delta)
    main_step = (_sign(main_delta), 0) if main_axis == 0 else (0, _sign(main_delta))
    side_step = (0, _sign(minor_delta)) if main_axis == 0 else (_sign(minor_delta), 0)
    detour_slots = set(random.sample(range(main_count), minor_count))

    rel_positions = _schedule_positions((0, 0), main_step, side_step, main_count, detour_slots)
    rel_sprite = set().union(*(_rect_cells(pos, size) for pos in rel_positions))

    rmin = min(r for r, _ in rel_sprite)
    rmax = max(r for r, _ in rel_sprite)
    cmin = min(c for _, c in rel_sprite)
    cmax = max(c for _, c in rel_sprite)
    row_choices = list(range(1 - rmin, H - 1 - rmax))
    col_choices = list(range(1 - cmin, W - 1 - cmax))
    if not row_choices or not col_choices:
        return None
    start = (random.choice(row_choices), random.choice(col_choices))

    positions = _schedule_positions(start, main_step, side_step, main_count, detour_slots)
    target = positions[-1]
    if target != (start[0] + dr, start[1] + dc):
        return None

    obstacles = _blocking_cells_for_schedule(start, detour_slots, main_count, size, main_step, side_step)
    occupied = set().union(*(_rect_cells(pos, size) for pos in positions)) | obstacles
    path_data = _place_path(shape, occupied, dr, dc)
    if path_data is None:
        return None
    red, blue, gray = path_data

    if obstacles & (_rect_cells(start, size) | _rect_cells(target, size)):
        return None

    grid = _render_input(shape, start, size, obstacles, red, blue, gray)
    sprite_top, parsed_size, sprite_cells = _find_sprite(grid)
    if sprite_top != start or parsed_size != size:
        return None

    indicator_cells = set(gray) | {red, blue}
    expected = _render_output(grid, positions, size, sprite_cells, indicator_cells)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    obstacle_mask = grid == ORANGE
    sim_positions = _simulate_positions(shape, start, target, size, obstacle_mask, main_axis)
    if sim_positions != positions:
        return None

    if minor_count > 0:
        signatures = _alt_output_signatures(shape, start, size, dr, dc, main_axis, sprite_cells, indicator_cells)
        if len(signatures) <= 1:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
