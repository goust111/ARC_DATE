from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

TOWER_COLORS = (BLUE, RED, YELLOW)
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIR_TO_CORNER = {
    (1, 1): (0, 0),
    (1, -1): (0, 1),
    (-1, 1): (1, 0),
    (-1, -1): (1, 1),
}
PAIR_MIX = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

GEN_ATTEMPTS = 600
RAY_ATTEMPTS = 120
MIN_SEG_LEN = 1
MAX_SEG_LEN = 7
GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _iter_segment(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _steps_to_boundary(start: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = start
    dr, dc = direction
    steps = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(r, c, h, w):
            return steps
        steps += 1


def _extend_to_boundary(start: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not _in_bounds(r, c, h, w):
            return cells
        cells.append((r, c))


def _tower_cells_from_green(green_rc: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    gr, gc = green_rc
    rr, cc = DIR_TO_CORNER[direction]
    r0 = gr - rr
    c0 = gc - cc
    return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]


def _mirror_cells(
    vertex: tuple[int, int],
    direction_before: tuple[int, int],
    direction_after: tuple[int, int],
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    vr, vc = vertex
    dr0, dc0 = direction_before
    dr1, dc1 = direction_after

    if dr0 == -dr1 and dc0 == dc1:
        required = (vr + dr0, vc)
        extras = [(required[0], required[1] - 1), (required[0], required[1] + 1)]
    elif dr0 == dr1 and dc0 == -dc1:
        required = (vr, vc + dc0)
        extras = [(required[0] - 1, required[1]), (required[0] + 1, required[1])]
    else:
        return None

    if not _in_bounds(required[0], required[1], h, w) or required in blocked:
        return None

    random.shuffle(extras)
    for extra in extras:
        if _in_bounds(extra[0], extra[1], h, w) and extra not in blocked:
            return [required, extra]
    return None


def _trace_ray(
    grid: np.ndarray,
    horiz_mirror: np.ndarray,
    vert_mirror: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    max_reflections: int = 3,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    reflections = 0

    while True:
        state = (r, c, dr, dc, reflections)
        assert state not in seen, "ray cycle detected"
        seen.add(state)

        hit_h = 0 <= r + dr < h and bool(horiz_mirror[r + dr, c])
        hit_v = 0 <= c + dc < w and bool(vert_mirror[r, c + dc])

        if hit_h or hit_v:
            if reflections >= max_reflections:
                break
            if hit_h:
                dr = -dr
            if hit_v:
                dc = -dc
            reflections += 1

        nr = r + dr
        nc = c + dc
        if not _in_bounds(nr, nc, h, w):
            break
        path.append((nr, nc))
        r, c = nr, nc

    return path


def _find_towers(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    towers: list[dict[str, object]] = []

    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if used[r0 : r0 + 2, c0 : c0 + 2].any():
                continue
            block = grid[r0 : r0 + 2, c0 : c0 + 2]
            green_locs = np.argwhere(block == GREEN)
            if green_locs.shape[0] != 1:
                continue
            others = [int(v) for v in block.flatten() if int(v) != GREEN]
            if len(set(others)) != 1:
                continue
            color = others[0]
            if color == BLACK:
                continue
            gr, gc = map(int, green_locs[0])
            direction = {
                (0, 0): (1, 1),
                (0, 1): (1, -1),
                (1, 0): (-1, 1),
                (1, 1): (-1, -1),
            }[(gr, gc)]
            cells = [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
            used[r0 : r0 + 2, c0 : c0 + 2] = True
            towers.append(
                {
                    "color": int(color),
                    "green": (r0 + gr, c0 + gc),
                    "direction": direction,
                    "cells": cells,
                }
            )

    return towers


def _mirror_orientation(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    orange = grid == ORANGE
    horiz = np.zeros((h, w), dtype=bool)
    vert = np.zeros((h, w), dtype=bool)

    for r, c in np.argwhere(orange):
        r = int(r)
        c = int(c)
        if (c > 0 and orange[r, c - 1]) or (c + 1 < w and orange[r, c + 1]):
            horiz[r, c] = True
        if (r > 0 and orange[r - 1, c]) or (r + 1 < h and orange[r + 1, c]):
            vert[r, c] = True

    return horiz, vert


def _mix_color(colors: set[int]) -> int | None:
    if len(colors) < 2:
        return None
    if colors == {RED, BLUE, YELLOW}:
        return GREY
    return PAIR_MIX.get(frozenset(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from 2x2 towers, reflect them off orange mirror walls,
    and mark crossings of differently colored rays with a mixed color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    towers = _find_towers(g)
    horiz_mirror, vert_mirror = _mirror_orientation(g)

    visits: dict[tuple[int, int], set[int]] = defaultdict(set)
    for tower in towers:
        color = int(tower["color"])
        start = tower["green"]
        direction = tower["direction"]
        ray_cells = _trace_ray(g, horiz_mirror, vert_mirror, start, direction)
        for r, c in set(ray_cells):
            visits[(r, c)].add(color)

    for (r, c), colors in visits.items():
        if int(g[r, c]) == BLACK:
            if len(colors) == 1:
                out[r, c] = next(iter(colors))
            else:
                mixed = _mix_color(set(map(int, colors)))
                if mixed is not None:
                    out[r, c] = int(mixed)

    return out


def _sample_segment_lengths(limit: int) -> list[int]:
    if limit <= 0:
        return []
    low = min(MIN_SEG_LEN, limit)
    high = min(MAX_SEG_LEN, limit)
    lengths = list(range(low, high + 1))
    random.shuffle(lengths)
    if 1 <= limit < low:
        return [limit]
    return lengths


def _valid_path_cells(
    cells: list[tuple[int, int]],
    blocked_objects: set[tuple[int, int]],
    blocked_paths: Counter,
    local_paths: set[tuple[int, int]],
    allowed_overlap: set[tuple[int, int]],
) -> bool:
    for cell in cells:
        if cell in blocked_objects:
            return False
        if cell in local_paths:
            return False
        if blocked_paths[cell] and cell not in allowed_overlap:
            return False
    return True


def _valid_object_cells(
    cells: list[tuple[int, int]],
    blocked_objects: set[tuple[int, int]],
    blocked_paths: Counter,
    local_paths: set[tuple[int, int]],
    local_objects: set[tuple[int, int]],
) -> bool:
    for cell in cells:
        if cell in blocked_objects or cell in local_objects or cell in local_paths:
            return False
        if blocked_paths[cell]:
            return False
    return True


def _build_prefix(
    p: tuple[int, int],
    arrival_dir: tuple[int, int],
    pre_reflections: int,
    h: int,
    w: int,
    blocked_objects: set[tuple[int, int]],
    blocked_paths: Counter,
    allowed_overlap: set[tuple[int, int]],
) -> dict[str, object] | None:
    backward_dir = (-arrival_dir[0], -arrival_dir[1])
    cur = p
    cur_dir = backward_dir
    back_cells: list[tuple[int, int]] = []
    local_paths: set[tuple[int, int]] = set()
    local_objects: set[tuple[int, int]] = set()
    mirrors: list[list[tuple[int, int]]] = []

    for _ in range(pre_reflections):
        options: list[tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], tuple[int, int]]] = []
        max_steps = _steps_to_boundary(cur, cur_dir, h, w)
        for length in _sample_segment_lengths(max_steps):
            seg = _iter_segment(cur, cur_dir, length)
            if not _valid_path_cells(seg, blocked_objects | local_objects, blocked_paths, local_paths, allowed_overlap):
                continue
            vertex = seg[-1]
            next_dirs = [(-cur_dir[0], cur_dir[1]), (cur_dir[0], -cur_dir[1])]
            random.shuffle(next_dirs)
            for next_dir in next_dirs:
                future_steps = _steps_to_boundary(vertex, next_dir, h, w)
                if future_steps <= 0:
                    continue
                mirror = _mirror_cells(
                    vertex,
                    cur_dir,
                    next_dir,
                    h,
                    w,
                    blocked_objects | local_objects | local_paths | set(seg) | {p},
                )
                if mirror is None:
                    continue
                options.append((seg, next_dir, mirror, vertex))
        if not options:
            return None
        seg, cur_dir, mirror, cur = random.choice(options)
        back_cells.extend(seg)
        local_paths.update(seg)
        local_objects.update(mirror)
        mirrors.append(mirror)

    max_steps = _steps_to_boundary(cur, cur_dir, h, w)
    finals: list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], tuple[int, int]]] = []
    for length in _sample_segment_lengths(max_steps):
        seg = _iter_segment(cur, cur_dir, length)
        if not seg:
            continue
        if not _valid_path_cells(seg, blocked_objects | local_objects, blocked_paths, local_paths, allowed_overlap):
            continue
        green_rc = seg[-1]
        initial_dir = (-cur_dir[0], -cur_dir[1])
        tower_cells = _tower_cells_from_green(green_rc, initial_dir)
        if not all(_in_bounds(r, c, h, w) for r, c in tower_cells):
            continue
        if not _valid_object_cells(tower_cells, blocked_objects, blocked_paths, local_paths | set(seg[:-1]), local_objects):
            continue
        finals.append((seg, tower_cells, green_rc, initial_dir))
    if not finals:
        return None

    seg, tower_cells, green_rc, initial_dir = random.choice(finals)
    back_cells.extend(seg)
    forward_cells = list(reversed(back_cells))
    prefix_cells = forward_cells[1:] + [p]

    return {
        "green": green_rc,
        "initial_dir": initial_dir,
        "path": prefix_cells,
        "path_set": set(prefix_cells),
        "objects": set(local_objects) | set(tower_cells),
        "mirrors": mirrors,
        "tower_cells": tower_cells,
    }


def _build_suffix(
    p: tuple[int, int],
    depart_dir: tuple[int, int],
    post_reflections: int,
    h: int,
    w: int,
    blocked_objects: set[tuple[int, int]],
    blocked_paths: Counter,
    allowed_overlap: set[tuple[int, int]],
) -> dict[str, object] | None:
    cur = p
    cur_dir = depart_dir
    path: list[tuple[int, int]] = []
    local_paths: set[tuple[int, int]] = set()
    local_objects: set[tuple[int, int]] = set()
    mirrors: list[list[tuple[int, int]]] = []

    for _ in range(post_reflections):
        options: list[tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], tuple[int, int]]] = []
        max_steps = _steps_to_boundary(cur, cur_dir, h, w)
        for length in _sample_segment_lengths(max_steps):
            seg = _iter_segment(cur, cur_dir, length)
            if not _valid_path_cells(seg, blocked_objects | local_objects, blocked_paths, local_paths, allowed_overlap):
                continue
            vertex = seg[-1]
            next_dirs = [(-cur_dir[0], cur_dir[1]), (cur_dir[0], -cur_dir[1])]
            random.shuffle(next_dirs)
            for next_dir in next_dirs:
                mirror = _mirror_cells(
                    vertex,
                    cur_dir,
                    next_dir,
                    h,
                    w,
                    blocked_objects | local_objects | local_paths | set(seg) | {p},
                )
                if mirror is None:
                    continue
                options.append((seg, next_dir, mirror, vertex))
        if not options:
            return None
        seg, cur_dir, mirror, cur = random.choice(options)
        path.extend(seg)
        local_paths.update(seg)
        local_objects.update(mirror)
        mirrors.append(mirror)

    final_seg = _extend_to_boundary(cur, cur_dir, h, w)
    if not _valid_path_cells(final_seg, blocked_objects | local_objects, blocked_paths, local_paths, allowed_overlap):
        return None
    path.extend(final_seg)

    return {
        "path": path,
        "path_set": set(path),
        "objects": set(local_objects),
        "mirrors": mirrors,
    }


def _choose_intersection_point(
    h: int,
    w: int,
    scenario: str,
    primary_dir: tuple[int, int],
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    if scenario == "pair":
        rows = list(range(2, h - 2))
        cols = list(range(2, w - 2))
        if not rows or not cols:
            return None
        return (random.choice(rows), random.choice(cols)), set()

    overlap_dir = (-primary_dir[0], -primary_dir[1])
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if _steps_to_boundary((r, c), overlap_dir, h, w) != 1:
                continue
            if _steps_to_boundary((r, c), primary_dir, h, w) < 2:
                continue
            other_axis = [d for d in DIAGS if d[0] * d[1] != primary_dir[0] * primary_dir[1]]
            if min(_steps_to_boundary((r, c), d, h, w) for d in other_axis) < 2:
                continue
            shared = {(r, c)}
            shared.update(_extend_to_boundary((r, c), overlap_dir, h, w))
            candidates.append(((r, c), shared))
    if not candidates:
        return None
    return random.choice(candidates)


def _ray_total_reflections(force_nonzero: bool) -> int:
    weights = [1, 3, 3, 2] if force_nonzero else [3, 4, 3, 2]
    return random.choices([0, 1, 2, 3], weights=weights, k=1)[0]


def _add_ray(
    specs: list[dict[str, object]],
    color: int,
    p: tuple[int, int],
    arrival_dir: tuple[int, int],
    h: int,
    w: int,
    blocked_objects: set[tuple[int, int]],
    blocked_paths: Counter,
    allowed_overlap: set[tuple[int, int]],
    total_reflections: int,
) -> dict[str, object] | None:
    for _ in range(RAY_ATTEMPTS):
        pre = random.randint(0, total_reflections)
        post = total_reflections - pre

        prefix = _build_prefix(p, arrival_dir, pre, h, w, blocked_objects, blocked_paths, allowed_overlap)
        if prefix is None:
            continue
        suffix = _build_suffix(p, arrival_dir, post, h, w, blocked_objects | set(prefix["objects"]), blocked_paths + Counter(prefix["path"]), allowed_overlap)
        if suffix is None:
            continue

        full_path = list(prefix["path"]) + list(suffix["path"])
        path_set = set(full_path)
        if any(blocked_paths[cell] and cell not in allowed_overlap for cell in path_set):
            continue
        return {
            "color": color,
            "green": prefix["green"],
            "direction": prefix["initial_dir"],
            "tower_cells": prefix["tower_cells"],
            "mirrors": list(prefix["mirrors"]) + list(suffix["mirrors"]),
            "path": full_path,
            "path_set": path_set,
            "objects": set(prefix["objects"]) | set(suffix["objects"]),
        }
    return None


def _render_input(h: int, w: int, rays: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)

    for ray in rays:
        color = int(ray["color"])
        green_rc = tuple(ray["green"])
        direction = tuple(ray["direction"])
        green_corner = DIR_TO_CORNER[direction]
        tower_cells = list(ray["tower_cells"])
        r0 = min(r for r, _ in tower_cells)
        c0 = min(c for _, c in tower_cells)
        for dr in range(2):
            for dc in range(2):
                cell = (r0 + dr, c0 + dc)
                grid[cell] = GREEN if (dr, dc) == green_corner else color

    for ray in rays:
        for mirror in ray["mirrors"]:
            for r, c in mirror:
                grid[r, c] = ORANGE

    return grid


def _mixed_pixel_counts(out: np.ndarray) -> dict[int, int]:
    counts = {}
    for color in (MAGENTA, ORANGE, GREEN, GREY):
        counts[color] = int(np.count_nonzero(out == color))
    return counts


def _paint_tower(grid: np.ndarray, green_rc: tuple[int, int], direction: tuple[int, int], color: int) -> list[tuple[int, int]]:
    cells = _tower_cells_from_green(green_rc, direction)
    h, w = grid.shape
    if not all(_in_bounds(r, c, h, w) for r, c in cells):
        raise ValueError("tower out of bounds")
    if any(int(grid[r, c]) != BLACK for r, c in cells):
        raise ValueError("tower overlap")

    rr, cc = DIR_TO_CORNER[direction]
    r0 = green_rc[0] - rr
    c0 = green_rc[1] - cc
    for dr in range(2):
        for dc in range(2):
            grid[r0 + dr, c0 + dc] = GREEN if (dr, dc) == (rr, cc) else color
    return cells


def _paint_vertical_reflection(
    grid: np.ndarray,
    vertex: tuple[int, int],
    direction_before: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    vr, vc = vertex
    dr, dc = direction_before
    required = (vr, vc + dc)
    extras = [(required[0] + 1, required[1]), (required[0] - 1, required[1])]
    h, w = grid.shape
    if not _in_bounds(required[0], required[1], h, w) or required in occupied:
        raise ValueError("mirror required cell blocked")
    for extra in extras:
        if _in_bounds(extra[0], extra[1], h, w) and extra not in occupied:
            cells = [required, extra]
            for r, c in cells:
                grid[r, c] = ORANGE
            return cells
    raise ValueError("mirror extension blocked")


def _paint_distractor_mirror(grid: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    choices: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            horiz = [(r, c), (r, c + 1)]
            if c + 1 < w and all(cell not in occupied for cell in horiz):
                choices.append(horiz)
            vert = [(r, c), (r + 1, c)]
            if r + 1 < h and all(cell not in occupied for cell in vert):
                choices.append(vert)
    if not choices:
        return False
    cells = random.choice(choices)
    for r, c in cells:
        grid[r, c] = ORANGE
    return True


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=int)


def _construct_pair_template() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if h < 10 or w < 10:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()

    pr = random.randint(3, h - 5)
    pc = random.randint(3, w - 4)
    p = (pr, pc)

    max_a = min(pr, pc, 5)
    max_b = min(pr, w - 1 - pc, 5)
    if max_a < 2 or max_b < 2:
        return None
    da = random.randint(2, max_a)
    db = random.randint(2, max_b)

    colors = list(random.choice([(RED, BLUE), (RED, YELLOW), (BLUE, YELLOW)]))
    random.shuffle(colors)

    try:
        occupied.update(_paint_tower(grid, (pr - da, pc - da), (1, 1), colors[0]))
        occupied.update(_paint_tower(grid, (pr - db, pc + db), (1, -1), colors[1]))
    except ValueError:
        return None

    num_reflections = random.choices([0, 1, 2, 3], weights=[2, 4, 3, 2], k=1)[0]
    current = p
    current_dir = (1, 1)
    occupied_paths = {p}

    for i in range(num_reflections):
        max_steps = _steps_to_boundary(current, current_dir, h, w) - 1
        if max_steps < 1:
            return None
        step_hi = min(max_steps, 3 + i * 2)
        step = random.randint(1, step_hi)
        seg = _iter_segment(current, current_dir, step)
        if any(cell in occupied for cell in seg):
            return None
        vertex = seg[-1]
        try:
            mirror_cells = _paint_vertical_reflection(grid, vertex, current_dir, occupied | set(seg))
        except ValueError:
            return None
        occupied.update(mirror_cells)
        occupied_paths.update(seg)
        current = vertex
        current_dir = (current_dir[0], -current_dir[1])

    if num_reflections == 0 and not _paint_distractor_mirror(grid, occupied):
        return None

    return _apply_random_transform(grid)


def _construct_triple_template() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if h < 11 or w < 10:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()

    pr = 2
    pc = random.randint(2, w - 3)
    p = (pr, pc)

    max_dc = min(h - 2, w - 2 - pc)
    if max_dc < 2:
        return None
    dc = random.randint(2, max_dc)
    colors = [RED, BLUE, YELLOW]
    random.shuffle(colors)

    try:
        occupied.update(_paint_tower(grid, (0, pc - 2), (1, 1), colors[0]))
        occupied.update(_paint_tower(grid, (0, pc + 2), (1, -1), colors[1]))
        occupied.update(_paint_tower(grid, (pr + dc, pc + dc), (-1, -1), colors[2]))
    except ValueError:
        return None

    num_reflections = random.choices([0, 1, 2], weights=[2, 4, 2], k=1)[0]
    current = p
    current_dir = (1, -1)
    for i in range(num_reflections):
        max_steps = _steps_to_boundary(current, current_dir, h, w) - 1
        if max_steps < 1:
            return None
        step_hi = min(max_steps, 3 + i * 2)
        step = random.randint(1, step_hi)
        seg = _iter_segment(current, current_dir, step)
        if any(cell in occupied for cell in seg):
            return None
        vertex = seg[-1]
        try:
            mirror_cells = _paint_vertical_reflection(grid, vertex, current_dir, occupied | set(seg))
        except ValueError:
            return None
        occupied.update(mirror_cells)
        current = vertex
        current_dir = (current_dir[0], -current_dir[1])

    if num_reflections == 0 and not _paint_distractor_mirror(grid, occupied):
        return None

    return _apply_random_transform(grid)


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(["pair", "triple"])
    inp = _construct_pair_template() if scenario == "pair" else _construct_triple_template()
    if inp is None:
        return None
    try:
        out = main(inp)
    except AssertionError:
        return None

    counts = _mixed_pixel_counts(out)
    if scenario == "pair":
        if counts[MAGENTA] + counts[ORANGE] + counts[GREEN] <= 0:
            return None
    else:
        if counts[GREY] <= 0:
            return None

    num_towers = int(np.count_nonzero(inp == GREEN))
    if num_towers not in (2, 3):
        return None
    if int(np.count_nonzero(inp == ORANGE)) < 2:
        return None

    towers = _find_towers(inp)
    if len(towers) != num_towers:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray ARC input")
