from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SATELLITE = 2
OBJECT_PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)
CW = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

MIN_SIDE = 14
MAX_SIDE = 20
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_AREA = 3
MAX_AREA = 14
GEN_RETRIES = 400
RAY_RETRIES = 300
OBJECT_RETRIES = 200


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if (
                        in_bounds(grid.shape, nr, nc)
                        and (nr, nc) not in seen
                        and int(grid[nr, nc]) == color
                    ):
                        seen.add((nr, nc))
                        q.append((nr, nc))
            components.append(comp)
    return components


def _trace_folded_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> tuple[list[tuple[int, int]], int]:
    assert length >= 1
    pos = start
    d = direction
    coords: list[tuple[int, int]] = []
    turns = 0
    for step in range(length):
        assert in_bounds(shape, pos[0], pos[1]), "ray left the grid"
        coords.append(pos)
        if step == length - 1:
            break
        nr = pos[0] + d[0]
        nc = pos[1] + d[1]
        while not in_bounds(shape, nr, nc):
            d = CW[d]
            turns += 1
            nr = pos[0] + d[0]
            nc = pos[1] + d[1]
        pos = (nr, nc)
    assert len(set(coords)) == len(coords), "ray must not self-overlap"
    return coords, turns


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    colors = sorted(set(map(int, g.ravel())))
    object_colors = [color for color in colors if color not in (bg, SATELLITE)]
    assert object_colors, "expected at least one object color"

    components: list[dict] = []
    cell_to_component: dict[tuple[int, int], int] = {}
    for color in object_colors:
        for coords in _components_of_color(g, color):
            comp_id = len(components)
            cell_set = set(coords)
            components.append(
                {
                    "id": comp_id,
                    "color": int(color),
                    "coords": coords,
                    "cells": cell_set,
                    "area": len(coords),
                }
            )
            for cell in coords:
                cell_to_component[cell] = comp_id

    satellites = [tuple(map(int, rc)) for rc in np.argwhere(g == SATELLITE)]
    assert satellites, "expected at least one red satellite pixel"
    assert len(satellites) == len(components), "expected one satellite per object"

    attached: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for sat in satellites:
        adjacent_object_cells = [cell for cell in neighbors4(*sat) if cell in cell_to_component]
        assert len(adjacent_object_cells) == 1, "each satellite must touch exactly one object cell"
        anchor = adjacent_object_cells[0]
        comp_id = cell_to_component[anchor]
        assert comp_id not in attached, "each object must have exactly one satellite"
        direction = (sat[0] - anchor[0], sat[1] - anchor[1])
        assert direction in DIRS, "satellite direction must be orthogonal"
        attached[comp_id] = (sat, direction)

    objects: list[dict] = []
    for comp in components:
        sat, direction = attached[comp["id"]]
        objects.append(
            {
                "color": comp["color"],
                "coords": comp["coords"],
                "cells": comp["cells"],
                "area": comp["area"],
                "satellite": sat,
                "direction": direction,
            }
        )
    return bg, objects


def _render_output(shape: tuple[int, int], bg: int, objects: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in objects:
        path, _turns = _trace_folded_ray(shape, obj["satellite"], obj["direction"], obj["area"])
        for cell in path:
            assert int(out[cell]) == bg, "rays must not overlap"
            out[cell] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each object into an equal-area ray that starts from its red nozzle and folds clockwise along borders.

    Input structure:
    - The grid has a black background, several separated single-color foreground objects, and one red satellite pixel for each object.
    - Each red satellite pixel lies outside its object and is orthogonally adjacent to exactly one object cell.
    - The satellite's side relative to that neighboring object cell encodes the emission direction.

    Rule:
    - Remove every object and every red satellite, leaving only the background canvas.
    - For each object, start at the satellite cell and draw a ray in the encoded direction using the object's color.
    - The ray length equals the object's area, counting the satellite cell as the first ray cell; whenever the next step would leave the grid, rotate the direction clockwise by 90 degrees and continue until the full length is drawn.

    Key insight:
    - The object shape only contributes its pixel count, while the single red satellite fixes both the ray's start and direction; long rays must be counted step by step through each clockwise border fold.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _analyze_scene(g)
    return _render_output(g.shape, bg, objects)


def _boundary_span(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> int:
    h, w = map(int, shape)
    r, c = start
    if direction == UP:
        return r + 1
    if direction == DOWN:
        return h - r
    if direction == LEFT:
        return c + 1
    return w - c


def _step(cell: tuple[int, int], direction: tuple[int, int], amount: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * amount, cell[1] + direction[1] * amount


def _sample_targets(n_objects: int) -> list[str]:
    if n_objects == 2:
        choices = [["straight", "fold"], ["fold", "corner"], ["straight", "corner"]]
        targets = random.choice(choices)
    elif n_objects == 3:
        targets = ["straight", "fold", "corner"]
    else:
        targets = ["straight", "fold", "corner", random.choice(["straight", "fold", "corner"])]
    random.shuffle(targets)
    return targets


def _boundary_start_for_turning(
    shape: tuple[int, int], require_corner_room: bool
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    side = random.choice(["top", "right", "bottom", "left"])
    if side == "top":
        if w < 3:
            return None
        c_hi = w - 3 if require_corner_room else w - 2
        if c_hi < 1:
            return None
        return (0, random.randint(1, c_hi)), UP
    if side == "right":
        if h < 3:
            return None
        r_hi = h - 3 if require_corner_room else h - 2
        if r_hi < 1:
            return None
        return (random.randint(1, r_hi), w - 1), RIGHT
    if side == "bottom":
        if w < 3:
            return None
        c_lo = 2 if require_corner_room else 1
        if c_lo > w - 2:
            return None
        return (h - 1, random.randint(c_lo, w - 2)), DOWN
    if h < 3:
        return None
    r_lo = 2 if require_corner_room else 1
    if r_lo > h - 2:
        return None
    return (random.randint(r_lo, h - 2), 0), LEFT


def _generic_straight_start(
    shape: tuple[int, int], preferred_direction: tuple[int, int] | None = None
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    directions = [preferred_direction] if preferred_direction is not None else list(DIRS)
    if preferred_direction is None:
        random.shuffle(directions)
    for direction in directions:
        candidates = []
        for r in range(h):
            for c in range(w):
                anchor = (r - direction[0], c - direction[1])
                if not in_bounds(shape, anchor[0], anchor[1]):
                    continue
                if _boundary_span(shape, (r, c), direction) >= MIN_AREA:
                    candidates.append((r, c))
        if candidates:
            return random.choice(candidates), direction
    return None


def _sample_ray_plan(
    shape: tuple[int, int],
    target: str,
    occupied_rays: set[tuple[int, int]],
    occupied_satellites: set[tuple[int, int]],
    preferred_direction: tuple[int, int] | None = None,
) -> dict | None:
    for _ in range(RAY_RETRIES):
        if target == "straight":
            sampled = _generic_straight_start(shape, preferred_direction)
        elif target == "fold":
            sampled = _boundary_start_for_turning(shape, require_corner_room=False)
        else:
            sampled = _boundary_start_for_turning(shape, require_corner_room=True)
        if sampled is None:
            continue

        start, direction = sampled
        if start in occupied_satellites:
            continue
        if any(max(abs(start[0] - rr), abs(start[1] - cc)) <= 1 for rr, cc in occupied_satellites):
            continue

        anchor = (start[0] - direction[0], start[1] - direction[1])
        if not in_bounds(shape, anchor[0], anchor[1]):
            continue

        span1 = _boundary_span(shape, start, direction)
        end1 = _step(start, direction, span1 - 1)
        direction2 = CW[direction]
        extra1 = _boundary_span(shape, end1, direction2) - 1
        end2 = _step(end1, direction2, extra1)
        direction3 = CW[direction2]
        extra2 = _boundary_span(shape, end2, direction3) - 1

        if target == "straight":
            lo = MIN_AREA
            hi = min(MAX_AREA, span1)
        elif target == "fold":
            lo = max(MIN_AREA, span1 + 1)
            hi = min(MAX_AREA, span1 + extra1)
        else:
            lo = max(MIN_AREA, span1 + extra1 + 1)
            hi = min(MAX_AREA, span1 + extra2 + extra1)
        if lo > hi:
            continue

        length = random.randint(lo, hi)
        path, turns = _trace_folded_ray(shape, start, direction, length)
        if target == "straight" and turns != 0:
            continue
        if target == "fold" and turns != 1:
            continue
        if target == "corner" and turns < 2:
            continue
        if any(cell in occupied_rays for cell in path):
            continue

        return {
            "satellite": start,
            "direction": direction,
            "anchor": anchor,
            "area": length,
            "path": path,
            "turns": turns,
        }
    return None


def _touches_any(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in occupied)


def _grow_object(
    shape: tuple[int, int],
    plan: dict,
    blocked_exact: set[tuple[int, int]],
    blocked_touching: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    sat = plan["satellite"]
    anchor = plan["anchor"]
    direction = plan["direction"]
    area = plan["area"]
    forbidden_orth = {cell for cell in neighbors4(*sat) if cell != anchor and in_bounds(shape, *cell)}
    if anchor in blocked_exact or _touches_any(anchor, blocked_touching):
        return None

    coords = {anchor}
    frontier = set()
    for nbr in neighbors4(*anchor):
        if in_bounds(shape, *nbr):
            frontier.add(nbr)

    for _ in range(area - 1):
        options = []
        for cell in list(frontier):
            if cell in coords or cell in blocked_exact or cell in forbidden_orth:
                continue
            if _touches_any(cell, blocked_touching):
                continue
            if cell == sat:
                continue
            attach = sum(nbr in coords for nbr in neighbors4(*cell))
            if attach == 0:
                continue
            dr = cell[0] - anchor[0]
            dc = cell[1] - anchor[1]
            behind = max(0, -(dr * direction[0] + dc * direction[1]))
            distance = abs(dr) + abs(dc)
            weight = ((attach + 1) ** 2) * (behind + 2) / (distance + 1)
            options.append((cell, weight))
        if not options:
            return None
        cells = [cell for cell, _weight in options]
        weights = [weight for _cell, weight in options]
        chosen = random.choices(cells, weights=weights, k=1)[0]
        coords.add(chosen)
        frontier.discard(chosen)
        for nbr in neighbors4(*chosen):
            if in_bounds(shape, *nbr):
                frontier.add(nbr)

    if any(cell in forbidden_orth for cell in coords):
        return None
    if len([cell for cell in neighbors4(*sat) if cell in coords]) != 1:
        return None
    return coords


def _validate_input_scene(grid: np.ndarray) -> list[dict]:
    bg, objects = _analyze_scene(grid)
    assert bg == BG, "generator must keep black as the background"
    touched: set[tuple[int, int]] = set()
    for obj in objects:
        assert obj["color"] != SATELLITE
        for cell in obj["cells"]:
            for other in touched:
                assert max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) > 1, "objects must stay separated"
        touched |= obj["cells"]
        touched.add(obj["satellite"])
        path, turns = _trace_folded_ray(grid.shape, obj["satellite"], obj["direction"], obj["area"])
        obj["path"] = path
        obj["turns"] = turns
    all_path_cells: set[tuple[int, int]] = set()
    for obj in objects:
        for cell in obj["path"]:
            assert cell not in all_path_cells, "rays must be non-overlapping"
            all_path_cells.add(cell)
    return objects


def _expected_from_plans(shape: tuple[int, int], plans: list[dict], colors: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for plan, color in zip(plans, colors):
        for cell in plan["path"]:
            assert int(out[cell]) == BG
            out[cell] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    shape = (h, w)
    n_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    targets = _sample_targets(n_objects)
    colors = random.sample(OBJECT_PALETTE, n_objects)

    preferred_dirs = list(DIRS)
    random.shuffle(preferred_dirs)

    plans: list[dict] = []
    occupied_rays: set[tuple[int, int]] = set()
    occupied_satellites: set[tuple[int, int]] = set()
    for i, target in enumerate(targets):
        preferred = preferred_dirs[i] if i < len(preferred_dirs) else None
        plan = _sample_ray_plan(shape, target, occupied_rays, occupied_satellites, preferred)
        if plan is None:
            return None
        plans.append(plan)
        occupied_rays |= set(plan["path"])
        occupied_satellites.add(plan["satellite"])

    all_satellites = {plan["satellite"] for plan in plans}
    object_cells_by_plan: list[set[tuple[int, int]]] = []
    placed_objects: set[tuple[int, int]] = set()
    for index, plan in enumerate(plans):
        other_satellites = all_satellites - {plan["satellite"]}
        blocked_exact = set(occupied_rays) | all_satellites | placed_objects
        blocked_touching = set(placed_objects) | set(other_satellites)
        coords = None
        for _ in range(OBJECT_RETRIES):
            coords = _grow_object(shape, plan, blocked_exact, blocked_touching)
            if coords is not None:
                break
        if coords is None:
            return None
        object_cells_by_plan.append(coords)
        placed_objects |= coords

    grid = np.full(shape, BG, dtype=int)
    for coords, plan, color in zip(object_cells_by_plan, plans, colors):
        for r, c in coords:
            grid[r, c] = int(color)
        sr, sc = plan["satellite"]
        assert int(grid[sr, sc]) == BG
        grid[sr, sc] = SATELLITE

    try:
        analyzed = _validate_input_scene(grid)
        expected = _expected_from_plans(shape, plans, colors)
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    actual_turn_buckets = {0: 0, 1: 0, 2: 0}
    for obj in analyzed:
        bucket = min(2, int(obj["turns"]))
        actual_turn_buckets[bucket] += 1
    for target in targets:
        if target == "straight" and actual_turn_buckets[0] == 0:
            return None
        if target == "fold" and actual_turn_buckets[1] == 0:
            return None
        if target == "corner" and actual_turn_buckets[2] == 0:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-ray-fold input")
