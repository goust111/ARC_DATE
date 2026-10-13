import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
MAROON = 9
SOURCE_COLORS = [1, 2, 3, 4, 6, 7]
MAX_GENERATE_ATTEMPTS = 400
NON_MAROON_KINDS = ["edge", "parallel", "turn_edge"]
MAROON_KINDS = ["maroon", "turn_maroon"]

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = [UP, RIGHT, DOWN, LEFT]

# The prose says "90° reflection" but the tie-break is contradictory.
# This implementation uses the visible bar endpoint that is hit:
# - vertical 3x1 mirrors turn horizontal rays up/down when their top/bottom
#   endpoint is hit; the middle cell absorbs.
# - horizontal 1x3 mirrors turn vertical rays left/right when their left/right
#   endpoint is hit; the middle cell absorbs.
SPRITE_BBOX_CELLS = {
    UP: [(0, 1), (1, 0), (1, 1)],
    RIGHT: [(0, 0), (1, 0), (1, 1)],
    DOWN: [(0, 0), (0, 1), (1, 0)],
    LEFT: [(0, 0), (0, 1), (1, 1)],
}
SPRITE_FRONT_OFFSETS = {
    UP: (0, 1),
    RIGHT: (1, 1),
    DOWN: (1, 0),
    LEFT: (0, 0),
}
SPRITE_BY_MASK = {
    frozenset(cells): direction for direction, cells in SPRITE_BBOX_CELLS.items()
}
SPRITE_FRONT_ANCHOR_OFFSETS = {
    direction: [
        (r - SPRITE_FRONT_OFFSETS[direction][0], c - SPRITE_FRONT_OFFSETS[direction][1])
        for r, c in cells
    ]
    for direction, cells in SPRITE_BBOX_CELLS.items()
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def sprite_cells_from_front(front: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    fr, fc = front
    return {
        (fr + dr, fc + dc)
        for dr, dc in SPRITE_FRONT_ANCHOR_OFFSETS[direction]
    }


def trace_segment(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def monochrome_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def detect_sources(grid: np.ndarray) -> list[dict]:
    reserved = {BG, GREY, MAROON}
    colors = {int(v) for v in np.unique(grid) if int(v) not in reserved}
    comps = monochrome_components(grid, colors)
    sources = []
    for comp in comps:
        coords = comp["coords"]
        if len(coords) != 3:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        mask = frozenset((r - r0, c - c0) for r, c in coords)
        if mask not in SPRITE_BY_MASK:
            continue
        direction = SPRITE_BY_MASK[mask]
        fr, fc = SPRITE_FRONT_OFFSETS[direction]
        sources.append(
            {
                "color": int(comp["color"]),
                "direction": direction,
                "front": (r0 + fr, c0 + fc),
                "coords": set(coords),
                "bbox": comp["bbox"],
            }
        )
    sources.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return sources


def detect_mirrors(grid: np.ndarray) -> dict[tuple[int, int], tuple[str, int]]:
    comps = monochrome_components(grid, {GREY})
    mirrors: dict[tuple[int, int], tuple[str, int]] = {}
    for comp in comps:
        coords = comp["coords"]
        if len(coords) != 3:
            continue
        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        if len(rows) == 1 and len(cols) == 3 and cols[-1] - cols[0] == 2:
            row = rows[0]
            for idx, col in enumerate(sorted(c for _, c in coords)):
                mirrors[(row, col)] = ("H", idx)
        elif len(cols) == 1 and len(rows) == 3 and rows[-1] - rows[0] == 2:
            col = cols[0]
            for idx, row in enumerate(sorted(r for r, _ in coords)):
                mirrors[(row, col)] = ("V", idx)
    return mirrors


def mirror_reflection(
    mirror_info: tuple[str, int], direction: tuple[int, int]
) -> tuple[int, int] | None:
    kind, index = mirror_info
    dr, dc = direction
    if kind == "H":
        if dc != 0:
            return None
        if index == 0:
            return LEFT
        if index == 2:
            return RIGHT
        return None
    if dr != 0:
        return None
    if index == 0:
        return UP
    if index == 2:
        return DOWN
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit an azure ray from each 3-pixel L sprite, reflect it off grey bar mirrors,
    then stamp a rotated copy of the source sprite at the ray endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    mirrors = detect_mirrors(g)
    sources = detect_sources(g)
    assert sources, "expected at least one source sprite"

    for source in sources:
        front = source["front"]
        direction = source["direction"]
        r, c = front
        dr, dc = direction
        seen_states = {(r, c, dr, dc)}

        while True:
            nr, nc = r + dr, c + dc
            if not in_bounds(g.shape, nr, nc):
                endpoint = (r, c)
                final_direction = (dr, dc)
                break

            cell = int(g[nr, nc])
            if cell == BG:
                out[nr, nc] = AZURE
                r, c = nr, nc
                continue

            if cell == MAROON:
                endpoint = (r, c)
                final_direction = (dr, dc)
                break

            if cell == GREY and (nr, nc) in mirrors:
                reflected = mirror_reflection(mirrors[(nr, nc)], (dr, dc))
                if reflected is None:
                    endpoint = (r, c)
                    final_direction = (dr, dc)
                    break
                dr, dc = reflected
                state = (r, c, dr, dc)
                if state in seen_states:
                    endpoint = (r, c)
                    final_direction = (dr, dc)
                    break
                seen_states.add(state)
                continue

            endpoint = (r, c)
            final_direction = (dr, dc)
            break

        for rr, cc in sprite_cells_from_front(endpoint, final_direction):
            assert in_bounds((h, w), rr, cc), "copied sprite must fit in-bounds"
            out[rr, cc] = source["color"]

    return out


def zone_bounds(
    shape: tuple[int, int], zone_name: str
) -> tuple[int, int, int, int]:
    h, w = shape
    mid_r = h // 2
    mid_c = w // 2
    if zone_name == "NW":
        return 0, mid_r - 1, 0, mid_c - 1
    if zone_name == "NE":
        return 0, mid_r - 1, mid_c + 1, w - 1
    if zone_name == "SW":
        return mid_r + 1, h - 1, 0, mid_c - 1
    return mid_r + 1, h - 1, mid_c + 1, w - 1


def zone_dirs(zone_name: str) -> tuple[tuple[int, int], tuple[int, int]]:
    vertical = UP if zone_name.startswith("N") else DOWN
    horizontal = LEFT if zone_name.endswith("W") else RIGHT
    return vertical, horizontal


def cell_in_zone(cell: tuple[int, int], zone: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, r1, c0, c1 = zone
    return r0 <= r <= r1 and c0 <= c <= c1


def cells_in_zone(cells: set[tuple[int, int]], zone: tuple[int, int, int, int]) -> bool:
    return all(cell_in_zone(cell, zone) for cell in cells)


def free_of_reserved(cells: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    return not (cells & reserved)


def fronts_on_edge(
    shape: tuple[int, int], zone: tuple[int, int, int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r0, r1, c0, c1 = zone
    candidates = []
    if direction == UP:
        row = 0
        for col in range(c0, c1 + 1):
            front = (row, col)
            cells = sprite_cells_from_front(front, direction)
            if all(in_bounds(shape, rr, cc) for rr, cc in cells) and cells_in_zone(cells, zone):
                candidates.append(front)
    elif direction == DOWN:
        row = h - 1
        for col in range(c0, c1 + 1):
            front = (row, col)
            cells = sprite_cells_from_front(front, direction)
            if all(in_bounds(shape, rr, cc) for rr, cc in cells) and cells_in_zone(cells, zone):
                candidates.append(front)
    elif direction == LEFT:
        col = 0
        for row in range(r0, r1 + 1):
            front = (row, col)
            cells = sprite_cells_from_front(front, direction)
            if all(in_bounds(shape, rr, cc) for rr, cc in cells) and cells_in_zone(cells, zone):
                candidates.append(front)
    else:
        col = w - 1
        for row in range(r0, r1 + 1):
            front = (row, col)
            cells = sprite_cells_from_front(front, direction)
            if all(in_bounds(shape, rr, cc) for rr, cc in cells) and cells_in_zone(cells, zone):
                candidates.append(front)
    return candidates


def move(point: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return point[0] + direction[0] * steps, point[1] + direction[1] * steps


def parallel_mirror_cells(
    hit: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]]:
    r, c = hit
    if direction == UP:
        return {(r, c), (r - 1, c), (r - 2, c)}
    if direction == DOWN:
        return {(r, c), (r + 1, c), (r + 2, c)}
    if direction == LEFT:
        return {(r, c), (r, c - 1), (r, c - 2)}
    return {(r, c), (r, c + 1), (r, c + 2)}


def turning_mirror_cells(
    turn_point: tuple[int, int], initial: tuple[int, int], final: tuple[int, int]
) -> set[tuple[int, int]]:
    hit = move(turn_point, initial)
    r, c = hit
    if initial in (LEFT, RIGHT):
        if final == UP:
            return {(r, c), (r + 1, c), (r + 2, c)}
        return {(r, c), (r - 1, c), (r - 2, c)}
    if final == LEFT:
        return {(r, c), (r, c + 1), (r, c + 2)}
    return {(r, c), (r, c - 1), (r, c - 2)}


def choose_source_front(
    start: tuple[int, int],
    direction: tuple[int, int],
    zone: tuple[int, int, int, int],
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    min_steps: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    candidates = []
    distance = min_steps
    current = move(start, direction, -distance)
    while cell_in_zone(current, zone):
        path = trace_segment(current, direction, distance)
        if path[-1] != start:
            raise AssertionError("internal path construction mismatch")
        source_cells = sprite_cells_from_front(current, direction)
        if (
            all(in_bounds(shape, rr, cc) for rr, cc in source_cells)
            and cells_in_zone(source_cells, zone)
            and free_of_reserved(source_cells | set(path), reserved)
        ):
            candidates.append((current, path))
        distance += 1
        current = move(start, direction, -distance)
    if not candidates:
        return None
    return random.choice(candidates)


def place_edge_scene(
    grid: np.ndarray,
    zone_name: str,
    direction: tuple[int, int],
    color: int,
    reserved: set[tuple[int, int]],
) -> dict | None:
    zone = zone_bounds(grid.shape, zone_name)
    endpoints = fronts_on_edge(grid.shape, zone, direction)
    random.shuffle(endpoints)
    for endpoint in endpoints:
        result = choose_source_front(endpoint, direction, zone, grid.shape, reserved, min_steps=2)
        if result is None:
            continue
        source_front, path = result
        source_cells = sprite_cells_from_front(source_front, direction)
        copy_cells = sprite_cells_from_front(endpoint, direction)
        return {
            "color": color,
            "direction": direction,
            "source_cells": source_cells,
            "mirror_cells": set(),
            "absorber": None,
            "reserved": source_cells | set(path) | copy_cells,
        }
    return None


def place_maroon_scene(
    grid: np.ndarray,
    zone_name: str,
    direction: tuple[int, int],
    color: int,
    reserved: set[tuple[int, int]],
) -> dict | None:
    zone = zone_bounds(grid.shape, zone_name)
    absorbers = fronts_on_edge(grid.shape, zone, direction)
    random.shuffle(absorbers)
    for absorber in absorbers:
        endpoint = move(absorber, direction, -1)
        copy_cells = sprite_cells_from_front(endpoint, direction)
        if (
            not in_bounds(grid.shape, *endpoint)
            or not cells_in_zone(copy_cells, zone)
            or not free_of_reserved(copy_cells | {absorber}, reserved)
        ):
            continue
        result = choose_source_front(endpoint, direction, zone, grid.shape, reserved | {absorber}, min_steps=2)
        if result is None:
            continue
        source_front, path = result
        source_cells = sprite_cells_from_front(source_front, direction)
        return {
            "color": color,
            "direction": direction,
            "source_cells": source_cells,
            "mirror_cells": set(),
            "absorber": absorber,
            "reserved": source_cells | set(path) | copy_cells | {absorber},
        }
    return None


def place_parallel_scene(
    grid: np.ndarray,
    zone_name: str,
    direction: tuple[int, int],
    color: int,
    reserved: set[tuple[int, int]],
) -> dict | None:
    zone = zone_bounds(grid.shape, zone_name)
    mirror_edge_hits = fronts_on_edge(grid.shape, zone, direction)
    # Move one step inward so the sprite stops before the mirror rather than at the boundary.
    hits = [move(hit, direction, -3) for hit in mirror_edge_hits]
    random.shuffle(hits)
    for hit in hits:
        mirror_cells = parallel_mirror_cells(hit, direction)
        endpoint = move(hit, direction, -1)
        copy_cells = sprite_cells_from_front(endpoint, direction)
        if (
            not all(in_bounds(grid.shape, rr, cc) for rr, cc in mirror_cells)
            or not cells_in_zone(mirror_cells | copy_cells, zone)
            or not free_of_reserved(mirror_cells | copy_cells, reserved)
        ):
            continue
        result = choose_source_front(
            endpoint,
            direction,
            zone,
            grid.shape,
            reserved | mirror_cells,
            min_steps=2,
        )
        if result is None:
            continue
        source_front, path = result
        source_cells = sprite_cells_from_front(source_front, direction)
        return {
            "color": color,
            "direction": direction,
            "source_cells": source_cells,
            "mirror_cells": mirror_cells,
            "absorber": None,
            "reserved": source_cells | set(path) | copy_cells | mirror_cells,
        }
    return None


def choose_turn_scene(
    grid: np.ndarray,
    zone_name: str,
    initial: tuple[int, int],
    final: tuple[int, int],
    color: int,
    reserved: set[tuple[int, int]],
    with_maroon: bool,
) -> dict | None:
    zone = zone_bounds(grid.shape, zone_name)
    endpoints = fronts_on_edge(grid.shape, zone, final)
    random.shuffle(endpoints)
    for edge_target in endpoints:
        absorber = edge_target if with_maroon else None
        endpoint = move(edge_target, final, -1) if with_maroon else edge_target
        copy_cells = sprite_cells_from_front(endpoint, final)
        if (
            not in_bounds(grid.shape, *endpoint)
            or not cells_in_zone(copy_cells, zone)
            or not free_of_reserved(copy_cells | ({absorber} if absorber else set()), reserved)
        ):
            continue
        turn_candidates = []
        current = move(endpoint, final, -2)
        while cell_in_zone(current, zone):
            mirror_cells = turning_mirror_cells(current, initial, final)
            turn_distance = abs(endpoint[0] - current[0]) + abs(endpoint[1] - current[1])
            segment2 = trace_segment(current, final, turn_distance)
            if (
                all(in_bounds(grid.shape, rr, cc) for rr, cc in mirror_cells)
                and cells_in_zone(mirror_cells | set(segment2), zone)
                and free_of_reserved(
                    mirror_cells | set(segment2) | copy_cells | ({absorber} if absorber else set()),
                    reserved,
                )
            ):
                turn_candidates.append(current)
            current = move(current, final, -1)
        random.shuffle(turn_candidates)
        for turn_point in turn_candidates:
            mirror_cells = turning_mirror_cells(turn_point, initial, final)
            turn_distance = abs(endpoint[0] - turn_point[0]) + abs(endpoint[1] - turn_point[1])
            path2 = trace_segment(turn_point, final, turn_distance)
            result = choose_source_front(
                turn_point,
                initial,
                zone,
                grid.shape,
                reserved | mirror_cells | copy_cells | ({absorber} if absorber else set()) | set(path2),
                min_steps=2,
            )
            if result is None:
                continue
            source_front, path1 = result
            source_cells = sprite_cells_from_front(source_front, initial)
            scene_reserved = source_cells | set(path1) | set(path2) | copy_cells | mirror_cells
            if absorber is not None:
                scene_reserved.add(absorber)
            return {
                "color": color,
                "direction": initial,
                "source_cells": source_cells,
                "mirror_cells": mirror_cells,
                "absorber": absorber,
                "reserved": scene_reserved,
            }
    return None


def scenario_spec(zone_name: str, kind: str) -> tuple[tuple[int, int], tuple[int, int] | None]:
    vertical, horizontal = zone_dirs(zone_name)
    if kind == "edge":
        return random.choice([(vertical, None), (horizontal, None)])
    if kind == "maroon":
        return random.choice([(vertical, None), (horizontal, None)])
    if kind == "parallel":
        return random.choice([(vertical, None), (horizontal, None)])
    if random.random() < 0.5:
        return vertical, horizontal
    return horizontal, vertical


def choose_kinds(n_sprites: int) -> list[str]:
    if n_sprites == 2:
        kinds = random.sample(NON_MAROON_KINDS, 2)
        if random.random() < 0.5:
            kinds[random.randrange(2)] = random.choice(MAROON_KINDS)
    elif n_sprites == 3:
        kinds = NON_MAROON_KINDS.copy()
        if random.random() < 0.65:
            kinds[random.randrange(3)] = random.choice(MAROON_KINDS)
    else:
        kinds = NON_MAROON_KINDS.copy()
        kinds.append(random.choice(MAROON_KINDS if random.random() < 0.7 else NON_MAROON_KINDS))
    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(19, 25)
    w = random.randint(19, 25)
    grid = np.full((h, w), BG, dtype=int)

    zone_names = ["NW", "NE", "SW", "SE"]
    n_sprites = random.randint(2, 4)
    chosen_zones = random.sample(zone_names, n_sprites)
    kinds = choose_kinds(n_sprites)
    colors = random.sample(SOURCE_COLORS, n_sprites)

    reserved: set[tuple[int, int]] = set()
    absorber_used = False
    placements = []

    for zone_name, kind, color in zip(chosen_zones, kinds, colors):
        direction, final = scenario_spec(zone_name, kind)
        if kind == "edge":
            placement = place_edge_scene(grid, zone_name, direction, color, reserved)
        elif kind == "maroon":
            if absorber_used:
                return None
            placement = place_maroon_scene(grid, zone_name, direction, color, reserved)
        elif kind == "parallel":
            placement = place_parallel_scene(grid, zone_name, direction, color, reserved)
        elif kind == "turn_edge":
            placement = choose_turn_scene(grid, zone_name, direction, final, color, reserved, with_maroon=False)
        else:
            if absorber_used:
                return None
            placement = choose_turn_scene(grid, zone_name, direction, final, color, reserved, with_maroon=True)

        if placement is None:
            return None
        placements.append(placement)
        reserved |= placement["reserved"]
        if placement["absorber"] is not None:
            absorber_used = True

    for placement in placements:
        for r, c in placement["source_cells"]:
            grid[r, c] = placement["color"]
        for r, c in placement["mirror_cells"]:
            grid[r, c] = GREY
        if placement["absorber"] is not None:
            ar, ac = placement["absorber"]
            grid[ar, ac] = MAROON

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out == AZURE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
