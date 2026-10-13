import copy
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


BG = 0
TOWER = 5
RECEIVER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

GridPoint = Tuple[int, int]
Direction = Tuple[int, int]
Lengths = Dict[Direction, int]

GEN_ATTEMPTS = 300
GROUP_ATTEMPTS = 140
ROUTE_ATTEMPTS = 100
DISTRACTOR_ATTEMPTS = 80


def add_pt(point: GridPoint, direction: Direction, steps: int = 1) -> GridPoint:
    return (point[0] + direction[0] * steps, point[1] + direction[1] * steps)


def in_bounds(shape: Tuple[int, int], point: GridPoint) -> bool:
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def is_edge(shape: Tuple[int, int], point: GridPoint) -> bool:
    r, c = point
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def perpendiculars(direction: Direction) -> Tuple[Direction, Direction]:
    if direction in (UP, DOWN):
        return (LEFT, RIGHT)
    return (UP, DOWN)


def tower_cells(center: GridPoint, lengths: Lengths) -> Set[GridPoint]:
    cells = {center}
    for direction, length in lengths.items():
        for step in range(1, length + 1):
            cells.add(add_pt(center, direction, step))
    return cells


def expand8(cells: Iterable[GridPoint]) -> Set[GridPoint]:
    out: Set[GridPoint] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def components_of_color(grid: np.ndarray, color: int) -> List[List[GridPoint]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: List[List[GridPoint]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[GridPoint] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def parse_tower(coords: Sequence[GridPoint]) -> Tuple[GridPoint, Lengths]:
    coord_set = set(coords)
    candidates: List[Tuple[GridPoint, Lengths]] = []
    for center in sorted(coord_set):
        lengths: Lengths = {}
        built = {center}
        for direction in DIRS:
            length = 0
            p1 = add_pt(center, direction, 1)
            if p1 in coord_set:
                length = 1
                built.add(p1)
                p2 = add_pt(center, direction, 2)
                if p2 in coord_set:
                    length = 2
                    built.add(p2)
            lengths[direction] = length
        arm_count = sum(length > 0 for length in lengths.values())
        if 2 <= arm_count <= 4 and built == coord_set:
            candidates.append((center, lengths))
    assert len(candidates) == 1, "each grey component must be exactly one cross-like tower"
    return candidates[0]


def infer_towers(grid: np.ndarray) -> Tuple[List[dict], Dict[GridPoint, int], Dict[GridPoint, Direction]]:
    towers: List[dict] = []
    pixel_to_tower: Dict[GridPoint, int] = {}
    pixel_to_arm_dir: Dict[GridPoint, Direction] = {}
    for coords in components_of_color(grid, TOWER):
        center, lengths = parse_tower(coords)
        tip_map: Dict[Direction, GridPoint] = {}
        tower_index = len(towers)
        pixel_to_tower[center] = tower_index
        for direction, length in lengths.items():
            if length == 0:
                continue
            tip_map[direction] = add_pt(center, direction, length)
            for step in range(1, length + 1):
                cell = add_pt(center, direction, step)
                pixel_to_tower[cell] = tower_index
                pixel_to_arm_dir[cell] = direction
        towers.append(
            {
                "center": center,
                "lengths": lengths,
                "tips": tip_map,
            }
        )
    assert towers, "expected at least one tower"
    return towers, pixel_to_tower, pixel_to_arm_dir


def trace_ray(
    grid: np.ndarray,
    towers: Sequence[dict],
    pixel_to_tower: Dict[GridPoint, int],
    pixel_to_arm_dir: Dict[GridPoint, Direction],
    start: GridPoint,
    direction: Direction,
) -> Tuple[List[GridPoint], Optional[int]]:
    path: List[GridPoint] = []
    seen_states: Set[Tuple[GridPoint, Direction]] = set()
    position = start
    current_dir = direction

    while True:
        if not in_bounds(grid.shape, position):
            return path, None

        state = (position, current_dir)
        assert state not in seen_states, "ray loop detected"
        seen_states.add(state)

        value = int(grid[position])
        if value == BG:
            path.append(position)
            if is_edge(grid.shape, position):
                return path, None
            position = add_pt(position, current_dir)
            continue

        if value == TOWER:
            assert position in pixel_to_arm_dir, "ray must hit a tower arm, not the center"
            turn_dir = pixel_to_arm_dir[position]
            tip = towers[pixel_to_tower[position]]["tips"][turn_dir]
            position = add_pt(tip, turn_dir)
            current_dir = turn_dir
            continue

        assert is_edge(grid.shape, position), "receivers must be single edge pixels"
        return path, value


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == TOWER), "expected grey towers"

    non_tower = np.argwhere((g != BG) & (g != TOWER))
    for r, c in non_tower:
        assert is_edge(g.shape, (int(r), int(c))), "non-grey colored cells must be edge receivers"

    towers, pixel_to_tower, pixel_to_arm_dir = infer_towers(g)
    paint: Dict[GridPoint, int] = {}

    for tower in sorted(towers, key=lambda item: item["center"]):
        lengths = tower["lengths"]
        for direction in DIRS:
            if lengths[direction] == 0 or lengths[OPPOSITE[direction]] != 0:
                continue
            start = add_pt(tower["tips"][direction], direction)
            path, receiver_color = trace_ray(g, towers, pixel_to_tower, pixel_to_arm_dir, start, direction)
            if receiver_color is None:
                continue
            for point in path:
                prev = paint.get(point)
                assert prev is None or prev == receiver_color, "conflicting receiver colors on the same path cell"
                paint[point] = receiver_color

    out = g.copy()
    for point, color in paint.items():
        out[point] = color
    return out


def copy_world(world: dict) -> dict:
    return {
        "shape": world["shape"],
        "towers": copy.deepcopy(world["towers"]),
        "tower_cells": set(world["tower_cells"]),
        "tower_buffer": set(world["tower_buffer"]),
        "path_cells": set(world["path_cells"]),
        "edge_used": set(world["edge_used"]),
        "receivers": dict(world["receivers"]),
        "routes": copy.deepcopy(world["routes"]),
        "free_colors": list(world["free_colors"]),
    }


def restore_world(world: dict, snapshot: dict) -> None:
    world.clear()
    world.update(snapshot)


def can_place_tower(world: dict, center: GridPoint, lengths: Lengths) -> bool:
    shape = world["shape"]
    cells = tower_cells(center, lengths)
    if any(not in_bounds(shape, cell) or is_edge(shape, cell) for cell in cells):
        return False
    if cells & world["tower_cells"]:
        return False
    if cells & world["path_cells"]:
        return False
    if cells & world["edge_used"]:
        return False
    if cells & world["tower_buffer"]:
        return False
    return True


def place_tower(world: dict, center: GridPoint, lengths: Lengths) -> None:
    cells = tower_cells(center, lengths)
    world["towers"].append({"center": center, "lengths": copy.deepcopy(lengths)})
    world["tower_cells"].update(cells)
    world["tower_buffer"].update(expand8(cells))


def segment_cells_until_hit(start: GridPoint, hit: GridPoint, direction: Direction, shape: Tuple[int, int]) -> Optional[List[GridPoint]]:
    cells: List[GridPoint] = []
    point = start
    while point != hit:
        if not in_bounds(shape, point):
            return None
        cells.append(point)
        point = add_pt(point, direction)
    return cells


def random_length(rng: np.random.Generator) -> int:
    return int(rng.integers(1, 3))


def single_emitter_lengths(launch_dir: Direction, rng: np.random.Generator) -> Lengths:
    lengths = {direction: 0 for direction in DIRS}
    lengths[launch_dir] = random_length(rng)
    lengths[OPPOSITE[launch_dir]] = 0
    for direction in perpendiculars(launch_dir):
        lengths[direction] = random_length(rng)
    return lengths


def double_emitter_lengths(first_dir: Direction, second_dir: Direction, rng: np.random.Generator) -> Lengths:
    lengths = {direction: 0 for direction in DIRS}
    lengths[first_dir] = random_length(rng)
    lengths[second_dir] = random_length(rng)
    lengths[OPPOSITE[first_dir]] = 0
    lengths[OPPOSITE[second_dir]] = 0
    return lengths


def full_cross_lengths(rng: np.random.Generator, forced_dir: Optional[Direction] = None, forced_length: Optional[int] = None) -> Lengths:
    lengths = {direction: random_length(rng) for direction in DIRS}
    if forced_dir is not None and forced_length is not None:
        lengths[forced_dir] = forced_length
    return lengths


def interior_centers(shape: Tuple[int, int]) -> List[GridPoint]:
    h, w = shape
    return [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]


def choose_bounce_centers(
    shape: Tuple[int, int],
    start: GridPoint,
    incoming_dir: Direction,
    outgoing_dir: Direction,
    hit_dist: int,
) -> List[GridPoint]:
    h, w = shape
    centers: List[GridPoint] = []
    if incoming_dir == RIGHT:
        row = start[0] + hit_dist if outgoing_dir == UP else start[0] - hit_dist
        for col in range(start[1] + 2, w - 2):
            centers.append((row, col))
    elif incoming_dir == LEFT:
        row = start[0] + hit_dist if outgoing_dir == UP else start[0] - hit_dist
        for col in range(start[1] - 2, 1, -1):
            centers.append((row, col))
    elif incoming_dir == DOWN:
        col = start[1] + hit_dist if outgoing_dir == LEFT else start[1] - hit_dist
        for row in range(start[0] + 2, h - 2):
            centers.append((row, col))
    else:
        col = start[1] + hit_dist if outgoing_dir == LEFT else start[1] - hit_dist
        for row in range(start[0] - 2, 1, -1):
            centers.append((row, col))
    return centers


def final_segment(
    world: dict,
    start: GridPoint,
    direction: Direction,
    with_receiver: bool,
) -> Optional[Tuple[List[GridPoint], GridPoint]]:
    shape = world["shape"]
    segment: List[GridPoint] = []
    point = start
    while in_bounds(shape, point):
        if is_edge(shape, point):
            exit_point = point
            break
        segment.append(point)
        point = add_pt(point, direction)
    else:
        return None

    if with_receiver:
        if not segment:
            return None
        if exit_point in world["edge_used"] or exit_point in world["tower_cells"]:
            return None
        if any(cell in world["tower_cells"] or cell in world["path_cells"] or cell in world["edge_used"] for cell in segment):
            return None
        return segment, exit_point

    segment_with_edge = segment + [exit_point]
    if len(segment_with_edge) < 2:
        return None
    if any(cell in world["tower_cells"] or cell in world["path_cells"] or cell in world["edge_used"] for cell in segment_with_edge):
        return None
    return segment_with_edge, exit_point


def build_route(
    world: dict,
    emitter_center: GridPoint,
    emitter_lengths: Lengths,
    launch_dir: Direction,
    bounce_count: int,
    with_receiver: bool,
    rng: np.random.Generator,
) -> bool:
    route_backup = copy_world(world)

    start = add_pt(emitter_center, launch_dir, emitter_lengths[launch_dir] + 1)
    if not in_bounds(world["shape"], start) or is_edge(world["shape"], start):
        return False
    if start in world["tower_cells"] or start in world["path_cells"] or start in world["edge_used"]:
        return False

    route_cells: List[GridPoint] = []
    current = start
    current_dir = launch_dir

    for bounce_index in range(bounce_count):
        success = False
        for _ in range(ROUTE_ATTEMPTS):
            outgoing_dir = perpendiculars(current_dir)[int(rng.integers(0, 2))]
            out_length = random_length(rng)
            hit_dist = int(rng.integers(1, out_length + 1))
            candidate_centers = choose_bounce_centers(world["shape"], current, current_dir, outgoing_dir, hit_dist)
            rng.shuffle(candidate_centers)
            for center in candidate_centers[:30]:
                lengths = full_cross_lengths(rng, forced_dir=outgoing_dir, forced_length=out_length)
                if not can_place_tower(world, center, lengths):
                    continue
                hit_point = add_pt(center, outgoing_dir, hit_dist)
                segment = segment_cells_until_hit(current, hit_point, current_dir, world["shape"])
                if segment is None or len(segment) < 2:
                    continue
                if any(cell in world["tower_cells"] or cell in world["path_cells"] or cell in world["edge_used"] for cell in segment):
                    continue
                next_start = add_pt(add_pt(center, outgoing_dir, out_length), outgoing_dir)
                if not in_bounds(world["shape"], next_start) or is_edge(world["shape"], next_start):
                    continue
                if next_start in world["tower_cells"] or next_start in world["path_cells"] or next_start in world["edge_used"]:
                    continue

                world["path_cells"].update(segment)
                route_cells.extend(segment)
                place_tower(world, center, lengths)
                current = next_start
                current_dir = outgoing_dir
                success = True
                break
            if success:
                break
        if not success:
            restore_world(world, route_backup)
            return False

    final = final_segment(world, current, current_dir, with_receiver)
    if final is None:
        restore_world(world, route_backup)
        return False
    segment, exit_point = final
    world["path_cells"].update(segment)
    route_cells.extend(segment)
    world["edge_used"].add(exit_point)

    receiver_color: Optional[int] = None
    if with_receiver:
        if not world["free_colors"]:
            restore_world(world, route_backup)
            return False
        color_index = int(rng.integers(0, len(world["free_colors"])))
        receiver_color = world["free_colors"].pop(color_index)
        world["receivers"][exit_point] = receiver_color

    if with_receiver and len(route_cells) < 4:
        restore_world(world, route_backup)
        return False

    world["routes"].append(
        {
            "launch_dir": launch_dir,
            "bounce_count": bounce_count,
            "receiver_color": receiver_color,
            "path_cells": list(route_cells),
            "exit_point": exit_point,
        }
    )
    return True


def place_single_group(world: dict, route_spec: dict, rng: np.random.Generator) -> bool:
    shape = world["shape"]
    centers = interior_centers(shape)
    rng.shuffle(centers)
    for _ in range(GROUP_ATTEMPTS):
        center = centers[int(rng.integers(0, len(centers)))]
        launch_dir = DIRS[int(rng.integers(0, len(DIRS)))]
        lengths = single_emitter_lengths(launch_dir, rng)
        if not can_place_tower(world, center, lengths):
            continue
        snapshot = copy_world(world)
        place_tower(world, center, lengths)
        ok = build_route(
            world,
            center,
            lengths,
            launch_dir,
            route_spec["bounce_count"],
            route_spec["with_receiver"],
            rng,
        )
        if ok:
            return True
        restore_world(world, snapshot)
    return False


def place_double_group(world: dict, route_specs: Sequence[dict], rng: np.random.Generator) -> bool:
    shape = world["shape"]
    centers = interior_centers(shape)
    rng.shuffle(centers)
    for _ in range(GROUP_ATTEMPTS):
        center = centers[int(rng.integers(0, len(centers)))]
        first_dir = DIRS[int(rng.integers(0, len(DIRS)))]
        second_dir = perpendiculars(first_dir)[int(rng.integers(0, 2))]
        lengths = double_emitter_lengths(first_dir, second_dir, rng)
        if not can_place_tower(world, center, lengths):
            continue
        snapshot = copy_world(world)
        place_tower(world, center, lengths)
        pairs = [
            (route_specs[0], first_dir),
            (route_specs[1], second_dir),
        ]
        pairs.sort(key=lambda item: (-item[0]["bounce_count"], not item[0]["with_receiver"]))
        ok = True
        for route_spec, launch_dir in pairs:
            if not build_route(
                world,
                center,
                lengths,
                launch_dir,
                route_spec["bounce_count"],
                route_spec["with_receiver"],
                rng,
            ):
                ok = False
                break
        if ok:
            return True
        restore_world(world, snapshot)
    return False


def add_distractors(world: dict, min_total_towers: int, rng: np.random.Generator) -> bool:
    target = max(min_total_towers, len(world["towers"]) + int(rng.integers(0, 3)))
    shape = world["shape"]
    centers = interior_centers(shape)
    rng.shuffle(centers)
    attempts = 0
    while len(world["towers"]) < target and attempts < DISTRACTOR_ATTEMPTS:
        attempts += 1
        center = centers[int(rng.integers(0, len(centers)))]
        lengths = full_cross_lengths(rng)
        if can_place_tower(world, center, lengths):
            place_tower(world, center, lengths)
    return len(world["towers"]) >= min_total_towers


def add_unused_receivers(world: dict, rng: np.random.Generator) -> None:
    extra_count = int(rng.integers(0, 3))
    shape = world["shape"]
    edge_positions: List[GridPoint] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            point = (r, c)
            if is_edge(shape, point):
                edge_positions.append(point)
    rng.shuffle(edge_positions)
    for point in edge_positions:
        if extra_count == 0 or not world["free_colors"]:
            break
        if point in world["edge_used"] or point in world["tower_cells"]:
            continue
        color_index = int(rng.integers(0, len(world["free_colors"])))
        color = world["free_colors"].pop(color_index)
        world["receivers"][point] = color
        world["edge_used"].add(point)
        extra_count -= 1


def build_grid(world: dict) -> np.ndarray:
    grid = np.full(world["shape"], BG, dtype=int)
    for tower in world["towers"]:
        for cell in tower_cells(tower["center"], tower["lengths"]):
            grid[cell] = TOWER
    for point, color in world["receivers"].items():
        assert grid[point] == BG
        grid[point] = color
    return grid


def expected_output(grid: np.ndarray, routes: Sequence[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for route in routes:
        color = route["receiver_color"]
        if color is None:
            continue
        for point in route["path_cells"]:
            out[point] = color
    return out


def choose_bounce_counts(route_count: int, rng: np.random.Generator) -> List[int]:
    counts = list(rng.permutation([0, 1, 2])[: min(route_count, 3)])
    while len(counts) < route_count:
        counts.append(int(rng.integers(0, 3)))
    rng.shuffle(counts)
    return counts


def choose_receiver_flags(route_count: int, rng: np.random.Generator) -> List[bool]:
    while True:
        flags = [bool(int(rng.integers(0, 2))) for _ in range(route_count)]
        if any(flags):
            if route_count == 1 or any(not flag for flag in flags) or rng.random() < 0.5:
                return flags


def init_world(shape: Tuple[int, int]) -> dict:
    return {
        "shape": shape,
        "towers": [],
        "tower_cells": set(),
        "tower_buffer": set(),
        "path_cells": set(),
        "edge_used": set(),
        "receivers": {},
        "routes": [],
        "free_colors": list(RECEIVER_COLORS),
    }


def _gen_attempt(rng: np.random.Generator) -> Optional[np.ndarray]:
    route_count = int(rng.integers(2, 5))
    bounce_counts = choose_bounce_counts(route_count, rng)
    receiver_flags = choose_receiver_flags(route_count, rng)

    min_size = 14 + route_count * 2 + max(bounce_counts)
    h = int(rng.integers(min_size, min_size + 6))
    w = int(rng.integers(min_size + 1, min_size + 8))
    shape = (h, w)
    world = init_world(shape)

    route_specs = [
        {
            "bounce_count": bounce_counts[i],
            "with_receiver": receiver_flags[i],
        }
        for i in range(route_count)
    ]

    use_double = route_count >= 2 and rng.random() < 0.5
    next_route = 0
    if use_double:
        pair = sorted(route_specs[:2], key=lambda item: (-item["bounce_count"], not item["with_receiver"]))
        if not place_double_group(world, pair, rng):
            return None
        next_route = 2

    while next_route < route_count:
        if not place_single_group(world, route_specs[next_route], rng):
            return None
        next_route += 1

    if not add_distractors(world, min_total_towers=max(3, route_count), rng=rng):
        return None
    add_unused_receivers(world, rng)

    grid = build_grid(world)
    expected = expected_output(grid, world["routes"])
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None

    colored_routes = [route for route in world["routes"] if route["receiver_color"] is not None]
    if not colored_routes:
        return None
    colored_path_cells = sum(len(route["path_cells"]) for route in colored_routes)
    if colored_path_cells < 6:
        return None
    if len({route["bounce_count"] for route in world["routes"]}) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
