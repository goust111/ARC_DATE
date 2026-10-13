from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
WALL_FRACTION = 0.6
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (2, 6)
DOOR_OPEN_RANGE = (0.35, 0.7)
GEN_ATTEMPTS = 500

SHAPES_BY_SIZE: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (
        ((0, 0),),
    ),
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
    ),
}


def _cluster_lines(counts: np.ndarray, threshold: int) -> list[int]:
    hits = [idx for idx, value in enumerate(counts.tolist()) if int(value) >= int(threshold)]
    if not hits:
        return []
    groups: list[list[int]] = [[hits[0]]]
    for idx in hits[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return [max(group, key=lambda pos: int(counts[pos])) for group in groups]


def _find_wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    row_counts = np.count_nonzero(g == WALL, axis=1)
    col_counts = np.count_nonzero(g == WALL, axis=0)
    row_threshold = int(np.ceil(WALL_FRACTION * g.shape[1]))
    col_threshold = int(np.ceil(WALL_FRACTION * g.shape[0]))
    wall_rows = _cluster_lines(row_counts, row_threshold)
    wall_cols = _cluster_lines(col_counts, col_threshold)
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected enclosing wall lattice"
    assert wall_rows[0] == 0 and wall_rows[-1] == g.shape[0] - 1, "outer horizontal walls missing"
    assert wall_cols[0] == 0 and wall_cols[-1] == g.shape[1] - 1, "outer vertical walls missing"
    assert all(b - a >= 3 for a, b in zip(wall_rows, wall_rows[1:], strict=False)), "rooms must be at least 2 cells tall"
    assert all(b - a >= 3 for a, b in zip(wall_cols, wall_cols[1:], strict=False)), "rooms must be at least 2 cells wide"
    return wall_rows, wall_cols


def _room_specs(wall_rows: list[int], wall_cols: list[int]) -> list[tuple[int, int, int, int, int, int]]:
    rooms: list[tuple[int, int, int, int, int, int]] = []
    for room_r, (wr0, wr1) in enumerate(zip(wall_rows, wall_rows[1:], strict=False)):
        for room_c, (wc0, wc1) in enumerate(zip(wall_cols, wall_cols[1:], strict=False)):
            rooms.append((wr0 + 1, wr1, wc0 + 1, wc1, room_r, room_c))
    return rooms


def _room_id(room_rows: int, room_cols: int, room_r: int, room_c: int) -> int:
    assert 0 <= room_r < room_rows and 0 <= room_c < room_cols
    return room_r * room_cols + room_c


def _adjacency_from_grid(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
) -> list[list[int]]:
    g = np.asarray(grid, dtype=int)
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    adjacency = [set() for _ in range(room_rows * room_cols)]

    for room_r in range(room_rows):
        top = wall_rows[room_r] + 1
        bottom = wall_rows[room_r + 1]
        for room_c in range(room_cols - 1):
            wall_c = wall_cols[room_c + 1]
            if np.any(g[top:bottom, wall_c] == BG):
                a = _room_id(room_rows, room_cols, room_r, room_c)
                b = _room_id(room_rows, room_cols, room_r, room_c + 1)
                adjacency[a].add(b)
                adjacency[b].add(a)

    for room_r in range(room_rows - 1):
        wall_r = wall_rows[room_r + 1]
        for room_c in range(room_cols):
            left = wall_cols[room_c] + 1
            right = wall_cols[room_c + 1]
            if np.any(g[wall_r, left:right] == BG):
                a = _room_id(room_rows, room_cols, room_r, room_c)
                b = _room_id(room_rows, room_cols, room_r + 1, room_c)
                adjacency[a].add(b)
                adjacency[b].add(a)

    return [sorted(neighbors) for neighbors in adjacency]


def _extract_seeds(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int, int, int]],
) -> dict[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    seeds: dict[int, tuple[int, int]] = {}
    seen_colors: set[int] = set()
    for room_id, (top, bottom, left, right, _room_r, _room_c) in enumerate(rooms):
        interior = g[top:bottom, left:right]
        colors = [
            int(value)
            for value in np.unique(interior)
            if int(value) not in (BG, WALL)
        ]
        assert len(colors) <= 1, "each room may contain at most one seed color"
        if not colors:
            continue
        color = colors[0]
        assert color not in seen_colors, "seed colors must be unique"
        seen_colors.add(color)
        power = int(np.count_nonzero(interior == color))
        assert 1 <= power <= 4, "seed size must be between 1 and 4"
        seeds[room_id] = (color, power)
    return seeds


def _bfs_distances(adjacency: list[list[int]], start: int) -> dict[int, int]:
    distances = {int(start): 0}
    queue = deque([int(start)])
    while queue:
        room_id = queue.popleft()
        for neighbor in adjacency[room_id]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[room_id] + 1
            queue.append(neighbor)
    return distances


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    wall_rows, wall_cols = _find_wall_lines(g)
    rooms = _room_specs(wall_rows, wall_cols)
    seeds = _extract_seeds(g, rooms)
    adjacency = _adjacency_from_grid(g, wall_rows, wall_cols)
    candidates: dict[int, list[tuple[int, int, int, int, int, int]]] = {
        room_id: [] for room_id in range(len(rooms))
    }

    for source_room, (color, power) in seeds.items():
        distances = _bfs_distances(adjacency, source_room)
        source_room_r = rooms[source_room][4]
        source_room_c = rooms[source_room][5]
        for target_room, distance in distances.items():
            if distance == 0 or distance > power or target_room in seeds:
                continue
            candidates[target_room].append(
                (power, distance, source_room_r, source_room_c, color, source_room)
            )

    room_colors: dict[int, int] = {}
    for room_id in range(len(rooms)):
        if room_id in seeds:
            room_colors[room_id] = seeds[room_id][0]
            continue
        if not candidates[room_id]:
            continue
        best = min(
            candidates[room_id],
            key=lambda item: (-item[0], item[1], item[2], item[3]),
        )
        room_colors[room_id] = best[4]

    return {
        "rooms": rooms,
        "seeds": seeds,
        "adjacency": adjacency,
        "candidates": candidates,
        "room_colors": room_colors,
        "wall_rows": wall_rows,
        "wall_cols": wall_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seed rooms and propagate their colors room-by-room through door gaps.

    The grey wall lattice defines rectangular rooms. A seed's pixel count is its
    propagation radius in rooms. Empty rooms choose the reaching seed with the
    largest radius, breaking equal-radius conflicts by smaller room distance.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)
    out = g.copy()
    rooms = analysis["rooms"]
    room_colors = analysis["room_colors"]
    for room_id, color in room_colors.items():
        top, bottom, left, right, _room_r, _room_c = rooms[room_id]
        out[top:bottom, left:right] = int(color)
    return out


def _sample_sizes(count: int) -> list[int]:
    lo, hi = ROOM_SIZE_RANGE
    return [random.randint(lo, hi) for _ in range(int(count))]


def _wall_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    for size in sizes:
        positions.append(positions[-1] + int(size) + 1)
    return positions


def _adjacency_from_doors(
    room_rows: int,
    room_cols: int,
    door_right: list[list[bool]],
    door_down: list[list[bool]],
) -> list[list[int]]:
    adjacency = [set() for _ in range(room_rows * room_cols)]
    for room_r in range(room_rows):
        for room_c in range(room_cols - 1):
            if not door_right[room_r][room_c]:
                continue
            a = _room_id(room_rows, room_cols, room_r, room_c)
            b = _room_id(room_rows, room_cols, room_r, room_c + 1)
            adjacency[a].add(b)
            adjacency[b].add(a)
    for room_r in range(room_rows - 1):
        for room_c in range(room_cols):
            if not door_down[room_r][room_c]:
                continue
            a = _room_id(room_rows, room_cols, room_r, room_c)
            b = _room_id(room_rows, room_cols, room_r + 1, room_c)
            adjacency[a].add(b)
            adjacency[b].add(a)
    return [sorted(neighbors) for neighbors in adjacency]


def _all_pairs_distances(adjacency: list[list[int]]) -> list[dict[int, int]]:
    return [_bfs_distances(adjacency, room_id) for room_id in range(len(adjacency))]


def _random_doors(room_rows: int, room_cols: int) -> tuple[list[list[bool]], list[list[bool]]]:
    open_prob = random.uniform(*DOOR_OPEN_RANGE)
    door_right = [
        [random.random() < open_prob for _ in range(room_cols - 1)]
        for _ in range(room_rows)
    ]
    door_down = [
        [random.random() < open_prob for _ in range(room_cols)]
        for _ in range(room_rows - 1)
    ]
    if not any(any(row) for row in door_right) and not any(any(row) for row in door_down):
        if room_cols > 1 and (room_rows == 1 or random.random() < 0.5):
            door_right[random.randrange(room_rows)][random.randrange(room_cols - 1)] = True
        else:
            door_down[random.randrange(room_rows - 1)][random.randrange(room_cols)] = True
    return door_right, door_down


def _generic_seed_plan(distances: list[dict[int, int]]) -> dict[str, object] | None:
    options: list[tuple[int, int, list[int]]] = []
    for source_room, room_distances in enumerate(distances):
        for power in range(1, 5):
            reachable = [
                target_room
                for target_room, distance in room_distances.items()
                if 0 < distance <= power
            ]
            if reachable:
                options.append((source_room, power, reachable))
    if not options:
        return None
    source_room, power, reachable = random.choice(options)
    return {
        "seeds": [(source_room, power)],
        "target": random.choice(reachable),
    }


def _power_seed_plan(distances: list[dict[int, int]]) -> dict[str, object] | None:
    options: list[tuple[int, int, int, int, int]] = []
    room_count = len(distances)
    for strong_room in range(room_count):
        for weak_room in range(room_count):
            if weak_room == strong_room:
                continue
            for target_room in range(room_count):
                if target_room in (strong_room, weak_room):
                    continue
                strong_distance = distances[strong_room].get(target_room)
                weak_distance = distances[weak_room].get(target_room)
                if strong_distance is None or weak_distance is None:
                    continue
                if 1 <= weak_distance < strong_distance <= 4:
                    options.append(
                        (strong_room, strong_distance, weak_room, weak_distance, target_room)
                    )
    if not options:
        return None
    strong_room, strong_power, weak_room, weak_power, target_room = random.choice(options)
    return {
        "seeds": [(strong_room, strong_power), (weak_room, weak_power)],
        "target": target_room,
        "winner": strong_room,
        "loser": weak_room,
    }


def _distance_seed_plan(distances: list[dict[int, int]]) -> dict[str, object] | None:
    options: list[tuple[int, int, int, int, int]] = []
    room_count = len(distances)
    for room_a in range(room_count):
        for room_b in range(room_a + 1, room_count):
            for target_room in range(room_count):
                if target_room in (room_a, room_b):
                    continue
                dist_a = distances[room_a].get(target_room)
                dist_b = distances[room_b].get(target_room)
                if dist_a is None or dist_b is None or dist_a == dist_b:
                    continue
                need = max(dist_a, dist_b)
                if need <= 4:
                    options.append((room_a, room_b, target_room, dist_a, dist_b))
    if not options:
        return None
    room_a, room_b, target_room, dist_a, dist_b = random.choice(options)
    power = random.randint(max(dist_a, dist_b), 4)
    winner = room_a if dist_a < dist_b else room_b
    loser = room_b if winner == room_a else room_a
    return {
        "seeds": [(room_a, power), (room_b, power)],
        "target": target_room,
        "winner": winner,
        "loser": loser,
    }


def _shape_extent(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return (
        max(dr for dr, _dc in shape) + 1,
        max(dc for _dr, dc in shape) + 1,
    )


def _seed_shape(size: int, room_h: int, room_w: int) -> tuple[tuple[int, int], ...]:
    fitting = []
    for shape in SHAPES_BY_SIZE[int(size)]:
        shape_h, shape_w = _shape_extent(shape)
        if shape_h <= room_h and shape_w <= room_w:
            fitting.append(shape)
    assert fitting, "every generated room must support its seed shape"
    return random.choice(fitting)


def _draw_seed(
    grid: np.ndarray,
    room: tuple[int, int, int, int, int, int],
    color: int,
    power: int,
) -> None:
    top, bottom, left, right, _room_r, _room_c = room
    room_h = bottom - top
    room_w = right - left
    shape = _seed_shape(power, room_h, room_w)
    shape_h, shape_w = _shape_extent(shape)
    offset_r = random.randint(0, room_h - shape_h)
    offset_c = random.randint(0, room_w - shape_w)
    for dr, dc in shape:
        grid[top + offset_r + dr, left + offset_c + dc] = int(color)


def _build_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    door_right: list[list[bool]],
    door_down: list[list[bool]],
    seeds: list[tuple[int, int]],
) -> np.ndarray:
    wall_rows = _wall_positions(row_sizes)
    wall_cols = _wall_positions(col_sizes)
    height = wall_rows[-1] + 1
    width = wall_cols[-1] + 1
    grid = np.zeros((height, width), dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    room_rows = len(row_sizes)
    room_cols = len(col_sizes)
    for room_r in range(room_rows):
        for room_c in range(room_cols - 1):
            if not door_right[room_r][room_c]:
                continue
            door_row = random.randint(wall_rows[room_r] + 1, wall_rows[room_r + 1] - 1)
            grid[door_row, wall_cols[room_c + 1]] = BG
    for room_r in range(room_rows - 1):
        for room_c in range(room_cols):
            if not door_down[room_r][room_c]:
                continue
            door_col = random.randint(wall_cols[room_c] + 1, wall_cols[room_c + 1] - 1)
            grid[wall_rows[room_r + 1], door_col] = BG

    rooms = _room_specs(wall_rows, wall_cols)
    colors = random.sample(SEED_COLORS, len(seeds))
    for color, (room_id, power) in zip(colors, seeds, strict=False):
        _draw_seed(grid, rooms[room_id], color, power)
    return grid


def _ambiguous_room(candidates: list[tuple[int, int, int, int, int, int]]) -> bool:
    if len(candidates) < 2:
        return False
    ordered = sorted(candidates, key=lambda item: (-item[0], item[1], item[2], item[3]))
    return ordered[0][0] == ordered[1][0] and ordered[0][1] == ordered[1][1]


def _mode_matches(
    mode: str,
    analysis: dict[str, object],
    meta: dict[str, object],
) -> bool:
    seeds: dict[int, tuple[int, int]] = analysis["seeds"]
    candidates: dict[int, list[tuple[int, int, int, int, int, int]]] = analysis["candidates"]
    room_colors: dict[int, int] = analysis["room_colors"]

    for room_id, room_candidates in candidates.items():
        if room_id in seeds:
            continue
        if _ambiguous_room(room_candidates):
            return False

    target = int(meta["target"])
    if target in seeds or target not in room_colors:
        return False

    if mode == "generic":
        return True

    source_candidates = {candidate[5]: candidate for candidate in candidates[target]}
    winner = int(meta["winner"])
    loser = int(meta["loser"])
    if winner not in source_candidates or loser not in source_candidates:
        return False

    winner_candidate = source_candidates[winner]
    loser_candidate = source_candidates[loser]
    ordered = sorted(
        candidates[target],
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )
    if ordered[0][5] != winner:
        return False

    if mode == "power":
        return winner_candidate[0] > loser_candidate[0] and winner_candidate[1] > loser_candidate[1]

    if mode == "distance":
        return winner_candidate[0] == loser_candidate[0] and winner_candidate[1] < loser_candidate[1]

    return False


def _attempt(mode: str) -> np.ndarray | None:
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    row_sizes = _sample_sizes(room_rows)
    col_sizes = _sample_sizes(room_cols)
    door_right, door_down = _random_doors(room_rows, room_cols)
    adjacency = _adjacency_from_doors(room_rows, room_cols, door_right, door_down)
    distances = _all_pairs_distances(adjacency)

    if mode == "generic":
        meta = _generic_seed_plan(distances)
    elif mode == "power":
        meta = _power_seed_plan(distances)
    else:
        meta = _distance_seed_plan(distances)
    if meta is None:
        return None

    grid = _build_grid(row_sizes, col_sizes, door_right, door_down, meta["seeds"])
    try:
        out = main(grid)
        analysis = _analyze_grid(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if not _mode_matches(mode, analysis, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    modes = ["generic", "power", "distance"]
    for _ in range(GEN_ATTEMPTS):
        mode = random.choice(modes)
        grid = _attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-diffusion puzzle")
