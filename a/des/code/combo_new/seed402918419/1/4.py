import random
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400
CANVAS_MIN = 18
CANVAS_MAX = 30
CLUSTER_PAD = 2
ROOM_SIZE_MIN = 3
ROOM_SIZE_MAX = 4
EXTRA_EDGE_PROB = 0.3
OPTIONAL_CLUSTER_PROB = 0.7

SINGLE_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
CONFLICT_SHAPES = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
EMPTY_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))


def _wall_color(grid: np.ndarray) -> int:
    values = grid[grid != BG]
    assert values.size > 0, "expected at least one non-background pixel"
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _nearest_wall(grid: np.ndarray, wall: int, r: int, c: int) -> Tuple[int | None, int | None, int | None, int | None]:
    top = None
    for rr in range(r - 1, -1, -1):
        if int(grid[rr, c]) == wall:
            top = rr
            break

    bottom = None
    for rr in range(r + 1, grid.shape[0]):
        if int(grid[rr, c]) == wall:
            bottom = rr
            break

    left = None
    for cc in range(c - 1, -1, -1):
        if int(grid[r, cc]) == wall:
            left = cc
            break

    right = None
    for cc in range(c + 1, grid.shape[1]):
        if int(grid[r, cc]) == wall:
            right = cc
            break

    return top, left, bottom, right


def _is_valid_room(grid: np.ndarray, wall: int, room: Tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = room
    if bottom - top < 2 or right - left < 2:
        return False

    if any(int(grid[r, c]) != wall for r, c in ((top, left), (top, right), (bottom, left), (bottom, right))):
        return False

    interior = grid[top + 1 : bottom, left + 1 : right]
    if interior.size == 0 or np.any(interior == wall):
        return False

    side_slices = (
        grid[top, left + 1 : right],
        grid[bottom, left + 1 : right],
        grid[top + 1 : bottom, left],
        grid[top + 1 : bottom, right],
    )
    for side in side_slices:
        side = np.asarray(side)
        gaps = side != wall
        if int(np.count_nonzero(gaps)) > 1:
            return False
        if np.any(side[gaps] != BG):
            return False

    return True


def _find_rooms(grid: np.ndarray, wall: int) -> List[Tuple[int, int, int, int]]:
    candidates: Counter[Tuple[int, int, int, int]] = Counter()
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == wall:
                continue
            top, left, bottom, right = _nearest_wall(grid, wall, r, c)
            if None in (top, left, bottom, right):
                continue
            candidates[(int(top), int(left), int(bottom), int(right))] += 1

    rooms = [room for room in candidates if _is_valid_room(grid, wall, room)]
    rooms.sort()
    return rooms


def _share_vertical_wall(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int] | None:
    if a[3] != b[1]:
        return None
    start = max(a[0] + 1, b[0] + 1)
    end = min(a[2] - 1, b[2] - 1)
    if start > end:
        return None
    return a[3], start, end


def _share_horizontal_wall(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int] | None:
    if a[2] != b[0]:
        return None
    start = max(a[1] + 1, b[1] + 1)
    end = min(a[3] - 1, b[3] - 1)
    if start > end:
        return None
    return a[2], start, end


def _room_graph(grid: np.ndarray, wall: int, rooms: Sequence[Tuple[int, int, int, int]]) -> List[Set[int]]:
    graph = [set() for _ in rooms]
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            left_first = rooms[i][1] <= rooms[j][1]
            a, b = (rooms[i], rooms[j]) if left_first else (rooms[j], rooms[i])
            shared_v = _share_vertical_wall(a, b)
            if shared_v is not None:
                col, start, end = shared_v
                if np.any(grid[start : end + 1, col] == BG):
                    graph[i].add(j)
                    graph[j].add(i)
                    continue

            top_first = rooms[i][0] <= rooms[j][0]
            a, b = (rooms[i], rooms[j]) if top_first else (rooms[j], rooms[i])
            shared_h = _share_horizontal_wall(a, b)
            if shared_h is not None:
                row, start, end = shared_h
                if np.any(grid[row, start : end + 1] == BG):
                    graph[i].add(j)
                    graph[j].add(i)
    return graph


def _room_seed_colors(grid: np.ndarray, wall: int, room: Tuple[int, int, int, int]) -> Set[int]:
    top, left, bottom, right = room
    interior = grid[top + 1 : bottom, left + 1 : right]
    return {int(color) for color in np.unique(interior) if int(color) not in {BG, wall}}


def _component_seed_sets(graph: Sequence[Set[int]], room_seeds: Sequence[Set[int]]) -> Tuple[List[int], List[Set[int]]]:
    component_of = [-1] * len(graph)
    component_colors: List[Set[int]] = []

    for start in range(len(graph)):
        if component_of[start] != -1:
            continue
        comp_index = len(component_colors)
        stack = [start]
        component_of[start] = comp_index
        colors: Set[int] = set()

        while stack:
            node = stack.pop()
            colors.update(room_seeds[node])
            for nxt in graph[node]:
                if component_of[nxt] == -1:
                    component_of[nxt] = comp_index
                    stack.append(nxt)

        component_colors.append(colors)

    return component_of, component_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each room with its uniquely reachable seed color while leaving conflict-connected rooms empty.

    Input structure:
    - The background is black.
    - Grey one-pixel rectangular borders define multiple axis-aligned rooms with black interiors.
    - Some shared walls have a single black doorway pixel that connects neighboring rooms.
    - A few rooms contain unique non-black, non-grey seed pixels.

    Rule:
    - Reconstruct every room from the rectangular grey walls, even when a wall contains a one-pixel doorway.
    - Build a room-connectivity graph by linking rooms whose shared wall contains a doorway.
    - For each connected room component, collect all seed colors present anywhere in that component.
    - If a component contains exactly one seed color, fill every black interior pixel of its rooms with that color; if it contains zero or multiple seed colors, leave black interiors unchanged.
    - Keep seed pixels and all grey wall pixels in place.

    Key insight:
    - The transformation is not pixel-level flood fill through black space; it first recovers room rectangles and then reasons over the doorway graph, so multi-seed components must stay empty instead of being painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall = _wall_color(g)
    rooms = _find_rooms(g, wall)
    assert rooms, "expected at least one room"

    room_seeds = [_room_seed_colors(g, wall, room) for room in rooms]
    graph = _room_graph(g, wall, rooms)
    component_of, component_colors = _component_seed_sets(graph, room_seeds)

    out = g.copy()
    for idx, room in enumerate(rooms):
        colors = component_colors[component_of[idx]]
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        top, left, bottom, right = room
        interior = out[top + 1 : bottom, left + 1 : right]
        interior[interior == BG] = color

    return out


def _grid_adjacencies(rows: int, cols: int) -> List[Tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            cur = r * cols + c
            if c + 1 < cols:
                edges.append((cur, cur + 1))
            if r + 1 < rows:
                edges.append((cur, cur + cols))
    return edges


def _connected_edges(rng: random.Random, rows: int, cols: int) -> List[Tuple[int, int]]:
    total = rows * cols
    if total <= 1:
        return []

    all_edges = _grid_adjacencies(rows, cols)
    neighbors: Dict[int, List[int]] = {i: [] for i in range(total)}
    for a, b in all_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    start = rng.randrange(total)
    visited = {start}
    tree_edges: Set[Tuple[int, int]] = set()

    while len(visited) < total:
        frontier = []
        for node in visited:
            for nxt in neighbors[node]:
                if nxt not in visited:
                    frontier.append((node, nxt))
        a, b = rng.choice(frontier)
        tree_edges.add(tuple(sorted((a, b))))
        visited.add(b)

    for edge in all_edges:
        edge = tuple(sorted(edge))
        if edge in tree_edges:
            continue
        if rng.random() < EXTRA_EDGE_PROB:
            tree_edges.add(edge)

    return sorted(tree_edges)


def _sample_room_sizes(rng: random.Random, count: int) -> List[int]:
    return [rng.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(count)]


def _cluster_shape(rng: random.Random, kind: str) -> Tuple[int, int]:
    if kind == "single":
        return rng.choice(SINGLE_SHAPES)
    if kind == "conflict":
        return rng.choice(CONFLICT_SHAPES)
    return rng.choice(EMPTY_SHAPES)


def _build_cluster(
    rng: random.Random,
    kind: str,
    seed_colors: Sequence[int],
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[Tuple[int, int]], List[Set[int]]]:
    rows, cols = _cluster_shape(rng, kind)
    heights = _sample_room_sizes(rng, rows)
    widths = _sample_room_sizes(rng, cols)

    row_lines = [0]
    for height in heights:
        row_lines.append(row_lines[-1] + height + 1)
    col_lines = [0]
    for width in widths:
        col_lines.append(col_lines[-1] + width + 1)

    cluster = np.full((row_lines[-1] + 1, col_lines[-1] + 1), BG, dtype=int)
    for r in row_lines:
        cluster[r, :] = WALL
    for c in col_lines:
        cluster[:, c] = WALL

    rooms = []
    for r in range(rows):
        for c in range(cols):
            rooms.append((row_lines[r], col_lines[c], row_lines[r + 1], col_lines[c + 1]))

    edges = _connected_edges(rng, rows, cols)
    for a, b in edges:
        ra, ca = divmod(a, cols)
        rb, cb = divmod(b, cols)
        room_a = rooms[a]
        if ra == rb:
            shared_col = room_a[3]
            door_row = rng.randint(room_a[0] + 1, room_a[2] - 1)
            cluster[door_row, shared_col] = BG
        else:
            shared_row = room_a[2]
            door_col = rng.randint(room_a[1] + 1, room_a[3] - 1)
            cluster[shared_row, door_col] = BG

    room_seeds = [set() for _ in rooms]
    room_indices = list(range(len(rooms)))
    rng.shuffle(room_indices)

    if kind == "single":
        seed_room = room_indices[0]
        room_seeds[seed_room].add(int(seed_colors[0]))
    elif kind == "conflict":
        seed_count = 2 + int(len(rooms) >= 3 and rng.random() < 0.35)
        for color, room_index in zip(seed_colors[:seed_count], room_indices):
            room_seeds[room_index].add(int(color))

    for idx, colors in enumerate(room_seeds):
        if not colors:
            continue
        color = next(iter(colors))
        top, left, bottom, right = rooms[idx]
        rr = rng.randint(top + 1, bottom - 1)
        cc = rng.randint(left + 1, right - 1)
        cluster[rr, cc] = color

    return cluster, rooms, edges, room_seeds


def _boxes_separated(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_clusters(
    rng: random.Random,
    canvas_shape: Tuple[int, int],
    clusters: Sequence[np.ndarray],
) -> List[Tuple[int, int]] | None:
    h, w = canvas_shape
    placements: List[Tuple[int, int]] = [(-1, -1)] * len(clusters)
    placed_boxes: List[Tuple[int, int, int, int]] = []

    order = sorted(range(len(clusters)), key=lambda i: clusters[i].shape[0] * clusters[i].shape[1], reverse=True)
    for idx in order:
        ch, cw = clusters[idx].shape
        if ch + 2 > h or cw + 2 > w:
            return None
        options = []
        for top in range(1, h - ch):
            for left in range(1, w - cw):
                box = (top, left, top + ch - 1, left + cw - 1)
                if all(_boxes_separated(box, other, CLUSTER_PAD) for other in placed_boxes):
                    options.append((top, left))
        if not options:
            return None
        top, left = rng.choice(options)
        placements[idx] = (top, left)
        placed_boxes.append((top, left, top + ch - 1, left + cw - 1))

    return placements


def _expected_output(
    grid: np.ndarray,
    rooms: Sequence[Tuple[int, int, int, int]],
    edges: Sequence[Tuple[int, int]],
    room_seeds: Sequence[Set[int]],
) -> np.ndarray:
    graph = [set() for _ in rooms]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    component_of, component_colors = _component_seed_sets(graph, room_seeds)

    out = grid.copy()
    for idx, room in enumerate(rooms):
        colors = component_colors[component_of[idx]]
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        top, left, bottom, right = room
        interior = out[top + 1 : bottom, left + 1 : right]
        interior[interior == BG] = color
    return out


def _globalize(
    local_rooms: Sequence[Tuple[int, int, int, int]],
    row_offset: int,
    col_offset: int,
) -> List[Tuple[int, int, int, int]]:
    return [
        (top + row_offset, left + col_offset, bottom + row_offset, right + col_offset)
        for top, left, bottom, right in local_rooms
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cluster_types = ["single", "conflict"]
    if rng.random() < OPTIONAL_CLUSTER_PROB:
        cluster_types.append(rng.choice(("empty", "single")))

    total_seed_budget = 4
    palette = list(SEED_COLORS)
    rng.shuffle(palette)
    if len(palette) < total_seed_budget:
        return None

    local_clusters = []
    color_index = 0
    for kind in cluster_types:
        needed = 1 if kind == "single" else 3 if kind == "conflict" else 0
        cluster, rooms, edges, room_seeds = _build_cluster(rng, kind, palette[color_index : color_index + needed])
        color_index += needed
        local_clusters.append((cluster, rooms, edges, room_seeds))

    min_h = max(cluster.shape[0] for cluster, _, _, _ in local_clusters) + 2
    min_w = max(cluster.shape[1] for cluster, _, _, _ in local_clusters) + 2
    canvas_h = rng.randint(max(CANVAS_MIN, min_h), CANVAS_MAX)
    canvas_w = rng.randint(max(CANVAS_MIN, min_w), CANVAS_MAX)

    placements = _place_clusters(rng, (canvas_h, canvas_w), [cluster for cluster, _, _, _ in local_clusters])
    if placements is None:
        return None

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    global_rooms: List[Tuple[int, int, int, int]] = []
    global_edges: List[Tuple[int, int]] = []
    global_room_seeds: List[Set[int]] = []

    room_offset = 0
    for (cluster, rooms, edges, room_seeds), (top, left) in zip(local_clusters, placements):
        ch, cw = cluster.shape
        grid[top : top + ch, left : left + cw] = cluster
        global_rooms.extend(_globalize(rooms, top, left))
        global_edges.extend([(a + room_offset, b + room_offset) for a, b in edges])
        global_room_seeds.extend([set(colors) for colors in room_seeds])
        room_offset += len(rooms)

    expected = _expected_output(grid, global_rooms, global_edges, global_room_seeds)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    room_fill_kinds = []
    graph = [set() for _ in global_rooms]
    for a, b in global_edges:
        graph[a].add(b)
        graph[b].add(a)
    component_of, component_colors = _component_seed_sets(graph, global_room_seeds)
    for idx in range(len(global_rooms)):
        room_fill_kinds.append(len(component_colors[component_of[idx]]))
    if 1 not in room_fill_kinds or not any(count >= 2 for count in room_fill_kinds):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
