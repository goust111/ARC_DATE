from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
FLOW_COLORS = [1, 2, 4, 6, 7, 8, 9]

MIN_SIDE = 18
MAX_SIDE = 28
MIN_ROOM_SIDE = 4
MIN_ROOMS = 4
MAX_ROOMS = 7
MAX_ATTEMPTS = 500

SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_rooms(blocked: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = blocked.shape
    labels = np.full((h, w), -1, dtype=int)
    rooms: list[list[tuple[int, int]]] = []
    room_id = 0
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or labels[r, c] != -1:
                continue
            queue = [(r, c)]
            labels[r, c] = -2
            cells: list[tuple[int, int]] = []
            touches_boundary = False
            qi = 0
            while qi < len(queue):
                rr, cc = queue[qi]
                qi += 1
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = -2
                        queue.append((nr, nc))
            if touches_boundary:
                for rr, cc in cells:
                    labels[rr, cc] = -1
                continue
            for rr, cc in cells:
                labels[rr, cc] = room_id
            rooms.append(cells)
            room_id += 1
    return labels, rooms


def _extract_structure(
    grid: np.ndarray, *, expect_source: bool
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[dict], tuple[int, int, int] | None]:
    g = np.asarray(grid, dtype=int)
    colored_mask = (g != BLACK) & (g != GREEN) & (g != GREY)
    blocked = (g == GREY) | colored_mask
    room_labels, rooms = _label_rooms(blocked)

    doors: list[dict] = []
    source: tuple[int, int, int] | None = None
    for r, c in np.argwhere(colored_mask):
        r = int(r)
        c = int(c)
        color = int(g[r, c])

        up = int(room_labels[r - 1, c]) if r > 0 else -1
        down = int(room_labels[r + 1, c]) if r + 1 < g.shape[0] else -1
        left = int(room_labels[r, c - 1]) if c > 0 else -1
        right = int(room_labels[r, c + 1]) if c + 1 < g.shape[1] else -1

        if up >= 0 and down >= 0 and up != down and left < 0 and right < 0:
            doors.append({"pos": (r, c), "rooms": (up, down), "color": color, "axis": "h"})
            continue
        if left >= 0 and right >= 0 and left != right and up < 0 and down < 0:
            doors.append({"pos": (r, c), "rooms": (left, right), "color": color, "axis": "v"})
            continue

        adjacent = {rid for rid in (up, down, left, right) if rid >= 0}
        if len(adjacent) == 1:
            if source is not None:
                raise AssertionError("expected exactly one source pixel")
            source = (r, c, color)
            continue
        raise AssertionError("non-grey colored pixel must be either a door or the source")

    if expect_source and source is None:
        raise AssertionError("source pixel not found")
    return room_labels, rooms, doors, source


def _build_graph(room_count: int, doors: list[dict]) -> dict[int, list[tuple[int, int, tuple[int, int]]]]:
    graph = {rid: [] for rid in range(room_count)}
    for door in doors:
        a, b = map(int, door["rooms"])
        color = int(door["color"])
        pos = tuple(door["pos"])
        graph[a].append((b, color, pos))
        graph[b].append((a, color, pos))
    return graph


def _source_room(room_labels: np.ndarray, source: tuple[int, int, int]) -> int:
    r, c, _ = source
    adjacent = {int(room_labels[nr, nc]) for nr, nc in _neighbors4(r, c) if 0 <= nr < room_labels.shape[0] and 0 <= nc < room_labels.shape[1] and room_labels[nr, nc] >= 0}
    if len(adjacent) != 1:
        raise AssertionError("source must belong to exactly one room")
    return adjacent.pop()


def _propagate_colors(
    room_count: int, graph: dict[int, list[tuple[int, int, tuple[int, int]]]], source_room: int, source_color: int
) -> tuple[dict[int, int], dict[int, int], dict[int, tuple[int, int] | None]]:
    room_color = {int(source_room): int(source_color)}
    parent = {int(source_room): -1}
    incoming_door: dict[int, tuple[int, int] | None] = {int(source_room): None}
    queue = deque([int(source_room)])
    while queue:
        room = queue.popleft()
        for nbr, door_color, door_pos in graph[room]:
            if nbr in room_color:
                continue
            room_color[nbr] = int(door_color)
            parent[nbr] = int(room)
            incoming_door[nbr] = tuple(door_pos)
            queue.append(int(nbr))
    return room_color, parent, incoming_door


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every green room object by the color that most recently flowed through a door from the source room.

    Input structure:
    - The grid has a black background and grey wall lines that partition the interior into closed rooms.
    - Each internal wall between neighboring rooms contains exactly one non-grey door pixel, and one room contains a single non-green colored source pixel.
    - Every room contains one or more 4-connected green objects on black floor cells.

    Rule:
    - Treat the door-linked rooms as a tree and start the flow in the source room, coloring that room's green objects with the source color.
    - Whenever the flow crosses a door into an adjacent room, the flow color immediately changes to that door's color.
    - Recolor all green objects in each room with the color carried when the flow enters that room; keep walls, door pixels, the source pixel, and black floor unchanged.

    Key insight:
    - A room's final color is not decided locally by anything inside that room; it is the color of the last door on the unique path from the source room to that room.
    """
    g = np.asarray(grid, dtype=int)
    room_labels, rooms, doors, source = _extract_structure(g, expect_source=True)
    assert source is not None
    graph = _build_graph(len(rooms), doors)
    source_room = _source_room(room_labels, source)
    room_color, _parent, _incoming = _propagate_colors(len(rooms), graph, source_room, int(source[2]))

    out = g.copy()
    for rid, color in room_color.items():
        mask = (room_labels == int(rid)) & (g == GREEN)
        out[mask] = int(color)
    return out


def _room_size(room: dict) -> tuple[int, int]:
    return room["r1"] - room["r0"] + 1, room["c1"] - room["c0"] + 1


def _feasible_splits(room: dict) -> list[str]:
    h, w = _room_size(room)
    options: list[str] = []
    if h >= 2 * MIN_ROOM_SIDE + 1:
        options.append("h")
    if w >= 2 * MIN_ROOM_SIDE + 1:
        options.append("v")
    return options


def _split_room(grid: np.ndarray, room: dict) -> tuple[list[dict], dict]:
    r0, c0, r1, c1 = room["r0"], room["c0"], room["r1"], room["c1"]
    h, w = _room_size(room)
    options = _feasible_splits(room)
    if len(options) == 2:
        slack_h = h - (2 * MIN_ROOM_SIDE + 1)
        slack_w = w - (2 * MIN_ROOM_SIDE + 1)
        orient = random.choices(options, weights=[slack_h + 1, slack_w + 1])[0]
    else:
        orient = options[0]

    door_color = random.choice(FLOW_COLORS)
    if orient == "h":
        wall_r = random.randint(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE)
        door_min = c0 + 1 if c1 - c0 >= 2 else c0
        door_max = c1 - 1 if c1 - c0 >= 2 else c1
        door_c = random.randint(door_min, door_max)
        grid[wall_r, c0 : c1 + 1] = GREY
        grid[wall_r, door_c] = door_color
        children = [
            {"r0": r0, "c0": c0, "r1": wall_r - 1, "c1": c1},
            {"r0": wall_r + 1, "c0": c0, "r1": r1, "c1": c1},
        ]
        door = {"pos": (wall_r, door_c), "axis": "h", "color": door_color}
    else:
        wall_c = random.randint(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE)
        door_min = r0 + 1 if r1 - r0 >= 2 else r0
        door_max = r1 - 1 if r1 - r0 >= 2 else r1
        door_r = random.randint(door_min, door_max)
        grid[r0 : r1 + 1, wall_c] = GREY
        grid[door_r, wall_c] = door_color
        children = [
            {"r0": r0, "c0": c0, "r1": r1, "c1": wall_c - 1},
            {"r0": r0, "c0": wall_c + 1, "r1": r1, "c1": c1},
        ]
        door = {"pos": (door_r, wall_c), "axis": "v", "color": door_color}
    return children, door


def _generate_room_partition() -> tuple[np.ndarray, list[dict]] | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    rooms = [{"r0": 1, "c0": 1, "r1": h - 2, "c1": w - 2}]
    target_rooms = random.randint(MIN_ROOMS, MAX_ROOMS)
    while len(rooms) < target_rooms:
        candidates = [i for i, room in enumerate(rooms) if _feasible_splits(room)]
        if not candidates:
            return None
        weights = []
        for idx in candidates:
            rh, rw = _room_size(rooms[idx])
            weights.append((rh * rw) ** 1.2)
        idx = random.choices(candidates, weights=weights)[0]
        room = rooms.pop(idx)
        children, _door = _split_room(grid, room)
        rooms.extend(children)

    return grid, rooms


def _bfs_distances(graph: dict[int, list[tuple[int, int, tuple[int, int]]]], start: int) -> dict[int, int]:
    dist = {int(start): 0}
    queue = deque([int(start)])
    while queue:
        node = queue.popleft()
        for nbr, _color, _pos in graph[node]:
            if nbr in dist:
                continue
            dist[nbr] = dist[node] + 1
            queue.append(nbr)
    return dist


def _pick_source_room(graph: dict[int, list[tuple[int, int, tuple[int, int]]]], room_by_id: dict[int, dict]) -> int | None:
    candidates = []
    for rid, room in room_by_id.items():
        h, w = _room_size(room)
        if h < 5 or w < 5:
            continue
        dist = _bfs_distances(graph, rid)
        if len(dist) != len(graph):
            continue
        if max(dist.values()) >= 2:
            candidates.append((max(dist.values()), h * w, rid))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    top_depth = candidates[0][0]
    filtered = [rid for depth, _area, rid in candidates if depth == top_depth]
    return random.choice(filtered)


def _candidate_cells(room: dict, margin: int) -> list[tuple[int, int]]:
    r0 = room["r0"] + margin
    c0 = room["c0"] + margin
    r1 = room["r1"] - margin
    c1 = room["c1"] - margin
    if r0 > r1 or c0 > c1:
        r0, c0, r1, c1 = room["r0"], room["c0"], room["r1"], room["c1"]
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(cells)
    return cells


def _can_place_shape(grid: np.ndarray, room: dict, shape: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    cells = {(top + dr, left + dc) for dr, dc in shape}
    for r, c in cells:
        if not (room["r0"] <= r <= room["r1"] and room["c0"] <= c <= room["c1"]):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                continue
            if room["r0"] <= nr <= room["r1"] and room["c0"] <= nc <= room["c1"] and int(grid[nr, nc]) != BLACK:
                return False
    return True


def _place_shape(grid: np.ndarray, room: dict, color: int, shape: tuple[tuple[int, int], ...]) -> bool:
    max_dr = max(dr for dr, _dc in shape)
    max_dc = max(dc for _dr, dc in shape)
    candidate_tops = _candidate_cells(
        {"r0": room["r0"], "c0": room["c0"], "r1": room["r1"] - max_dr, "c1": room["c1"] - max_dc},
        margin=1,
    )
    for top, left in candidate_tops:
        if _can_place_shape(grid, room, shape, top, left):
            for dr, dc in shape:
                grid[top + dr, left + dc] = int(color)
            return True
    return False


def _object_count(room: dict, has_source: bool) -> int:
    area = (_room_size(room)[0]) * (_room_size(room)[1])
    if area >= 42:
        low, high = 2, 3
    elif area >= 24:
        low, high = 1, 2
    else:
        low, high = 1, 1
    if has_source and high > 1:
        high -= 1
    return random.randint(low, max(low, high))


def _place_features(grid: np.ndarray, rooms: list[dict], room_by_id: dict[int, dict], source_room: int) -> bool:
    source_color = random.choice(FLOW_COLORS)
    if not _place_shape(grid, room_by_id[source_room], source_color, ((0, 0),)):
        return False

    for rid, room in room_by_id.items():
        count = _object_count(room, rid == source_room)
        for _ in range(count):
            shape_choices = SHAPES[:]
            random.shuffle(shape_choices)
            placed = False
            for shape in shape_choices:
                if _place_shape(grid, room, GREEN, shape):
                    placed = True
                    break
            if not placed:
                return False
    return True


def _graph_is_tree(graph: dict[int, list[tuple[int, int, tuple[int, int]]]]) -> bool:
    undirected_edges = set()
    for a, nbrs in graph.items():
        for b, _color, _pos in nbrs:
            undirected_edges.add(tuple(sorted((a, b))))
    if len(undirected_edges) != len(graph) - 1:
        return False
    seen = set()
    queue = deque([next(iter(graph))])
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for nbr, _color, _pos in graph[node]:
            if nbr not in seen:
                queue.append(nbr)
    return len(seen) == len(graph)


def _semantic_checks(grid: np.ndarray, room_by_id_expected: dict[int, dict]) -> bool:
    try:
        room_labels, rooms, doors, source = _extract_structure(grid, expect_source=True)
        assert source is not None
        graph = _build_graph(len(rooms), doors)
        assert len(rooms) == len(room_by_id_expected)
        assert _graph_is_tree(graph)

        source_room = _source_room(room_labels, source)
        dist = _bfs_distances(graph, source_room)
        assert len(dist) == len(rooms)
        assert max(dist.values()) >= 2

        green_rooms = set()
        for rid in range(len(rooms)):
            if np.any((room_labels == rid) & (grid == GREEN)):
                green_rooms.add(rid)
        assert len(green_rooms) == len(rooms)

        room_color, parent, incoming = _propagate_colors(len(rooms), graph, source_room, int(source[2]))
        assert len(room_color) == len(rooms)
        assert len(set(room_color.values())) >= 2

        overwrite_needed = False
        for rid, d in dist.items():
            if d < 2:
                continue
            par = parent[rid]
            assert par >= 0
            if room_color[rid] != room_color[par]:
                overwrite_needed = True
                break
        assert overwrite_needed

        out = main(grid)
        for rid, color in room_color.items():
            mask = (room_labels == rid) & (grid == GREEN)
            assert mask.any()
            assert np.all(out[mask] == int(color))

        non_green_mask = grid != GREEN
        assert np.array_equal(out[non_green_mask], grid[non_green_mask])
        for rid, room in room_by_id_expected.items():
            probe = (room["r0"], room["c0"])
            assert int(room_labels[probe]) == int(rid)
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    generated = _generate_room_partition()
    if generated is None:
        return None
    grid, room_rects = generated

    try:
        room_labels, rooms, doors, _source = _extract_structure(grid, expect_source=False)
    except Exception:
        return None
    if len(rooms) != len(room_rects):
        return None

    room_by_id: dict[int, dict] = {}
    for room in room_rects:
        rid = int(room_labels[room["r0"], room["c0"]])
        if rid < 0 or rid in room_by_id:
            return None
        room_by_id[rid] = room
    if len(room_by_id) != len(room_rects):
        return None

    graph = _build_graph(len(rooms), doors)
    if not _graph_is_tree(graph):
        return None
    source_room = _pick_source_room(graph, room_by_id)
    if source_room is None:
        return None

    if not _place_features(grid, room_rects, room_by_id, source_room):
        return None
    if not _semantic_checks(grid, room_by_id):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
