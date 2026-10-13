from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
COLOR_CYCLE = (2, 6, 3, 1)

GEN_ATTEMPTS = 400
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_SIZE_RANGE = (3, 5)
EXTRA_EDGE_PROB = 0.25
DISCONNECTED_PROB = 0.5
CHAINY_PROB = 0.5


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _detect_colors(g: np.ndarray) -> tuple[int, int, int, tuple[int, int]]:
    corner_vals = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    if len(set(corner_vals)) == 1:
        wall = corner_vals[0]
    else:
        border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
        vals, cnts = np.unique(border, return_counts=True)
        wall = int(vals[cnts.argmax()])

    colors = [int(v) for v in np.unique(g)]
    non_wall = [color for color in colors if color != wall]
    assert len(non_wall) >= 2, "expected background plus key color"

    counts = {color: int(np.count_nonzero(g == color)) for color in non_wall}
    bg = max(non_wall, key=lambda color: counts[color])
    key_colors = [color for color in non_wall if color != bg]
    assert len(key_colors) == 1, "expected exactly one non-wall non-background color"
    key_color = key_colors[0]

    key_coords = np.argwhere(g == key_color)
    assert key_coords.shape == (1, 2), "expected exactly one key pixel"
    key_pos = (int(key_coords[0, 0]), int(key_coords[0, 1]))
    return bg, wall, key_color, key_pos


def _find_doors(g: np.ndarray, wall: int) -> np.ndarray:
    h, w = g.shape
    passable = g != wall
    doors = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not passable[r, c]:
                continue
            vertical = bool((g[r - 1, c] == wall) and (g[r + 1, c] == wall) and passable[r, c - 1] and passable[r, c + 1])
            horizontal = bool((g[r, c - 1] == wall) and (g[r, c + 1] == wall) and passable[r - 1, c] and passable[r + 1, c])
            if vertical ^ horizontal:
                doors[r, c] = True
    return doors


def _label_rooms(g: np.ndarray, wall: int, doors: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = g.shape
    room_mask = (g != wall) & (~doors)
    labels = np.full((h, w), -1, dtype=int)
    rooms: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not room_mask[r, c] or labels[r, c] >= 0:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = len(rooms)
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and room_mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = labels[r, c]
                        q.append((nr, nc))

            coords_arr = np.asarray(coords, dtype=int)
            r0 = int(coords_arr[:, 0].min())
            r1 = int(coords_arr[:, 0].max())
            c0 = int(coords_arr[:, 1].min())
            c1 = int(coords_arr[:, 1].max())
            assert np.all(room_mask[r0 : r1 + 1, c0 : c1 + 1]), "rooms must be solid rectangles once doors are sealed"
            rooms.append(coords_arr)

    assert rooms, "expected at least one room"
    return labels, rooms


def _room_graph(labels: np.ndarray, doors: np.ndarray, g: np.ndarray, wall: int) -> list[set[int]]:
    h, w = g.shape
    graph = [set() for _ in range(int(labels.max()) + 1)]
    passable = g != wall

    for r, c in np.argwhere(doors):
        rr = int(r)
        cc = int(c)
        vertical = bool((g[rr - 1, cc] == wall) and (g[rr + 1, cc] == wall) and passable[rr, cc - 1] and passable[rr, cc + 1])
        horizontal = bool((g[rr, cc - 1] == wall) and (g[rr, cc + 1] == wall) and passable[rr - 1, cc] and passable[rr + 1, cc])

        if vertical:
            a = int(labels[rr, cc - 1])
            b = int(labels[rr, cc + 1])
        else:
            assert horizontal
            a = int(labels[rr - 1, cc])
            b = int(labels[rr + 1, cc])

        assert a >= 0 and b >= 0 and a != b, "a door must connect two distinct rooms"
        graph[a].add(b)
        graph[b].add(a)

    return graph


def _bfs_distances(graph: list[set[int]], start: int) -> list[int]:
    dist = [-1] * len(graph)
    dist[start] = 0
    q: deque[int] = deque([start])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if dist[nxt] < 0:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood rectangular rooms from the keyed room according to door-graph distance, cycling through a fixed palette ring.

    Input structure:
    - The background is black and thin gray walls partition the grid into enclosed axis-aligned rectangular rooms.
    - A shared wall between two rooms may contain a single background gap that acts as a door.
    - Exactly one room contains a single colored key pixel; other room interiors are background.

    Rule:
    - Identify door cells as one-cell wall gaps, temporarily seal them, and recover each room as a separate rectangular interior component.
    - Build a room-adjacency graph by reconnecting the two rooms on opposite sides of every door.
    - Fill the keyed room with the key color; rooms at graph distance 1, 2, 3, ... use the next colors in the task's fixed cyclic order.
    - Rooms not reachable from the keyed room stay background, and walls plus door cells remain unchanged.

    Key insight:
    - The puzzle depends on two hidden structures at once: the sealed-wall decomposition reveals the room nodes, and the door gaps reveal the edges whose BFS layers choose the palette-ring shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, wall, key_color, key_pos = _detect_colors(g)
    assert key_color in COLOR_CYCLE, "key color must belong to the palette ring"

    doors = _find_doors(g, wall)
    labels, rooms = _label_rooms(g, wall, doors)
    start = int(labels[key_pos])
    assert start >= 0, "key must be inside a room"

    graph = _room_graph(labels, doors, g, wall)
    dist = _bfs_distances(graph, start)
    phase0 = COLOR_CYCLE.index(key_color)

    out = g.copy()
    for room_id, room_cells in enumerate(rooms):
        if dist[room_id] < 0:
            continue
        fill = COLOR_CYCLE[(phase0 + dist[room_id]) % len(COLOR_CYCLE)]
        out[room_cells[:, 0], room_cells[:, 1]] = fill
    return out


def _room_id(i: int, j: int, cols: int) -> int:
    return i * cols + j


def _build_layout(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]], dict[tuple[int, int], dict[str, int]]]:
    rows = len(row_sizes)
    cols = len(col_sizes)
    h = sum(row_sizes) + rows + 1
    w = sum(col_sizes) + cols + 1
    grid = np.full((h, w), BG, dtype=int)

    wall_rows = [0]
    pos = 0
    for size in row_sizes:
        pos += size + 1
        wall_rows.append(pos)
    wall_cols = [0]
    pos = 0
    for size in col_sizes:
        pos += size + 1
        wall_cols.append(pos)

    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    room_boxes: list[tuple[int, int, int, int]] = []
    edge_specs: dict[tuple[int, int], dict[str, int]] = {}

    row_starts = []
    pos = 1
    for size in row_sizes:
        row_starts.append(pos)
        pos += size + 1
    col_starts = []
    pos = 1
    for size in col_sizes:
        col_starts.append(pos)
        pos += size + 1

    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            room_boxes.append((r0, c0, r0 + row_sizes[i] - 1, c0 + col_sizes[j] - 1))

    for i in range(rows):
        for j in range(cols - 1):
            a = _room_id(i, j, cols)
            b = _room_id(i, j + 1, cols)
            r0, _c0, r1, c1 = room_boxes[a]
            edge_specs[(a, b)] = {
                "kind": 0,
                "wall_coord": c1 + 1,
                "span0": r0,
                "span1": r1,
            }

    for i in range(rows - 1):
        for j in range(cols):
            a = _room_id(i, j, cols)
            b = _room_id(i + 1, j, cols)
            r0, c0, r1, _c1 = room_boxes[a]
            edge_specs[(a, b)] = {
                "kind": 1,
                "wall_coord": r1 + 1,
                "span0": c0,
                "span1": c0 + col_sizes[j] - 1,
            }

    return grid, room_boxes, edge_specs


def _adjacency_lists(rows: int, cols: int) -> tuple[list[set[int]], list[tuple[int, int]]]:
    total = rows * cols
    graph = [set() for _ in range(total)]
    edges: list[tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            a = _room_id(i, j, cols)
            if j + 1 < cols:
                b = _room_id(i, j + 1, cols)
                graph[a].add(b)
                graph[b].add(a)
                edges.append((a, b))
            if i + 1 < rows:
                b = _room_id(i + 1, j, cols)
                graph[a].add(b)
                graph[b].add(a)
                edges.append((a, b))
    return graph, edges


def _norm_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _pick_room_sizes(count: int) -> list[int]:
    return [random.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1]) for _ in range(count)]


def _grow_reachable(graph: list[set[int]], start: int, target: int, chainy: bool) -> set[int]:
    reachable = {start}
    order = [start]
    while len(reachable) < target:
        frontier: list[tuple[int, int]] = []
        frontier.extend((node, nbr) for node in reachable for nbr in graph[node] if nbr not in reachable)
        assert frontier, "room grid must stay connected before doors are removed"

        if chainy and len(order) >= 1:
            tail = order[-1]
            tail_frontier = [(u, v) for (u, v) in frontier if u == tail]
            if tail_frontier:
                frontier = tail_frontier

        u, v = random.choice(frontier)
        reachable.add(v)
        order.append(v)
    return reachable


def _select_door_edges(rows: int, cols: int) -> tuple[int, set[tuple[int, int]], bool]:
    total = rows * cols
    graph, all_edges = _adjacency_lists(rows, cols)
    start = random.randrange(total)
    disconnected = total >= 4 and random.random() < DISCONNECTED_PROB
    target = total if not disconnected else random.randint(3, total - 1)
    reachable = _grow_reachable(graph, start, target, chainy=(random.random() < CHAINY_PROB))

    chosen: set[tuple[int, int]] = set()
    built = {start}
    while built != reachable:
        candidates = [_norm_edge(u, v) for u in built for v in graph[u] if v in reachable and v not in built]
        edge = random.choice(candidates)
        chosen.add(edge)
        built.add(edge[0] if edge[1] in built else edge[1])
        built.add(edge[1] if edge[0] in built else edge[0])

    for a, b in all_edges:
        edge = _norm_edge(a, b)
        if edge in chosen:
            continue
        if a in reachable and b in reachable and random.random() < EXTRA_EDGE_PROB:
            chosen.add(edge)
        elif a not in reachable and b not in reachable and random.random() < EXTRA_EDGE_PROB:
            chosen.add(edge)

    return start, chosen, disconnected


def _graph_from_edges(total: int, edges: set[tuple[int, int]]) -> list[set[int]]:
    graph = [set() for _ in range(total)]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _expected_output(
    grid: np.ndarray,
    room_boxes: list[tuple[int, int, int, int]],
    door_edges: set[tuple[int, int]],
    start: int,
    key_color: int,
) -> np.ndarray:
    phase0 = COLOR_CYCLE.index(key_color)
    dist = _bfs_distances(_graph_from_edges(len(room_boxes), door_edges), start)
    out = grid.copy()
    for room_id, (r0, c0, r1, c1) in enumerate(room_boxes):
        if dist[room_id] < 0:
            continue
        fill = COLOR_CYCLE[(phase0 + dist[room_id]) % len(COLOR_CYCLE)]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _generate_case() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rows = random.randint(ROOM_ROWS_RANGE[0], ROOM_ROWS_RANGE[1])
        cols = random.randint(ROOM_COLS_RANGE[0], ROOM_COLS_RANGE[1])
        row_sizes = _pick_room_sizes(rows)
        col_sizes = _pick_room_sizes(cols)

        grid, room_boxes, edge_specs = _build_layout(row_sizes, col_sizes)
        total = rows * cols
        start, door_edges, disconnected = _select_door_edges(rows, cols)

        for edge in door_edges:
            spec = edge_specs[edge]
            pos = random.randint(spec["span0"], spec["span1"])
            if spec["kind"] == 0:
                grid[pos, spec["wall_coord"]] = BG
            else:
                grid[spec["wall_coord"], pos] = BG

        key_color = random.choice(COLOR_CYCLE)
        r0, c0, r1, c1 = room_boxes[start]
        key_r = random.randint(r0, r1)
        key_c = random.randint(c0, c1)
        grid[key_r, key_c] = key_color

        expected = _expected_output(grid, room_boxes, door_edges, start, key_color)
        graph = _graph_from_edges(total, door_edges)
        dist = _bfs_distances(graph, start)
        max_dist = max(dist)
        unreachable = sum(d < 0 for d in dist)

        if max_dist < 2:
            continue
        if disconnected and unreachable == 0:
            continue
        if not disconnected and unreachable != 0:
            continue
        if len({int(v) for v in np.unique(expected)} - {BG, WALL}) < 3:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid room-topology puzzle")


def generate_input() -> np.ndarray:
    return _generate_case()
