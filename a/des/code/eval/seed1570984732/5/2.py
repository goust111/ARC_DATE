from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


GREY = 5
MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(values) > 0, "expected a non-background wall color"
    return int(values[np.argmax(counts)])


def _detect_doors(grid: np.ndarray, wall: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    door_mask = np.zeros((h, w), dtype=bool)
    door_axis = np.zeros((h, w), dtype=np.int8)  # 1=vertical wall door, 2=horizontal wall door

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) == wall:
                continue

            up = int(grid[r - 1, c]) == wall
            down = int(grid[r + 1, c]) == wall
            left = int(grid[r, c - 1]) == wall
            right = int(grid[r, c + 1]) == wall

            vertical = up and down and not left and not right
            horizontal = left and right and not up and not down
            if vertical ^ horizontal:
                door_mask[r, c] = True
                door_axis[r, c] = 1 if vertical else 2

    return door_mask, door_axis


def _label_rooms(blocked: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = blocked.shape
    labels = np.zeros((h, w), dtype=int)
    rooms: dict[int, list[tuple[int, int]]] = {}
    room_id = 0

    for r in range(h):
        for c in range(w):
            if blocked[r, c] or labels[r, c] != 0:
                continue
            room_id += 1
            queue = deque([(r, c)])
            labels[r, c] = room_id
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = room_id
                        queue.append((nr, nc))
            rooms[room_id] = cells

    return labels, rooms


def _build_graph(
    grid: np.ndarray, labels: np.ndarray, door_mask: np.ndarray, door_axis: np.ndarray
) -> dict[int, list[tuple[int, int, tuple[int, int]]]]:
    graph: dict[int, list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)

    for r, c in np.argwhere(door_mask):
        rr = int(r)
        cc = int(c)
        axis = int(door_axis[rr, cc])
        if axis == 1:
            a = int(labels[rr, cc - 1])
            b = int(labels[rr, cc + 1])
        else:
            a = int(labels[rr - 1, cc])
            b = int(labels[rr + 1, cc])
        if a <= 0 or b <= 0 or a == b:
            continue
        door_color = int(grid[rr, cc])
        pos = (rr, cc)
        graph[a].append((b, door_color, pos))
        graph[b].append((a, door_color, pos))

    for room_id in graph:
        graph[room_id].sort(key=lambda item: (item[2][0], item[2][1], item[0], item[1]))
    return graph


def _start_info(
    grid: np.ndarray, rooms: dict[int, list[tuple[int, int]]], labels: np.ndarray, bg: int
) -> tuple[int, int]:
    starts = np.argwhere(grid == MAROON)
    assert len(starts) == 1, "expected exactly one maroon start marker"
    sr, sc = map(int, starts[0])
    start_room = int(labels[sr, sc])
    assert start_room > 0, "start marker must lie inside a room"

    signal_counts = Counter(
        int(grid[r, c])
        for r, c in rooms[start_room]
        if int(grid[r, c]) not in {bg, MAROON}
    )
    assert signal_counts, "start room must contain an initial signal color"
    start_signal = int(signal_counts.most_common(1)[0][0])
    return start_room, start_signal


def _solve(grid: np.ndarray, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    wall = _wall_color(g, bg)

    door_mask, door_axis = _detect_doors(g, wall)
    blocked = (g == wall) | door_mask
    labels, rooms = _label_rooms(blocked)
    assert rooms, "expected at least one room"

    graph = _build_graph(g, labels, door_mask, door_axis)
    start_room, start_signal = _start_info(g, rooms, labels, bg)

    room_colors = {start_room: start_signal}
    queue = deque([start_room])
    same_count = 0
    diff_count = 0

    while queue:
        room = queue.popleft()
        current = room_colors[room]
        for neighbor, door_color, _ in graph.get(room, []):
            if neighbor in room_colors:
                continue
            if door_color == current:
                next_color = current
                same_count += 1
            else:
                next_color = door_color
                diff_count += 1
            room_colors[neighbor] = next_color
            queue.append(neighbor)

    assert len(room_colors) == len(rooms), "all rooms must be reachable from the start room"

    out = g.copy()
    for room_id, color in room_colors.items():
        for r, c in rooms[room_id]:
            if int(g[r, c]) == bg:
                out[r, c] = color

    if return_trace:
        return out, {"same": same_count, "different": diff_count, "room_count": len(rooms)}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangular rooms according to door-colored BFS signal propagation.

    Input structure:
    - bg is the most common color.
    - Grey walls partition the canvas into rectangular rooms.
    - Each shared wall segment has one colored one-pixel door.
    - One room contains a maroon start marker and one extra colored signal pixel.

    Rule:
    - Treat each room as a node and each door as an edge.
    - Start from the maroon room carrying its local signal color.
    - Crossing a same-colored door keeps the signal color; crossing a different-colored door replaces it.
    - Rooms are filled in first-arrival BFS order, filling only their bg cells.
    - Walls, doors, and the maroon marker stay unchanged.
    """
    return _solve(grid)


GEN_RETRIES = 400
MIN_ROOMS = 4
MAX_ROOMS = 8
MIN_SIDE = 12
MAX_SIDE = 24
MIN_ROOM_SIDE = 2
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8]


def _room_area(room: tuple[int, int, int, int]) -> int:
    top, left, bottom, right = room
    return (bottom - top - 1) * (right - left - 1)


def _room_cells(room: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    top, left, bottom, right = room
    return [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]


def _draw_outer_walls(grid: np.ndarray) -> None:
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY


def _split_options(room: tuple[int, int, int, int]) -> list[tuple[str, int, int]]:
    top, left, bottom, right = room
    options: list[tuple[str, int, int]] = []
    if bottom - top - 1 >= 2 * MIN_ROOM_SIDE + 1:
        options.append(("h", top + MIN_ROOM_SIDE + 1, bottom - MIN_ROOM_SIDE - 1))
    if right - left - 1 >= 2 * MIN_ROOM_SIDE + 1:
        options.append(("v", left + MIN_ROOM_SIDE + 1, right - MIN_ROOM_SIDE - 1))
    return options


def _partition_rooms(
    rng: random.Random, bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h = rng.randint(MIN_SIDE, MAX_SIDE)
    w = rng.randint(MIN_SIDE, MAX_SIDE)
    grid = np.full((h, w), bg, dtype=int)
    _draw_outer_walls(grid)

    rooms = [(0, 0, h - 1, w - 1)]
    target_rooms = rng.randint(MIN_ROOMS, MAX_ROOMS)

    for _ in range(target_rooms * 5):
        if len(rooms) >= target_rooms:
            break
        candidates = []
        for idx, room in enumerate(rooms):
            for axis, lo, hi in _split_options(room):
                candidates.append((idx, room, axis, lo, hi, _room_area(room)))
        if not candidates:
            break
        chosen_idx = rng.choices(
            range(len(candidates)),
            weights=[item[5] for item in candidates],
            k=1,
        )[0]
        room_idx, room, axis, lo, hi, _ = candidates[chosen_idx]
        top, left, bottom, right = room
        cut = rng.randint(lo, hi)
        rooms.pop(room_idx)
        if axis == "h":
            grid[cut, left + 1 : right] = GREY
            rooms.append((top, left, cut, right))
            rooms.append((cut, left, bottom, right))
        else:
            grid[top + 1 : bottom, cut] = GREY
            rooms.append((top, left, bottom, cut))
            rooms.append((top, cut, bottom, right))

    if len(rooms) < MIN_ROOMS:
        return None
    return grid, rooms


def _find_adjacencies(
    rooms: list[tuple[int, int, int, int]]
) -> list[dict[str, int | str]]:
    edges: list[dict[str, int | str]] = []
    for i, (t1, l1, b1, r1) in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            t2, l2, b2, r2 = rooms[j]
            if r1 == l2 or r2 == l1:
                wall_col = r1 if r1 == l2 else r2
                row_lo = max(t1, t2) + 1
                row_hi = min(b1, b2) - 1
                if row_lo <= row_hi:
                    edges.append(
                        {
                            "u": i,
                            "v": j,
                            "axis": "v",
                            "line": wall_col,
                            "lo": row_lo,
                            "hi": row_hi,
                        }
                    )
            elif b1 == t2 or b2 == t1:
                wall_row = b1 if b1 == t2 else b2
                col_lo = max(l1, l2) + 1
                col_hi = min(r1, r2) - 1
                if col_lo <= col_hi:
                    edges.append(
                        {
                            "u": i,
                            "v": j,
                            "axis": "h",
                            "line": wall_row,
                            "lo": col_lo,
                            "hi": col_hi,
                        }
                    )
    return edges


def _bfs_distances(adj: dict[int, list[int]], start: int) -> dict[int, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def _choose_room_colors(
    rng: random.Random,
    adj: dict[int, list[int]],
    dist: dict[int, int],
    start: int,
    palette: list[int],
) -> dict[int, int]:
    room_colors = {start: rng.choice(palette)}
    order = sorted(dist, key=lambda room_id: (dist[room_id], room_id))
    for room_id in order[1:]:
        parents = [nbr for nbr in adj[room_id] if dist[nbr] == dist[room_id] - 1]
        assert parents, "every non-start room must have a previous-layer parent"
        if rng.random() < 0.5:
            room_colors[room_id] = room_colors[rng.choice(parents)]
        else:
            forbidden = {room_colors[parent] for parent in parents}
            options = [color for color in palette if color not in forbidden]
            if not options:
                options = [color for color in palette if color != room_colors[parents[0]]]
            room_colors[room_id] = rng.choice(options)
    return room_colors


def _expected_output(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    room_colors: dict[int, int],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for room_id, room in enumerate(rooms):
        color = room_colors[room_id]
        for r, c in _room_cells(room):
            if int(out[r, c]) == bg:
                out[r, c] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(BG_CHOICES)
    partition = _partition_rooms(rng, bg)
    if partition is None:
        return None
    grid, rooms = partition

    edges = _find_adjacencies(rooms)
    if len(edges) < len(rooms) - 1:
        return None

    adj: dict[int, list[int]] = {i: [] for i in range(len(rooms))}
    for edge in edges:
        u = int(edge["u"])
        v = int(edge["v"])
        adj[u].append(v)
        adj[v].append(u)

    start = rng.randrange(len(rooms))
    dist = _bfs_distances(adj, start)
    if len(dist) != len(rooms):
        return None

    palette_pool = [color for color in range(10) if color not in {bg, GREY, MAROON}]
    palette = rng.sample(palette_pool, rng.randint(3, min(5, len(palette_pool))))
    room_colors = _choose_room_colors(rng, adj, dist, start, palette)

    for edge in edges:
        u = int(edge["u"])
        v = int(edge["v"])
        if dist[u] + 1 == dist[v]:
            door_color = room_colors[v]
        elif dist[v] + 1 == dist[u]:
            door_color = room_colors[u]
        else:
            door_color = rng.choice(palette)

        if edge["axis"] == "v":
            r = rng.randint(int(edge["lo"]), int(edge["hi"]))
            c = int(edge["line"])
        else:
            r = int(edge["line"])
            c = rng.randint(int(edge["lo"]), int(edge["hi"]))
        grid[r, c] = door_color

    start_cells = _room_cells(rooms[start])
    if len(start_cells) < 2:
        return None
    start_marker, signal_marker = rng.sample(start_cells, 2)
    grid[start_marker] = MAROON
    grid[signal_marker] = room_colors[start]

    expected = _expected_output(grid, rooms, room_colors, bg)
    try:
        solved, trace = _solve(grid, return_trace=True)
    except AssertionError:
        return None
    if trace["same"] == 0 or trace["different"] == 0:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-propagation ARC input")
