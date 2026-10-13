from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

FILL_CYCLE = (RED, ORANGE, YELLOW, GREEN)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 300
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (3, 6)
LONG_PATH_PROB = 0.5


def _door_segments(grid: np.ndarray) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    h, w = grid.shape
    doors: list[tuple[str, tuple[tuple[int, int], ...]]] = []

    for c in range(1, w - 1):
        r = 1
        while r < h - 1:
            if int(grid[r, c]) == BLUE:
                r += 1
                continue
            start = r
            while r < h - 1 and int(grid[r, c]) != BLUE:
                r += 1
            end = r - 1
            length = end - start + 1
            if (
                1 <= length <= 2
                and int(grid[start - 1, c]) == BLUE
                and int(grid[end + 1, c]) == BLUE
                and all(int(grid[rr, c - 1]) != BLUE and int(grid[rr, c + 1]) != BLUE for rr in range(start, end + 1))
            ):
                cells = tuple((rr, c) for rr in range(start, end + 1))
                doors.append(("vertical", cells))

    for r in range(1, h - 1):
        c = 1
        while c < w - 1:
            if int(grid[r, c]) == BLUE:
                c += 1
                continue
            start = c
            while c < w - 1 and int(grid[r, c]) != BLUE:
                c += 1
            end = c - 1
            length = end - start + 1
            if (
                1 <= length <= 2
                and int(grid[r, start - 1]) == BLUE
                and int(grid[r, end + 1]) == BLUE
                and all(int(grid[r - 1, cc]) != BLUE and int(grid[r + 1, cc]) != BLUE for cc in range(start, end + 1))
            ):
                cells = tuple((r, cc) for cc in range(start, end + 1))
                doors.append(("horizontal", cells))

    return doors


def _seal_doors(grid: np.ndarray, doors: list[tuple[str, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    sealed = np.array(grid, dtype=int, copy=True)
    for _kind, cells in doors:
        for r, c in cells:
            sealed[r, c] = BLUE
    return sealed


def _room_components(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    labels = np.full((h, w), -1, dtype=int)
    rooms: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == BLUE:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_boundary = False

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_boundary = True
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) != BLUE:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            if touches_boundary:
                continue

            room_id = len(rooms)
            arr = np.array(coords, dtype=int)
            labels[arr[:, 0], arr[:, 1]] = room_id
            rooms.append(arr)

    return rooms, labels


def _room_graph(
    labels: np.ndarray,
    room_count: int,
    doors: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> list[set[int]]:
    graph = [set() for _ in range(room_count)]

    for kind, cells in doors:
        if kind == "vertical":
            left = {int(labels[r, c - 1]) for r, c in cells if int(labels[r, c - 1]) >= 0}
            right = {int(labels[r, c + 1]) for r, c in cells if int(labels[r, c + 1]) >= 0}
            if len(left) != 1 or len(right) != 1:
                continue
            a = next(iter(left))
            b = next(iter(right))
        else:
            up = {int(labels[r - 1, c]) for r, c in cells if int(labels[r - 1, c]) >= 0}
            down = {int(labels[r + 1, c]) for r, c in cells if int(labels[r + 1, c]) >= 0}
            if len(up) != 1 or len(down) != 1:
                continue
            a = next(iter(up))
            b = next(iter(down))
        if a == b:
            continue
        graph[a].add(b)
        graph[b].add(a)

    return graph


def _bfs_distances(graph: list[set[int]], start: int) -> dict[int, int]:
    dist = {int(start): 0}
    queue = deque([int(start)])
    while queue:
        node = queue.popleft()
        for nxt in sorted(graph[node]):
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect rooms enclosed by blue walls, infer door connections, and color rooms
    by door-distance from the unique red-marked start room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_coords = np.argwhere(g == RED)
    assert red_coords.shape == (1, 2), "expected exactly one red start marker"

    doors = _door_segments(g)
    sealed = _seal_doors(g, doors)
    rooms, labels = _room_components(sealed)
    assert rooms, "expected at least one enclosed room"

    start_r, start_c = map(int, red_coords[0])
    start_room = int(labels[start_r, start_c])
    assert start_room >= 0, "start marker must lie inside an enclosed room"

    graph = _room_graph(labels, len(rooms), doors)
    dist = _bfs_distances(graph, start_room)

    out = g.copy()
    for room_id, coords in enumerate(rooms):
        if room_id not in dist:
            continue
        color = int(FILL_CYCLE[dist[room_id] % len(FILL_CYCLE)])
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _room_boxes(
    row_sizes: list[int], col_sizes: list[int]
) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]]:
    wall_rows = [0]
    for size in row_sizes:
        wall_rows.append(wall_rows[-1] + int(size) + 1)
    wall_cols = [0]
    for size in col_sizes:
        wall_cols.append(wall_cols[-1] + int(size) + 1)

    boxes: list[tuple[int, int, int, int]] = []
    for rr in range(len(row_sizes)):
        for cc in range(len(col_sizes)):
            boxes.append(
                (
                    int(wall_rows[rr] + 1),
                    int(wall_rows[rr + 1] - 1),
                    int(wall_cols[cc] + 1),
                    int(wall_cols[cc + 1] - 1),
                )
            )

    return wall_rows, wall_cols, boxes


def _candidate_edges(
    row_sizes: list[int],
    col_sizes: list[int],
    wall_rows: list[int],
    wall_cols: list[int],
) -> list[dict[str, int]]:
    room_rows = len(row_sizes)
    room_cols = len(col_sizes)
    edges: list[dict[str, int]] = []

    def room_id(rr: int, cc: int) -> int:
        return int(rr * room_cols + cc)

    for rr in range(room_rows):
        top = int(wall_rows[rr] + 1)
        bottom = int(wall_rows[rr + 1] - 1)
        for cc in range(room_cols - 1):
            edges.append(
                {
                    "a": room_id(rr, cc),
                    "b": room_id(rr, cc + 1),
                    "kind": 0,
                    "fixed": int(wall_cols[cc + 1]),
                    "start": top,
                    "end": bottom,
                }
            )

    for rr in range(room_rows - 1):
        fixed = int(wall_rows[rr + 1])
        for cc in range(room_cols):
            left = int(wall_cols[cc] + 1)
            right = int(wall_cols[cc + 1] - 1)
            edges.append(
                {
                    "a": room_id(rr, cc),
                    "b": room_id(rr + 1, cc),
                    "kind": 1,
                    "fixed": fixed,
                    "start": left,
                    "end": right,
                }
            )

    return edges


def _choose_edges(room_count: int, edges: list[dict[str, int]]) -> list[int]:
    chosen: list[int] = []
    seen = {0}
    remaining = set(range(1, room_count))

    while remaining:
        frontier = [
            idx
            for idx, edge in enumerate(edges)
            if (edge["a"] in seen) ^ (edge["b"] in seen)
        ]
        if not frontier:
            raise RuntimeError("room grid adjacency unexpectedly disconnected")
        idx = int(np.random.choice(frontier))
        chosen.append(idx)
        edge = edges[idx]
        seen.add(int(edge["a"]))
        seen.add(int(edge["b"]))
        remaining = set(range(room_count)) - seen

    leftovers = [idx for idx in range(len(edges)) if idx not in chosen]
    extra_cap = min(len(leftovers), max(0, room_count // 2))
    if extra_cap > 0:
        extra_count = int(np.random.randint(0, extra_cap + 1))
        if extra_count:
            extra = np.random.choice(leftovers, size=extra_count, replace=False)
            chosen.extend(int(idx) for idx in extra)

    return chosen


def _graph_from_edges(room_count: int, edges: list[dict[str, int]], chosen: list[int]) -> list[set[int]]:
    graph = [set() for _ in range(room_count)]
    for idx in chosen:
        edge = edges[idx]
        a = int(edge["a"])
        b = int(edge["b"])
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _farthest_node(graph: list[set[int]], start: int) -> tuple[int, dict[int, int]]:
    dist = _bfs_distances(graph, start)
    farthest = max(dist.items(), key=lambda item: (item[1], item[0]))[0]
    return int(farthest), dist


def _carve_door(grid: np.ndarray, edge: dict[str, int]) -> bool:
    span = int(edge["end"] - edge["start"] + 1)
    widths = [width for width in (1, 2) if span >= width + 2]
    if not widths:
        return False

    width = int(np.random.choice(widths))
    gap_start = int(np.random.randint(edge["start"] + 1, edge["end"] - width + 2))
    fixed = int(edge["fixed"])

    if int(edge["kind"]) == 0:
        grid[gap_start : gap_start + width, fixed] = BLACK
    else:
        grid[fixed, gap_start : gap_start + width] = BLACK
    return True


def _expected_output(
    inp: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    graph: list[set[int]],
    start_room: int,
) -> np.ndarray:
    out = np.array(inp, copy=True)
    dist = _bfs_distances(graph, start_room)
    for room_id, (r0, r1, c0, c1) in enumerate(boxes):
        color = int(FILL_CYCLE[dist[room_id] % len(FILL_CYCLE)])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    room_rows = int(np.random.randint(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1] + 1))
    room_cols = int(np.random.randint(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1] + 1))
    row_sizes = [int(x) for x in np.random.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1] + 1, size=room_rows)]
    col_sizes = [int(x) for x in np.random.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1] + 1, size=room_cols)]

    wall_rows, wall_cols, boxes = _room_boxes(row_sizes, col_sizes)
    h = int(wall_rows[-1] + 1)
    w = int(wall_cols[-1] + 1)
    room_count = room_rows * room_cols

    inp = np.full((h, w), BLUE, dtype=int)
    for r0, r1, c0, c1 in boxes:
        inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    edges = _candidate_edges(row_sizes, col_sizes, wall_rows, wall_cols)
    chosen = _choose_edges(room_count, edges)
    for idx in chosen:
        if not _carve_door(inp, edges[idx]):
            return None

    graph = _graph_from_edges(room_count, edges, chosen)
    want_long = room_count >= 5 and bool(np.random.rand() < LONG_PATH_PROB)

    if want_long:
        seed = int(np.random.randint(room_count))
        end_a, _ = _farthest_node(graph, seed)
        end_b, dist_a = _farthest_node(graph, end_a)
        start_room = int(np.random.choice([end_a, end_b]))
        dist = dist_a if start_room == end_a else _bfs_distances(graph, start_room)
    else:
        start_room = int(np.random.randint(room_count))
        dist = _bfs_distances(graph, start_room)

    max_dist = max(dist.values())
    if max_dist < (4 if want_long else 2):
        return None

    r0, r1, c0, c1 = boxes[start_room]
    mark_r = int(np.random.randint(r0, r1 + 1))
    mark_c = int(np.random.randint(c0, c1 + 1))
    inp[mark_r, mark_c] = RED

    expected = _expected_output(inp, boxes, graph, start_room)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
