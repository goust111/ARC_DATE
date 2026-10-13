from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

SOURCE_COLORS = (1, 4, 6, 7, 8, 9)
GEN_RETRIES = 400

Room = tuple[int, int]
Edge = tuple[Room, Room]


def _edge_key(a: Room, b: Room) -> Edge:
    return (a, b) if a <= b else (b, a)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _adjacent_rooms(room: Room, n_rows: int, n_cols: int) -> list[Room]:
    r, c = room
    out: list[Room] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n_rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n_cols:
        out.append((r, c + 1))
    return out


def _axis_layout(lengths: list[int]) -> tuple[list[int], list[int], list[int], int]:
    starts: list[int] = []
    ends: list[int] = []
    walls = [0]
    pos = 1
    for span in lengths:
        starts.append(pos)
        ends.append(pos + span - 1)
        pos += span
        walls.append(pos)
        pos += 1
    return starts, ends, walls, pos


def _room_boxes(row_sizes: list[int], col_sizes: list[int]) -> tuple[dict[Room, tuple[int, int, int, int]], list[int], list[int], list[int], list[int], int, int]:
    row_starts, row_ends, row_walls, h = _axis_layout(row_sizes)
    col_starts, col_ends, col_walls, w = _axis_layout(col_sizes)
    boxes: dict[Room, tuple[int, int, int, int]] = {}
    for rr in range(len(row_sizes)):
        for cc in range(len(col_sizes)):
            boxes[(rr, cc)] = (row_starts[rr], row_ends[rr], col_starts[cc], col_ends[cc])
    return boxes, row_starts, row_ends, col_starts, col_ends, h, w


def _label_regions(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[int]]:
    h, w = grid.shape
    blocked = np.isin(grid, (GREY, RED, GREEN))
    labels = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    colors: list[int] = []

    for r in range(h):
        for c in range(w):
            if blocked[r, c] or labels[r, c] != -1:
                continue
            rid = len(regions)
            q = deque([(r, c)])
            labels[r, c] = rid
            cells: list[tuple[int, int]] = []
            color = int(grid[r, c])

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                assert int(grid[cr, cc]) == color, "container interiors must be monochrome"
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = rid
                        q.append((nr, nc))

            regions.append(cells)
            colors.append(color)

    return labels, regions, colors


def _valve_candidates(labels: np.ndarray, r: int, c: int) -> list[tuple[str, int, int]]:
    h, w = labels.shape
    out: list[tuple[str, int, int]] = []
    if 0 < r < h - 1:
        upper = int(labels[r - 1, c])
        lower = int(labels[r + 1, c])
        if upper >= 0 and lower >= 0 and upper != lower:
            out.append(("horizontal", upper, lower))
    if 0 < c < w - 1:
        left = int(labels[r, c - 1])
        right = int(labels[r, c + 1])
        if left >= 0 and right >= 0 and left != right:
            out.append(("vertical", left, right))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat grey walls as barriers, recover valve-induced container adjacency,
    then flood all containers reachable from the unique liquid source.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    liquid_colors = sorted(set(map(int, np.unique(g))) - {BLACK, RED, GREEN, GREY})
    assert len(liquid_colors) == 1, "expected exactly one liquid color"
    liquid = liquid_colors[0]

    labels, regions, region_colors = _label_regions(g)
    assert regions, "expected at least one container"
    assert set(region_colors) <= {BLACK, liquid}, "unexpected interior colors"

    source_ids = {int(labels[r, c]) for r, c in np.argwhere(g == liquid)}
    assert len(source_ids) == 1, "liquid must fill exactly one source container"
    source_id = next(iter(source_ids))

    graph = [set() for _ in regions]
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in (RED, GREEN):
                continue
            candidates = _valve_candidates(labels, r, c)
            assert len(candidates) == 1, "each valve must separate exactly two containers"
            orient, a, b = candidates[0]
            if color == GREEN:
                graph[a].add(b)
                graph[b].add(a)
            else:
                assert orient == "horizontal", "red valves only belong on horizontal walls"
                graph[a].add(b)

    reachable = {source_id}
    q = deque([source_id])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in reachable:
                reachable.add(nxt)
                q.append(nxt)

    out = g.copy()
    for rid in reachable:
        for r, c in regions[rid]:
            out[r, c] = liquid
    return out


def _room_graph(valves: dict[Edge, str]) -> dict[Room, set[Room]]:
    graph: dict[Room, set[Room]] = {}
    for (a, b), kind in valves.items():
        graph.setdefault(a, set())
        graph.setdefault(b, set())
        if kind == "green":
            graph[a].add(b)
            graph[b].add(a)
        else:
            upper, lower = (a, b) if a[0] < b[0] else (b, a)
            assert upper[1] == lower[1] and upper[0] + 1 == lower[0]
            graph[upper].add(lower)
    return graph


def _distances(source: Room, valves: dict[Edge, str]) -> dict[Room, int]:
    graph = _room_graph(valves)
    dist = {source: 0}
    q = deque([source])
    while q:
        cur = q.popleft()
        for nxt in graph.get(cur, ()):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _set_valve(valves: dict[Edge, str], a: Room, b: Room, kind: str) -> None:
    key = _edge_key(a, b)
    assert key not in valves
    if a[0] == b[0]:
        assert kind == "green", "vertical adjacencies only admit green valves"
    valves[key] = kind


def _cascade_candidates(
    source: Room,
    reached: set[Room],
    valves: dict[Edge, str],
    blocked_room: Room,
    n_rows: int,
    n_cols: int,
) -> list[tuple[Room, Room, str]]:
    non_source: list[tuple[Room, Room, str]] = []
    fallback: list[tuple[Room, Room, str]] = []

    for room in reached:
        for nbr in _adjacent_rooms(room, n_rows, n_cols):
            key = _edge_key(room, nbr)
            if nbr in reached or key in valves or blocked_room in key:
                continue
            choices = ["green"]
            if room[1] == nbr[1] and room[0] < nbr[0]:
                choices.append("red")
            for kind in choices:
                candidate = (room, nbr, kind)
                if room == source:
                    fallback.append(candidate)
                else:
                    non_source.append(candidate)
    return non_source if non_source else fallback


def _all_adjacencies(n_rows: int, n_cols: int) -> list[Edge]:
    edges: list[Edge] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r + 1 < n_rows:
                edges.append(_edge_key((r, c), (r + 1, c)))
            if c + 1 < n_cols:
                edges.append(_edge_key((r, c), (r, c + 1)))
    return edges


def _render_case(
    row_sizes: list[int],
    col_sizes: list[int],
    source: Room,
    source_color: int,
    valves: dict[Edge, str],
) -> tuple[np.ndarray, dict[Room, tuple[int, int, int, int]]]:
    boxes, _row_starts, row_ends, _col_starts, col_ends, h, w = _room_boxes(row_sizes, col_sizes)
    grid = np.full((h, w), BLACK, dtype=int)

    for wall_r in {0, h - 1} | {row_ends[r] + 1 for r in range(len(row_sizes) - 1)}:
        grid[wall_r, :] = GREY
    for wall_c in {0, w - 1} | {col_ends[c] + 1 for c in range(len(col_sizes) - 1)}:
        grid[:, wall_c] = GREY

    for room, (r0, r1, c0, c1) in boxes.items():
        fill = source_color if room == source else BLACK
        grid[r0 : r1 + 1, c0 : c1 + 1] = fill

    for (a, b), kind in valves.items():
        if a[0] != b[0]:
            upper = a if a[0] < b[0] else b
            wall_r = boxes[upper][1] + 1
            _, _, c0, c1 = boxes[upper]
            valve_c = random.randint(c0, c1)
            grid[wall_r, valve_c] = GREEN if kind == "green" else RED
        else:
            assert kind == "green"
            left = a if a[1] < b[1] else b
            wall_c = boxes[left][3] + 1
            r0, r1, _, _ = boxes[left]
            valve_r = random.randint(r0, r1)
            grid[valve_r, wall_c] = GREEN

    return grid, boxes


def _expected_output(
    inp: np.ndarray,
    boxes: dict[Room, tuple[int, int, int, int]],
    reached: set[Room],
    source_color: int,
) -> np.ndarray:
    out = inp.copy()
    for room in reached:
        r0, r1, c0, c1 = boxes[room]
        out[r0 : r1 + 1, c0 : c1 + 1] = source_color
    return out


def _generate_attempt() -> np.ndarray | None:
    n_rows = random.randint(3, 4)
    n_cols = random.randint(2, 4)
    row_sizes = [random.randint(2, 5) for _ in range(n_rows)]
    col_sizes = [random.randint(2, 5) for _ in range(n_cols)]
    source_color = random.choice(SOURCE_COLORS)

    source = (random.randint(1, n_rows - 2), random.randrange(n_cols))
    blocked_room = (source[0] - 1, source[1])
    below_room = (source[0] + 1, source[1])
    side_choices = []
    if source[1] > 0:
        side_choices.append((source[0], source[1] - 1))
    if source[1] + 1 < n_cols:
        side_choices.append((source[0], source[1] + 1))
    side_room = random.choice(side_choices)

    valves: dict[Edge, str] = {}
    _set_valve(valves, blocked_room, source, "red")
    _set_valve(valves, source, below_room, "red")
    _set_valve(valves, source, side_room, "green")

    reached = set(_distances(source, valves))
    cascade = _cascade_candidates(source, reached, valves, blocked_room, n_rows, n_cols)
    if not cascade:
        return None
    room_a, room_b, kind = random.choice(cascade)
    _set_valve(valves, room_a, room_b, kind)

    extra_edges = _all_adjacencies(n_rows, n_cols)
    random.shuffle(extra_edges)
    for a, b in extra_edges:
        if _edge_key(a, b) in valves or blocked_room in (a, b):
            continue
        if a[0] == b[0]:
            if random.random() < 0.45:
                _set_valve(valves, a, b, "green")
        else:
            p = random.random()
            if p < 0.25:
                _set_valve(valves, a, b, "green")
            elif p < 0.50:
                _set_valve(valves, a, b, "red")

    dist = _distances(source, valves)
    reached_rooms = set(dist)
    if blocked_room in reached_rooms:
        return None
    if below_room not in reached_rooms or side_room not in reached_rooms:
        return None
    if len(reached_rooms) < 4:
        return None
    if max(dist.values()) < 2:
        return None

    inp, boxes = _render_case(row_sizes, col_sizes, source, source_color, valves)
    expected = _expected_output(inp, boxes, reached_rooms, source_color)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid valve-container ARC input")
