from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


_GEN_CALL_COUNT = 0
_GEN_ATTEMPTS = 400
_ROOM_COUNT_RANGE = (2, 5)
_ROOM_SIZE_RANGE = (2, 5)


def _divider_indices(grey_counts: np.ndarray, axis_len: int) -> list[int]:
    return [int(i) for i, count in enumerate(grey_counts.tolist()) if int(count) > axis_len // 2]


def _spans_from_dividers(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for divider in dividers:
        end = int(divider) - 1
        assert start <= end, "expected non-empty room spans between divider lines"
        spans.append((start, end))
        start = int(divider) + 1
    assert start <= length - 1, "expected room cells after the final divider line"
    spans.append((start, length - 1))
    return spans


def _locate_index(value: int, spans: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(spans):
        if lo <= value <= hi:
            return idx
    raise AssertionError("marker must lie inside exactly one room span")


def _find_single(g: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(g == int(color))
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _room_graph(
    g: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    row_dividers: list[int],
    col_dividers: list[int],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    room_rows = len(row_spans)
    room_cols = len(col_spans)
    graph = {(r, c): set() for r in range(room_rows) for c in range(room_cols)}

    for r, divider_row in enumerate(row_dividers):
        for c, (c0, c1) in enumerate(col_spans):
            if np.any(g[int(divider_row), c0 : c1 + 1] != GREY):
                a = (r, c)
                b = (r + 1, c)
                graph[a].add(b)
                graph[b].add(a)

    for c, divider_col in enumerate(col_dividers):
        for r, (r0, r1) in enumerate(row_spans):
            if np.any(g[r0 : r1 + 1, int(divider_col)] != GREY):
                a = (r, c)
                b = (r, c + 1)
                graph[a].add(b)
                graph[b].add(a)

    return graph


def _bfs_dist(graph: dict[tuple[int, int], set[tuple[int, int]]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _unique_shortest_path(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    dist = _bfs_dist(graph, start)
    assert goal in dist, "expected the required rooms to be connected by doors"
    rev = _bfs_dist(graph, goal)
    total = dist[goal]

    shortest_counts: dict[tuple[int, int], int] = {start: 1}
    for node in sorted(dist, key=dist.get):
        base = shortest_counts.get(node, 0)
        if base == 0:
            continue
        for nxt in graph[node]:
            if dist.get(nxt) == dist[node] + 1:
                shortest_counts[nxt] = min(2, shortest_counts.get(nxt, 0) + base)
    assert shortest_counts.get(goal, 0) == 1, "expected a unique shortest room path"

    path = [goal]
    cur = goal
    while cur != start:
        preds = [
            prev
            for prev in graph[cur]
            if dist.get(prev) == dist[cur] - 1 and dist.get(prev, 0) + rev.get(prev, 10**9) == total
        ]
        assert len(preds) == 1, "expected a unique shortest room path"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downscale a divider-grid floorplan to room cells and mark the unique shortest room path that visits the key.

    Input structure:
    - The background is black, with gray full rows and gray full columns partitioning the canvas into a rectangular array of rooms.
    - Some gray divider segments contain a one-cell gap; each gap is a door joining the two rooms on opposite sides of that divider segment.
    - Exactly one red pixel marks the start room, one green pixel marks the key room, and one blue pixel marks the goal room.

    Rule:
    - Count the room partitions induced by the gray divider rows and columns; the output grid has one cell per room.
    - Build a room-adjacency graph whose edges come from divider gaps, then find the unique shortest room path from the start room to the key room and from the key room to the goal room.
    - Paint the output rooms on that combined path: start red, key green, goal blue, all other visited rooms yellow, and every untouched room black.

    Key insight:
    - The puzzle lives at the room level rather than the pixel level, so door positions and marker offsets matter only for recovering the coarse room graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_dividers = _divider_indices((g == GREY).sum(axis=1), g.shape[1])
    col_dividers = _divider_indices((g == GREY).sum(axis=0), g.shape[0])
    row_spans = _spans_from_dividers(g.shape[0], row_dividers)
    col_spans = _spans_from_dividers(g.shape[1], col_dividers)

    start_px = _find_single(g, RED)
    key_px = _find_single(g, GREEN)
    goal_px = _find_single(g, BLUE)

    start_room = (_locate_index(start_px[0], row_spans), _locate_index(start_px[1], col_spans))
    key_room = (_locate_index(key_px[0], row_spans), _locate_index(key_px[1], col_spans))
    goal_room = (_locate_index(goal_px[0], row_spans), _locate_index(goal_px[1], col_spans))
    assert len({start_room, key_room, goal_room}) == 3, "markers must occupy three distinct rooms"

    graph = _room_graph(g, row_spans, col_spans, row_dividers, col_dividers)
    path_to_key = _unique_shortest_path(graph, start_room, key_room)
    path_to_goal = _unique_shortest_path(graph, key_room, goal_room)
    route = path_to_key + path_to_goal[1:]

    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for room in route:
        out[room] = YELLOW
    out[start_room] = RED
    out[key_room] = GREEN
    out[goal_room] = BLUE
    return out


def _room_neighbors(node: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < room_rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < room_cols:
        out.append((r, c + 1))
    return out


def _random_tree(room_rows: int, room_cols: int) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph = {(r, c): set() for r in range(room_rows) for c in range(room_cols)}
    start = (random.randrange(room_rows), random.randrange(room_cols))
    stack = [start]
    seen = {start}
    while stack:
        cur = stack[-1]
        options = [nxt for nxt in _room_neighbors(cur, room_rows, room_cols) if nxt not in seen]
        if not options:
            stack.pop()
            continue
        nxt = random.choice(options)
        graph[cur].add(nxt)
        graph[nxt].add(cur)
        seen.add(nxt)
        stack.append(nxt)
    return graph


def _tree_path(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    parent = {start: None}
    q: deque[tuple[int, int]] = deque([start])
    while q and goal not in parent:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    assert goal in parent
    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _choose_special_rooms(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    mode: str,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    nodes = list(graph)
    candidates = []

    for start in nodes:
        for goal in nodes:
            if start == goal:
                continue
            direct = _tree_path(graph, start, goal)
            if mode == "on_path":
                if len(direct) < 4:
                    continue
                for key in direct[1:-1]:
                    candidates.append((start, key, goal, direct))
            else:
                for key in nodes:
                    if key in direct:
                        continue
                    route = _tree_path(graph, start, key) + _tree_path(graph, key, goal)[1:]
                    if len(route) >= 5:
                        candidates.append((start, key, goal, route))

    assert candidates, "failed to choose start/key/goal rooms with the requested branch"
    return random.choice(candidates)


def _render_layout(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    dividers: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        spans.append((pos, pos + size - 1))
        pos += size
        if idx + 1 < len(sizes):
            dividers.append(pos)
            pos += 1
    return spans, dividers, pos


def _expected_output(
    room_rows: int,
    room_cols: int,
    start: tuple[int, int],
    key: tuple[int, int],
    goal: tuple[int, int],
    route: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((room_rows, room_cols), dtype=int)
    for room in route:
        out[room] = YELLOW
    out[start] = RED
    out[key] = GREEN
    out[goal] = BLUE
    return out


def _place_marker(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], color: int) -> None:
    r = random.randint(row_span[0], row_span[1])
    c = random.randint(col_span[0], col_span[1])
    grid[r, c] = color


def _sample_room_counts() -> tuple[int, int]:
    for _ in range(50):
        room_rows = random.randint(*_ROOM_COUNT_RANGE)
        room_cols = random.randint(*_ROOM_COUNT_RANGE)
        if room_rows * room_cols >= 6:
            return room_rows, room_cols
    raise RuntimeError("failed to sample a sufficiently rich room grid")


def _build_input(mode: str) -> tuple[np.ndarray, np.ndarray]:
    room_rows, room_cols = _sample_room_counts()
    row_sizes = [random.randint(*_ROOM_SIZE_RANGE) for _ in range(room_rows)]
    col_sizes = [random.randint(*_ROOM_SIZE_RANGE) for _ in range(room_cols)]

    row_spans, row_dividers, height = _render_layout(row_sizes)
    col_spans, col_dividers, width = _render_layout(col_sizes)
    assert height <= 30 and width <= 30

    graph = _random_tree(room_rows, room_cols)
    start, key, goal, route = _choose_special_rooms(graph, mode)
    expected = _expected_output(room_rows, room_cols, start, key, goal, route)

    grid = np.zeros((height, width), dtype=int)
    grid[row_dividers, :] = GREY
    grid[:, col_dividers] = GREY

    for a, neighbors in graph.items():
        for b in neighbors:
            if a > b:
                continue
            if a[0] != b[0]:
                divider_row = row_dividers[min(a[0], b[0])]
                c0, c1 = col_spans[a[1]]
                gap_col = random.randint(c0, c1)
                grid[divider_row, gap_col] = BLACK
            else:
                divider_col = col_dividers[min(a[1], b[1])]
                r0, r1 = row_spans[a[0]]
                gap_row = random.randint(r0, r1)
                grid[gap_row, divider_col] = BLACK

    _place_marker(grid, row_spans[start[0]], col_spans[start[1]], RED)
    _place_marker(grid, row_spans[key[0]], col_spans[key[1]], GREEN)
    _place_marker(grid, row_spans[goal[0]], col_spans[goal[1]], BLUE)
    return grid, expected


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    modes = ("detour", "on_path")
    mode = modes[_GEN_CALL_COUNT % len(modes)]
    _GEN_CALL_COUNT += 1

    for _ in range(_GEN_ATTEMPTS):
        try:
            grid, expected = _build_input(mode)
        except AssertionError:
            continue

        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        solved = main(grid)
        if np.array_equal(solved, expected):
            return grid

    raise RuntimeError("failed to generate a valid room-with-key puzzle instance")
