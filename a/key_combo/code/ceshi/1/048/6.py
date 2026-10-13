"""Room-graph abstraction with a mandatory key waypoint."""

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5


def _divider_indices(grid):
    gray = grid == GRAY
    row_dividers = [r for r in range(grid.shape[0]) if int(gray[r].sum()) > grid.shape[1] // 2]
    col_dividers = [c for c in range(grid.shape[1]) if int(gray[:, c].sum()) > grid.shape[0] // 2]
    return row_dividers, col_dividers


def _intervals(length, dividers):
    bounds = [-1] + list(dividers) + [length]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def _room_of(point, row_dividers, col_dividers):
    r, c = point
    rr = sum(d < r for d in row_dividers)
    cc = sum(d < c for d in col_dividers)
    return rr, cc


def _room_graph(grid, row_dividers, col_dividers):
    row_spans = _intervals(grid.shape[0], row_dividers)
    col_spans = _intervals(grid.shape[1], col_dividers)
    graph = {(r, c): set() for r in range(len(row_spans)) for c in range(len(col_spans))}

    for r, divider in enumerate(row_dividers):
        for c, (left, right) in enumerate(col_spans):
            if np.any(grid[divider, left:right] != GRAY):
                a, b = (r, c), (r + 1, c)
                graph[a].add(b)
                graph[b].add(a)
    for c, divider in enumerate(col_dividers):
        for r, (top, bottom) in enumerate(row_spans):
            if np.any(grid[top:bottom, divider] != GRAY):
                a, b = (r, c), (r, c + 1)
                graph[a].add(b)
                graph[b].add(a)
    return graph, len(row_spans), len(col_spans)


def _graph_shortest(graph, start, goal, blocked_edge=None, count_paths=False):
    queue = deque([start])
    distance = {start: 0}
    ways = {start: 1}
    previous = {}
    while queue:
        node = queue.popleft()
        for nxt in sorted(graph[node]):
            if blocked_edge is not None and frozenset((node, nxt)) == blocked_edge:
                continue
            candidate = distance[node] + 1
            if nxt not in distance:
                distance[nxt] = candidate
                ways[nxt] = ways[node]
                previous[nxt] = node
                queue.append(nxt)
            elif distance[nxt] == candidate:
                ways[nxt] = min(2, ways[nxt] + ways[node])
    if goal not in distance:
        return None, 0, None
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return distance[goal], ways[goal] if count_paths else 1, path


def main(grid: np.ndarray) -> np.ndarray:
    """Abstract divided rooms and mark the shortest route through the key room.

    Input structure: Gray one-cell dividers form rectangular rooms; single-cell
    gaps in divider segments are doors. Red, green, and blue pixels identify the
    start, key, and goal rooms respectively.
    Rule: Treat rooms as nodes and doors as edges, concatenate the shortest path
    from start to key with the shortest path from key to goal, and output one cell
    per room. Preserve marker colors, color other route rooms yellow, and leave
    all rooms outside the route black.
    Key insight: Pixel positions inside rooms are irrelevant; pathfinding occurs
    on the smaller room adjacency graph and the green room is mandatory.
    """
    grid = np.asarray(grid)
    row_dividers, col_dividers = _divider_indices(grid)
    graph, room_rows, room_cols = _room_graph(grid, row_dividers, col_dividers)
    marker_rooms = {}
    for color in (RED, GREEN, BLUE):
        coords = np.argwhere(grid == color)
        if len(coords) != 1:
            raise ValueError("each room marker must occur exactly once")
        marker_rooms[color] = _room_of(tuple(coords[0]), row_dividers, col_dividers)

    _, _, first = _graph_shortest(graph, marker_rooms[RED], marker_rooms[GREEN])
    _, _, second = _graph_shortest(graph, marker_rooms[GREEN], marker_rooms[BLUE])
    if first is None or second is None:
        raise ValueError("mandatory waypoint route is disconnected")

    out = np.zeros((room_rows, room_cols), dtype=int)
    for room in set(first + second):
        out[room] = YELLOW
    for color in (RED, GREEN, BLUE):
        out[marker_rooms[color]] = color
    return out


def _all_grid_edges(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _make_graph(rows, cols):
    graph = {(r, c): set() for r in range(rows) for c in range(cols)}
    for a, b in _all_grid_edges(rows, cols):
        if random.random() < random.uniform(0.48, 0.68):
            graph[a].add(b)
            graph[b].add(a)
    return graph


def _has_alternative(graph, path, shortest_distance):
    for a, b in zip(path, path[1:]):
        distance, _, _ = _graph_shortest(
            graph, path[0], path[-1], blocked_edge=frozenset((a, b))
        )
        if distance is not None and distance > shortest_distance:
            return True
    return False


def _choose_markers(graph):
    nodes = list(graph)
    for _ in range(300):
        start, key, goal = random.sample(nodes, 3)
        d1, ways1, p1 = _graph_shortest(graph, start, key, count_paths=True)
        d2, ways2, p2 = _graph_shortest(graph, key, goal, count_paths=True)
        direct, direct_ways, direct_path = _graph_shortest(graph, start, goal, count_paths=True)
        if None in (d1, d2, direct) or ways1 != 1 or ways2 != 1 or direct_ways != 1:
            continue
        if d1 < 2 or d2 < 2 or direct >= d1 + d2 or key in direct_path:
            continue
        if set(p1).intersection(p2) != {key}:
            continue
        if not _has_alternative(graph, p1, d1) or not _has_alternative(graph, p2, d2):
            continue
        return start, key, goal
    return None


def generate_input() -> np.ndarray:
    for _ in range(6000):
        room_rows = random.randint(3, 5)
        room_cols = random.randint(3, 5)
        graph = _make_graph(room_rows, room_cols)
        markers = _choose_markers(graph)
        if markers is None:
            continue
        room_h = random.randint(2, 4)
        room_w = random.randint(2, 4)
        height = room_rows * room_h + room_rows - 1
        width = room_cols * room_w + room_cols - 1
        grid = np.zeros((height, width), dtype=int)
        row_dividers = [(r + 1) * room_h + r for r in range(room_rows - 1)]
        col_dividers = [(c + 1) * room_w + c for c in range(room_cols - 1)]
        grid[row_dividers, :] = GRAY
        grid[:, col_dividers] = GRAY

        for a, b in _all_grid_edges(room_rows, room_cols):
            if b not in graph[a]:
                continue
            if a[0] != b[0]:
                divider = row_dividers[min(a[0], b[0])]
                left = a[1] * (room_w + 1)
                grid[divider, left + random.randrange(room_w)] = BG
            else:
                divider = col_dividers[min(a[1], b[1])]
                top = a[0] * (room_h + 1)
                grid[top + random.randrange(room_h), divider] = BG

        for room, color in zip(markers, (RED, GREEN, BLUE)):
            top = room[0] * (room_h + 1)
            left = room[1] * (room_w + 1)
            grid[top + random.randrange(room_h), left + random.randrange(room_w)] = color

        found_rows, found_cols = _divider_indices(grid)
        found_graph, _, _ = _room_graph(grid, found_rows, found_cols)
        if found_rows == row_dividers and found_cols == col_dividers and found_graph == graph:
            return grid
    raise RuntimeError("failed to generate a non-degenerate room graph")
