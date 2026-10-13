"""Mileage-weighted route selection on a square-and-track graph."""

from collections import deque
import heapq
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _components(mask, connectivity=4):
    directions = DIRS if connectivity == 4 else DIRS8
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        component = []
        while queue:
            r, c = queue.pop()
            component.append((r, c))
            for dr, dc in directions:
                nxt = r + dr, c + dc
                if 0 <= nxt[0] < mask.shape[0] and 0 <= nxt[1] < mask.shape[1]:
                    if mask[nxt] and not seen[nxt]:
                        seen[nxt] = True
                        queue.append(nxt)
        result.append(component)
    return result


def _nodes(grid):
    nodes = []
    for color in (RED, BLUE):
        for component in _components(grid == color):
            rs = [p[0] for p in component]
            cs = [p[1] for p in component]
            top, bottom = min(rs), max(rs)
            left, right = min(cs), max(cs)
            if bottom - top != right - left or len(component) != (bottom - top + 1) ** 2:
                raise ValueError("node is not a solid square")
            nodes.append({"color": color, "cells": set(component)})
    return nodes


def _extract_graph(grid):
    nodes = _nodes(grid)
    graph = {i: [] for i in range(len(nodes))}
    edges = []
    cell_to_node = {cell: i for i, node in enumerate(nodes) for cell in node["cells"]}
    for track in _components(grid == GRAY, connectivity=8):
        touched = set()
        for r, c in track:
            for dr, dc in DIRS8:
                node = cell_to_node.get((r + dr, c + dc))
                if node is not None:
                    touched.add(node)
        if len(touched) != 2:
            raise ValueError("each track component must connect exactly two nodes")
        a, b = sorted(touched)
        edge_id = len(edges)
        edges.append({"nodes": (a, b), "cells": set(track), "weight": len(track)})
        graph[a].append((b, len(track), edge_id))
        graph[b].append((a, len(track), edge_id))
    return nodes, edges, graph


def _dijkstra(graph, start, goal):
    distance = {start: 0}
    ways = {start: 1}
    previous = {}
    heap = [(0, start)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist != distance[node]:
            continue
        for nxt, weight, edge_id in graph[node]:
            candidate = dist + weight
            if candidate < distance.get(nxt, 10 ** 9):
                distance[nxt] = candidate
                ways[nxt] = ways[node]
                previous[nxt] = (node, edge_id)
                heapq.heappush(heap, (candidate, nxt))
            elif candidate == distance.get(nxt):
                ways[nxt] = min(2, ways[nxt] + ways[node])
    if goal not in distance:
        return None, 0, None, None
    node_path = [goal]
    edge_path = []
    while node_path[-1] != start:
        parent, edge_id = previous[node_path[-1]]
        edge_path.append(edge_id)
        node_path.append(parent)
    node_path.reverse()
    edge_path.reverse()
    return distance[goal], ways[goal], node_path, edge_path


def _hop_shortest(graph, start, goal):
    queue = deque([start])
    distance = {start: 0}
    previous = {}
    while queue:
        node = queue.popleft()
        for nxt, _, _ in graph[node]:
            if nxt not in distance:
                distance[nxt] = distance[node] + 1
                previous[nxt] = node
                queue.append(nxt)
    if goal not in distance:
        return None, None
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return distance[goal], path


def main(grid: np.ndarray) -> np.ndarray:
    """Highlight the terminal route with the fewest total track pixels.

    Input structure: Two red solid square terminals and several blue solid square
    stations are joined pairwise by separate one-cell-wide gray track components
    on the dominant background color.
    Rule: Treat squares as graph nodes and each gray component as an edge whose
    weight is its number of track cells. Recolor tracks of the unique minimum-total-
    mileage terminal route yellow and its intermediate station squares green.
    Key insight: Route cost is geometric track mileage, not the number of graph
    edges; a route visiting fewer stations can therefore lose.
    """
    grid = np.asarray(grid)
    nodes, edges, graph = _extract_graph(grid)
    terminals = [i for i, node in enumerate(nodes) if node["color"] == RED]
    if len(terminals) != 2:
        raise ValueError("expected exactly two red terminal squares")
    _, _, node_path, edge_path = _dijkstra(graph, terminals[0], terminals[1])
    if node_path is None:
        raise ValueError("terminal nodes are disconnected")
    out = grid.copy()
    for edge_id in edge_path:
        for point in edges[edge_id]["cells"]:
            out[point] = YELLOW
    for node_id in node_path[1:-1]:
        for point in nodes[node_id]["cells"]:
            out[point] = GREEN
    return out


def _polyline(points):
    cells = []
    for a, b in zip(points, points[1:]):
        if a[0] == b[0]:
            step = 1 if b[1] >= a[1] else -1
            segment = [(a[0], c) for c in range(a[1], b[1] + step, step)]
        elif a[1] == b[1]:
            step = 1 if b[0] >= a[0] else -1
            segment = [(r, a[1]) for r in range(a[0], b[0] + step, step)]
        else:
            raise ValueError("polyline segments must be axis aligned")
        cells.extend(segment if not cells else segment[1:])
    return cells


def _paint_node(grid, top, left, size, color):
    grid[top:top + size, left:left + size] = color


def generate_input() -> np.ndarray:
    for _ in range(2000):
        height = random.randint(21, 27)
        width = random.randint(25, 30)
        size = 2
        top_row = random.randint(1, 3)
        terminal_row = random.randint(8, 10)
        bottom_row = height - size - 2
        start_col = 1
        goal_col = width - size - 1
        top_count = random.randint(2, 4)
        usable_left, usable_right = 6, width - 7
        top_cols = [
            round(usable_left + i * (usable_right - usable_left) / max(1, top_count - 1))
            for i in range(top_count)
        ]
        if any(b - a < size + 2 for a, b in zip(top_cols, top_cols[1:])):
            continue
        bottom_col = width // 2 - 1
        grid = np.zeros((height, width), dtype=int)
        node_specs = [
            (terminal_row, start_col, RED),
            (terminal_row, goal_col, RED),
            *[(top_row, c, BLUE) for c in top_cols],
            (bottom_row, bottom_col, BLUE),
        ]
        for r, c, color in node_specs:
            _paint_node(grid, r, c, size, color)

        paths = []
        paths.append(_polyline([
            (terminal_row - 1, start_col),
            (top_row + size + 1, start_col),
            (top_row + size + 1, top_cols[0]),
            (top_row + size, top_cols[0]),
        ]))
        for left, right in zip(top_cols, top_cols[1:]):
            paths.append(_polyline([(top_row, left + size), (top_row, right - 1)]))
        paths.append(_polyline([
            (top_row + size, top_cols[-1] + 1),
            (top_row + size + 1, top_cols[-1] + 1),
            (top_row + size + 1, goal_col + 1),
            (terminal_row - 1, goal_col + 1),
        ]))
        paths.append(_polyline([
            (terminal_row + size, start_col + 1),
            (bottom_row, start_col + 1),
            (bottom_row, bottom_col - 1),
        ]))
        paths.append(_polyline([
            (bottom_row + 1, bottom_col + size),
            (bottom_row + 1, goal_col),
            (terminal_row + size, goal_col),
        ]))

        occupied = set()
        valid = True
        for path in paths:
            if occupied.intersection(path) or any(grid[p] != BG for p in path):
                valid = False
                break
            occupied.update(path)
            for point in path:
                grid[point] = GRAY
        if not valid:
            continue

        background = random.choice([0, 6, 7, 8, 9])
        if background != BG:
            grid[grid == BG] = background
        turns = random.randrange(4)
        grid = np.rot90(grid, turns).copy()
        try:
            nodes, edges, graph = _extract_graph(grid)
        except ValueError:
            continue
        terminals = [i for i, node in enumerate(nodes) if node["color"] == RED]
        if len(terminals) != 2 or len(edges) != len(paths):
            continue
        mileage, ways, node_path, _ = _dijkstra(graph, terminals[0], terminals[1])
        hops, hop_path = _hop_shortest(graph, terminals[0], terminals[1])
        if ways != 1 or node_path is None or hops is None:
            continue
        weighted_hops = len(node_path) - 1
        if not (hops < weighted_hops and set(hop_path) != set(node_path)):
            continue
        hop_mileage = 0
        for a, b in zip(hop_path, hop_path[1:]):
            hop_mileage += next(weight for nxt, weight, _ in graph[a] if nxt == b)
        if hop_mileage <= mileage:
            continue
        return grid
    raise RuntimeError("failed to generate a non-degenerate mileage graph")
