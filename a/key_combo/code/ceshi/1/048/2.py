"""Periodic-wall completion followed by shortest-path tracing."""

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
CYAN = 8
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _infer_period(walls):
    if not walls.any():
        raise ValueError("wall generator is missing")
    for period in range(1, walls.shape[0] // 2 + 1):
        if walls[2 * period:].any():
            continue
        if np.array_equal(walls[:period], walls[period:2 * period]):
            return period
    raise ValueError("top wall pattern is not two equal periods")


def _shortest(walls, start, goal, *, count_paths=False, blocked_edge=None):
    h, w = walls.shape
    distance = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=np.int64)
    previous = {}
    distance[start] = 0
    ways[start] = 1
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        r, c = cell
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if not (0 <= nr < h and 0 <= nc < w) or walls[nr, nc]:
                continue
            if blocked_edge is not None and frozenset((cell, nxt)) == blocked_edge:
                continue
            candidate = distance[cell] + 1
            if distance[nxt] < 0:
                distance[nxt] = candidate
                ways[nxt] = ways[cell]
                previous[nxt] = cell
                queue.append(nxt)
            elif distance[nxt] == candidate:
                ways[nxt] = min(2, ways[nxt] + ways[cell])
    if distance[goal] < 0:
        return None, 0, None
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return int(distance[goal]), int(ways[goal]) if count_paths else 1, path


def main(grid: np.ndarray) -> np.ndarray:
    """Complete a vertical wall period, then mark the unique shortest endpoint path.

    Input structure: Black background, cyan walls showing exactly two complete
    periods at the top, and two red endpoints below the shown wall generator.
    Rule: Repeat the cyan wall mask vertically to the bottom, find the shortest
    four-neighbor route through non-wall cells, and color its intermediate cells
    yellow while preserving both endpoints and all completed walls.
    Key insight: Apparently empty lower cells include implied walls, so periodic
    completion must occur before pathfinding.
    """
    grid = np.asarray(grid)
    visible_walls = grid == CYAN
    period = _infer_period(visible_walls)
    full_walls = np.zeros(grid.shape, dtype=bool)
    for r in range(grid.shape[0]):
        full_walls[r] = visible_walls[r % period]

    endpoints = [tuple(x) for x in np.argwhere(grid == RED)]
    if len(endpoints) != 2 or any(full_walls[p] for p in endpoints):
        raise ValueError("expected two endpoints on open cells")
    _, _, path = _shortest(full_walls, endpoints[0], endpoints[1])
    if path is None:
        raise ValueError("endpoints are disconnected")

    out = grid.copy()
    out[full_walls] = CYAN
    for point in path[1:-1]:
        out[point] = YELLOW
    out[endpoints[0]] = RED
    out[endpoints[1]] = RED
    return out


def _primitive(pattern):
    p = pattern.shape[0]
    for d in range(1, p):
        if p % d == 0 and np.array_equal(pattern, np.tile(pattern[:d], (p // d, 1))):
            return False
    return True


def _has_longer_alternative(walls, path, shortest_distance):
    for a, b in zip(path, path[1:]):
        distance, _, _ = _shortest(walls, path[0], path[-1], blocked_edge=frozenset((a, b)))
        if distance is not None and distance > shortest_distance:
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(12000):
        period = random.randint(3, 6)
        width = random.randint(9, 16)
        height = random.randint(2 * period + 5, min(24, 2 * period + 12))
        pattern = np.random.random((period, width)) < random.uniform(0.20, 0.34)
        pattern[-1, random.randrange(width)] = True
        if not _primitive(pattern):
            continue
        full_walls = np.tile(pattern, (height // period + 1, 1))[:height]
        candidates = [
            (r, c)
            for r in range(2 * period + 1, height)
            for c in range(width)
            if not full_walls[r, c]
        ]
        if len(candidates) < 2:
            continue
        for _ in range(50):
            start, goal = random.sample(candidates, 2)
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 5:
                continue
            distance, ways, path = _shortest(full_walls, start, goal, count_paths=True)
            if path is None or ways != 1 or distance <= 5:
                continue
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if distance <= manhattan or not _has_longer_alternative(full_walls, path, distance):
                continue
            visible_walls = np.zeros_like(full_walls)
            visible_walls[:2 * period] = full_walls[:2 * period]
            visible_distance, _, visible_path = _shortest(visible_walls, start, goal)
            if visible_path is None or visible_distance >= distance:
                continue
            if not any(full_walls[p] for p in visible_path):
                continue
            grid = np.zeros((height, width), dtype=int)
            grid[:2 * period][full_walls[:2 * period]] = CYAN
            grid[start] = RED
            grid[goal] = RED
            return grid
    raise RuntimeError("failed to generate a non-degenerate periodic maze")
