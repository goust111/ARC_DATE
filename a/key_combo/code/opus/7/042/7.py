from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors4, neighbors8


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (13, 18)
PATH_LEN_RANGE = (8, 17)
MIN_MANHATTAN_SEPARATION = 6
MIN_TURNS = 2
MIN_SECOND_RING_CELLS = 3


def _shortest_path_interior(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    dist = np.full((h, w), -1, dtype=int)
    paths = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    paths[start] = 1

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) == Color.BLUE:
                continue
            nd = int(dist[r, c]) + 1
            if int(dist[nr, nc]) == -1:
                dist[nr, nc] = nd
                paths[nr, nc] = paths[r, c]
                q.append((nr, nc))
            elif int(dist[nr, nc]) == nd:
                paths[nr, nc] = min(2, int(paths[nr, nc]) + int(paths[r, c]))

    assert int(dist[goal]) >= 0, "expected connected red endpoints"
    assert int(paths[goal]) == 1, "expected a unique shortest path"

    full_path = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        predecessors = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if in_bounds((h, w), nr, nc) and int(dist[nr, nc]) == int(dist[r, c]) - 1
        ]
        assert len(predecessors) == 1, "expected a uniquely reconstructible shortest path"
        cur = predecessors[0]
        full_path.append(cur)

    full_path.reverse()
    for r, c in full_path[1:-1]:
        assert int(grid[r, c]) == Color.BLACK, "shortest-path interior must lie on black corridor cells"
    return full_path[1:-1]


def _distance_from_path(grid: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, grid.shape)
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) == Color.BLUE or int(dist[nr, nc]) != -1:
                continue
            dist[nr, nc] = int(dist[r, c]) + 1
            q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the unique maze path yellow, then fill even obstacle-constrained Chebyshev rings around that path in green.

    Input structure:
    - The grid contains black corridor cells, blue wall cells, and exactly two red endpoint pixels.
    - The red endpoints lie inside the corridor space, and blue walls block movement and ring expansion.
    - Between the two endpoints there is one unique shortest 4-neighbor route through corridor cells.

    Rule:
    - Find the unique shortest 4-neighbor path between the two red endpoints while treating blue cells as impassable.
    - Recolor the black interior cells of that path yellow, keeping the red endpoints and blue walls unchanged.
    - Starting from the full path plus both endpoints, run an 8-neighbor BFS that cannot cross blue cells.
    - For every original black cell reached at positive even distance, recolor it green; keep odd-distance black cells unchanged.

    Key insight:
    - The blue maze walls simultaneously define the unique route and clip the square-ring expansion, so the output is an interference pattern of even Chebyshev layers around the solved path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.RED}, "unexpected input colors"

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.RED)]
    assert len(red_coords) == 2, "expected exactly two red endpoints"
    start, goal = red_coords

    path_interior = _shortest_path_interior(g, start, goal)
    seeds = [start, *path_interior, goal]
    ring_dist = _distance_from_path(g, seeds)

    out = g.copy()
    for r, c in path_interior:
        out[r, c] = Color.YELLOW
    even_ring = (g == Color.BLACK) & (ring_dist > 0) & (ring_dist % 2 == 0)
    out[even_ring] = Color.GREEN
    return out


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            turns += 1
    return turns


def _can_attach(cell: tuple[int, int], allowed_neighbors: set[tuple[int, int]], open_cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = cell
    if not in_bounds(shape, r, c) or cell in open_cells:
        return False
    if r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1:
        return False
    neighbors = {(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in open_cells}
    return neighbors <= allowed_neighbors


def _path_candidate_dirs(
    current: tuple[int, int],
    previous: tuple[int, int] | None,
    visited: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    dirs = DIRS4[:]
    random.shuffle(dirs)
    candidates: list[tuple[int, int]] = []
    for dr, dc in dirs:
        nr, nc = current[0] + dr, current[1] + dc
        nxt = (nr, nc)
        if not in_bounds(shape, nr, nc):
            continue
        if nr < 2 or nc < 2 or nr > shape[0] - 3 or nc > shape[1] - 3:
            continue
        if nxt in visited:
            continue
        allowed = {current}
        if previous is not None:
            allowed.add(previous)
        if not _can_attach(nxt, {current}, visited, shape):
            continue
        candidates.append(nxt)
    return candidates


def _build_main_path(shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    target_len = random.randint(*PATH_LEN_RANGE)
    start = (random.randint(2, h - 3), random.randint(2, w - 3))
    path = [start]
    visited = {start}

    def dfs() -> bool:
        if len(path) == target_len:
            return (
                abs(path[0][0] - path[-1][0]) + abs(path[0][1] - path[-1][1]) >= MIN_MANHATTAN_SEPARATION
                and _turn_count(path) >= MIN_TURNS
            )

        current = path[-1]
        previous = path[-2] if len(path) >= 2 else None
        candidates = _path_candidate_dirs(current, previous, visited, shape)
        random.shuffle(candidates)
        for nxt in candidates:
            path.append(nxt)
            visited.add(nxt)
            if dfs():
                return True
            visited.remove(nxt)
            path.pop()
        return False

    if dfs():
        return path
    return None


def _branch_dirs_for_cell(path: list[tuple[int, int]], index: int) -> list[tuple[int, int]]:
    r, c = path[index]
    blocked = set()
    if index > 0:
        pr, pc = path[index - 1]
        blocked.add((pr - r, pc - c))
    if index + 1 < len(path):
        nr, nc = path[index + 1]
        blocked.add((nr - r, nc - c))
    dirs = [direction for direction in DIRS4 if direction not in blocked]
    random.shuffle(dirs)
    return dirs


def _add_branches(path: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[set[tuple[int, int]], int]:
    open_cells = set(path)
    second_ring_cells = 0
    branch_budget = max(4, len(path) // 2)
    branch_sites = list(range(1, len(path) - 1))
    random.shuffle(branch_sites)

    for index in branch_sites:
        if branch_budget <= 0:
            break
        base = path[index]
        for dr, dc in _branch_dirs_for_cell(path, index):
            if branch_budget <= 0:
                break
            if random.random() < 0.35:
                continue
            first = (base[0] + dr, base[1] + dc)
            if not _can_attach(first, {base}, open_cells, shape):
                continue
            branch = [first]
            second = (first[0] + dr, first[1] + dc)
            if random.random() < 0.7 and _can_attach(second, {first}, open_cells | {first}, shape):
                branch.append(second)
            for cell in branch:
                open_cells.add(cell)
            if len(branch) == 2:
                second_ring_cells += 1
            branch_budget -= 1
            if random.random() < 0.55:
                break

    return open_cells, second_ring_cells


def _candidate_grid() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    path = _build_main_path((h, w))
    if path is None:
        return None

    open_cells, second_ring_cells = _add_branches(path, (h, w))
    if second_ring_cells < MIN_SECOND_RING_CELLS:
        return None

    grid = np.full((h, w), Color.BLUE, dtype=int)
    for r, c in open_cells:
        grid[r, c] = Color.BLACK

    start, goal = path[0], path[-1]
    grid[start] = Color.RED
    grid[goal] = Color.RED

    interior = path[1:-1]
    if len(interior) < 6:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.sum(out == Color.YELLOW)) != len(interior):
        return None
    if int(np.sum(out == Color.GREEN)) < 4:
        return None
    if not np.any((grid == Color.BLACK) & (out == Color.GREEN)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _candidate_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
