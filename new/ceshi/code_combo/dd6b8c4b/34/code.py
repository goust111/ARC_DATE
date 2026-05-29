# Combo variant of dd6b8c4b (index 34)
# combo_spec: D9/2+D16/0
# Name: 迷宫式墙壁+加权距离背景

from __future__ import annotations

import heapq
import random
import sys
from collections import deque

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import neighbors4


# Design analysis for the fused variant:
# - Concrete behavior shift:
#   Original dd6b8c4b only needed ordinary BFS through one floor color, so a
#   token on a 6-step route always beat one on an 8-step route. Here the maze
#   still gates reachability, but there are two corridor colors: orange costs 1
#   per entered cell and grey costs 2. Example: a token behind a 6-step
#   orange+grey route with total cost 10 can lose to a farther token on an
#   8-step all-orange route with total cost 8. The solver must therefore rank
#   by weighted path cost instead of raw step count.
# - Complexity calibration:
#   The layout uses only two passable background colors and keeps costs binary
#   (1 vs 2). That is enough to visibly change the ranking without introducing
#   opaque arithmetic. The maze is a single-cell corridor system with dead ends,
#   so the wall change is obvious by eye and does not collapse into a loose
#   frame. To make weighting observable, the generator forces more than 9
#   reachable maroons, so the destination can only accept a subset.
# - Hidden assumptions broken by the fusion:
#   The original code assumed one floor color, unweighted BFS, and "erase moved
#   maroon back to inferred floor". All three are unsafe now. This solver uses a
#   fixed palette for the fused task, replaces BFS with Dijkstra, and the
#   generator guarantees every maroon sits on an orange endpoint so erasing a
#   moved token back to orange is always correct.
# - Placement legality and observability:
#   The generator starts from a wall field, carves a non-overlapping corridor
#   maze, keeps exactly one opening adjacent to the 3x3 target block, and places
#   an isolated unreachable distractor in a sealed pocket. The weighted and
#   unweighted top-9 selections are validated to differ, preventing accidental
#   regression back to the original plain-distance behavior.


GRID_H = GRID_W = 11
CENTER_R = CENTER_C = 5

CHEAP_FLOOR = 7   # orange
HEAVY_FLOOR = 5   # grey
WALL_COLOR = 6    # magenta
DEST_COLOR = 3    # green
RED_CENTER = 2
MAROON = 9

PASSABLE_COST = {
    CHEAP_FLOOR: 1,
    HEAVY_FLOOR: 2,
    MAROON: 1,  # generator places maroons on orange corridor endpoints
}

CANONICAL_ORANGE = {
    (3, 5), (2, 5), (1, 5),
    (1, 4), (1, 3), (2, 3), (2, 2), (2, 1),
    (3, 3), (3, 2), (3, 1), (4, 1), (5, 1), (6, 1), (6, 2), (7, 1), (8, 1),
    (9, 1), (9, 2), (9, 3), (8, 3),
    (1, 9), (2, 9), (4, 9), (5, 9), (6, 9), (6, 8), (7, 9), (8, 9),
    (9, 9), (9, 8), (9, 7), (8, 7),
    (8, 5),  # isolated pocket for an unreachable distractor
}

CANONICAL_GREY = {
    (1, 6), (1, 7), (1, 8),
    (2, 7), (2, 8),
    (3, 7), (3, 8), (3, 9),
}

REACHABLE_TOKEN_CELLS = [
    (2, 1), (4, 1), (6, 2), (8, 1), (8, 3),
    (1, 9), (2, 9), (4, 9), (6, 8), (8, 9),
]

UNREACHABLE_TOKEN_CELLS = [(8, 5)]
GEN_RETRIES = 32


def _find_center(grid: np.ndarray) -> tuple[int, int]:
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    return int(centers[0, 0]), int(centers[0, 1])


def _dest_cells(center: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    h, w = shape
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _start_cells(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    h, w = grid.shape
    for r, c in dest_set:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set:
                if int(grid[rr, cc]) in PASSABLE_COST and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    starts.append((rr, cc))
    assert starts, "there must be at least one passable opening next to the target block"
    return starts


def _weighted_distances(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), 10**9, dtype=int)
    heap: list[tuple[int, int, int]] = []
    for r, c in _start_cells(grid, dest_set):
        dist[r, c] = 0
        heapq.heappush(heap, (0, r, c))

    while heap:
        d, r, c = heapq.heappop(heap)
        if d != int(dist[r, c]):
            continue
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                color = int(grid[rr, cc])
                if color not in PASSABLE_COST:
                    continue
                nd = d + int(PASSABLE_COST[color])
                if nd < int(dist[rr, cc]):
                    dist[rr, cc] = nd
                    heapq.heappush(heap, (nd, rr, cc))
    return dist


def _unweighted_distances(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), 10**9, dtype=int)
    q = deque()
    for r, c in _start_cells(grid, dest_set):
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in PASSABLE_COST:
                if int(dist[rr, cc]) == 10**9:
                    dist[rr, cc] = int(dist[r, c]) + 1
                    q.append((rr, cc))
    return dist


def _rank_reachable_maroons(grid: np.ndarray, *, weighted: bool) -> list[tuple[int, int, int]]:
    center = _find_center(grid)
    dest_cells = _dest_cells(center, grid.shape)
    dest_set = set(dest_cells)
    dist = _weighted_distances(grid, dest_set) if weighted else _unweighted_distances(grid, dest_set)
    ranked = [
        (int(dist[r, c]), int(r), int(c))
        for r, c in np.argwhere(grid == MAROON)
        if int(dist[r, c]) < 10**9
    ]
    ranked.sort()
    return ranked


def _solve_variant(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    center = _find_center(g)
    dest_cells = _dest_cells(center, g.shape)
    ranked = _rank_reachable_maroons(g, weighted=True)
    k = min(len(dest_cells), len(ranked))
    chosen = {(r, c) for _, r, c in ranked[:k]}

    for r, c in chosen:
        out[r, c] = CHEAP_FLOOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the lowest-cost reachable maroon pixels into the 3x3 block around the unique red center.

    Input structure:
    - A green 3x3 destination block is centered on a unique red pixel.
    - Magenta walls carve the board into a narrow maze.
    - Orange and grey corridor cells are both traversable; scattered maroon pixels sit on corridor endpoints.

    Rule:
    - Start from corridor cells adjacent to the destination block and search through orange, grey, and maroon cells.
    - Entering orange or maroon costs 1, while entering grey costs 2.
    - Rank reachable maroons by total path cost, breaking ties by row then column.
    - Move the best-ranked maroons into the destination in row-major order and erase their source cells back to orange.

    Key insight: The maze still controls reachability, but shortest step count is no longer enough:
    a longer orange route can outrank a shorter route that passes through too many grey cells.
    """

    return _solve_variant(grid)


def _base_maze() -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), WALL_COLOR, dtype=int)
    for r in range(CENTER_R - 1, CENTER_R + 2):
        for c in range(CENTER_C - 1, CENTER_C + 2):
            grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    for r, c in CANONICAL_ORANGE:
        grid[r, c] = CHEAP_FLOOR
    for r, c in CANONICAL_GREY:
        grid[r, c] = HEAVY_FLOOR
    return grid


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    weighted = _rank_reachable_maroons(grid, weighted=True)
    plain = _rank_reachable_maroons(grid, weighted=False)

    weighted_keep = {(r, c) for _, r, c in weighted[9:]}
    plain_keep = {(r, c) for _, r, c in plain[9:]}
    return {
        "reachable_count": len(weighted),
        "weighted_keep": weighted_keep,
        "plain_keep": plain_keep,
        "keep_diff": len(weighted_keep ^ plain_keep),
        "cheap_count": int((grid == CHEAP_FLOOR).sum()),
        "heavy_count": int((grid == HEAVY_FLOOR).sum()),
    }


def _build_candidate() -> np.ndarray:
    grid = _base_maze()
    for r, c in REACHABLE_TOKEN_CELLS:
        grid[r, c] = MAROON
    for r, c in UNREACHABLE_TOKEN_CELLS:
        grid[r, c] = MAROON
    grid = _random_symmetry(grid)

    metrics = _fusion_metrics(grid)
    assert metrics["reachable_count"] == len(REACHABLE_TOKEN_CELLS)
    assert metrics["reachable_count"] > 9
    assert metrics["heavy_count"] > 0 and metrics["cheap_count"] > 0
    assert metrics["keep_diff"] > 0, "weighted and unweighted rankings must disagree"

    out = _solve_variant(grid)
    assert int((out == MAROON).sum()) == len(REACHABLE_TOKEN_CELLS) + len(UNREACHABLE_TOKEN_CELLS)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _build_candidate()
        except AssertionError:
            continue
    raise RuntimeError(f"failed to generate a valid maze-weighted sample after {GEN_RETRIES} attempts")
