from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIDE_RANGE = (17, 29)  # odd sizes only, inclusive
GEN_RETRIES = 300
_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _bfs_distances(walkable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(walkable.shape, -1, dtype=int)
    if not walkable[start]:
        return dist
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(walkable.shape, r, c):
            if not walkable[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _seed_position(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert pts.shape == (1, 2), f"expected exactly one seed of color {color}"
    return tuple(map(int, pts[0]))


def _paint_territories(grid: np.ndarray) -> np.ndarray:
    walkable = grid != GREY
    red_seed = _seed_position(grid, RED)
    blue_seed = _seed_position(grid, BLUE)
    red_dist = _bfs_distances(walkable, red_seed)
    blue_dist = _bfs_distances(walkable, blue_seed)

    out = grid.copy()
    corridor = grid == BLACK
    for r, c in np.argwhere(corridor):
        dr = int(red_dist[r, c])
        db = int(blue_dist[r, c])
        if dr == -1 and db == -1:
            continue
        if dr == db:
            out[r, c] = YELLOW
        elif db == -1 or (dr != -1 and dr < db):
            out[r, c] = RED
        else:
            out[r, c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each corridor cell by which seed reaches it first through the maze.

    Input structure:
    - Grey wall pixels stand on a black background and form a 4-connected corridor maze.
    - Inside the corridors there is exactly one red seed pixel and exactly one blue seed pixel.

    Rule:
    - For every black corridor cell, measure the 4-connected shortest-path distance to the
      red seed and to the blue seed while staying inside the corridors and seed cells.
    - Recolor a corridor cell red if it is strictly closer to the red seed, blue if it is
      strictly closer to the blue seed, and yellow if the two shortest-path distances are equal.
    - Any black corridor cell unreachable from both seeds stays black.
    - Grey walls and the red/blue seed pixels themselves stay unchanged.

    Key insight:
    - Ownership is decided by maze shortest paths, not by straight-line or raw Manhattan
      distance, so walls can force a cell that looks nearer to one seed to belong to the other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30
    colors = set(map(int, np.unique(g)))
    assert colors.issubset({BLACK, BLUE, RED, GREY}), "unexpected input colors"
    return _paint_territories(g)


def _build_tree_maze(h: int, w: int, rng: random.Random) -> np.ndarray:
    grid = np.full((h, w), GREY, dtype=int)
    cells = [(r, c) for r in range(1, h, 2) for c in range(1, w, 2)]
    start = cells[rng.randrange(len(cells))]
    stack = [start]
    visited = {start}
    grid[start] = BLACK

    while stack:
        r, c = stack[-1]
        options = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < h - 1 and 1 <= nc < w - 1 and (nr, nc) not in visited:
                options.append((nr, nc, r + dr // 2, c + dc // 2))
        if not options:
            stack.pop()
            continue
        nr, nc, wr, wc = options[rng.randrange(len(options))]
        grid[wr, wc] = BLACK
        grid[nr, nc] = BLACK
        visited.add((nr, nc))
        stack.append((nr, nc))

    return grid


def _corridor_degrees(walkable: np.ndarray) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(walkable):
        rr, cc = int(r), int(c)
        degrees[(rr, cc)] = sum(1 for nr, nc in _neighbors4(walkable.shape, rr, cc) if walkable[nr, nc])
    return degrees


def _branches_from_midpoint(walkable: np.ndarray, midpoint: tuple[int, int]) -> list[dict[int, list[tuple[int, int]]]]:
    used = {midpoint}
    branches: list[dict[int, list[tuple[int, int]]]] = []
    for nr, nc in _neighbors4(walkable.shape, *midpoint):
        if not walkable[nr, nc] or (nr, nc) in used:
            continue
        dist_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        q: deque[tuple[int, int, int]] = deque([(nr, nc, 1)])
        used.add((nr, nc))
        while q:
            r, c, d = q.popleft()
            dist_groups[d].append((r, c))
            for rr, cc in _neighbors4(walkable.shape, r, c):
                if not walkable[rr, cc] or (rr, cc) == midpoint or (rr, cc) in used:
                    continue
                used.add((rr, cc))
                q.append((rr, cc, d + 1))
        branches.append(dist_groups)
    return branches


def _pick_seed_pair(walkable: np.ndarray, rng: random.Random, *, require_spare_branch: bool):
    degrees = _corridor_degrees(walkable)
    midpoints = [cell for cell, degree in degrees.items() if degree >= 3]
    rng.shuffle(midpoints)

    for midpoint in midpoints:
        branches = _branches_from_midpoint(walkable, midpoint)
        if len(branches) < 3:
            continue
        spare_ok = False
        if require_spare_branch:
            spare_ok = any(max(branch) >= 3 for branch in branches if branch)
        candidates = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                common = sorted(set(branches[i]) & set(branches[j]))
                common = [d for d in common if d >= 3]
                if common:
                    if require_spare_branch:
                        has_spare = any(
                            k not in (i, j) and branches[k] and max(branches[k]) >= 3
                            for k in range(len(branches))
                        )
                        if not has_spare:
                            continue
                    candidates.append((common[-1], i, j))
        if not candidates or (require_spare_branch and not spare_ok):
            continue
        best_dist = max(item[0] for item in candidates)
        far_pairs = [item for item in candidates if item[0] == best_dist]
        dist_from_mid, i, j = far_pairs[rng.randrange(len(far_pairs))]
        red_seed = branches[i][dist_from_mid][rng.randrange(len(branches[i][dist_from_mid]))]
        blue_seed = branches[j][dist_from_mid][rng.randrange(len(branches[j][dist_from_mid]))]
        return red_seed, blue_seed, midpoint, branches, (i, j)
    return None


def _seal_unreachable_branch(
    grid: np.ndarray,
    midpoint: tuple[int, int],
    branches: list[dict[int, list[tuple[int, int]]]],
    used_pair: tuple[int, int],
    rng: random.Random,
) -> bool:
    spare = [
        (max(branch), idx)
        for idx, branch in enumerate(branches)
        if idx not in used_pair and branch and max(branch) >= 3
    ]
    if not spare:
        return False
    best_depth = max(depth for depth, _idx in spare)
    candidates = [idx for depth, idx in spare if depth == best_depth]
    branch_idx = candidates[rng.randrange(len(candidates))]
    plug = branches[branch_idx][1][rng.randrange(len(branches[branch_idx][1]))]
    grid[plug] = GREY
    return True


def _has_shortest_path_counterexample(grid: np.ndarray) -> bool:
    red_seed = _seed_position(grid, RED)
    blue_seed = _seed_position(grid, BLUE)
    walkable = grid != GREY
    red_dist = _bfs_distances(walkable, red_seed)
    blue_dist = _bfs_distances(walkable, blue_seed)

    for r, c in np.argwhere(grid == BLACK):
        rr, cc = int(r), int(c)
        dr = int(red_dist[rr, cc])
        db = int(blue_dist[rr, cc])
        if dr == -1 or db == -1 or dr == db:
            continue
        man_r = abs(rr - red_seed[0]) + abs(cc - red_seed[1])
        man_b = abs(rr - blue_seed[0]) + abs(cc - blue_seed[1])
        if man_r == man_b:
            continue
        if (man_r < man_b) != (dr < db):
            return True
    return False


def _validate_generated_case(grid: np.ndarray, *, require_unreachable: bool) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED, GREY})

    out = main(grid)
    walkable = grid != GREY
    red_seed = _seed_position(grid, RED)
    blue_seed = _seed_position(grid, BLUE)
    red_dist = _bfs_distances(walkable, red_seed)
    blue_dist = _bfs_distances(walkable, blue_seed)

    corridor = grid == BLACK
    assert np.any((out == RED) & corridor), "expected red-owned corridor cells"
    assert np.any((out == BLUE) & corridor), "expected blue-owned corridor cells"
    assert np.any((out == YELLOW) & corridor), "expected at least one equidistant corridor cell"

    unreachable = corridor & (red_dist == -1) & (blue_dist == -1)
    if require_unreachable:
        assert np.any(unreachable), "expected a disconnected corridor branch"
    assert np.all(out[unreachable] == BLACK), "unreachable corridors must stay black"

    reachable = corridor & ~unreachable
    assert np.any(reachable), "expected reachable corridors"

    expected = _paint_territories(grid)
    assert np.array_equal(out, expected), "output must match shortest-path coloring"
    assert _has_shortest_path_counterexample(grid), "maze must defeat direct-distance shortcuts"


def _generate_attempt(rng: random.Random, mode: int) -> np.ndarray | None:
    side_lo, side_hi = GRID_SIDE_RANGE
    h = rng.randrange(side_lo, side_hi + 1, 2)
    w = rng.randrange(side_lo, side_hi + 1, 2)

    maze = _build_tree_maze(h, w, rng)
    walkable = maze == BLACK
    require_unreachable = mode != 0
    picked = _pick_seed_pair(walkable, rng, require_spare_branch=require_unreachable)
    if picked is None:
        return None

    red_seed, blue_seed, midpoint, branches, used_pair = picked
    grid = maze.copy()
    if require_unreachable and not _seal_unreachable_branch(grid, midpoint, branches, used_pair, rng):
        return None

    grid[red_seed] = RED
    grid[blue_seed] = BLUE

    try:
        _validate_generated_case(grid, require_unreachable=require_unreachable)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    mode = _GEN_CALL_COUNT % 3

    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze instance")
