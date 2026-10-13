from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
START_COLOR = 2
GOAL_COLOR = 3
PATH_COLOR = 4
WALL_COLOR = 5

BLOCK_SIZE = 3
NAV_SHAPE = (5, 5)

OPEN = 0
WALL = 1
START = 2
GOAL = 3

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BRANCHES = ("non_detour", "detour", "axis_detour")
GENERATION_RETRIES = 120
WALL_POOL_SIZE = 12

_generation_index = random.randrange(len(BRANCHES))


def _classify_block(block: np.ndarray) -> int:
    if np.all(block == START_COLOR):
        return START
    if np.all(block == GOAL_COLOR):
        return GOAL

    colors = set(map(int, np.unique(block)))
    if WALL_COLOR in colors:
        assert colors <= {BACKGROUND, WALL_COLOR}, "wall blocks may contain only black and grey"
        return WALL

    assert colors == {BACKGROUND}, "channel blocks must be fully black"
    return OPEN


def _compress_navigation(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % BLOCK_SIZE == 0 and width % BLOCK_SIZE == 0, "grid must split into 3x3 blocks"

    nav_rows = height // BLOCK_SIZE
    nav_cols = width // BLOCK_SIZE
    nav = np.empty((nav_rows, nav_cols), dtype=int)
    start = None
    goal = None

    for br in range(nav_rows):
        for bc in range(nav_cols):
            block = g[
                br * BLOCK_SIZE : (br + 1) * BLOCK_SIZE,
                bc * BLOCK_SIZE : (bc + 1) * BLOCK_SIZE,
            ]
            kind = _classify_block(block)
            nav[br, bc] = kind
            if kind == START:
                assert start is None, "expected exactly one start block"
                start = (br, bc)
            elif kind == GOAL:
                assert goal is None, "expected exactly one goal block"
                goal = (br, bc)

    assert start is not None and goal is not None, "expected one start and one goal"
    return nav, start, goal


def _in_bounds(nav: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < nav.shape[0] and 0 <= c < nav.shape[1]


def _bfs_with_parent(
    nav: np.ndarray, start: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int] | None]]]:
    rows, cols = map(int, nav.shape)
    dist = np.full((rows, cols), -1, dtype=int)
    ways = np.zeros((rows, cols), dtype=np.int64)
    parent: list[list[tuple[int, int] | None]] = [[None for _ in range(cols)] for _ in range(rows)]

    sr, sc = start
    dist[sr, sc] = 0
    ways[sr, sc] = 1
    q = deque([start])

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if not _in_bounds(nav, rr, cc) or int(nav[rr, cc]) == WALL:
                continue

            nd = int(dist[r, c] + 1)
            if int(dist[rr, cc]) == -1:
                dist[rr, cc] = nd
                ways[rr, cc] = ways[r, c]
                parent[rr][cc] = (r, c)
                q.append((rr, cc))
            elif int(dist[rr, cc]) == nd:
                ways[rr, cc] += ways[r, c]

    return dist, ways, parent


def _reconstruct_unique_shortest_path(
    nav: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    dist, ways, parent = _bfs_with_parent(nav, start)
    gr, gc = goal
    assert int(dist[gr, gc]) >= 0, "start must reach goal"
    assert int(ways[gr, gc]) == 1, "expected a unique shortest path"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur[0]][cur[1]]
    path.reverse()

    assert path[0] == start and path[-1] == goal
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a 15x15 partitioned grid into a 5x5 navigation map, find the unique shortest route, and paint it back blockwise.

    Input structure:
    - The input is a black 15x15 grid interpreted as a 5x5 layout of 3x3 blocks.
    - A channel block is fully black, a wall block contains a grey marker pixel on black, the start block is a solid red 3x3 square, and the goal block is a solid green 3x3 square.
    - Exactly one start block and one goal block are present, and the intended route is defined on 4-neighbor block adjacency.

    Rule:
    - Compress the grid by classifying each 3x3 block as channel, wall, start, or goal.
    - On the resulting 5x5 navigation grid, find the unique shortest 4-neighbor path from the start block to the goal block while never entering wall blocks.
    - Map every interior path cell back to its original 3x3 block and fill that whole block yellow.
    - Keep the red start block, green goal block, grey wall markers, and all off-path channel blocks unchanged.

    Key insight:
    - The visible 15x15 pixels are only a scaled encoding of a hidden 5x5 maze, so the solver must reason in the compressed block graph and then project the abstract path back onto the original blocks.
    """
    g = np.asarray(grid, dtype=int)
    nav, start, goal = _compress_navigation(g)
    path = _reconstruct_unique_shortest_path(nav, start, goal)

    out = g.copy()
    for br, bc in path[1:-1]:
        r0 = br * BLOCK_SIZE
        c0 = bc * BLOCK_SIZE
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = PATH_COLOR
    return out


def _bfs_distances_and_counts(nav: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    dist, ways, _parent = _bfs_with_parent(nav, start)
    return dist, ways


def _path_cells_via_distance_counts(
    nav: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> set[tuple[int, int]]:
    dist_s, ways_s = _bfs_distances_and_counts(nav, start)
    dist_g, ways_g = _bfs_distances_and_counts(nav, goal)
    total_dist = int(dist_s[goal])
    total_ways = int(ways_s[goal])
    assert total_dist >= 0 and total_ways == 1, "generated cases must have one shortest path"

    cells: set[tuple[int, int]] = set()
    for r in range(nav.shape[0]):
        for c in range(nav.shape[1]):
            if int(nav[r, c]) == WALL:
                continue
            if int(dist_s[r, c]) < 0 or int(dist_g[r, c]) < 0:
                continue
            if int(dist_s[r, c]) + int(dist_g[r, c]) != total_dist:
                continue
            if int(ways_s[r, c]) * int(ways_g[r, c]) != total_ways:
                continue
            cells.add((r, c))

    assert start in cells and goal in cells
    return cells


def _expected_output_via_counts(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    nav, start, goal = _compress_navigation(g)
    path_cells = _path_cells_via_distance_counts(nav, start, goal)

    out = g.copy()
    for br, bc in path_cells:
        if (br, bc) in (start, goal):
            continue
        r0 = br * BLOCK_SIZE
        c0 = bc * BLOCK_SIZE
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = PATH_COLOR
    return out


def _navigation_from_case(
    start: tuple[int, int], goal: tuple[int, int], walls: set[tuple[int, int]]
) -> np.ndarray:
    nav = np.full(NAV_SHAPE, OPEN, dtype=int)
    for wr, wc in walls:
        nav[wr, wc] = WALL
    nav[start] = START
    nav[goal] = GOAL
    return nav


def _path_signature(
    nav: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> tuple[int, int, tuple[tuple[int, int], ...] | None]:
    dist, ways, parent = _bfs_with_parent(nav, start)
    gr, gc = goal
    goal_dist = int(dist[gr, gc])
    goal_ways = int(ways[gr, gc])
    if goal_dist < 0 or goal_ways != 1:
        return goal_dist, goal_ways, None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur[0]][cur[1]]
    path.reverse()
    return goal_dist, goal_ways, tuple(path)


def _turn_count(path: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> int:
    turns = 0
    for idx in range(1, len(path) - 1):
        prev_step = (
            int(path[idx][0] - path[idx - 1][0]),
            int(path[idx][1] - path[idx - 1][1]),
        )
        next_step = (
            int(path[idx + 1][0] - path[idx][0]),
            int(path[idx + 1][1] - path[idx][1]),
        )
        if prev_step != next_step:
            turns += 1
    return turns


def _all_walls_are_active(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: set[tuple[int, int]],
    base_path: tuple[tuple[int, int], ...],
) -> bool:
    base_sig = (len(base_path) - 1, 1, base_path)
    for wall in walls:
        reduced = set(walls)
        reduced.remove(wall)
        reduced_nav = _navigation_from_case(start, goal, reduced)
        if _path_signature(reduced_nav, start, goal) == base_sig:
            return False
    return True


def _branch_conditions_met(
    branch: str,
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: set[tuple[int, int]],
) -> bool:
    nav = _navigation_from_case(start, goal, walls)
    dist, ways, path = _path_signature(nav, start, goal)
    if ways != 1 or path is None or dist < 4:
        return False

    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    turns = _turn_count(path)
    open_off_path = NAV_SHAPE[0] * NAV_SHAPE[1] - len(walls) - len(path)
    if open_off_path < 2:
        return False
    if not _all_walls_are_active(start, goal, walls, path):
        return False

    if branch == "non_detour":
        return start[0] != goal[0] and start[1] != goal[1] and dist == manhattan and turns >= 1
    if branch == "detour":
        return start[0] != goal[0] and start[1] != goal[1] and dist > manhattan and turns >= 2
    if branch == "axis_detour":
        return (start[0] == goal[0] or start[1] == goal[1]) and dist > manhattan and turns >= 2
    raise AssertionError(f"unknown branch: {branch}")


def _candidate_wall_pool(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    gr, gc = goal
    primary = [
        cell
        for cell in itertools.product(range(NAV_SHAPE[0]), range(NAV_SHAPE[1]))
        if cell not in (start, goal)
        and min(sr, gr) - 1 <= cell[0] <= max(sr, gr) + 1
        and min(sc, gc) - 1 <= cell[1] <= max(sc, gc) + 1
    ]
    random.shuffle(primary)
    if len(primary) > WALL_POOL_SIZE:
        primary = primary[:WALL_POOL_SIZE]

    if len(primary) < WALL_POOL_SIZE:
        extras = [
            cell
            for cell in itertools.product(range(NAV_SHAPE[0]), range(NAV_SHAPE[1]))
            if cell not in (start, goal) and cell not in primary
        ]
        random.shuffle(extras)
        extras.sort(
            key=lambda cell: abs(cell[0] - sr)
            + abs(cell[1] - sc)
            + abs(cell[0] - gr)
            + abs(cell[1] - gc)
        )
        primary.extend(extras[: WALL_POOL_SIZE - len(primary)])

    return primary


def _sample_start_goal(branch: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if branch == "axis_detour":
        if random.random() < 0.5:
            row = random.randrange(NAV_SHAPE[0])
            c0, c1 = sorted(random.sample(range(NAV_SHAPE[1]), 2))
            if c1 - c0 < 2:
                return None
            return (row, c0), (row, c1)

        col = random.randrange(NAV_SHAPE[1])
        r0, r1 = sorted(random.sample(range(NAV_SHAPE[0]), 2))
        if r1 - r0 < 2:
            return None
        return (r0, col), (r1, col)

    cells = list(itertools.product(range(NAV_SHAPE[0]), range(NAV_SHAPE[1])))
    start, goal = random.sample(cells, 2)
    if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 3:
        return None
    if branch in {"non_detour", "detour"} and (start[0] == goal[0] or start[1] == goal[1]):
        return None
    return start, goal


def _wall_count_order(branch: str) -> tuple[int, ...]:
    if branch == "non_detour":
        first = random.choice((3, 4))
        remaining = [count for count in (3, 4, 5) if count != first]
        random.shuffle(remaining)
        return (first, *remaining)
    counts = [3, 4, 5]
    random.shuffle(counts)
    return tuple(counts)


def _search_case(branch: str) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    for _ in range(GENERATION_RETRIES):
        sampled = _sample_start_goal(branch)
        if sampled is None:
            continue
        start, goal = sampled
        pool = _candidate_wall_pool(start, goal)

        for wall_count in _wall_count_order(branch):
            combos = list(itertools.combinations(pool, wall_count))
            random.shuffle(combos)
            for combo in combos:
                walls = set(combo)
                if _branch_conditions_met(branch, start, goal, walls):
                    return start, goal, walls

    raise RuntimeError(f"failed to build a valid {branch} case")


def _render_case(
    start: tuple[int, int], goal: tuple[int, int], walls: set[tuple[int, int]]
) -> np.ndarray:
    grid = np.full((NAV_SHAPE[0] * BLOCK_SIZE, NAV_SHAPE[1] * BLOCK_SIZE), BACKGROUND, dtype=int)

    for br in range(NAV_SHAPE[0]):
        for bc in range(NAV_SHAPE[1]):
            r0 = br * BLOCK_SIZE
            c0 = bc * BLOCK_SIZE
            if (br, bc) == start:
                grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = START_COLOR
            elif (br, bc) == goal:
                grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = GOAL_COLOR
            elif (br, bc) in walls:
                rr = random.randrange(BLOCK_SIZE)
                cc = random.randrange(BLOCK_SIZE)
                grid[r0 + rr, c0 + cc] = WALL_COLOR

    return grid


def _validate_generated_case(grid: np.ndarray, branch: str) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (15, 15), "generated puzzle must be 15x15"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, START_COLOR, GOAL_COLOR, WALL_COLOR}

    nav, start, goal = _compress_navigation(g)
    walls = {
        (r, c)
        for r in range(nav.shape[0])
        for c in range(nav.shape[1])
        if int(nav[r, c]) == WALL
    }

    assert 3 <= len(walls) <= 5, "generator must place 3 to 5 wall blocks"
    assert abs(start[0] - goal[0]) + abs(start[1] - goal[1]) > 1, "start and goal must not be adjacent"

    for wr, wc in walls:
        block = g[
            wr * BLOCK_SIZE : (wr + 1) * BLOCK_SIZE,
            wc * BLOCK_SIZE : (wc + 1) * BLOCK_SIZE,
        ]
        assert int(np.count_nonzero(block == WALL_COLOR)) == 1, "wall blocks must contain one grey marker"
        assert int(np.count_nonzero(block != BACKGROUND)) == 1, "wall blocks must otherwise stay black"

    expected = _expected_output_via_counts(g)
    actual = main(g)
    assert np.array_equal(actual, expected), "independent solver must match main()"

    assert _branch_conditions_met(branch, start, goal, walls), "generated example missed its intended branch"


def generate_input() -> np.ndarray:
    global _generation_index

    branch = BRANCHES[_generation_index % len(BRANCHES)]
    _generation_index += 1

    start, goal, walls = _search_case(branch)
    grid = _render_case(start, goal, walls)
    _validate_generated_case(grid, branch)
    return grid
