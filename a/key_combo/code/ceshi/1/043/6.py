from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

FLOOR_COLOR = 0
WALL_COLOR = 8
SOURCE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

MAZE_SIZES = (9, 11, 13, 15)
MIN_SOURCE_DISTANCE = 8
MIN_ASSIGNED_CELLS = 8
MIN_GEODESIC_DIFFERENCES = 3
MIN_TIES = 1
MAX_TIES = 12
GEN_ATTEMPTS = 500
_GEN_COUNTER = 0


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield int(r) + dr, int(c) + dc


def bfs_distances(passable: np.ndarray, start: Coord) -> np.ndarray:
    mask = np.asarray(passable, dtype=bool)
    assert bool(mask[start]), "the source must be part of the walkable floor network"
    dist = np.full(mask.shape, -1, dtype=int)
    dist[start] = 0
    q: deque[Coord] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(mask.shape, nr, nc):
                continue
            if not bool(mask[nr, nc]) or int(dist[nr, nc]) >= 0:
                continue
            dist[nr, nc] = int(dist[r, c]) + 1
            q.append((nr, nc))
    return dist


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert bool(np.any(g == FLOOR_COLOR)), "expected black floor cells"
    assert bool(np.any(g == WALL_COLOR)), "expected teal wall cells"

    source_colors = sorted(
        int(color)
        for color in np.unique(g)
        if int(color) not in {FLOOR_COLOR, WALL_COLOR}
    )
    assert len(source_colors) == 2, "expected exactly two source colors"

    sources = []
    for color in source_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), "each source color must mark exactly one cell"
        sources.append(
            {
                "color": int(color),
                "coord": (int(coords[0, 0]), int(coords[0, 1])),
            }
        )

    floor_mask = g == FLOOR_COLOR
    walkable = floor_mask.copy()
    for source in sources:
        walkable[source["coord"]] = True

    distances = [bfs_distances(walkable, source["coord"]) for source in sources]
    assert bool(np.all((distances[0] >= 0) | (distances[1] >= 0) | ~floor_mask)), (
        "every floor cell must be reachable from at least one source"
    )
    return {
        "grid": g,
        "floor_mask": floor_mask,
        "walkable": walkable,
        "sources": sources,
    }


def _distance_fields(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        bfs_distances(scene["walkable"], source["coord"])
        for source in scene["sources"]
    )


def _expected_output(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    floor = scene["floor_mask"]
    dist0, dist1 = _distance_fields(scene)

    nearer0 = floor & (dist0 >= 0) & ((dist1 < 0) | (dist0 < dist1))
    nearer1 = floor & (dist1 >= 0) & ((dist0 < 0) | (dist1 < dist0))
    out[nearer0] = int(scene["sources"][0]["color"])
    out[nearer1] = int(scene["sources"][1]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition black floor cells by true around-the-wall distance to the nearer colored source, leaving ties black.

    Input structure:
    - Black cells form a maze floor and teal cells form impassable walls.
    - Two distinct non-black, non-teal single-pixel sources replace ordinary floor cells.
    - A source may lie at a corridor end, inside a narrow corridor, or at a junction; source cells remain walkable.

    Rule:
    - Compute 4-neighbor shortest-path distance from each source through black floor and both source cells.
    - Recolor every black floor cell with the color of the strictly nearer source.
    - Keep equally distant floor cells black, and preserve teal walls and both source markers.

    Key insight:
    - Ownership uses geodesic distance through the maze rather than straight-line distance, and source markers do not block paths through their floor positions.
    """
    return _expected_output(_parse_scene(grid))


def _walkable_neighbors(mask: np.ndarray, cell: Coord) -> list[Coord]:
    return [
        (nr, nc)
        for nr, nc in neighbors4(*cell)
        if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc])
    ]


def _carve_loop_maze(rows: int, cols: int) -> np.ndarray:
    maze = np.zeros((rows, cols), dtype=bool)
    coarse_h = (rows - 1) // 2
    coarse_w = (cols - 1) // 2
    nodes = [(r, c) for r in range(coarse_h) for c in range(coarse_w)]
    node_coord = lambda node: (1 + 2 * node[0], 1 + 2 * node[1])

    root = random.choice(nodes)
    visited = {root}
    stack = [root]
    maze[node_coord(root)] = True
    tree_edges: set[frozenset[Coord]] = set()

    while stack:
        cur = stack[-1]
        choices = []
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < coarse_h and 0 <= nxt[1] < coarse_w and nxt not in visited:
                choices.append(nxt)
        if not choices:
            stack.pop()
            continue
        nxt = random.choice(choices)
        visited.add(nxt)
        stack.append(nxt)
        tree_edges.add(frozenset((cur, nxt)))
        a = node_coord(cur)
        b = node_coord(nxt)
        maze[a] = True
        maze[b] = True
        maze[(a[0] + b[0]) // 2, (a[1] + b[1]) // 2] = True

    openings = []
    for node in nodes:
        a = node_coord(node)
        for dr, dc in ((1, 0), (0, 1)):
            nxt = (node[0] + dr, node[1] + dc)
            if nxt[0] >= coarse_h or nxt[1] >= coarse_w:
                continue
            edge = frozenset((node, nxt))
            if edge in tree_edges:
                continue
            b = node_coord(nxt)
            openings.append(((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))

    random.shuffle(openings)
    loop_count = random.randint(2, min(7, len(openings)))
    for cell in openings[:loop_count]:
        maze[cell] = True
    return maze
def _one_shortest_path(mask: np.ndarray, start: Coord, goal: Coord) -> list[Coord] | None:
    q: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nxt in _walkable_neighbors(mask, cur):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    return None


def _has_alternate_route(
    mask: np.ndarray,
    start: Coord,
    goal: Coord,
    shortest_path: list[Coord],
) -> bool:
    for a, b in zip(shortest_path, shortest_path[1:]):
        banned = frozenset((a, b))
        q: deque[Coord] = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            if cur == goal:
                return True
            for nxt in _walkable_neighbors(mask, cur):
                if nxt in seen or frozenset((cur, nxt)) == banned:
                    continue
                seen.add(nxt)
                q.append(nxt)
    return False


def _ownership_signature(
    floor: np.ndarray,
    dist0: np.ndarray,
    dist1: np.ndarray,
) -> np.ndarray:
    signature = np.full(floor.shape, -1, dtype=int)
    signature[floor & (dist0 == dist1)] = 2
    signature[floor & (dist0 >= 0) & ((dist1 < 0) | (dist0 < dist1))] = 0
    signature[floor & (dist1 >= 0) & ((dist0 < 0) | (dist1 < dist0))] = 1
    return signature


def _pair_is_semantic(
    floor: np.ndarray,
    source_a: Coord,
    source_b: Coord,
    expect_ties: bool,
) -> bool:
    dist0 = bfs_distances(floor, source_a)
    if int(dist0[source_b]) < MIN_SOURCE_DISTANCE:
        return False
    dist1 = bfs_distances(floor, source_b)

    path = _one_shortest_path(floor, source_a, source_b)
    if path is None or not _has_alternate_route(floor, source_a, source_b, path):
        return False

    tie_mask = floor & (dist0 == dist1)
    tie_count = int(np.count_nonzero(tie_mask))
    if expect_ties:
        if not (MIN_TIES <= tie_count <= MAX_TIES):
            return False
    elif tie_count != 0:
        return False

    assigned0 = int(np.count_nonzero(floor & (dist0 < dist1)))
    assigned1 = int(np.count_nonzero(floor & (dist1 < dist0)))
    if min(assigned0, assigned1) < MIN_ASSIGNED_CELLS:
        return False

    rr, cc = np.indices(floor.shape)
    md0 = np.abs(rr - source_a[0]) + np.abs(cc - source_a[1])
    md1 = np.abs(rr - source_b[0]) + np.abs(cc - source_b[1])
    geodesic = _ownership_signature(floor, dist0, dist1)
    manhattan = np.full(floor.shape, -1, dtype=int)
    manhattan[floor & (md0 == md1)] = 2
    manhattan[floor & (md0 < md1)] = 0
    manhattan[floor & (md1 < md0)] = 1
    if int(np.count_nonzero(floor & (geodesic != manhattan))) < MIN_GEODESIC_DIFFERENCES:
        return False

    degrees = (
        len(_walkable_neighbors(floor, source_a)),
        len(_walkable_neighbors(floor, source_b)),
    )
    if min(degrees) < 1 or max(degrees) > 3:
        return False
    return True


def _select_sources(
    floor: np.ndarray,
    expect_ties: bool,
) -> tuple[Coord, Coord] | None:
    cells = [tuple(map(int, coord)) for coord in np.argwhere(floor)]
    random.shuffle(cells)
    for _ in range(600):
        source_a, source_b = random.sample(cells, 2)
        if _pair_is_semantic(floor, source_a, source_b, expect_ties):
            return source_a, source_b
    return None


def _semantic_validation(grid: np.ndarray, expect_ties: bool) -> bool:
    try:
        scene = _parse_scene(grid)
    except AssertionError:
        return False

    dist0, dist1 = _distance_fields(scene)
    tie_count = int(np.count_nonzero(scene["floor_mask"] & (dist0 == dist1)))
    if expect_ties:
        if not (MIN_TIES <= tie_count <= MAX_TIES):
            return False
    elif tie_count != 0:
        return False

    if not _pair_is_semantic(
        scene["walkable"],
        scene["sources"][0]["coord"],
        scene["sources"][1]["coord"],
        expect_ties,
    ):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.all(out[grid == WALL_COLOR] == WALL_COLOR):
        return False
    for source in scene["sources"]:
        if int(out[source["coord"]]) != int(source["color"]):
            return False
    return True


def _gen_attempt(expect_ties: bool) -> np.ndarray | None:
    rows = random.choice(MAZE_SIZES)
    cols = random.choice(MAZE_SIZES)
    floor = _carve_loop_maze(rows, cols)
    sources = _select_sources(floor, expect_ties)
    if sources is None:
        return None

    grid = np.full((rows, cols), WALL_COLOR, dtype=int)
    grid[floor] = FLOOR_COLOR
    colors = random.sample(SOURCE_COLORS, 2)
    grid[sources[0]] = int(colors[0])
    grid[sources[1]] = int(colors[1])
    if not _semantic_validation(grid, expect_ties):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    _GEN_COUNTER += 1
    expect_ties = _GEN_COUNTER % 2 == 0
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(expect_ties)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid geodesic territory input")
