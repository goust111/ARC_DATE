from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PORTAL_COLORS = (BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
RNG = np.random.default_rng()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _portal_lookup(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for color in map(int, np.unique(grid)):
        if color in (BLACK, TEAL, MAROON):
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each portal color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        lookup[a] = b
        lookup[b] = a
    assert lookup, "expected at least one portal pair"
    return lookup


def _graph_neighbors(
    passable: np.ndarray,
    portal_lookup: dict[tuple[int, int], tuple[int, int]],
    cell: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds(passable.shape, nr, nc) and passable[nr, nc]:
            out.append((nr, nc))
    twin = portal_lookup.get(cell)
    if twin is not None and twin not in out:
        out.append(twin)
    return out


def _route_info(
    grid: np.ndarray,
    enabled_portal_colors: set[int] | None = None,
) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    endpoints = [tuple(map(int, p)) for p in np.argwhere(g == MAROON)]
    assert len(endpoints) == 2, "expected exactly two maroon endpoints"

    passable = g != BLACK
    portal_lookup = _portal_lookup(g)
    if enabled_portal_colors is not None:
        portal_lookup = {
            cell: twin
            for cell, twin in portal_lookup.items()
            if int(g[cell]) in enabled_portal_colors
        }
    start, goal = endpoints

    dist = {start: 0}
    count = {start: 1}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])

    while queue:
        cell = queue.popleft()
        for nxt in _graph_neighbors(passable, portal_lookup, cell):
            new_dist = dist[cell] + 1
            if nxt not in dist:
                dist[nxt] = new_dist
                count[nxt] = count[cell]
                parent[nxt] = cell
                queue.append(nxt)
            elif new_dist == dist[nxt]:
                count[nxt] = min(2, count[nxt] + count[cell])

    assert goal in dist, "endpoints must be connected"

    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        assert cur is not None
        path.append(cur)
    path.reverse()
    return {"dist": int(dist[goal]), "count": int(count[goal]), "path": path}


def _shortest_path(grid: np.ndarray) -> list[tuple[int, int]]:
    info = _route_info(grid)
    assert int(info["count"]) == 1, "expected a unique shortest route"
    return info["path"]


def _path_uses_teleport(
    path: list[tuple[int, int]],
    portal_lookup: dict[tuple[int, int], tuple[int, int]],
) -> bool:
    return any(portal_lookup.get(a) == b for a, b in zip(path, path[1:]))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the teal cells on the unique shortest endpoint route when same-colored portal pairs allow teleport jumps.

    Input structure
    - A black background contains one teal 4-connected track network, exactly two maroon endpoint pixels, and several non-teal non-maroon portal colors.
    - Each portal color appears exactly twice and sits on the track as a normal walkable pixel.

    Rule
    - Move along the track with 4-neighbor steps through every non-black track pixel.
    - When the route stands on one portal pixel, it may spend one extra step to jump to its same-colored twin.
    - Among all endpoint-to-endpoint walks there is exactly one shortest route; recolor only the teal cells on that route to maroon, while leaving endpoints, portal pixels, and off-route track unchanged.

    Key insight
    - The paired colors are not destinations to connect directly; they add shortcut edges to the track graph, so the correct answer is the unique shortest path in that augmented graph.
    """
    g = np.asarray(grid, dtype=int)
    path = _shortest_path(g)
    out = g.copy()
    for r, c in path:
        if out[r, c] == TEAL:
            out[r, c] = MAROON
    return out


def _track_neighbors(
    cell: tuple[int, int],
    track: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in track]


def _build_loop_track() -> tuple[int, int, set[tuple[int, int]]]:
    h = int(RNG.choice(np.array([13, 15, 17, 19], dtype=int)))
    w = int(RNG.choice(np.array([13, 15, 17, 19], dtype=int)))
    coarse_h = (h - 1) // 2
    coarse_w = (w - 1) // 2
    nodes = [(r, c) for r in range(coarse_h) for c in range(coarse_w)]

    root = nodes[int(RNG.integers(0, len(nodes)))]
    visited = {root}
    stack = [root]
    node_coord = lambda node: (1 + 2 * node[0], 1 + 2 * node[1])
    track = {node_coord(root)}

    while stack:
        cur = stack[-1]
        choices = []
        for dr, dc in DIRS4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < coarse_h and 0 <= nxt[1] < coarse_w and nxt not in visited:
                choices.append(nxt)
        if not choices:
            stack.pop()
            continue
        nxt = choices[int(RNG.integers(0, len(choices)))]
        visited.add(nxt)
        stack.append(nxt)
        a = node_coord(cur)
        b = node_coord(nxt)
        track.update({a, b, ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)})

    openings = []
    for r, c in nodes:
        a = node_coord((r, c))
        for dr, dc in ((1, 0), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt[0] >= coarse_h or nxt[1] >= coarse_w:
                continue
            b = node_coord(nxt)
            midpoint = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            if midpoint not in track:
                openings.append(midpoint)
    RNG.shuffle(openings)
    loop_count = int(RNG.integers(5, min(12, len(openings)) + 1))
    track.update(openings[:loop_count])
    return h, w, track


def _track_route_info(
    track: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    portal_lookup: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> dict | None:
    q = deque([start])
    dist = {start: 0}
    count = {start: 1}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        choices = _track_neighbors(cur, track)
        if portal_lookup is not None and cur in portal_lookup:
            choices.append(portal_lookup[cur])
        for nxt in choices:
            nd = int(dist[cur]) + 1
            if nxt not in dist:
                dist[nxt] = nd
                count[nxt] = int(count[cur])
                parent[nxt] = cur
                q.append(nxt)
            elif int(dist[nxt]) == nd:
                count[nxt] = min(2, int(count[nxt]) + int(count[cur]))
    if goal not in dist:
        return None

    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return {"dist": int(dist[goal]), "count": int(count[goal]), "path": path}


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    physical_steps = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(path, path[1:])
        if abs(b[0] - a[0]) + abs(b[1] - a[1]) == 1
    ]
    for a, b in zip(physical_steps, physical_steps[1:]):
        turns += a != b
    return int(turns)


def _alternate_track_distance(
    track: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]],
) -> int | None:
    alternatives = []
    for a, b in zip(path, path[1:]):
        banned = frozenset((a, b))
        q = deque([start])
        dist = {start: 0}
        while q:
            cur = q.popleft()
            for nxt in _track_neighbors(cur, track):
                if frozenset((cur, nxt)) == banned or nxt in dist:
                    continue
                dist[nxt] = int(dist[cur]) + 1
                q.append(nxt)
        if goal in dist:
            alternatives.append(int(dist[goal]))
    return min(alternatives) if alternatives else None


def _select_endpoints(
    track: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], dict] | None:
    nodes = [cell for cell in track if cell[0] % 2 == 1 and cell[1] % 2 == 1]
    RNG.shuffle(nodes)
    junctions = sum(len(_track_neighbors(cell, track)) >= 3 for cell in track)
    if junctions < 8:
        return None

    for start in nodes[: min(30, len(nodes))]:
        goals = nodes.copy()
        RNG.shuffle(goals)
        for goal in goals:
            if goal == start:
                continue
            info = _track_route_info(track, start, goal)
            if info is None or not (18 <= int(info["dist"]) <= 42):
                continue
            if _path_turns(info["path"]) < 4:
                continue
            alternate = _alternate_track_distance(track, start, goal, info["path"])
            if alternate is None:
                continue
            return start, goal, info
    return None


def _select_competing_portals(
    track: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    plain_info: dict,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], dict] | None:
    candidates = [
        cell
        for cell in track
        if cell not in {start, goal} and len(_track_neighbors(cell, track)) == 2
    ]
    if len(candidates) < 6:
        return None

    pair_count = int(RNG.integers(3, min(5, len(candidates) // 2) + 1))
    for _ in range(250):
        indices = RNG.choice(len(candidates), size=2 * pair_count, replace=False)
        points = [candidates[int(i)] for i in indices]
        pairs = [(points[2 * i], points[2 * i + 1]) for i in range(pair_count)]
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) < 5 for a, b in pairs):
            continue

        lookup: dict[tuple[int, int], tuple[int, int]] = {}
        pair_index: dict[tuple[int, int], int] = {}
        for i, (a, b) in enumerate(pairs):
            lookup[a] = b
            lookup[b] = a
            pair_index[a] = pair_index[b] = int(i)

        full = _track_route_info(track, start, goal, lookup)
        if full is None or int(full["count"]) != 1 or int(full["dist"]) >= int(plain_info["dist"]):
            continue
        used = {
            pair_index[a]
            for a, b in zip(full["path"], full["path"][1:])
            if lookup.get(a) == b
        }
        if not (2 <= len(used) < pair_count):
            continue

        single_distances = []
        for a, b in pairs:
            single = _track_route_info(track, start, goal, {a: b, b: a})
            assert single is not None
            single_distances.append(int(single["dist"]))
        if int(full["dist"]) >= min(single_distances):
            continue
        if not any(
            single_distances[i] < int(plain_info["dist"])
            for i in range(pair_count)
            if i not in used
        ):
            continue
        return pairs, full
    return None


def _render_loop_grid(
    h: int,
    w: int,
    track: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    portal_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in track:
        grid[r, c] = TEAL
    grid[start] = MAROON
    grid[goal] = MAROON

    colors = RNG.choice(
        np.array(PORTAL_COLORS, dtype=int),
        size=len(portal_pairs),
        replace=False,
    )
    for color, (a, b) in zip(colors.tolist(), portal_pairs):
        grid[a] = int(color)
        grid[b] = int(color)
    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, k=int(RNG.integers(0, 4)))
    if RNG.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _used_portal_colors(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    lookup: dict[tuple[int, int], tuple[int, int]],
) -> set[int]:
    return {
        int(grid[a])
        for a, b in zip(path, path[1:])
        if lookup.get(a) == b
    }


def _physical_path_length(grid: np.ndarray) -> int:
    return int(_route_info(grid, enabled_portal_colors=set())["dist"])


def _gen_attempt() -> np.ndarray | None:
    h, w, track = _build_loop_track()
    selected = _select_endpoints(track)
    if selected is None:
        return None
    start, goal, plain_info = selected

    portal_selection = _select_competing_portals(track, start, goal, plain_info)
    if portal_selection is None:
        return None
    portal_pairs, _expected_info = portal_selection

    grid = _render_loop_grid(h, w, track, start, goal, portal_pairs)
    grid = _random_transform(grid)
    try:
        info = _route_info(grid)
        path = _shortest_path(grid)
    except AssertionError:
        return None

    lookup = _portal_lookup(grid)
    used_colors = _used_portal_colors(grid, path, lookup)
    all_colors = {int(grid[cell]) for cell in lookup}
    if not (2 <= len(used_colors) < len(all_colors)):
        return None

    plain_info_grid = _route_info(grid, enabled_portal_colors=set())
    if int(info["dist"]) >= int(plain_info_grid["dist"]):
        return None
    single_distances = {
        color: int(_route_info(grid, enabled_portal_colors={color})["dist"])
        for color in all_colors
    }
    if int(info["dist"]) >= min(single_distances.values()):
        return None
    if not any(
        single_distances[color] < int(plain_info_grid["dist"])
        for color in all_colors - used_colors
    ):
        return None

    teal_total = int(np.count_nonzero(grid == TEAL))
    teal_on_path = sum(int(grid[r, c]) == TEAL for r, c in path)
    if teal_on_path < 5 or teal_total - teal_on_path < 12:
        return None

    out = main(grid)
    if np.array_equal(out, grid) or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"failed to generate a competing portal-path input after {GEN_ATTEMPTS} attempts"
    )
