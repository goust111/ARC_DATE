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


def _shortest_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    endpoints = [tuple(map(int, p)) for p in np.argwhere(g == MAROON)]
    assert len(endpoints) == 2, "expected exactly two maroon endpoints"

    passable = g != BLACK
    portal_lookup = _portal_lookup(g)
    start, goal = endpoints

    dist = {start: 0}
    count = {start: 1}
    queue = deque([start])

    while queue:
        cell = queue.popleft()
        for nxt in _graph_neighbors(passable, portal_lookup, cell):
            new_dist = dist[cell] + 1
            if nxt not in dist:
                dist[nxt] = new_dist
                count[nxt] = count[cell]
                queue.append(nxt)
            elif new_dist == dist[nxt]:
                count[nxt] = min(2, count[nxt] + count[cell])

    assert goal in dist, "endpoints must be connected"
    assert count[goal] == 1, "expected a unique shortest route"

    path = [goal]
    cur = goal
    while cur != start:
        preds = [
            prev
            for prev in _graph_neighbors(passable, portal_lookup, cur)
            if dist.get(prev) == dist[cur] - 1
        ]
        assert len(preds) == 1, "shortest-path predecessor chain must be unique"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


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


def _physical_degree(cell: tuple[int, int], track: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in track for dr, dc in DIRS4)


def _build_snake_track() -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    bars = int(RNG.integers(4, 7))
    span = int(RNG.integers(8, 15))
    left_margin = int(RNG.integers(2, 6))
    right_margin = int(RNG.integers(2, 6))
    top_margin = int(RNG.integers(2, 6))
    bottom_margin = int(RNG.integers(2, 6))

    h = top_margin + (2 * bars - 1) + bottom_margin
    w = left_margin + span + right_margin
    if h > 30 or w > 30:
        raise AssertionError("generator exceeded ARC size bound")

    r_top = top_margin
    c0 = left_margin
    c1 = c0 + span - 1

    main_path: list[tuple[int, int]] = []
    connector_cells: list[tuple[int, int]] = []
    for i in range(bars):
        r = r_top + 2 * i
        cols = range(c0, c1 + 1) if i % 2 == 0 else range(c1, c0 - 1, -1)
        main_path.extend((r, c) for c in cols)
        if i < bars - 1:
            connector_col = c1 if i % 2 == 0 else c0
            connector = (r + 1, connector_col)
            connector_cells.append(connector)
            main_path.append(connector)

    track = set(main_path)

    for r, c in connector_cells:
        if RNG.random() >= 0.55:
            continue
        step = 1 if c == c1 else -1
        room = (w - 1 - c) if step == 1 else c
        if room <= 0:
            continue
        length = int(RNG.integers(1, min(3, room) + 1))
        for k in range(1, length + 1):
            track.add((r, c + step * k))

    if top_margin > 0 and RNG.random() < 0.65:
        branch_col = int(RNG.integers(c0 + 2, c1 - 1))
        length = int(RNG.integers(1, min(3, top_margin) + 1))
        for k in range(1, length + 1):
            track.add((r_top - k, branch_col))

    bottom_row = r_top + 2 * (bars - 1)
    if bottom_margin > 0 and RNG.random() < 0.65:
        branch_col = int(RNG.integers(c0 + 1, c1 - 2))
        length = int(RNG.integers(1, min(3, bottom_margin) + 1))
        for k in range(1, length + 1):
            track.add((bottom_row + k, branch_col))

    return track, main_path


def _select_portal_pairs(
    track: set[tuple[int, int]],
    main_path: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = [
        idx
        for idx, cell in enumerate(main_path)
        if 3 <= idx < len(main_path) - 3 and _physical_degree(cell, track) == 2
    ]
    max_pairs = min(4, len(candidates) // 2, max(2, len(main_path) // 12))
    if max_pairs < 2:
        raise AssertionError("not enough interior cells for portal pairs")

    total_pairs = int(RNG.integers(2, max_pairs + 1))
    min_gap = max(3, len(main_path) // 18)
    long_gap = max(6, len(main_path) // 12)
    min_between = 2

    for _ in range(200):
        picks = sorted(RNG.choice(candidates, size=2 * total_pairs, replace=False).tolist())
        pairs: list[tuple[int, int]] = []
        total_saving = 0
        has_long_jump = False
        ok = True

        for j in range(total_pairs):
            a = int(picks[2 * j])
            b = int(picks[2 * j + 1])
            if b - a < min_gap:
                ok = False
                break
            if pairs and a - pairs[-1][1] < min_between:
                ok = False
                break
            pairs.append((a, b))
            total_saving += b - a - 1
            has_long_jump |= (b - a) >= long_gap

        if ok and has_long_jump and total_saving >= max(6, total_pairs * 2):
            return pairs

    raise AssertionError("failed to place non-overlapping portal shortcuts")


def _random_transform(grid: np.ndarray) -> np.ndarray:
    k = int(RNG.integers(0, 4))
    out = np.rot90(grid, k=k)
    if RNG.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _build_grid(
    track: set[tuple[int, int]],
    main_path: list[tuple[int, int]],
    portal_pairs: list[tuple[int, int]],
) -> np.ndarray:
    rows = [r for r, _ in track]
    cols = [c for _, c in track]
    h = max(rows) + 1
    w = max(cols) + 1
    grid = np.zeros((h, w), dtype=int)
    for r, c in track:
        grid[r, c] = TEAL

    start = main_path[0]
    goal = main_path[-1]
    grid[start] = MAROON
    grid[goal] = MAROON

    colors = RNG.choice(np.array(PORTAL_COLORS, dtype=int), size=len(portal_pairs), replace=False)
    for color, (a_idx, b_idx) in zip(colors.tolist(), portal_pairs):
        ar, ac = main_path[a_idx]
        br, bc = main_path[b_idx]
        grid[ar, ac] = int(color)
        grid[br, bc] = int(color)
    return grid


def _physical_path_length(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    endpoints = [tuple(map(int, p)) for p in np.argwhere(g == MAROON)]
    start, goal = endpoints
    queue = deque([start])
    dist = {start: 0}

    while queue:
        cell = queue.popleft()
        if cell == goal:
            return dist[cell]
        r, c = cell
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if _in_bounds(g.shape, nr, nc) and g[nr, nc] != BLACK and nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    raise AssertionError("physical track must connect the endpoints")


def _gen_attempt() -> np.ndarray | None:
    try:
        track, main_path = _build_snake_track()
        portal_pairs = _select_portal_pairs(track, main_path)
    except AssertionError:
        return None
    grid = _build_grid(track, main_path, portal_pairs)
    grid = _random_transform(grid)

    try:
        path = _shortest_path(grid)
    except AssertionError:
        return None

    portal_lookup = _portal_lookup(grid)
    if not _path_uses_teleport(path, portal_lookup):
        return None

    plain_len = _physical_path_length(grid)
    if len(path) - 1 >= plain_len:
        return None

    teal_total = int(np.count_nonzero(grid == TEAL))
    teal_on_path = sum(grid[r, c] == TEAL for r, c in path)
    if teal_on_path < 6 or teal_total - teal_on_path < 4:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid portal-path input after {GEN_ATTEMPTS} attempts")
