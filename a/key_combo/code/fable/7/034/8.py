from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8
DOOR_COLORS = (YELLOW, TEAL)

Room = tuple[int, int]
Edge = tuple[Room, Room]


def _edge_key(a: Room, b: Room) -> Edge:
    return (a, b) if a < b else (b, a)


def _parse_lattice(grid: np.ndarray):
    """Parse wall coordinates, rooms, colored doors, and terminal rooms."""
    g = np.asarray(grid, dtype=int)
    wall_palette = np.isin(g, (GREY, YELLOW, TEAL))
    wall_rows = [r for r in range(g.shape[0]) if bool(np.all(wall_palette[r, :]))]
    wall_cols = [c for c in range(g.shape[1]) if bool(np.all(wall_palette[:, c]))]

    assert len(wall_rows) >= 3 and len(wall_cols) >= 3
    assert wall_rows[0] == 0 and wall_rows[-1] == g.shape[0] - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == g.shape[1] - 1
    assert all(b - a >= 2 for a, b in zip(wall_rows, wall_rows[1:]))
    assert all(b - a >= 2 for a, b in zip(wall_cols, wall_cols[1:]))

    nr, nc = len(wall_rows) - 1, len(wall_cols) - 1
    adj: dict[Room, list[tuple[Room, int, tuple[int, int]]]] = {
        (r, c): [] for r in range(nr) for c in range(nc)
    }

    # A horizontal separator segment belongs to exactly the rooms above/below it.
    for i in range(1, nr):
        wr = wall_rows[i]
        for j in range(nc):
            segment = g[wr, wall_cols[j] + 1 : wall_cols[j + 1]]
            offsets = np.flatnonzero(np.isin(segment, DOOR_COLORS))
            assert len(offsets) <= 1
            if len(offsets) == 1:
                c = wall_cols[j] + 1 + int(offsets[0])
                a, b = (i - 1, j), (i, j)
                color = int(g[wr, c])
                adj[a].append((b, color, (wr, c)))
                adj[b].append((a, color, (wr, c)))

    # A vertical separator segment belongs to exactly the rooms left/right of it.
    for j in range(1, nc):
        wc = wall_cols[j]
        for i in range(nr):
            segment = g[wall_rows[i] + 1 : wall_rows[i + 1], wc]
            offsets = np.flatnonzero(np.isin(segment, DOOR_COLORS))
            assert len(offsets) <= 1
            if len(offsets) == 1:
                r = wall_rows[i] + 1 + int(offsets[0])
                a, b = (i, j - 1), (i, j)
                color = int(g[r, wc])
                adj[a].append((b, color, (r, wc)))
                adj[b].append((a, color, (r, wc)))

    terminals: list[Room] = []
    for i in range(nr):
        for j in range(nc):
            interior = g[
                wall_rows[i] + 1 : wall_rows[i + 1],
                wall_cols[j] + 1 : wall_cols[j + 1],
            ]
            reds = int(np.count_nonzero(interior == RED))
            assert reds <= 1
            assert np.all(np.isin(interior, (BACKGROUND, RED)))
            if reds:
                terminals.append((i, j))
    assert len(terminals) == 2
    return wall_rows, wall_cols, adj, terminals


def _unique_alternating_path(
    adj: dict[Room, list[tuple[Room, int, tuple[int, int]]]],
    start: Room,
    goal: Room,
):
    """BFS on (room, previous-door-color), counting shortest state paths."""
    initial = (start, -1)
    dist = {initial: 0}
    ways = {initial: 1}
    prev: dict[tuple[Room, int], tuple[tuple[Room, int], tuple[int, int]]] = {}
    queue = deque([initial])

    while queue:
        state = queue.popleft()
        room, last_color = state
        for neighbor, color, door in adj[room]:
            if color == last_color:
                continue
            nxt = (neighbor, color)
            nd = dist[state] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[state]
                prev[nxt] = (state, door)
                queue.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(2, ways[nxt] + ways[state])

    goals = [state for state in dist if state[0] == goal]
    assert goals, "the terminals must have an alternating-door route"
    best_distance = min(dist[state] for state in goals)
    best_goals = [state for state in goals if dist[state] == best_distance]
    assert sum(ways[state] for state in best_goals) == 1, "shortest legal route must be unique"

    state = best_goals[0]
    rooms = [state[0]]
    doors: list[tuple[int, int]] = []
    while state != initial:
        prior, door = prev[state]
        doors.append(door)
        state = prior
        rooms.append(state[0])
    rooms.reverse()
    doors.reverse()
    return rooms, doors


def main(grid: np.ndarray) -> np.ndarray:
    """Illuminate the unique shortest room route whose yellow and teal doors alternate.

    Input structure:
    - Complete grey horizontal and vertical separator lines partition a black grid into a regular room array.
    - A one-cell yellow or teal gap in an internal separator is a door between its two adjacent rooms.
    - Exactly two room interiors contain one red terminal pixel each; all other interior pixels are black.

    Rule:
    - Treat rooms as nodes and doors as colored edges, and choose the unique fewest-door route between the terminal rooms whose successive door colors strictly alternate; either color may come first.
    - Recolor every black interior pixel of every route room green, preserve the two red pixels, and recolor each used door grey. Leave all other rooms, doors, and walls unchanged.

    Key insight:
    - Distance must be computed in the state space of both the current room and previous door color; an ordinary spatial or graph shortest path can be blocked by two consecutive same-color doors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BACKGROUND, RED, GREEN, YELLOW, GREY, TEAL}
    wall_rows, wall_cols, adj, terminals = _parse_lattice(g)
    rooms, doors = _unique_alternating_path(adj, terminals[0], terminals[1])

    out = g.copy()
    for i, j in rooms:
        view = out[
            wall_rows[i] + 1 : wall_rows[i + 1],
            wall_cols[j] + 1 : wall_cols[j + 1],
        ]
        view[view == BACKGROUND] = GREEN
    for r, c in doors:
        out[r, c] = GREY
    return out


def _path_edges(path: list[Room]) -> list[Edge]:
    return [_edge_key(a, b) for a, b in zip(path, path[1:])]


def _alternating_colors(length: int, first: int) -> list[int]:
    other = TEAL if first == YELLOW else YELLOW
    return [first if i % 2 == 0 else other for i in range(length)]


def _all_simple_terminal_paths(
    graph: dict[Room, list[Room]], start: Room, goal: Room
) -> list[list[Room]]:
    """Independent small-graph oracle used only to reject semantically weak samples."""
    found: list[list[Room]] = []

    def visit(node: Room, path: list[Room], seen: set[Room]) -> None:
        if node == goal:
            found.append(path.copy())
            return
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                path.append(nxt)
                visit(nxt, path, seen)
                path.pop()
                seen.remove(nxt)

    visit(start, [start], {start})
    return found


def _render(
    rows: int,
    cols: int,
    room_h: int,
    room_w: int,
    edge_colors: dict[Edge, int],
    terminals: tuple[Room, Room],
):
    height = rows * room_h + rows + 1
    width = cols * room_w + cols + 1
    grid = np.full((height, width), BACKGROUND, dtype=int)
    wall_rows = [i * (room_h + 1) for i in range(rows + 1)]
    wall_cols = [j * (room_w + 1) for j in range(cols + 1)]
    grid[wall_rows, :] = GREY
    grid[:, wall_cols] = GREY

    door_pixels: dict[Edge, tuple[int, int]] = {}
    for edge, color in edge_colors.items():
        a, b = edge
        if a[0] == b[0]:
            r = wall_rows[a[0]] + random.randint(1, room_h)
            c = wall_cols[max(a[1], b[1])]
        else:
            r = wall_rows[max(a[0], b[0])]
            c = wall_cols[a[1]] + random.randint(1, room_w)
        grid[r, c] = color
        door_pixels[edge] = (r, c)

    for i, j in terminals:
        r = wall_rows[i] + random.randint(1, room_h)
        c = wall_cols[j] + random.randint(1, room_w)
        grid[r, c] = RED
    return grid, wall_rows, wall_cols, door_pixels


def _gen_attempt() -> np.ndarray | None:
    # A theta graph supplies three visibly distinct terminal routes: the direct
    # route is shorter but illegal, while two detours alternate and have unequal
    # lengths, so selecting the shortest *legal* detour is also indispensable.
    rows = random.randint(6, 8)
    cols = random.randint(4, 7)
    candidates = [r for r in range(2, rows - 2) if r != rows - 1 - r]
    mid = random.choice(candidates)
    start, goal = (mid, 0), (mid, cols - 1)

    direct = [(mid, c) for c in range(cols)]
    upper = [(r, 0) for r in range(mid, -1, -1)]
    upper += [(0, c) for c in range(1, cols)]
    upper += [(r, cols - 1) for r in range(1, mid + 1)]
    lower = [(r, 0) for r in range(mid, rows)]
    lower += [(rows - 1, c) for c in range(1, cols)]
    lower += [(r, cols - 1) for r in range(rows - 2, mid - 1, -1)]

    paths = [direct, upper, lower]
    edge_colors: dict[Edge, int] = {}
    for detour in (upper, lower):
        colors = _alternating_colors(len(detour) - 1, random.choice(DOOR_COLORS))
        edge_colors.update(zip(_path_edges(detour), colors))

    direct_colors = _alternating_colors(len(direct) - 1, random.choice(DOOR_COLORS))
    # The final two direct doors repeat a color, while earlier doors preserve a
    # tempting alternating prefix and (for length >= 3) both colors stay visible.
    direct_colors[-1] = direct_colors[-2]
    edge_colors.update(zip(_path_edges(direct), direct_colors))

    active = set().union(*map(set, paths))
    all_rooms = {(r, c) for r in range(rows) for c in range(cols)}
    # Add several acyclic dead-end doors. They prevent the diagram from reducing
    # visually to only three pristine lines without creating another terminal path.
    for _ in range(random.randint(2, 5)):
        frontier = []
        for a in active:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                b = (a[0] + dr, a[1] + dc)
                if b in all_rooms and b not in active:
                    frontier.append((a, b))
        if not frontier:
            break
        a, b = random.choice(frontier)
        edge_colors[_edge_key(a, b)] = random.choice(DOOR_COLORS)
        active.add(b)

    graph = {room: [] for room in all_rooms}
    for a, b in edge_colors:
        graph[a].append(b)
        graph[b].append(a)

    # Semantic oracle: enumerate graph paths independently of solver state BFS.
    all_paths = _all_simple_terminal_paths(graph, start, goal)
    legal = []
    for path in all_paths:
        colors = [edge_colors[e] for e in _path_edges(path)]
        if all(a != b for a, b in zip(colors, colors[1:])):
            legal.append(path)
    if len(legal) < 2:
        return None
    plain_distance = min(len(path) - 1 for path in all_paths)
    legal_distance = min(len(path) - 1 for path in legal)
    best = [path for path in legal if len(path) - 1 == legal_distance]
    if len(best) != 1 or plain_distance >= legal_distance:
        return None
    chosen = best[0]
    chosen_colors = [edge_colors[e] for e in _path_edges(chosen)]
    if set(chosen_colors) != set(DOOR_COLORS):
        return None

    room_h, room_w = random.randint(2, 4), random.randint(2, 4)
    grid, wall_rows, wall_cols, door_pixels = _render(
        rows, cols, room_h, room_w, edge_colors, (start, goal)
    )
    expected = grid.copy()
    for i, j in chosen:
        view = expected[
            wall_rows[i] + 1 : wall_rows[i + 1],
            wall_cols[j] + 1 : wall_cols[j + 1],
        ]
        view[view == BACKGROUND] = GREEN
    for edge in _path_edges(chosen):
        expected[door_pixels[edge]] = GREY

    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected) or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, uniquely solvable alternating-door room maze."""
    for _ in range(200):
        grid = _gen_attempt()
        if grid is not None:
            # Random transposition covers both horizontal and vertical terminal
            # arrangements without changing the room-relative semantics.
            return grid.T.copy() if random.random() < 0.5 else grid
    raise RuntimeError("failed to generate a valid alternating-door maze")
