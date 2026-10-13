import random
from collections import deque

import numpy as np


BG = 0
MAGENTA = 6
ORANGE = 7
WALL_COLORS = (1, 2, 3, 4, 5, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_rooms(passable: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = passable.shape
    labels = -np.ones((h, w), dtype=int)
    rooms: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = len(rooms)
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = len(rooms)
                        q.append((nr, nc))
            rooms.append(np.array(cells, dtype=int))

    return labels, rooms


def _door_connection(g: np.ndarray, room_ids: np.ndarray, r: int, c: int):
    h, w = g.shape
    if int(g[r, c]) != ORANGE:
        return None

    specials = {BG, MAGENTA, ORANGE}
    candidates = []

    if 0 < c < w - 1:
        left = int(room_ids[r, c - 1])
        right = int(room_ids[r, c + 1])
        if left != -1 and right != -1 and left != right and 0 < r < h - 1:
            wall_color = int(g[r - 1, c])
            if wall_color == int(g[r + 1, c]) and wall_color not in specials:
                candidates.append((left, right, wall_color))

    if 0 < r < h - 1:
        up = int(room_ids[r - 1, c])
        down = int(room_ids[r + 1, c])
        if up != -1 and down != -1 and up != down and 0 < c < w - 1:
            wall_color = int(g[r, c - 1])
            if wall_color == int(g[r, c + 1]) and wall_color not in specials:
                candidates.append((up, down, wall_color))

    if len(candidates) != 1:
        return None
    a, b, wall_color = candidates[0]
    return a, b, wall_color


def _fill_room(out: np.ndarray, room: np.ndarray, color: int) -> None:
    rr = room[:, 0]
    cc = room[:, 1]
    mask = out[rr, cc] == BG
    out[rr[mask], cc[mask]] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recursively color rooms according to the wall color of the door that reaches them.

    Input structure:
    - bg = black. Colored 1-pixel walls partition the full grid into rectangular rooms.
    - Some wall pixels are replaced by orange door markers, each opening a passage between two adjacent rooms.
    - Exactly one room contains a magenta source pixel; other room interiors are otherwise black.

    Rule:
    - Treat every bg-or-magenta 4-connected interior region as one room, while walls and orange door markers stay fixed.
    - The room containing the magenta source is depth 0, and all of its bg cells become magenta.
    - Each orange door connects two neighboring rooms; the wall color around that door is the transmission color for crossing it.
    - Run a breadth-first expansion from the source room: every newly reached room is filled once, using the color of the door-wall that first reached it.

    Key insight:
    - Orange marks a doorway in an otherwise solid wall, and the propagated fill color comes from that wall segment, not from the source room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    passable = (g == BG) | (g == MAGENTA)
    room_ids, rooms = _label_rooms(passable)
    assert rooms, "expected at least one room"

    source_cells = np.argwhere(g == MAGENTA)
    assert source_cells.shape == (1, 2), "expected exactly one magenta source cell"
    sr, sc = map(int, source_cells[0])
    source_room = int(room_ids[sr, sc])
    assert source_room >= 0, "source cell must lie inside a room"

    adjacency: list[list[tuple[tuple[int, int], int, int]]] = [[] for _ in rooms]
    for r, c in np.argwhere(g == ORANGE):
        door = _door_connection(g, room_ids, int(r), int(c))
        if door is None:
            continue
        a, b, wall_color = door
        pos = (int(r), int(c))
        adjacency[a].append((pos, b, wall_color))
        adjacency[b].append((pos, a, wall_color))

    for edges in adjacency:
        edges.sort(key=lambda item: item[0])

    out = g.copy()
    seen = np.zeros(len(rooms), dtype=bool)
    q = deque([source_room])
    seen[source_room] = True
    _fill_room(out, rooms[source_room], MAGENTA)

    while q:
        room_id = q.popleft()
        for _door_pos, nbr, wall_color in adjacency[room_id]:
            if seen[nbr]:
                continue
            seen[nbr] = True
            _fill_room(out, rooms[nbr], wall_color)
            q.append(nbr)

    return out


# --------------- generator ---------------

MIN_GRID = 14
MAX_GRID = 28
MIN_ROOM_SIZE = 3
MIN_ROOMS = 4
MAX_ROOMS = 8
REACH_ALL_PROB = 0.5
GEN_RETRIES = 400


Rect = tuple[int, int, int, int]


def _rect_area(rect: Rect) -> int:
    r0, r1, c0, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _draw_outer_border(grid: np.ndarray, rng: random.Random) -> None:
    top, bottom, left, right = rng.choices(WALL_COLORS, k=4)
    grid[0, :] = top
    grid[-1, :] = bottom
    grid[:, 0] = left
    grid[:, -1] = right


def _split_room(grid: np.ndarray, room: Rect, orientation: str, rng: random.Random) -> tuple[Rect, Rect]:
    r0, r1, c0, c1 = room
    color = rng.choice(WALL_COLORS)
    if orientation == "v":
        wall_col = rng.randint(c0 + MIN_ROOM_SIZE, c1 - MIN_ROOM_SIZE)
        grid[r0 - 1 : r1 + 2, wall_col] = color
        left = (r0, r1, c0, wall_col - 1)
        right = (r0, r1, wall_col + 1, c1)
        return left, right

    wall_row = rng.randint(r0 + MIN_ROOM_SIZE, r1 - MIN_ROOM_SIZE)
    grid[wall_row, c0 - 1 : c1 + 2] = color
    top = (r0, wall_row - 1, c0, c1)
    bottom = (wall_row + 1, r1, c0, c1)
    return top, bottom


def _splittable_orientations(room: Rect) -> list[str]:
    r0, r1, c0, c1 = room
    opts = []
    if c1 - c0 + 1 >= 2 * MIN_ROOM_SIZE + 1:
        opts.append("v")
    if r1 - r0 + 1 >= 2 * MIN_ROOM_SIZE + 1:
        opts.append("h")
    return opts


def _build_partition(rng: random.Random):
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)
    grid = np.full((h, w), BG, dtype=int)
    _draw_outer_border(grid, rng)

    rooms: list[Rect] = [(1, h - 2, 1, w - 2)]
    target_rooms = rng.randint(MIN_ROOMS, MAX_ROOMS)

    while len(rooms) < target_rooms:
        candidates = [(idx, _splittable_orientations(room)) for idx, room in enumerate(rooms)]
        candidates = [(idx, opts) for idx, opts in candidates if opts]
        if not candidates:
            break

        weights = [_rect_area(rooms[idx]) for idx, _opts in candidates]
        idx, opts = rng.choices(candidates, weights=weights, k=1)[0]
        orientation = rng.choice(opts)
        room = rooms.pop(idx)
        child_a, child_b = _split_room(grid, room, orientation, rng)
        rooms.extend([child_a, child_b])

    if len(rooms) < MIN_ROOMS:
        return None
    return grid, rooms


def _vertical_positions(grid: np.ndarray, left: Rect, right: Rect) -> list[tuple[int, int]]:
    lr0, lr1, _lc0, lc1 = left
    rr0, rr1, rc0, _rc1 = right
    if lc1 + 2 != rc0:
        return []
    wall_col = lc1 + 1
    start = max(lr0, rr0) + 1
    stop = min(lr1, rr1) - 1
    positions = []
    for r in range(start, stop + 1):
        color = int(grid[r, wall_col])
        if color in (BG, MAGENTA, ORANGE):
            continue
        if (
            int(grid[r - 1, wall_col]) == color
            and int(grid[r + 1, wall_col]) == color
            and grid[r, wall_col - 1] == BG
            and grid[r, wall_col + 1] == BG
        ):
            positions.append((r, wall_col))
    return positions


def _horizontal_positions(grid: np.ndarray, top: Rect, bottom: Rect) -> list[tuple[int, int]]:
    tr0, tr1, tc0, tc1 = top
    br0, _br1, bc0, bc1 = bottom
    if tr1 + 2 != br0:
        return []
    wall_row = tr1 + 1
    start = max(tc0, bc0) + 1
    stop = min(tc1, bc1) - 1
    positions = []
    for c in range(start, stop + 1):
        color = int(grid[wall_row, c])
        if color in (BG, MAGENTA, ORANGE):
            continue
        if (
            int(grid[wall_row, c - 1]) == color
            and int(grid[wall_row, c + 1]) == color
            and grid[wall_row - 1, c] == BG
            and grid[wall_row + 1, c] == BG
        ):
            positions.append((wall_row, c))
    return positions


def _candidate_edges(grid: np.ndarray, rooms: list[Rect]):
    edges = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]

            if a[3] < b[2]:
                positions = _vertical_positions(grid, a, b)
                if positions:
                    edges.append({"u": i, "v": j, "color": int(grid[positions[0]]), "positions": positions})
                    continue
            if b[3] < a[2]:
                positions = _vertical_positions(grid, b, a)
                if positions:
                    edges.append({"u": i, "v": j, "color": int(grid[positions[0]]), "positions": positions})
                    continue
            if a[1] < b[0]:
                positions = _horizontal_positions(grid, a, b)
                if positions:
                    edges.append({"u": i, "v": j, "color": int(grid[positions[0]]), "positions": positions})
                    continue
            if b[1] < a[0]:
                positions = _horizontal_positions(grid, b, a)
                if positions:
                    edges.append({"u": i, "v": j, "color": int(grid[positions[0]]), "positions": positions})
    return edges


def _graph_from_edges(n_rooms: int, edges) -> list[list[tuple[int, int]]]:
    graph = [[] for _ in range(n_rooms)]
    for eid, edge in enumerate(edges):
        u = int(edge["u"])
        v = int(edge["v"])
        graph[u].append((v, eid))
        graph[v].append((u, eid))
    return graph


def _bfs_metadata(graph: list[list[tuple[int, int]]], start: int):
    dist = {start: 0}
    parent: dict[int, tuple[int, int]] = {}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt, eid in graph[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            parent[nxt] = (cur, eid)
            q.append(nxt)
    return dist, parent


def _choose_tree(rng: random.Random, n_rooms: int, edges):
    graph = _graph_from_edges(n_rooms, edges)
    source_candidates = []
    for room_id in range(n_rooms):
        dist, _parent = _bfs_metadata(graph, room_id)
        if len(dist) == n_rooms and max(dist.values()) >= 2:
            source_candidates.append(room_id)
    if not source_candidates:
        return None

    source = rng.choice(source_candidates)
    dist, parent = _bfs_metadata(graph, source)
    max_dist = max(dist.values())
    target = rng.choice([room for room, d in dist.items() if d == max_dist])

    selected_edges: set[int] = set()
    connected_rooms = {target}
    cur = target
    while cur != source:
        prev, eid = parent[cur]
        selected_edges.add(eid)
        connected_rooms.add(prev)
        cur = prev

    if len(connected_rooms) < 3:
        return None

    if n_rooms == len(connected_rooms) or rng.random() < REACH_ALL_PROB:
        target_size = n_rooms
    else:
        target_size = rng.randint(len(connected_rooms), n_rooms - 1)

    while len(connected_rooms) < target_size:
        frontier = []
        for room_id in connected_rooms:
            for nxt, eid in graph[room_id]:
                if nxt not in connected_rooms:
                    frontier.append((nxt, eid))
        if not frontier:
            break
        nxt, eid = rng.choice(frontier)
        selected_edges.add(eid)
        connected_rooms.add(nxt)

    edge_colors = {int(edges[eid]["color"]) for eid in selected_edges}
    if len(edge_colors) < 2:
        return None

    return source, selected_edges, connected_rooms


def _paint_selected_doors(grid: np.ndarray, edges, selected_edges: set[int], rng: random.Random) -> None:
    for eid in selected_edges:
        pos = rng.choice(edges[eid]["positions"])
        grid[pos] = ORANGE


def _place_source(grid: np.ndarray, room: Rect, rng: random.Random) -> None:
    r0, r1, c0, c1 = room
    sr = rng.randint(r0, r1)
    sc = rng.randint(c0, c1)
    grid[sr, sc] = MAGENTA


def _manual_expected(inp: np.ndarray, rooms: list[Rect], edges, source: int, selected_edges: set[int]) -> np.ndarray:
    expected = inp.copy()
    tree = [[] for _ in rooms]
    for eid in selected_edges:
        edge = edges[eid]
        u = int(edge["u"])
        v = int(edge["v"])
        color = int(edge["color"])
        tree[u].append((v, color))
        tree[v].append((u, color))

    seen = {source}
    q = deque([source])

    def fill_rect(rect: Rect, color: int) -> None:
        r0, r1, c0, c1 = rect
        patch = expected[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = color

    fill_rect(rooms[source], MAGENTA)
    while q:
        cur = q.popleft()
        for nxt, color in tree[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            fill_rect(rooms[nxt], color)
            q.append(nxt)
    return expected


def _attempt() -> np.ndarray | None:
    rng = random.Random()
    built = _build_partition(rng)
    if built is None:
        return None
    grid, rooms = built

    edges = _candidate_edges(grid, rooms)
    if len(edges) < 3:
        return None

    chosen = _choose_tree(rng, len(rooms), edges)
    if chosen is None:
        return None
    source, selected_edges, connected_rooms = chosen
    if len(selected_edges) < 2 or len(connected_rooms) < 3:
        return None

    inp = grid.copy()
    _paint_selected_doors(inp, edges, selected_edges, rng)
    _place_source(inp, rooms[source], rng)

    if int(np.count_nonzero(inp == MAGENTA)) != 1:
        return None
    if int(np.count_nonzero(inp == ORANGE)) != len(selected_edges):
        return None

    expected = _manual_expected(inp, rooms, edges, source, selected_edges)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door propagation puzzle")
