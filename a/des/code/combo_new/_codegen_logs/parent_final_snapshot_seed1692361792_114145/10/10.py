import random
from collections import Counter

import numpy as np


BLACK = 0
WALL = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

ROOM_ROW_RANGE = (2, 4)
ROOM_COL_RANGE = (3, 5)
ROOM_HEIGHT_RANGE = (3, 4)
ROOM_WIDTH_RANGE = (3, 4)
MAX_ACTIVE_PATHS = 2
MAX_DORMANT_PAIRS = 2
GEN_RETRIES = 400


def _component_labels(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    labels = np.full((h, w), -1, dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = comp_id
            while stack:
                cr, cc = stack.pop()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
            comp_id += 1
    return labels


def _infer_wall_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != BLACK)
    assert counts, "expected walls and special colored pixels"
    return int(counts.most_common(1)[0][0])


def _is_door_cell(grid: np.ndarray, r: int, c: int, wall: int) -> bool:
    h, w = grid.shape
    if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
        return False
    if int(grid[r, c]) in (BLACK, wall):
        return False

    up_open = int(grid[r - 1, c]) != wall
    down_open = int(grid[r + 1, c]) != wall
    left_open = int(grid[r, c - 1]) != wall
    right_open = int(grid[r, c + 1]) != wall

    return (left_open and right_open and not up_open and not down_open) or (
        up_open and down_open and not left_open and not right_open
    )


def _locate_pairs(grid: np.ndarray, wall: int) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    doors: dict[int, tuple[int, int]] = {}
    keys: dict[int, tuple[int, int]] = {}

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, wall):
                continue
            target = doors if _is_door_cell(grid, r, c, wall) else keys
            assert color not in target, f"duplicate special color {color}"
            target[color] = (r, c)

    assert doors, "expected at least one colored door"
    assert set(doors) == set(keys), "doors and keys must match by color"
    return doors, keys


def _passable_mask(grid: np.ndarray, wall: int, doors: dict[int, tuple[int, int]], unopened: set[int]) -> np.ndarray:
    passable = grid != wall
    for color in unopened:
        r, c = doors[color]
        passable[r, c] = False
    return passable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open colored doors when their matching keys become reachable, then flood each opened room cluster with the last key color used there.

    Input structure:
    - The background floor is black and grey walls split the grid into closed rectangular rooms.
    - A colored pixel embedded in a wall is a door; each door color appears exactly once.
    - A colored pixel inside a room is a key; each key color appears exactly once and matches one door.

    Rule:
    - Match keys to doors by color.
    - A door opens only when its matching key lies in a currently reachable room whose boundary includes that door.
    - Opening a door turns the door pixel black, removes the used key to black, and may expose more keys that can now open their own doors; repeat until no more doors can open.
    - For each final connected open area, fill every black pixel in that area with the color of the last key used there. Areas with no used key stay black, and grey walls never change.

    Key insight:
    - The colored wall pixels are blocked doors rather than room contents, so the puzzle is an iterative reachability process on rooms where each newly opened door can unlock the next key-door pair in a chain.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2

    wall = _infer_wall_color(grid)
    doors, keys = _locate_pairs(grid, wall)

    state = grid.copy()
    unopened = set(doors)
    used_events: list[tuple[int, int, int, int, tuple[int, int]]] = []
    round_id = 0

    while True:
        passable = _passable_mask(state, wall, doors, unopened)
        labels = _component_labels(passable)

        eligible: list[int] = []
        for color in sorted(unopened, key=lambda value: doors[value]):
            kr, kc = keys[color]
            comp = int(labels[kr, kc])
            assert comp >= 0, "keys must sit inside passable room cells"
            dr, dc = doors[color]
            if any(
                0 <= dr + drow < state.shape[0]
                and 0 <= dc + dcol < state.shape[1]
                and passable[dr + drow, dc + dcol]
                and int(labels[dr + drow, dc + dcol]) == comp
                for drow, dcol in NEIGHBORS4
            ):
                eligible.append(color)

        if not eligible:
            break

        for color in eligible:
            dr, dc = doors[color]
            kr, kc = keys[color]
            state[dr, dc] = BLACK
            state[kr, kc] = BLACK
            unopened.remove(color)
            used_events.append((round_id, dr, dc, color, (kr, kc)))
        round_id += 1

    passable = _passable_mask(state, wall, doors, unopened)
    labels = _component_labels(passable)
    out = state.copy()

    last_color_by_component: dict[int, tuple[tuple[int, int, int], int]] = {}
    for round_id, dr, dc, color, key_pos in used_events:
        comp = int(labels[key_pos])
        rank = (round_id, dr, dc)
        if comp not in last_color_by_component or rank > last_color_by_component[comp][0]:
            last_color_by_component[comp] = (rank, color)

    for comp, (_rank, color) in last_color_by_component.items():
        fill_mask = (labels == comp) & (out == BLACK)
        out[fill_mask] = int(color)

    return out


def _build_room_layout() -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]], dict[tuple[int, int], list[tuple[int, int]]], dict[tuple[int, int], list[tuple[int, int]]]]:
    room_rows = random.randint(*ROOM_ROW_RANGE)
    room_cols = random.randint(*ROOM_COL_RANGE)
    row_heights = [random.randint(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
    col_widths = [random.randint(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]

    height = sum(row_heights) + room_rows + 1
    width = sum(col_widths) + room_cols + 1
    assert height <= 30 and width <= 30

    grid = np.full((height, width), WALL, dtype=int)
    bounds: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    room_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    room_cores: dict[tuple[int, int], list[tuple[int, int]]] = {}

    r0 = 1
    for rr, room_h in enumerate(row_heights):
        c0 = 1
        for cc, room_w in enumerate(col_widths):
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            room = (rr, cc)
            bounds[room] = (r0, c0, r1, c1)
            room_cells[room] = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            room_cores[room] = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            c0 = c1 + 2
        r0 = r1 + 2

    return grid, bounds, room_cells, room_cores


def _adjacency(bounds: dict[tuple[int, int], tuple[int, int, int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    rooms = list(bounds)
    adj = {room: [] for room in rooms}
    for rr, cc in rooms:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if (nr, nc) in adj:
                adj[(rr, cc)].append((nr, nc))
    return adj


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a < b else (b, a)


def _door_candidates(
    bounds: dict[tuple[int, int], tuple[int, int, int, int]],
    room_a: tuple[int, int],
    room_b: tuple[int, int],
) -> list[tuple[int, int]]:
    if room_a[0] == room_b[0]:
        left, right = (room_a, room_b) if room_a[1] < room_b[1] else (room_b, room_a)
        r0, _c0, r1, c1 = bounds[left]
        wall_c = c1 + 1
        return [(r, wall_c) for r in range(r0 + 1, r1)]

    top, bottom = (room_a, room_b) if room_a[0] < room_b[0] else (room_b, room_a)
    r0, c0, r1, c1 = bounds[top]
    wall_r = r1 + 1
    return [(wall_r, c) for c in range(c0 + 1, c1)]


def _simple_paths(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    available: set[tuple[int, int]],
    min_edges: int,
    max_edges: int,
) -> list[tuple[tuple[int, int], ...]]:
    paths: list[tuple[tuple[int, int], ...]] = []

    def dfs(path: list[tuple[int, int]]) -> None:
        edge_count = len(path) - 1
        if edge_count >= min_edges:
            paths.append(tuple(path))
        if edge_count == max_edges:
            return
        for nxt in adj[path[-1]]:
            if nxt in available and nxt not in path:
                path.append(nxt)
                dfs(path)
                path.pop()

    for start in sorted(available):
        dfs([start])
    return paths


def _remaining_edges(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    available: set[tuple[int, int]],
    used_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for room in sorted(available):
        for nxt in adj[room]:
            edge = _edge_key(room, nxt)
            if nxt in available and edge not in used_edges and room < nxt:
                edges.append((room, nxt))
    return edges


def _choose_structure(
    bounds: dict[tuple[int, int], tuple[int, int, int, int]]
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] | None:
    adj = _adjacency(bounds)
    available = set(bounds)

    first_paths = _simple_paths(adj, available, min_edges=2, max_edges=3)
    if not first_paths:
        return None
    active_paths = [random.choice(first_paths)]
    available -= set(active_paths[0])

    if len(active_paths) < MAX_ACTIVE_PATHS and random.random() < 0.45:
        second_paths = _simple_paths(adj, available, min_edges=1, max_edges=2)
        if second_paths:
            second = random.choice(second_paths)
            active_paths.append(second)
            available -= set(second)

    used_edges = {
        _edge_key(path[idx], path[idx + 1])
        for path in active_paths
        for idx in range(len(path) - 1)
    }

    dormant_pairs: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    dormant_available = set(available)
    target_dormant = 0
    if len(dormant_available) >= 3 and random.random() < 0.75:
        target_dormant = 1
        if random.random() < 0.35:
            target_dormant = 2

    while len(dormant_pairs) < min(target_dormant, MAX_DORMANT_PAIRS):
        edge_options = []
        for room_a, room_b in _remaining_edges(adj, dormant_available, used_edges):
            key_rooms = [room for room in sorted(dormant_available) if room not in (room_a, room_b)]
            if key_rooms:
                edge_options.append((room_a, room_b, key_rooms))
        if not edge_options:
            break
        room_a, room_b, key_rooms = random.choice(edge_options)
        key_room = random.choice(key_rooms)
        dormant_pairs.append((room_a, room_b, key_room))
        used_edges.add(_edge_key(room_a, room_b))
        dormant_available.remove(room_a)
        dormant_available.remove(room_b)
        dormant_available.remove(key_room)

    return active_paths, dormant_pairs


def _pick_door_position(
    bounds: dict[tuple[int, int], tuple[int, int, int, int]],
    room_a: tuple[int, int],
    room_b: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = _door_candidates(bounds, room_a, room_b)
    isolated = [
        pos
        for pos in candidates
        if pos not in occupied and all(abs(pos[0] - orow) + abs(pos[1] - ocol) > 1 for orow, ocol in occupied)
    ]
    options = isolated or [pos for pos in candidates if pos not in occupied]
    if not options:
        return None
    return random.choice(options)


def _render_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid, bounds, room_cells, room_cores = _build_room_layout()
    structure = _choose_structure(bounds)
    if structure is None:
        return None
    active_paths, dormant_pairs = structure

    active_edge_count = sum(len(path) - 1 for path in active_paths)
    total_pairs = active_edge_count + len(dormant_pairs)
    if total_pairs <= 0 or total_pairs > len(PAIR_COLORS):
        return None

    colors = random.sample(PAIR_COLORS, total_pairs)
    color_idx = 0
    occupied_doors: set[tuple[int, int]] = set()
    key_assignments: dict[tuple[int, int], tuple[int, int]] = {}
    active_components = []

    for path in active_paths:
        edges = []
        for idx in range(len(path) - 1):
            room_a, room_b = path[idx], path[idx + 1]
            door_pos = _pick_door_position(bounds, room_a, room_b, occupied_doors)
            if door_pos is None:
                return None
            occupied_doors.add(door_pos)
            key_room = room_a
            if key_room in key_assignments:
                return None
            key_cells = room_cores[key_room]
            if not key_cells:
                return None
            key_pos = random.choice(key_cells)
            key_assignments[key_room] = key_pos
            color = colors[color_idx]
            color_idx += 1
            edges.append(
                {
                    "color": color,
                    "door_pos": door_pos,
                    "door_rooms": (room_a, room_b),
                    "key_room": key_room,
                    "key_pos": key_pos,
                }
            )
        active_components.append({"path": path, "edges": edges, "fill_color": edges[-1]["color"]})

    dormant_specs = []
    for room_a, room_b, key_room in dormant_pairs:
        if key_room in key_assignments:
            return None
        door_pos = _pick_door_position(bounds, room_a, room_b, occupied_doors)
        if door_pos is None:
            return None
        occupied_doors.add(door_pos)
        key_cells = room_cores[key_room]
        if not key_cells:
            return None
        key_pos = random.choice(key_cells)
        key_assignments[key_room] = key_pos
        color = colors[color_idx]
        color_idx += 1
        dormant_specs.append(
            {
                "color": color,
                "door_pos": door_pos,
                "door_rooms": (room_a, room_b),
                "key_room": key_room,
                "key_pos": key_pos,
            }
        )

    inp = grid.copy()
    for component in active_components:
        for edge in component["edges"]:
            inp[edge["door_pos"]] = int(edge["color"])
            inp[edge["key_pos"]] = int(edge["color"])
    for spec in dormant_specs:
        inp[spec["door_pos"]] = int(spec["color"])
        inp[spec["key_pos"]] = int(spec["color"])

    out = inp.copy()
    for component in active_components:
        fill_color = int(component["fill_color"])
        for room in component["path"]:
            r0, c0, r1, c1 = bounds[room]
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
        for edge in component["edges"]:
            dr, dc = edge["door_pos"]
            out[dr, dc] = fill_color

    if np.array_equal(main(inp), out):
        used_rooms = {room for component in active_components for room in component["path"]}
        dormant_rooms = {spec["key_room"] for spec in dormant_specs}
        assert used_rooms.isdisjoint(dormant_rooms)
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rendered = _render_case()
        if rendered is not None:
            return rendered[0]
    raise RuntimeError("failed to generate a valid door-key ARC input")
