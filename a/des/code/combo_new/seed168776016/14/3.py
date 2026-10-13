from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAGENTA = 6
ORANGE = 7

SIGNAL_COLORS = [GREEN, MAGENTA, ORANGE]
DIRS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}
LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

ROOM_CONNECTOR_OFFSETS = {"N": (0, 1), "E": (1, 2), "S": (2, 1), "W": (1, 0)}
NODE_CONNECTOR_OFFSETS = {"N": (0, 0), "E": (0, 1), "S": (1, 1), "W": (1, 0)}
NODE_MARKER_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]

TEMP_SIZE = 60
GEN_RETRIES = 500


def _move(pos: tuple[int, int], side: str, dist: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[side]
    return pos[0] + dr * dist, pos[1] + dc * dist


def _in_bounds(pos: tuple[int, int], shape: tuple[int, int]) -> bool:
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        c0, c1 = sorted((a[1], b[1]))
        return [(a[0], c) for c in range(c0, c1 + 1)]
    assert a[1] == b[1]
    r0, r1 = sorted((a[0], b[0]))
    return [(r, a[1]) for r in range(r0, r1 + 1)]


def _box_cells(kind: str, top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = top_left
    if kind == "room":
        return [(r + dr, c + dc) for dr in range(3) for dc in range(3)]
    assert kind == "node"
    return [(r + dr, c + dc) for dr in range(2) for dc in range(2)]


def _room_border_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = top_left
    return [
        (r + dr, c + dc)
        for dr in range(3)
        for dc in range(3)
        if dr in (0, 2) or dc in (0, 2)
    ]


def _room_center(top_left: tuple[int, int]) -> tuple[int, int]:
    return top_left[0] + 1, top_left[1] + 1


def _room_outside(top_left: tuple[int, int], side: str) -> tuple[int, int]:
    base = ROOM_CONNECTOR_OFFSETS[side]
    return _move((top_left[0] + base[0], top_left[1] + base[1]), side)


def _room_candidate_connectors(top_left: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    return {side: [_room_outside(top_left, side)] for side in DIRS}


def _room_top_left_from_outside(outside: tuple[int, int], side: str) -> tuple[int, int]:
    if side == "N":
        return outside[0] + 1, outside[1] - 1
    if side == "E":
        return outside[0] - 1, outside[1] - 3
    if side == "S":
        return outside[0] - 3, outside[1] - 1
    assert side == "W"
    return outside[0] - 1, outside[1] + 1


def _node_outside(top_left: tuple[int, int], side: str) -> tuple[int, int]:
    base = NODE_CONNECTOR_OFFSETS[side]
    return _move((top_left[0] + base[0], top_left[1] + base[1]), side)


def _node_candidate_connectors(top_left: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    r, c = top_left
    return {
        "N": [(r - 1, c), (r - 1, c + 1)],
        "E": [(r, c + 2), (r + 1, c + 2)],
        "S": [(r + 2, c), (r + 2, c + 1)],
        "W": [(r, c - 1), (r + 1, c - 1)],
    }


def _node_top_left_from_outside(outside: tuple[int, int], side: str) -> tuple[int, int]:
    if side == "N":
        return outside[0] + 1, outside[1]
    if side == "E":
        return outside[0], outside[1] - 2
    if side == "S":
        return outside[0] - 2, outside[1] - 1
    assert side == "W"
    return outside[0] - 1, outside[1] + 1


def _event_outside(event: dict[str, Any], side: str) -> tuple[int, int]:
    if event["kind"] == "room":
        return _room_outside(event["top_left"], side)
    return _node_outside(event["top_left"], side)


def _event_candidate_connectors(event: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    if event["kind"] == "room":
        return _room_candidate_connectors(event["top_left"])
    return _node_candidate_connectors(event["top_left"])


def _draw_room(canvas: np.ndarray, top_left: tuple[int, int], center_color: int) -> None:
    for r, c in _room_border_cells(top_left):
        canvas[r, c] = GREY
    cr, cc = _room_center(top_left)
    canvas[cr, cc] = center_color


def _draw_node(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    marker_color: int,
    marker_offset: tuple[int, int],
) -> None:
    r, c = top_left
    canvas[r : r + 2, c : c + 2] = GREY
    canvas[r + marker_offset[0], c + marker_offset[1]] = marker_color


def _cells_clear(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    allowed_protected: set[tuple[int, int]] | None = None,
) -> bool:
    allow = allowed_protected or set()
    for cell in cells:
        if not _in_bounds(cell, shape):
            return False
        if cell in blocked:
            return False
        if cell in protected and cell not in allow:
            return False
    return True


def _place_initial_room(
    canvas: np.ndarray,
    blocked: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    events: list[dict[str, Any]],
    top_left: tuple[int, int],
    center_color: int,
) -> int | None:
    box = _box_cells("room", top_left)
    event = {
        "kind": "room",
        "top_left": top_left,
        "center_color": center_color,
        "connections": set(),
    }
    connectors = [cell for cells in _event_candidate_connectors(event).values() for cell in cells]
    shape = canvas.shape
    if not _cells_clear(box, shape, blocked, protected):
        return None
    if not _cells_clear(connectors, shape, blocked, protected):
        return None
    _draw_room(canvas, top_left, center_color)
    blocked.update(box)
    protected.update(connectors)
    events.append(event)
    return len(events) - 1


def _connect_new_event(
    canvas: np.ndarray,
    blocked: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    events: list[dict[str, Any]],
    from_idx: int,
    side: str,
    kind: str,
    distance: int,
    center_color: int = BLACK,
    marker_color: int = RED,
    marker_offset: tuple[int, int] = (0, 0),
) -> int | None:
    src = events[from_idx]
    if side in src["connections"]:
        return None

    start = _event_outside(src, side)
    end = _move(start, side, distance)
    incoming_side = OPPOSITE[side]
    if kind == "room":
        top_left = _room_top_left_from_outside(end, incoming_side)
    else:
        top_left = _node_top_left_from_outside(end, incoming_side)

    box = _box_cells(kind, top_left)
    shape = canvas.shape
    new_event = {
        "kind": kind,
        "top_left": top_left,
        "connections": {incoming_side},
    }
    if kind == "room":
        new_event["center_color"] = center_color
    else:
        new_event["marker_color"] = marker_color
        new_event["marker_offset"] = marker_offset

    connectors = [cell for cells in _event_candidate_connectors(new_event).values() for cell in cells if _in_bounds(cell, shape)]
    seg = _segment_cells(start, end)
    if not _cells_clear(seg, shape, blocked, protected, allowed_protected={start}):
        return None
    if not _cells_clear(box, shape, blocked, protected):
        return None
    if not _cells_clear(connectors, shape, blocked, protected):
        return None

    for r, c in seg:
        canvas[r, c] = GREY
    blocked.update(seg)
    if kind == "room":
        _draw_room(canvas, top_left, center_color)
    else:
        _draw_node(canvas, top_left, marker_color, marker_offset)
    blocked.update(box)
    protected.update(connectors)

    src["connections"].add(side)
    events.append(new_event)
    return len(events) - 1


def _crop_non_black(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BLACK)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    if r1 - r0 + 3 <= 30 and c1 - c0 + 3 <= 30:
        r0 = max(0, r0 - 1)
        c0 = max(0, c0 - 1)
        r1 = min(grid.shape[0] - 1, r1 + 1)
        c1 = min(grid.shape[1] - 1, c1 + 1)

    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _detect_rooms(grid: np.ndarray) -> list[dict[str, Any]]:
    h, w = grid.shape
    rooms: list[dict[str, Any]] = []
    border_offsets = [(dr, dc) for dr in range(3) for dc in range(3) if dr in (0, 2) or dc in (0, 2)]
    for r in range(h - 2):
        for c in range(w - 2):
            if all(int(grid[r + dr, c + dc]) == GREY for dr, dc in border_offsets):
                center_color = int(grid[r + 1, c + 1])
                if center_color == GREY:
                    continue
                outside: dict[str, tuple[int, int]] = {}
                for side, cells in _room_candidate_connectors((r, c)).items():
                    cell = cells[0]
                    if _in_bounds(cell, grid.shape) and int(grid[cell]) == GREY:
                        outside[side] = cell
                rooms.append(
                    {
                        "kind": "room",
                        "top_left": (r, c),
                        "center_color": center_color,
                        "center_pos": (r + 1, c + 1),
                        "outside": outside,
                    }
                )
    return rooms


def _detect_nodes(grid: np.ndarray) -> list[dict[str, Any]]:
    h, w = grid.shape
    nodes: list[dict[str, Any]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            patch = grid[r : r + 2, c : c + 2]
            vals = [int(v) for v in patch.ravel()]
            marker_vals = [v for v in vals if v in (RED, BLUE)]
            if len(marker_vals) != 1:
                continue
            if sum(v == GREY for v in vals) != 3:
                continue
            outside: dict[str, tuple[int, int]] = {}
            for side, candidates in _node_candidate_connectors((r, c)).items():
                hits = [cell for cell in candidates if _in_bounds(cell, grid.shape) and int(grid[cell]) == GREY]
                if hits:
                    outside[side] = hits[0]
            nodes.append(
                {
                    "kind": "node",
                    "top_left": (r, c),
                    "marker_color": marker_vals[0],
                    "outside": outside,
                }
            )
    return nodes


def _build_event_map(
    shape: tuple[int, int],
    rooms: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    event_map = np.full(shape, -1, dtype=int)
    events: list[dict[str, Any]] = []

    for room in rooms:
        idx = len(events)
        ev = dict(room)
        ev["id"] = idx
        events.append(ev)
        for cell in _box_cells("room", room["top_left"]):
            event_map[cell] = idx

    for node in nodes:
        idx = len(events)
        ev = dict(node)
        ev["id"] = idx
        ev["sides"] = set(node["outside"].keys())
        events.append(ev)
        for cell in _box_cells("node", node["top_left"]):
            event_map[cell] = idx

    return event_map, events


def _trace_to_event(
    grid: np.ndarray,
    event_map: np.ndarray,
    start: tuple[int, int],
    side: str,
) -> int:
    assert int(grid[start]) == GREY
    assert int(event_map[start]) == -1
    cur = start
    for _ in range(grid.shape[0] * grid.shape[1]):
        nxt = _move(cur, side)
        assert _in_bounds(nxt, grid.shape)
        if int(event_map[nxt]) >= 0:
            return int(event_map[nxt])
        assert int(grid[nxt]) == GREY
        cur = nxt
    raise AssertionError("trace did not reach another event")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the signal color through a grey pipe network and fill the terminal room it reaches.

    Input structure:
    - The background is black.
    - Grey 1-pixel-wide horizontal and vertical pipe segments connect 2x2 grey routing nodes.
    - Each routing node contains exactly one red or blue marker pixel inside the 2x2 block.
    - Pipe endpoints terminate at 3x3 grey rooms whose centers are black, except for one start room whose center already holds the signal color.

    Rule:
    - Start from the unique filled room and follow its attached pipe.
    - Whenever the signal enters a node, use the marker color relative to the entering direction: red prefers a 90-degree right turn, blue prefers a 90-degree left turn.
    - If the preferred turn has no outgoing pipe, continue straight instead; never return along the incoming side.
    - Continue along straight pipe segments until reaching the next node or a terminal room.
    - Fill the reached room center with the same signal color, leaving all pipes and nodes unchanged.

    Key insight:
    - The red/blue marker inside each 2x2 node is a local routing instruction defined relative to the signal's entering direction, so solving requires tracing the network topology rather than reading directions globally.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _detect_rooms(g)
    nodes = _detect_nodes(g)
    assert rooms and nodes

    start_rooms = [room for room in rooms if room["center_color"] not in (BLACK, GREY)]
    assert len(start_rooms) == 1
    start_room = start_rooms[0]
    signal_color = int(start_room["center_color"])
    assert signal_color not in (RED, BLUE)

    event_map, events = _build_event_map(g.shape, rooms, nodes)
    start_sides = list(start_room["outside"].keys())
    assert len(start_sides) == 1
    start_side = start_sides[0]
    current = _trace_to_event(g, event_map, start_room["outside"][start_side], start_side)
    enter_dir = start_side

    out = g.copy()
    seen: set[tuple[int, str]] = set()
    while True:
        event = events[current]
        if event["kind"] == "room":
            cr, cc = event["center_pos"]
            out[cr, cc] = signal_color
            return out

        state = (current, enter_dir)
        assert state not in seen
        seen.add(state)

        marker_color = int(event["marker_color"])
        preferred = RIGHT[enter_dir] if marker_color == RED else LEFT[enter_dir]
        if preferred in event["sides"]:
            exit_dir = preferred
        elif enter_dir in event["sides"]:
            exit_dir = enter_dir
        else:
            choices = [side for side in event["sides"] if side != OPPOSITE[enter_dir]]
            assert len(choices) == 1
            exit_dir = choices[0]

        current = _trace_to_event(g, event_map, event["outside"][exit_dir], exit_dir)
        enter_dir = exit_dir


def _make_modes(num_nodes: int) -> list[str]:
    modes = ["turn", "straight"]
    while len(modes) < num_nodes:
        modes.append(random.choice(["turn", "straight"]))
    random.shuffle(modes)
    return modes


def _make_marker_colors(num_nodes: int) -> list[int]:
    colors = [RED, BLUE]
    while len(colors) < num_nodes:
        colors.append(random.choice([RED, BLUE]))
    random.shuffle(colors)
    return colors


def _decoy_candidates(node: dict[str, Any]) -> list[str]:
    enter_dir = node["enter_dir"]
    marker_color = node["marker_color"]
    preferred = RIGHT[enter_dir] if marker_color == RED else LEFT[enter_dir]
    other_turn = LEFT[enter_dir] if marker_color == RED else RIGHT[enter_dir]
    if node["mode"] == "turn":
        order = [enter_dir, other_turn]
    else:
        order = [other_turn]
    return [side for side in order if side not in node["connections"]]


def _generate_candidate() -> np.ndarray | None:
    canvas = np.full((TEMP_SIZE, TEMP_SIZE), BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    events: list[dict[str, Any]] = []

    start_top_left = (
        TEMP_SIZE // 2 - 1 + random.randint(-3, 3),
        TEMP_SIZE // 2 - 1 + random.randint(-3, 3),
    )
    start_idx = _place_initial_room(
        canvas,
        blocked,
        protected,
        events,
        start_top_left,
        random.choice(SIGNAL_COLORS),
    )
    if start_idx is None:
        return None

    num_nodes = random.randint(2, 3)
    modes = _make_modes(num_nodes)
    marker_colors = _make_marker_colors(num_nodes)

    current_idx = start_idx
    travel_dir = random.choice(list(DIRS))
    path_nodes: list[int] = []
    for i in range(num_nodes):
        next_idx = _connect_new_event(
            canvas,
            blocked,
            protected,
            events,
            current_idx,
            travel_dir,
            "node",
            distance=random.randint(2, 3),
            marker_color=marker_colors[i],
            marker_offset=random.choice(NODE_MARKER_OFFSETS),
        )
        if next_idx is None:
            return None

        node = events[next_idx]
        node["mode"] = modes[i]
        node["enter_dir"] = travel_dir
        node["marker_color"] = marker_colors[i]

        if modes[i] == "turn":
            travel_dir = RIGHT[travel_dir] if marker_colors[i] == RED else LEFT[travel_dir]
        node["main_exit"] = travel_dir if modes[i] == "straight" else travel_dir

        current_idx = next_idx
        path_nodes.append(next_idx)

    end_idx = _connect_new_event(
        canvas,
        blocked,
        protected,
        events,
        current_idx,
        travel_dir,
        "room",
        distance=random.randint(2, 3),
        center_color=BLACK,
    )
    if end_idx is None:
        return None

    random.shuffle(path_nodes)
    for node_idx in path_nodes:
        node = events[node_idx]
        if random.random() < 0.3:
            continue
        for side in _decoy_candidates(node):
            branch_idx = _connect_new_event(
                canvas,
                blocked,
                protected,
                events,
                node_idx,
                side,
                "room",
                distance=random.randint(2, 3),
                center_color=BLACK,
            )
            if branch_idx is not None:
                break

    cropped = _crop_non_black(canvas)
    if max(cropped.shape) > 30:
        return None
    return cropped


def _count_filled_rooms(grid: np.ndarray) -> int:
    return sum(room["center_color"] not in (BLACK, GREY) for room in _detect_rooms(grid))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if grid.shape != out.shape:
            continue
        if _count_filled_rooms(grid) != 1:
            continue
        if _count_filled_rooms(out) != 2:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
