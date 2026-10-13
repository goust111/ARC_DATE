import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PAYLOAD_COLORS = (1, 3, 4, 6, 7, 8, 9)

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}

_GEN_RETRIES = 300
_BASE_SHAPES = [
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (2, 0)}),
]


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate(shape):
    return _normalize({(c, -r) for r, c in shape})


def _reflect(shape):
    return _normalize({(r, -c) for r, c in shape})


def _shape_variants():
    out = set()
    for base in _BASE_SHAPES:
        cur = base
        for _ in range(4):
            out.add(_normalize(cur))
            out.add(_normalize(_reflect(cur)))
            cur = _rotate(cur)
    return tuple(sorted(out, key=lambda shp: (len(shp), max(r for r, _ in shp), max(c for _, c in shp), tuple(sorted(shp)))))


PAYLOAD_SHAPES = _shape_variants()


def _extract_spans(divider_mask):
    spans = []
    i = 0
    n = len(divider_mask)
    while i < n:
        if divider_mask[i]:
            j = i
            while j < n and divider_mask[j]:
                j += 1
            assert j == i + 1, "room dividers must be one cell thick"
            i = j
            continue
        j = i
        while j < n and not divider_mask[j]:
            j += 1
        spans.append((i, j - 1))
        i = j
    assert spans, "expected at least one room span"
    lengths = {end - start + 1 for start, end in spans}
    assert len(lengths) == 1, "rooms must have equal size"
    return spans


def _room_origins(grid):
    row_spans = _extract_spans(np.all(grid == GREY, axis=1))
    col_spans = _extract_spans(np.all(grid == GREY, axis=0))
    return row_spans, col_spans


def _room_slice(grid, row_spans, col_spans, rr, cc):
    r0, r1 = row_spans[rr]
    c0, c1 = col_spans[cc]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _parse_arrow(room):
    coords = np.argwhere(room == RED)
    if coords.size == 0:
        return None
    assert len(coords) == 5, "arrow must contain exactly five red pixels"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0) == 2 and int(c1 - c0) == 2, "arrow bbox must be 3x3"
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    for name, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return name
    raise AssertionError("unrecognized arrow shape")


def _parse_payload(room):
    payload = []
    for r in range(room.shape[0]):
        for c in range(room.shape[1]):
            value = int(room[r, c])
            if value not in (BLACK, RED, GREY):
                payload.append((r, c, value))
    return payload


def _parse_scene(grid):
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _room_origins(g)
    arrows = {}
    payloads = {}
    for rr in range(len(row_spans)):
        for cc in range(len(col_spans)):
            room = _room_slice(g, row_spans, col_spans, rr, cc)
            arrow = _parse_arrow(room)
            payload = _parse_payload(room)
            if arrow is not None:
                arrows[(rr, cc)] = arrow
            if payload:
                payloads[(rr, cc)] = payload
    return row_spans, col_spans, arrows, payloads


def _step_room(room, direction, room_rows, room_cols):
    dr, dc = DIR_VECS[direction]
    nr = room[0] + dr
    nc = room[1] + dc
    if 0 <= nr < room_rows and 0 <= nc < room_cols:
        return nr, nc
    return None


def _follow_chain(start, arrows, room_rows, room_cols):
    current = start
    seen = {start}
    while current in arrows:
        nxt = _step_room(current, arrows[current], room_rows, room_cols)
        if nxt is None or nxt in seen:
            break
        current = nxt
        seen.add(current)
    return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the arrows and carry each room's payload along its red arrow chain into the final destination room.

    Input structure:
    - The background is black and one-cell-thick grey rows and columns partition the canvas into equal rectangular rooms.
    - A room may contain one red 5-pixel arrow, drawn as a 3-cell shaft plus a 2-pixel V-shaped head, pointing up, down, left, or right.
    - A room may also contain a non-black, non-grey, non-red payload pattern at arbitrary local coordinates inside that room.

    Rule:
    - Every room that contains both an arrow and a payload sends that payload along the arrow direction into the adjacent room while preserving its room-local coordinates.
    - If the entered room also contains an arrow, keep following that arrow chain; if the current arrow points outside the room layout, stop in that boundary room.
    - Rooms that have payloads but no arrow keep their payloads, all arrows are removed, and overlapping arriving payloads are painted in room scan order so later source rooms overwrite earlier ones.

    Key insight:
    - Arrow-only rooms are relay stations, so the true destination is the end of the whole arrow chain rather than the immediate neighbor, and the payload keeps the same local offset in every room it traverses.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, arrows, payloads = _parse_scene(g)
    room_rows = len(row_spans)
    room_cols = len(col_spans)

    out = g.copy()
    out[out == RED] = BLACK

    source_rooms = []
    for room, payload in payloads.items():
        if room in arrows:
            source_rooms.append(room)
            r0, _ = row_spans[room[0]]
            c0, _ = col_spans[room[1]]
            for lr, lc, _color in payload:
                out[r0 + lr, c0 + lc] = BLACK

    for room in sorted(source_rooms):
        dest = _follow_chain(room, arrows, room_rows, room_cols)
        dst_r0 = row_spans[dest[0]][0]
        dst_c0 = col_spans[dest[1]][0]
        for lr, lc, color in payloads[room]:
            out[dst_r0 + lr, dst_c0 + lc] = int(color)

    return out


def _neighbors(room, room_rows, room_cols):
    rr, cc = room
    result = []
    for dr, dc in DIR_VECS.values():
        nr = rr + dr
        nc = cc + dc
        if 0 <= nr < room_rows and 0 <= nc < room_cols:
            result.append((nr, nc))
    return result


def _bfs_path(start, goal, room_rows, room_cols, blocked, rng):
    queue = deque([start])
    prev = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        nbrs = _neighbors(current, room_rows, room_cols)
        rng.shuffle(nbrs)
        for nxt in nbrs:
            if nxt in prev:
                continue
            if nxt in blocked and nxt != goal:
                continue
            prev[nxt] = current
            queue.append(nxt)
    if goal not in prev:
        return None
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path


def _choose_path(rng, room_rows, room_cols, starts, goals, blocked, min_len, max_len):
    starts = list(starts)
    goals = list(goals)
    rng.shuffle(starts)
    rng.shuffle(goals)
    for start in starts:
        for goal in goals:
            if start == goal:
                continue
            path = _bfs_path(start, goal, room_rows, room_cols, blocked, rng)
            if path is None:
                continue
            if min_len <= len(path) <= max_len:
                return path
    return None


def _dir_from_step(src, dst):
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for name, vec in DIR_VECS.items():
        if vec == (dr, dc):
            return name
    raise AssertionError("rooms must be adjacent")


def _assign_path_arrows(path, arrows):
    for src, dst in zip(path, path[1:]):
        arrows[src] = _dir_from_step(src, dst)


def _edge_rooms(room_rows, room_cols):
    out = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            if rr in (0, room_rows - 1) or cc in (0, room_cols - 1):
                out.append((rr, cc))
    return out


def _outward_directions(room, room_rows, room_cols):
    rr, cc = room
    out = []
    if rr == 0:
        out.append("up")
    if rr == room_rows - 1:
        out.append("down")
    if cc == 0:
        out.append("left")
    if cc == room_cols - 1:
        out.append("right")
    return out


def _random_arrow_box(room_h, room_w, rng):
    return rng.randint(0, room_h - 3), rng.randint(0, room_w - 3)


def _arrow_cells(top, left, direction):
    return {(top + r, left + c) for r, c in ARROW_PATTERNS[direction]}


def _all_local_cells(room_h, room_w):
    return {(r, c) for r in range(room_h) for c in range(room_w)}


def _place_shape(allowed, room_h, room_w, rng, must_touch=None, require_partial_overlap=False):
    candidates = []
    for shape in PAYLOAD_SHAPES:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        for dr in range(room_h - max_r):
            for dc in range(room_w - max_c):
                placed = frozenset((r + dr, c + dc) for r, c in shape)
                if not placed <= allowed:
                    continue
                if must_touch is not None:
                    overlap = placed & must_touch
                    if not overlap:
                        continue
                    if require_partial_overlap:
                        if len(overlap) == len(placed) or len(overlap) == len(must_touch):
                            continue
                candidates.append(placed)
    if not candidates:
        return None
    return set(rng.choice(candidates))


def _room_origin(rr, cc, room_h, room_w):
    return 1 + rr * (room_h + 1), 1 + cc * (room_w + 1)


def _build_grid(room_rows, room_cols, room_h, room_w, arrows, arrow_boxes, payloads):
    height = room_rows * room_h + room_rows + 1
    width = room_cols * room_w + room_cols + 1
    grid = np.full((height, width), BLACK, dtype=int)
    for r in range(0, height, room_h + 1):
        grid[r, :] = GREY
    for c in range(0, width, room_w + 1):
        grid[:, c] = GREY

    for room, (color, cells) in payloads.items():
        r0, c0 = _room_origin(room[0], room[1], room_h, room_w)
        for lr, lc in cells:
            grid[r0 + lr, c0 + lc] = int(color)

    for room, direction in arrows.items():
        top, left = arrow_boxes[room]
        r0, c0 = _room_origin(room[0], room[1], room_h, room_w)
        for lr, lc in _arrow_cells(top, left, direction):
            assert grid[r0 + lr, c0 + lc] == BLACK
            grid[r0 + lr, c0 + lc] = RED
    return grid


def _validate_case(grid, room_rows, room_cols, arrows, payloads, want_collision, collision_pair, want_boundary, boundary_terminal, stationary_room, decoy_room):
    row_spans, col_spans, parsed_arrows, parsed_payloads = _parse_scene(grid)
    if len(row_spans) != room_rows or len(col_spans) != room_cols:
        return False
    if parsed_arrows != arrows:
        return False
    expected_payloads = {
        room: sorted((r, c, color) for r, c in cells)
        for room, (color, cells) in payloads.items()
    }
    actual_payloads = {
        room: sorted((int(r), int(c), int(color)) for r, c, color in cells)
        for room, cells in parsed_payloads.items()
    }
    if expected_payloads != actual_payloads:
        return False

    sources = [room for room in payloads if room in arrows]
    if not sources:
        return False
    if stationary_room in {_follow_chain(room, arrows, room_rows, room_cols) for room in sources}:
        return False
    if decoy_room in sources:
        return False
    if want_collision:
        if collision_pair is None:
            return False
        a, b = collision_pair
        if _follow_chain(a, arrows, room_rows, room_cols) != _follow_chain(b, arrows, room_rows, room_cols):
            return False
        if not (payloads[a][1] & payloads[b][1]):
            return False
    if want_boundary:
        if boundary_terminal is None or boundary_terminal not in arrows:
            return False
        if _step_room(boundary_terminal, arrows[boundary_terminal], room_rows, room_cols) is not None:
            return False
    out = main(grid)
    if out.shape != grid.shape or np.any(out == RED):
        return False
    return True


def _gen_attempt(rng):
    room_rows = rng.randint(3, 4)
    room_cols = rng.randint(3, 4)
    room_h = rng.randint(5, 7)
    room_w = rng.randint(5, 7)
    if room_rows * room_h + room_rows + 1 > 30 or room_cols * room_w + room_cols + 1 > 30:
        return None

    rooms = [(rr, cc) for rr in range(room_rows) for cc in range(room_cols)]
    arrows = {}
    used = set()
    source_rooms = []
    terminal_rooms = set()

    base_candidates = [room for room in rooms]
    base_path = _choose_path(
        rng,
        room_rows,
        room_cols,
        base_candidates,
        base_candidates,
        blocked=set(),
        min_len=3,
        max_len=4,
    )
    if base_path is None:
        return None
    _assign_path_arrows(base_path, arrows)
    used.update(base_path)
    source_rooms.append(base_path[0])
    terminal_rooms.add(base_path[-1])

    want_collision = rng.random() < 0.5
    collision_pair = None
    if want_collision:
        joins = list(base_path[1:])
        rng.shuffle(joins)
        for join in joins:
            free_starts = [room for room in rooms if room not in used]
            path = _choose_path(
                rng,
                room_rows,
                room_cols,
                free_starts,
                [join],
                blocked=used - {join},
                min_len=2,
                max_len=4,
            )
            if path is None:
                continue
            _assign_path_arrows(path, arrows)
            used.update(path[:-1])
            source_rooms.append(path[0])
            collision_pair = (base_path[0], path[0])
            break
        if collision_pair is None:
            return None

    want_boundary = rng.random() < 0.5
    boundary_terminal = None
    if want_boundary:
        edge_choices = [room for room in _edge_rooms(room_rows, room_cols) if room not in used]
        rng.shuffle(edge_choices)
        for edge_room in edge_choices:
            free_starts = [room for room in rooms if room not in used and room != edge_room]
            path = _choose_path(
                rng,
                room_rows,
                room_cols,
                free_starts,
                [edge_room],
                blocked=used - {edge_room},
                min_len=2,
                max_len=4,
            )
            if path is None:
                continue
            _assign_path_arrows(path, arrows)
            arrows[edge_room] = rng.choice(_outward_directions(edge_room, room_rows, room_cols))
            used.update(path)
            source_rooms.append(path[0])
            terminal_rooms.add(edge_room)
            boundary_terminal = edge_room
            break
        if boundary_terminal is None:
            return None

    stationary_candidates = [room for room in rooms if room not in used]
    if not stationary_candidates:
        return None
    stationary_room = rng.choice(stationary_candidates)

    decoy_candidates = [room for room in rooms if room not in used and room != stationary_room and room not in arrows]
    if not decoy_candidates:
        return None
    decoy_room = rng.choice(decoy_candidates)
    arrows[decoy_room] = rng.choice(tuple(DIR_VECS))

    extra_source_candidates = [room for room in arrows if room not in source_rooms and room != decoy_room]
    if extra_source_candidates and rng.random() < 0.4:
        source_rooms.append(rng.choice(extra_source_candidates))

    payload_rooms = set(source_rooms)
    payload_rooms.add(stationary_room)

    no_arrow_terminals = [room for room in terminal_rooms if room not in arrows]
    if no_arrow_terminals and rng.random() < 0.5:
        payload_rooms.add(rng.choice(no_arrow_terminals))

    extra_payload_candidates = [
        room
        for room in rooms
        if room not in arrows and room not in payload_rooms and room not in terminal_rooms and rng.random() < 0.3
    ]
    for room in extra_payload_candidates[:2]:
        payload_rooms.add(room)

    arrow_boxes = {room: _random_arrow_box(room_h, room_w, rng) for room in arrows}
    allowed_by_room = {}
    all_cells = _all_local_cells(room_h, room_w)
    for room in rooms:
        forbidden = set()
        if room in arrows:
            forbidden = _arrow_cells(*arrow_boxes[room], arrows[room])
        allowed_by_room[room] = all_cells - forbidden
        if len(allowed_by_room[room]) < 2:
            return None

    payloads = {}
    assigned_collision = set()
    if collision_pair is not None:
        a, b = collision_pair
        common = allowed_by_room[a] & allowed_by_room[b]
        if not common:
            return None
        for _ in range(60):
            cells_a = _place_shape(allowed_by_room[a], room_h, room_w, rng, must_touch=common)
            if cells_a is None:
                continue
            cells_b = _place_shape(allowed_by_room[b], room_h, room_w, rng, must_touch=cells_a, require_partial_overlap=True)
            if cells_b is None:
                continue
            color_a, color_b = rng.sample(PAYLOAD_COLORS, 2)
            payloads[a] = (color_a, cells_a)
            payloads[b] = (color_b, cells_b)
            assigned_collision = {a, b}
            break
        if assigned_collision != {a, b}:
            return None

    for room in sorted(payload_rooms):
        if room in payloads:
            continue
        cells = _place_shape(allowed_by_room[room], room_h, room_w, rng)
        if cells is None:
            return None
        payloads[room] = (rng.choice(PAYLOAD_COLORS), cells)

    grid = _build_grid(room_rows, room_cols, room_h, room_w, arrows, arrow_boxes, payloads)
    if not _validate_case(
        grid,
        room_rows,
        room_cols,
        arrows,
        payloads,
        want_collision,
        collision_pair,
        want_boundary,
        boundary_terminal,
        stationary_room,
        decoy_room,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-arrow payload puzzle")
