import random

import numpy as np


BG = 0
WALL = 1
MARKER_COLORS = (2, 3, 4, 6)

# --------------- solver ---------------


def _group_consecutive(indexes: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    if not indexes:
        return groups
    start = prev = indexes[0]
    for value in indexes[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _grid_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], int, int]:
    row_groups = _group_consecutive(np.flatnonzero(np.all(grid == WALL, axis=1)).tolist())
    col_groups = _group_consecutive(np.flatnonzero(np.all(grid == WALL, axis=0)).tolist())
    assert len(row_groups) >= 2 and len(col_groups) >= 2

    h, w = grid.shape
    assert row_groups[0][0] == 0 and row_groups[-1][1] == h - 1
    assert col_groups[0][0] == 0 and col_groups[-1][1] == w - 1

    wall_mask = np.zeros_like(grid, dtype=bool)
    for r0, r1 in row_groups:
        wall_mask[r0 : r1 + 1, :] = True
    for c0, c1 in col_groups:
        wall_mask[:, c0 : c1 + 1] = True
    assert np.array_equal(grid == WALL, wall_mask)

    row_spans = [(row_groups[i][1] + 1, row_groups[i + 1][0] - 1) for i in range(len(row_groups) - 1)]
    col_spans = [(col_groups[i][1] + 1, col_groups[i + 1][0] - 1) for i in range(len(col_groups) - 1)]
    assert all(r0 <= r1 for r0, r1 in row_spans)
    assert all(c0 <= c1 for c0, c1 in col_spans)

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    room_h = row_sizes.pop()
    room_w = col_sizes.pop()
    assert room_h >= 1 and room_w >= 1
    return row_groups, col_groups, row_spans, col_spans, room_h, room_w


def _classify_room(patch: np.ndarray) -> dict | None:
    coords = np.argwhere((patch != BG) & (patch != WALL))
    if len(coords) == 0:
        return None

    colors = {int(patch[int(r), int(c)]) for r, c in coords}
    assert len(colors) == 1
    color = colors.pop()
    assert color in MARKER_COLORS or color not in (BG, WALL)

    if len(coords) == 1:
        return {"kind": "lock", "color": color, "coords": [(int(coords[0, 0]), int(coords[0, 1]))]}

    assert len(coords) == 2
    (r0, c0), (r1, c1) = coords.tolist()
    assert abs(r0 - r1) + abs(c0 - c1) == 1
    return {"kind": "key", "color": color, "coords": [(int(r0), int(c0)), (int(r1), int(c1))]}


def _opening_cell(
    room_a: tuple[int, int],
    room_b: tuple[int, int],
    row_groups: list[tuple[int, int]],
    col_groups: list[tuple[int, int]],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    ra, ca = room_a
    rb, cb = room_b
    if ra == rb:
        left = min(ca, cb)
        wall_c = col_groups[left + 1][0]
        room_r0, room_r1 = row_spans[ra]
        wall_r = room_r0 + (room_r1 - room_r0) // 2
        return wall_r, wall_c

    assert ca == cb
    top = min(ra, rb)
    wall_r = row_groups[top + 1][0]
    room_c0, room_c1 = col_spans[ca]
    wall_c = room_c0 + (room_c1 - room_c0) // 2
    return wall_r, wall_c


def main(grid: np.ndarray) -> np.ndarray:
    """Open doorways between adjacent rooms when a key touches a same-colored neighboring lock."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_groups, col_groups, row_spans, col_spans, _room_h, _room_w = _grid_structure(g)
    room_rows = len(row_spans)
    room_cols = len(col_spans)

    rooms: list[list[dict | None]] = []
    for ri, (r0, r1) in enumerate(row_spans):
        row: list[dict | None] = []
        for ci, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            info = _classify_room(patch)
            if info is not None:
                info = info.copy()
                info["room"] = (ri, ci)
                info["coords"] = [(r0 + rr, c0 + cc) for rr, cc in info["coords"]]
            row.append(info)
        rooms.append(row)

    openings: set[tuple[int, int]] = set()
    to_remove: set[tuple[int, int]] = set()
    for ri in range(room_rows):
        for ci in range(room_cols):
            info = rooms[ri][ci]
            if info is None or info["kind"] != "key":
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = ri + dr, ci + dc
                if not (0 <= nr < room_rows and 0 <= nc < room_cols):
                    continue
                other = rooms[nr][nc]
                if other is None or other["kind"] != "lock" or other["color"] != info["color"]:
                    continue
                openings.add(_opening_cell((ri, ci), (nr, nc), row_groups, col_groups, row_spans, col_spans))
                to_remove.update(info["coords"])
                to_remove.update(other["coords"])

    out = g.copy()
    for r, c in openings:
        out[int(r), int(c)] = BG
    for r, c in to_remove:
        out[int(r), int(c)] = BG
    return out


# --------------- generator ---------------

ROOM_HEIGHTS = (3, 4, 5)
ROOM_WIDTHS = (3, 4, 5)
ROOM_ROW_COUNTS = (2, 3, 4, 5)
ROOM_COL_COUNTS = (2, 3, 4, 5)
MAX_GRID_SIDE = 30
EXTRA_MODES = ("matched_only", "key_only", "lock_only", "both")
GEN_ATTEMPTS = 400

LAYOUTS = [
    (room_rows, room_cols, room_h, room_w)
    for room_rows in ROOM_ROW_COUNTS
    for room_cols in ROOM_COL_COUNTS
    for room_h in ROOM_HEIGHTS
    for room_w in ROOM_WIDTHS
    if room_rows * room_h + room_rows + 1 <= MAX_GRID_SIDE
    and room_cols * room_w + room_cols + 1 <= MAX_GRID_SIDE
]


def _room_bounds(room: tuple[int, int], room_h: int, room_w: int) -> tuple[int, int, int, int]:
    rr, cc = room
    r0 = 1 + rr * (room_h + 1)
    c0 = 1 + cc * (room_w + 1)
    return r0, c0, r0 + room_h - 1, c0 + room_w - 1


def _draw_base_grid(room_rows: int, room_cols: int, room_h: int, room_w: int) -> np.ndarray:
    height = room_rows * room_h + room_rows + 1
    width = room_cols * room_w + room_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    for r in range(0, height, room_h + 1):
        grid[r, :] = WALL
    for c in range(0, width, room_w + 1):
        grid[:, c] = WALL
    return grid


def _neighbor_rooms(room: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    rr, cc = room
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < room_rows and 0 <= nc < room_cols:
            out.append((nr, nc))
    return out


def _all_edges(room_rows: int, room_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            edges.append(((rr, cc), (rr, cc + 1)))
    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            edges.append(((rr, cc), (rr + 1, cc)))
    return edges


def _pick_disjoint_edges(room_rows: int, room_cols: int, pair_count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = _all_edges(room_rows, room_cols)
    best: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for _ in range(80):
        chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
        used: set[tuple[int, int]] = set()
        while len(chosen) < pair_count:
            candidates = [edge for edge in edges if edge[0] not in used and edge[1] not in used]
            if not candidates:
                break
            side_by_side = [edge for edge in candidates if edge[0][0] == edge[1][0]]
            stacked = [edge for edge in candidates if edge[0][1] == edge[1][1]]
            if side_by_side and stacked:
                candidates = side_by_side if random.random() < 0.5 else stacked
            edge = random.choice(candidates)
            chosen.append(edge)
            used.update(edge)
        if len(chosen) > len(best):
            best = chosen
        if len(best) == pair_count:
            return best
    return best


def _mode_min_extra(mode: str) -> int:
    return {"matched_only": 0, "key_only": 1, "lock_only": 1, "both": 2}[mode]


def _choose_extra_counts(mode: str, free_count: int) -> tuple[int, int]:
    if mode == "matched_only":
        return 0, 0
    if mode == "key_only":
        return random.randint(1, min(2, free_count)), 0
    if mode == "lock_only":
        return 0, random.randint(1, min(2, free_count))

    key_count = random.randint(1, min(2, free_count - 1))
    lock_count = random.randint(1, min(2, free_count - key_count))
    remaining = free_count - key_count - lock_count
    if remaining > 0 and random.random() < 0.5:
        if random.random() < 0.5:
            key_count += 1
        else:
            lock_count += 1
    return key_count, lock_count


def _allowed_colors(
    objects: dict[tuple[int, int], dict],
    room: tuple[int, int],
    kind: str,
    room_rows: int,
    room_cols: int,
) -> list[int]:
    other_kind = "lock" if kind == "key" else "key"
    forbidden = {
        int(objects[neighbor]["color"])
        for neighbor in _neighbor_rooms(room, room_rows, room_cols)
        if neighbor in objects and objects[neighbor]["kind"] == other_kind
    }
    return [color for color in MARKER_COLORS if color not in forbidden]


def _place_unmatched(
    objects: dict[tuple[int, int], dict],
    free_rooms: list[tuple[int, int]],
    kind: str,
    count: int,
    room_rows: int,
    room_cols: int,
) -> bool:
    for _ in range(count):
        candidates: list[tuple[tuple[int, int], list[int]]] = []
        for room in free_rooms:
            colors = _allowed_colors(objects, room, kind, room_rows, room_cols)
            if colors:
                candidates.append((room, colors))
        if not candidates:
            return False
        room, colors = random.choice(candidates)
        objects[room] = {"kind": kind, "color": random.choice(colors)}
        free_rooms.remove(room)
    return True


def _paint_object(
    grid: np.ndarray,
    room: tuple[int, int],
    room_h: int,
    room_w: int,
    kind: str,
    color: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _room_bounds(room, room_h, room_w)
    if kind == "lock":
        r = random.randint(r0, r1)
        c = random.randint(c0, c1)
        grid[r, c] = color
        return [(r, c)]

    choices: list[tuple[int, int]] = []
    if room_w >= 2:
        choices.append((0, 1))
    if room_h >= 2:
        choices.append((1, 0))
    dr, dc = random.choice(choices)
    if dr == 0:
        r = random.randint(r0, r1)
        c = random.randint(c0, c1 - 1)
    else:
        r = random.randint(r0, r1 - 1)
        c = random.randint(c0, c1)
    grid[r, c] = color
    grid[r + dr, c + dc] = color
    return [(r, c), (r + dr, c + dc)]


def _expected_opening(room_a: tuple[int, int], room_b: tuple[int, int], room_h: int, room_w: int) -> tuple[int, int]:
    ra, ca = room_a
    rb, cb = room_b
    if ra == rb:
        left = min(ca, cb)
        r0, _, _r1, _c1 = _room_bounds((ra, left), room_h, room_w)
        return r0 + (room_h - 1) // 2, (left + 1) * (room_w + 1)
    top = min(ra, rb)
    _, c0, _r1, _c1 = _room_bounds((top, ca), room_h, room_w)
    return (top + 1) * (room_h + 1), c0 + (room_w - 1) // 2


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols, room_h, room_w = random.choice(LAYOUTS)
    total_rooms = room_rows * room_cols
    mode = random.choice(EXTRA_MODES)
    max_pairs = min(len(MARKER_COLORS), (total_rooms - _mode_min_extra(mode)) // 2)
    if max_pairs < 1:
        return None

    pair_count = random.randint(1, max_pairs)
    pair_edges = _pick_disjoint_edges(room_rows, room_cols, pair_count)
    if len(pair_edges) != pair_count:
        return None

    objects: dict[tuple[int, int], dict] = {}
    matched_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for color, edge in zip(random.sample(MARKER_COLORS, pair_count), pair_edges):
        room_a, room_b = edge
        if random.random() < 0.5:
            key_room, lock_room = room_a, room_b
        else:
            key_room, lock_room = room_b, room_a
        objects[key_room] = {"kind": "key", "color": color}
        objects[lock_room] = {"kind": "lock", "color": color}
        matched_pairs.append((key_room, lock_room))

    free_rooms = [(rr, cc) for rr in range(room_rows) for cc in range(room_cols) if (rr, cc) not in objects]
    extra_keys, extra_locks = _choose_extra_counts(mode, len(free_rooms))

    tasks = [("key", extra_keys), ("lock", extra_locks)]
    random.shuffle(tasks)
    for kind, count in tasks:
        if not _place_unmatched(objects, free_rooms, kind, count, room_rows, room_cols):
            return None

    inp = _draw_base_grid(room_rows, room_cols, room_h, room_w)
    for room, info in objects.items():
        info["coords"] = _paint_object(inp, room, room_h, room_w, info["kind"], int(info["color"]))

    out = inp.copy()
    for key_room, lock_room in matched_pairs:
        for room in (key_room, lock_room):
            for r, c in objects[room]["coords"]:
                out[int(r), int(c)] = BG
        wall_r, wall_c = _expected_opening(key_room, lock_room, room_h, room_w)
        out[int(wall_r), int(wall_c)] = BG

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
