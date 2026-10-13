import random
from collections import Counter, defaultdict

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _paint_l(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r1, c1 = start
    r2, c2 = end
    c_lo, c_hi = sorted((c1, c2))
    out[r1, c_lo : c_hi + 1] = color
    r_lo, r_hi = sorted((r1, r2))
    out[r_lo : r_hi + 1, c2] = color


def _other_room(door: dict, room_id: int) -> int:
    a, b = door["rooms"]
    return b if a == room_id else a


def _door_anchor(door: dict, room_id: int) -> tuple[int, int]:
    r, c = door["gap"]
    side = door["sides"][room_id]
    if side == "top":
        return (r + 1, c)
    if side == "bottom":
        return (r - 1, c)
    if side == "left":
        return (r, c + 1)
    return (r, c - 1)


def _room_center(room: dict) -> tuple[int, int]:
    return ((room["top"] + room["bottom"]) // 2, (room["left"] + room["right"]) // 2)


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected non-background structure"
    wall = max(colors, key=lambda c: int(np.count_nonzero(g == c)))

    row_threshold = max(2, (7 * w + 9) // 10)
    col_threshold = max(2, (7 * h + 9) // 10)
    row_lines = [r for r in range(h) if int(np.count_nonzero(g[r] != bg)) >= row_threshold]
    col_lines = [c for c in range(w) if int(np.count_nonzero(g[:, c] != bg)) >= col_threshold]
    assert len(row_lines) >= 2 and len(col_lines) >= 2
    assert row_lines[0] == 0 and row_lines[-1] == h - 1
    assert col_lines[0] == 0 and col_lines[-1] == w - 1

    room_ids: dict[tuple[int, int], int] = {}
    rooms: list[dict] = []
    room_of = -np.ones((h, w), dtype=int)
    for ri in range(len(row_lines) - 1):
        top = row_lines[ri]
        bottom = row_lines[ri + 1]
        assert bottom - top >= 2
        for ci in range(len(col_lines) - 1):
            left = col_lines[ci]
            right = col_lines[ci + 1]
            assert right - left >= 2
            room_id = len(rooms)
            room_ids[(ri, ci)] = room_id
            rooms.append(
                {
                    "id": room_id,
                    "ri": ri,
                    "ci": ci,
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                }
            )
            room_of[top + 1 : bottom, left + 1 : right] = room_id

    doors: list[dict] = []
    room_to_doors: dict[int, list[int]] = defaultdict(list)
    marker_colors: set[int] = set()

    for ri in range(len(row_lines) - 2):
        wall_row = row_lines[ri + 1]
        for ci in range(len(col_lines) - 1):
            left = col_lines[ci]
            right = col_lines[ci + 1]
            room_a = room_ids[(ri, ci)]
            room_b = room_ids[(ri + 1, ci)]
            for c in range(left + 2, right - 1):
                if int(g[wall_row, c]) != bg:
                    continue
                marker = int(g[wall_row, c - 1])
                if marker in (bg, wall):
                    continue
                if int(g[wall_row, c + 1]) != marker:
                    continue
                door_id = len(doors)
                door = {
                    "id": door_id,
                    "rooms": (room_a, room_b),
                    "color": marker,
                    "gap": (wall_row, c),
                    "sides": {room_a: "bottom", room_b: "top"},
                }
                doors.append(door)
                room_to_doors[room_a].append(door_id)
                room_to_doors[room_b].append(door_id)
                marker_colors.add(marker)

    for ri in range(len(row_lines) - 1):
        top = row_lines[ri]
        bottom = row_lines[ri + 1]
        for ci in range(len(col_lines) - 2):
            wall_col = col_lines[ci + 1]
            room_a = room_ids[(ri, ci)]
            room_b = room_ids[(ri, ci + 1)]
            for r in range(top + 2, bottom - 1):
                if int(g[r, wall_col]) != bg:
                    continue
                marker = int(g[r - 1, wall_col])
                if marker in (bg, wall):
                    continue
                if int(g[r + 1, wall_col]) != marker:
                    continue
                door_id = len(doors)
                door = {
                    "id": door_id,
                    "rooms": (room_a, room_b),
                    "color": marker,
                    "gap": (r, wall_col),
                    "sides": {room_a: "right", room_b: "left"},
                }
                doors.append(door)
                room_to_doors[room_a].append(door_id)
                room_to_doors[room_b].append(door_id)
                marker_colors.add(marker)

    sources: list[dict] = []
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value in marker_colors or value in (bg, wall):
                continue
            room_id = int(room_of[r, c])
            if room_id < 0:
                continue
            sources.append({"pos": (r, c), "room": room_id, "color": value})

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "rooms": rooms,
        "room_of": room_of,
        "doors": doors,
        "room_to_doors": {room["id"]: list(room_to_doors.get(room["id"], [])) for room in rooms},
        "marker_colors": marker_colors,
        "sources": sources,
    }


def _paths_from_state(parsed: dict, room_id: int, entry_door_id: int, seen: set[tuple[int, int]]) -> list[list[int]]:
    state = (room_id, entry_door_id)
    if state in seen:
        return []

    entry_color = parsed["doors"][entry_door_id]["color"]
    exits = [
        door_id
        for door_id in parsed["room_to_doors"][room_id]
        if door_id != entry_door_id and parsed["doors"][door_id]["color"] != entry_color
    ]
    if not exits:
        return [[]]

    paths: list[list[int]] = []
    next_seen = seen | {state}
    for exit_door_id in exits:
        next_room = _other_room(parsed["doors"][exit_door_id], room_id)
        for suffix in _paths_from_state(parsed, next_room, exit_door_id, next_seen):
            paths.append([exit_door_id] + suffix)
    return paths


def _find_unique_path(parsed: dict, source_room: int) -> list[int]:
    paths: list[list[int]] = []
    for start_door_id in parsed["room_to_doors"][source_room]:
        next_room = _other_room(parsed["doors"][start_door_id], source_room)
        for suffix in _paths_from_state(parsed, next_room, start_door_id, set()):
            paths.append([start_door_id] + suffix)
    assert len(paths) == 1, "expected a unique room path from the source"
    return paths[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the unique door-color path through the room graph and draw the source-colored pipe.
    """
    parsed = _parse_instance(grid)
    out = np.array(parsed["grid"], copy=True)

    for source in parsed["sources"]:
        source_room = source["room"]
        path = _find_unique_path(parsed, source_room)
        if not path:
            continue

        room_chain = [source_room]
        current_room = source_room
        for door_id in path:
            current_room = _other_room(parsed["doors"][door_id], current_room)
            room_chain.append(current_room)

        color = source["color"]
        start_anchor = _door_anchor(parsed["doors"][path[0]], source_room)
        _paint_l(out, source["pos"], start_anchor, color)

        for door_id in path:
            gap_r, gap_c = parsed["doors"][door_id]["gap"]
            out[gap_r, gap_c] = color

        for idx in range(1, len(room_chain) - 1):
            room_id = room_chain[idx]
            entry_anchor = _door_anchor(parsed["doors"][path[idx - 1]], room_id)
            exit_anchor = _door_anchor(parsed["doors"][path[idx]], room_id)
            _paint_l(out, entry_anchor, exit_anchor, color)

        terminal_room = room_chain[-1]
        terminal_entry = _door_anchor(parsed["doors"][path[-1]], terminal_room)
        terminal_center = _room_center(parsed["rooms"][terminal_room])
        _paint_l(out, terminal_entry, terminal_center, color)

    return out


_GEN_ATTEMPTS = 400
_ROOM_SIZE_OPTIONS = (3, 5, 7)
_COLOR_IDS = tuple(range(10))


def _snake_order(n_room_rows: int, n_room_cols: int, by_rows: bool) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    if by_rows:
        for ri in range(n_room_rows):
            cols = range(n_room_cols) if ri % 2 == 0 else range(n_room_cols - 1, -1, -1)
            for ci in cols:
                order.append((ri, ci))
    else:
        for ci in range(n_room_cols):
            rows = range(n_room_rows) if ci % 2 == 0 else range(n_room_rows - 1, -1, -1)
            for ri in rows:
                order.append((ri, ci))
    return order


def _pick_layout(rng: random.Random) -> tuple[int, int]:
    layouts = []
    for n_room_rows in range(1, 5):
        for n_room_cols in range(1, 5):
            rooms = n_room_rows * n_room_cols
            if 2 <= rooms <= 10:
                layouts.append((n_room_rows, n_room_cols))
    return rng.choice(layouts)


def _sample_sizes(rng: random.Random, count: int, limit: int) -> list[int] | None:
    for _ in range(80):
        sizes = [rng.choice(_ROOM_SIZE_OPTIONS) for _ in range(count)]
        if sum(sizes) + count + 1 <= limit:
            return sizes
    return None


def _choose_door_colors(rng: random.Random, edge_count: int, palette: list[int]) -> list[int]:
    if edge_count == 1:
        return [rng.choice(palette)]
    usable = palette[: max(2, min(len(palette), rng.randint(2, min(4, len(palette)))))]
    colors = [rng.choice(usable)]
    for _ in range(edge_count - 1):
        options = [color for color in usable if color != colors[-1]]
        colors.append(rng.choice(options))
    return colors


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    n_room_rows, n_room_cols = _pick_layout(rng)
    heights = _sample_sizes(rng, n_room_rows, 31)
    widths = _sample_sizes(rng, n_room_cols, 31)
    if heights is None or widths is None:
        return None

    h = sum(heights) + n_room_rows + 1
    w = sum(widths) + n_room_cols + 1
    if h > 31 or w > 31:
        return None

    colors = list(_COLOR_IDS)
    rng.shuffle(colors)
    bg, wall = colors[:2]
    remaining = colors[2:]
    if len(remaining) < 3:
        return None

    by_rows = bool(rng.randint(0, 1))
    path_rooms = _snake_order(n_room_rows, n_room_cols, by_rows)
    if rng.random() < 0.5:
        path_rooms = list(reversed(path_rooms))
    edge_count = len(path_rooms) - 1
    door_colors = _choose_door_colors(rng, edge_count, remaining)
    source_color_choices = [color for color in remaining if color not in set(door_colors)]
    if not source_color_choices:
        return None
    source_color = rng.choice(source_color_choices)

    grid = np.full((h, w), bg, dtype=int)

    row_lines = [0]
    acc = 0
    for size in heights:
        acc += size + 1
        row_lines.append(acc)

    col_lines = [0]
    acc = 0
    for size in widths:
        acc += size + 1
        col_lines.append(acc)

    grid[row_lines, :] = wall
    grid[:, col_lines] = wall

    room_meta = {}
    for ri in range(n_room_rows):
        for ci in range(n_room_cols):
            room_meta[(ri, ci)] = {
                "top": row_lines[ri],
                "bottom": row_lines[ri + 1],
                "left": col_lines[ci],
                "right": col_lines[ci + 1],
            }

    source_room = path_rooms[0]
    source_info = room_meta[source_room]

    for idx, color in enumerate(door_colors):
        a = path_rooms[idx]
        b = path_rooms[idx + 1]
        if a[0] == b[0]:
            ri = a[0]
            ci = min(a[1], b[1])
            wall_col = col_lines[ci + 1]
            top = row_lines[ri]
            bottom = row_lines[ri + 1]
            gap_r = rng.randint(top + 2, bottom - 2)
            grid[gap_r, wall_col] = bg
            grid[gap_r - 1, wall_col] = color
            grid[gap_r + 1, wall_col] = color
        else:
            ri = min(a[0], b[0])
            ci = a[1]
            wall_row = row_lines[ri + 1]
            left = col_lines[ci]
            right = col_lines[ci + 1]
            gap_c = rng.randint(left + 2, right - 2)
            grid[wall_row, gap_c] = bg
            grid[wall_row, gap_c - 1] = color
            grid[wall_row, gap_c + 1] = color

    source_cells = []
    for r in range(source_info["top"] + 1, source_info["bottom"]):
        for c in range(source_info["left"] + 1, source_info["right"]):
            source_cells.append((r, c))
    if not source_cells:
        return None
    source_r, source_c = rng.choice(source_cells)
    grid[source_r, source_c] = source_color

    if int(np.count_nonzero(grid == bg)) <= int(np.count_nonzero(grid == wall)):
        return None

    parsed = _parse_instance(grid)
    if len(parsed["rooms"]) != n_room_rows * n_room_cols:
        return None
    if len(parsed["sources"]) != 1:
        return None
    if any(len(parsed["room_to_doors"][room["id"]]) == 0 for room in parsed["rooms"]):
        return None
    if len(parsed["doors"]) != edge_count:
        return None

    path = _find_unique_path(parsed, parsed["sources"][0]["room"])
    if len(path) != edge_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle")
