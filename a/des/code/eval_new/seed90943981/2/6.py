import random
from collections import Counter

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_FROM_VEC = {v: k for k, v in DIRS.items()}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

WALL_COLOR = 0
GEN_RETRIES = 400
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (5, 6)
EXTRA_DOOR_PROB = 0.45
MAX_PATH_LEN = 6


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_wall_color(grid: np.ndarray, bg: int) -> int:
    if bg != WALL_COLOR and np.any(grid == WALL_COLOR):
        return WALL_COLOR

    h, w = grid.shape
    best_color = None
    best_score = -1
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = grid == color
        row_score = int(np.count_nonzero(mask.sum(axis=1) > w // 2))
        col_score = int(np.count_nonzero(mask.sum(axis=0) > h // 2))
        score = row_score + col_score
        if score > best_score:
            best_score = score
            best_color = color
    if best_color is None:
        raise ValueError("failed to identify wall color")
    return int(best_color)


def _wall_lines(grid: np.ndarray, wall: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    wall_rows = [int(r) for r in range(h) if int(np.count_nonzero(grid[r] == wall)) > w // 2]
    wall_cols = [int(c) for c in range(w) if int(np.count_nonzero(grid[:, c] == wall)) > h // 2]
    if len(wall_rows) < 2 or len(wall_cols) < 2:
        raise ValueError("not enough wall lines")
    return wall_rows, wall_cols


def _spans_from_lines(lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(lines, lines[1:]):
        if b - a >= 2:
            spans.append((int(a + 1), int(b - 1)))
    if not spans:
        raise ValueError("no room spans found")
    return spans


def _boundary_sides(r: int, c: int, r0: int, r1: int, c0: int, c1: int) -> set[str]:
    sides = set()
    if r == r0:
        sides.add("up")
    if r == r1:
        sides.add("down")
    if c == c0:
        sides.add("left")
    if c == c1:
        sides.add("right")
    return sides


def _parse_rooms(grid: np.ndarray) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int], list[tuple[int, int]]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    wall = _find_wall_color(g, bg)
    wall_rows, wall_cols = _wall_lines(g, wall)
    row_spans = _spans_from_lines(wall_rows)
    col_spans = _spans_from_lines(wall_cols)

    rooms: dict[tuple[int, int], dict] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            interior = g[r0 : r1 + 1, c0 : c1 + 1]
            markers: list[tuple[tuple[int, int], int]] = []
            for rr, cc in np.argwhere((interior != bg) & (interior != wall)):
                markers.append(((int(r0 + rr), int(c0 + cc)), int(interior[rr, cc])))

            counts = Counter(color for _, color in markers)
            flag_color = None
            for color, count in counts.items():
                if count != 2:
                    continue
                pts = [pos for pos, col in markers if col == color]
                if abs(pts[0][0] - pts[1][0]) + abs(pts[0][1] - pts[1][1]) == 1:
                    flag_color = int(color)
                    break
            if flag_color is None:
                raise ValueError("room is missing a valid flag")

            flag_pts = [pos for pos, col in markers if col == flag_color]
            p0, p1 = flag_pts
            b0 = _boundary_sides(p0[0], p0[1], r0, r1, c0, c1)
            b1 = _boundary_sides(p1[0], p1[1], r0, r1, c0, c1)
            if bool(b0) == bool(b1):
                raise ValueError("flag pole is ambiguous")
            pole = p0 if b0 else p1
            flag = p1 if pole == p0 else p0
            direction = (int(flag[0] - pole[0]), int(flag[1] - pole[1]))
            if direction not in DIR_FROM_VEC:
                raise ValueError("flag direction must be axial")

            extras = [(pos, col) for pos, col in markers if pos not in flag_pts]
            if len(extras) > 1:
                raise ValueError("room has too many extra markers")
            source_color = int(extras[0][1]) if extras else None

            rooms[(ri, ci)] = {
                "bbox": (r0, r1, c0, c1),
                "exit": DIR_FROM_VEC[direction],
                "source_color": source_color,
            }

    doors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            room = (ri, ci)
            room_doors: dict[str, list[tuple[int, int]]] = {}
            if ri > 0:
                wall_r = wall_rows[ri]
                pts = [(wall_r, c) for c in range(c0, c1 + 1) if int(g[wall_r, c]) != wall]
                room_doors["up"] = pts
            if ri + 1 < len(row_spans):
                wall_r = wall_rows[ri + 1]
                pts = [(wall_r, c) for c in range(c0, c1 + 1) if int(g[wall_r, c]) != wall]
                room_doors["down"] = pts
            if ci > 0:
                wall_c = wall_cols[ci]
                pts = [(r, wall_c) for r in range(r0, r1 + 1) if int(g[r, wall_c]) != wall]
                room_doors["left"] = pts
            if ci + 1 < len(col_spans):
                wall_c = wall_cols[ci + 1]
                pts = [(r, wall_c) for r in range(r0, r1 + 1) if int(g[r, wall_c]) != wall]
                room_doors["right"] = pts
            doors[room] = room_doors

    return rooms, doors, bg


def _neighbor(room: tuple[int, int], direction: str) -> tuple[int, int]:
    ri, ci = room
    dr, dc = DIRS[direction]
    return (ri + dr, ci + dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each room's flag direction to propagate the source color through connected rooms.

    Input structure:
    - The most common color is the background, while black wall pixels partition the grid into rectangular rooms with one-cell door openings.
    - Every room contains a two-pixel monochrome flag: one pole pixel touches a room wall and the adjacent flag pixel points in the room's exit direction.
    - Exactly one room also contains an extra singleton source marker whose color is the liquid color.

    Rule:
    - Start at the unique source room and recolor that room's background cells to the source color.
    - For each reached room, check only the wall side indicated by its flag; if a door exists there and the neighboring room is still unvisited, the liquid crosses that door and recolors the neighboring room as well.
    - Stop when the directed flow points to a closed wall or would re-enter an already colored room; preserve all wall, flag, and source-marker pixels.

    Key insight:
    - The two-pixel flag must first be decoded into a directed exit, then those per-room exits define a directed room-to-room traversal rather than an undirected flood fill.
    """
    g = np.asarray(grid, dtype=int)
    rooms, doors, bg = _parse_rooms(g)

    sources = [room for room, info in rooms.items() if info["source_color"] is not None]
    if len(sources) != 1:
        raise ValueError("expected exactly one source room")
    source_room = sources[0]
    liquid = int(rooms[source_room]["source_color"])

    out = g.copy()
    visited: set[tuple[int, int]] = set()
    current = source_room

    while current not in visited:
        visited.add(current)
        r0, r1, c0, c1 = rooms[current]["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = liquid

        direction = rooms[current]["exit"]
        door_cells = doors[current].get(direction, [])
        if not door_cells:
            break

        nxt = _neighbor(current, direction)
        if nxt not in rooms or nxt in visited:
            break

        for rr, cc in door_cells:
            if out[rr, cc] == bg:
                out[rr, cc] = liquid
        current = nxt

    return out


def _room_neighbors(room: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    ri, ci = room
    out = []
    if ri > 0:
        out.append((ri - 1, ci))
    if ri + 1 < rows:
        out.append((ri + 1, ci))
    if ci > 0:
        out.append((ri, ci - 1))
    if ci + 1 < cols:
        out.append((ri, ci + 1))
    return out


def _random_path(rows: int, cols: int, length: int) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(starts)

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            return list(path)
        options = _room_neighbors(path[-1], rows, cols)
        random.shuffle(options)
        for nxt in options:
            if nxt in used:
                continue
            used.add(nxt)
            path.append(nxt)
            found = dfs(path, used)
            if found is not None:
                return found
            path.pop()
            used.remove(nxt)
        return None

    for start in starts:
        found = dfs([start], {start})
        if found is not None:
            return found
    return None


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    return DIR_FROM_VEC[(dr, dc)]


def _balanced_direction(open_dirs: list[str]) -> str:
    closed_dirs = [d for d in DIRS if d not in open_dirs]
    prefer_open = random.random() < 0.5
    if prefer_open and open_dirs:
        return random.choice(open_dirs)
    if (not prefer_open) and closed_dirs:
        return random.choice(closed_dirs)
    if open_dirs:
        return random.choice(open_dirs)
    return random.choice(closed_dirs)


def _draw_flag(grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str, color: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, r1, c0, c1 = bbox
    if direction == "up":
        c = random.randint(c0 + 1, c1 - 1)
        pole = (r1, c)
        flag = (r1 - 1, c)
    elif direction == "down":
        c = random.randint(c0 + 1, c1 - 1)
        pole = (r0, c)
        flag = (r0 + 1, c)
    elif direction == "left":
        r = random.randint(r0 + 1, r1 - 1)
        pole = (r, c1)
        flag = (r, c1 - 1)
    else:
        r = random.randint(r0 + 1, r1 - 1)
        pole = (r, c0)
        flag = (r, c0 + 1)
    grid[pole] = color
    grid[flag] = color
    return pole, flag


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]

    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    if h > 30 or w > 30:
        return None

    bg = random.choice([c for c in range(1, 10) if c != WALL_COLOR])
    grid = np.full((h, w), bg, dtype=int)

    wall_rows = [0]
    cur = 0
    for size in heights:
        cur += size + 1
        wall_rows.append(cur)
    wall_cols = [0]
    cur = 0
    for size in widths:
        cur += size + 1
        wall_cols.append(cur)

    for r in wall_rows:
        grid[r, :] = WALL_COLOR
    for c in wall_cols:
        grid[:, c] = WALL_COLOR

    row_spans = [(wall_rows[i] + 1, wall_rows[i + 1] - 1) for i in range(room_rows)]
    col_spans = [(wall_cols[i] + 1, wall_cols[i + 1] - 1) for i in range(room_cols)]
    bboxes = {
        (ri, ci): (row_spans[ri][0], row_spans[ri][1], col_spans[ci][0], col_spans[ci][1])
        for ri in range(room_rows)
        for ci in range(room_cols)
    }

    max_len = min(room_rows * room_cols, MAX_PATH_LEN)
    path_len = random.randint(1, max_len)
    path = _random_path(room_rows, room_cols, path_len)
    if path is None:
        return None
    required_edges = {_edge_key(a, b) for a, b in zip(path, path[1:])}

    doors: dict[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]] = {}
    open_dirs = {(ri, ci): [] for ri in range(room_rows) for ci in range(room_cols)}

    for ri in range(room_rows):
        for ci in range(room_cols - 1):
            a = (ri, ci)
            b = (ri, ci + 1)
            edge = _edge_key(a, b)
            if edge in required_edges or random.random() < EXTRA_DOOR_PROB:
                r0, r1 = row_spans[ri]
                wall_c = wall_cols[ci + 1]
                door = (random.randint(r0, r1), wall_c)
                grid[door] = bg
                doors[edge] = door
                open_dirs[a].append("right")
                open_dirs[b].append("left")

    for ri in range(room_rows - 1):
        for ci in range(room_cols):
            a = (ri, ci)
            b = (ri + 1, ci)
            edge = _edge_key(a, b)
            if edge in required_edges or random.random() < EXTRA_DOOR_PROB:
                c0, c1 = col_spans[ci]
                wall_r = wall_rows[ri + 1]
                door = (wall_r, random.randint(c0, c1))
                grid[door] = bg
                doors[edge] = door
                open_dirs[a].append("down")
                open_dirs[b].append("up")

    if path_len > 1 and not all(_edge_key(a, b) in doors for a, b in zip(path, path[1:])):
        return None

    source_room = path[0]
    source_color = random.choice([c for c in range(1, 10) if c not in {bg, WALL_COLOR}])
    flag_palette = [c for c in range(1, 10) if c not in {bg, WALL_COLOR, source_color}]

    exit_dirs: dict[tuple[int, int], str] = {}
    for a, b in zip(path, path[1:]):
        exit_dirs[a] = _direction_between(a, b)

    last_room = path[-1]
    last_open = list(open_dirs[last_room])
    if path_len > 1 and random.random() < 0.5:
        exit_dirs[last_room] = _direction_between(last_room, path[-2])
    else:
        closed = [d for d in DIRS if d not in last_open]
        if closed:
            exit_dirs[last_room] = random.choice(closed)
        elif path_len > 1:
            exit_dirs[last_room] = _direction_between(last_room, path[-2])
        else:
            exit_dirs[last_room] = random.choice(list(DIRS))

    for room in bboxes:
        if room not in exit_dirs:
            exit_dirs[room] = _balanced_direction(open_dirs[room])

    used_room_cells: dict[tuple[int, int], set[tuple[int, int]]] = {room: set() for room in bboxes}
    for idx, room in enumerate(sorted(bboxes)):
        flag_color = random.choice(flag_palette)
        pole, flag = _draw_flag(grid, bboxes[room], exit_dirs[room], flag_color)
        used_room_cells[room].update((pole, flag))

    r0, r1, c0, c1 = bboxes[source_room]
    free_cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in used_room_cells[source_room]
    ]
    if not free_cells:
        return None
    source_cell = random.choice(free_cells)
    grid[source_cell] = source_color

    bg_count = int(np.count_nonzero(grid == bg))
    wall_count = int(np.count_nonzero(grid == WALL_COLOR))
    if bg_count <= wall_count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
