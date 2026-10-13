import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
VERTICAL_DOOR_COLORS = (2, 3)
HORIZONTAL_DOOR_COLORS = (4, 6)
GENERATION_RETRIES = 400
SHAPE_RETRIES = 40
PLACEMENT_RETRIES = 60


def _bands(lines: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2, "wall lines must leave a non-empty room band"
        bands.append((int(a + 1), int(b - 1)))
    return bands


def _find_wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape

    def is_wall_row(r: int) -> bool:
        row = grid[r]
        if int(row[0]) != BLUE or int(row[w - 1]) != BLUE:
            return False
        for c in range(1, w - 1):
            if int(row[c]) != BLACK:
                continue
            left = int(row[c - 1])
            right = int(row[c + 1])
            if left != right or left in (BLACK, BLUE):
                return False
        return True

    def is_wall_col(c: int) -> bool:
        col = grid[:, c]
        if int(col[0]) != BLUE or int(col[h - 1]) != BLUE:
            return False
        for r in range(1, h - 1):
            if int(col[r]) != BLACK:
                continue
            up = int(col[r - 1])
            down = int(col[r + 1])
            if up != down or up in (BLACK, BLUE):
                return False
        return True

    wall_rows = [
        int(r)
        for r in range(h)
        if is_wall_row(r)
    ]
    wall_cols = [
        int(c)
        for c in range(w)
        if is_wall_col(c)
    ]

    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected border walls around the rooms"
    assert wall_rows[0] == 0 and wall_rows[-1] == h - 1, "top/bottom border must be walls"
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1, "left/right border must be walls"
    return wall_rows, wall_cols


def _room_bboxes(
    wall_rows: list[int], wall_cols: list[int]
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = _bands(wall_rows)
    col_bands = _bands(wall_cols)
    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            rooms[(int(ri), int(ci))] = (int(r0), int(c0), int(r1), int(c1))
    return rooms, row_bands, col_bands


def _collect_doors(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    room_doors = {room_id: [] for room_id in rooms}
    doors: list[dict] = []
    door_id = 0

    for wall_idx in range(1, len(wall_cols) - 1):
        c = int(wall_cols[wall_idx])
        left_room_col = int(wall_idx - 1)
        right_room_col = int(wall_idx)
        for room_row, (r0, r1) in enumerate(row_bands):
            matches: list[tuple[int, int]] = []
            for r in range(r0 + 1, r1):
                if int(grid[r, c]) != BLACK:
                    continue
                up = int(grid[r - 1, c])
                down = int(grid[r + 1, c])
                if up == down and up not in (BLACK, BLUE):
                    matches.append((int(r), int(up)))
            assert len(matches) == 1, "each vertical wall segment must contain exactly one colored door"
            gap_r, color = matches[0]
            door = {
                "id": int(door_id),
                "orientation": "vertical",
                "gap": (int(gap_r), int(c)),
                "color": int(color),
                "rooms": ((int(room_row), int(left_room_col)), (int(room_row), int(right_room_col))),
            }
            doors.append(door)
            room_doors[(int(room_row), int(left_room_col))].append(door)
            room_doors[(int(room_row), int(right_room_col))].append(door)
            door_id += 1

    for wall_idx in range(1, len(wall_rows) - 1):
        r = int(wall_rows[wall_idx])
        top_room_row = int(wall_idx - 1)
        bottom_room_row = int(wall_idx)
        for room_col, (c0, c1) in enumerate(col_bands):
            matches: list[tuple[int, int]] = []
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) != BLACK:
                    continue
                left = int(grid[r, c - 1])
                right = int(grid[r, c + 1])
                if left == right and left not in (BLACK, BLUE):
                    matches.append((int(c), int(left)))
            assert len(matches) == 1, "each horizontal wall segment must contain exactly one colored door"
            gap_c, color = matches[0]
            door = {
                "id": int(door_id),
                "orientation": "horizontal",
                "gap": (int(r), int(gap_c)),
                "color": int(color),
                "rooms": ((int(top_room_row), int(room_col)), (int(bottom_room_row), int(room_col))),
            }
            doors.append(door)
            room_doors[(int(top_room_row), int(room_col))].append(door)
            room_doors[(int(bottom_room_row), int(room_col))].append(door)
            door_id += 1

    return doors, room_doors


def _is_single_component(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return True
    h, w = mask.shape
    start = tuple(map(int, points[0]))
    seen = np.zeros_like(mask, dtype=bool)
    queue = deque([start])
    seen[start] = True
    count = 0
    while queue:
        r, c = queue.popleft()
        count += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                queue.append((nr, nc))
    return int(count) == int(mask.sum())


def _extract_room_object(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict | None:
    r0, c0, r1, c1 = bbox
    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    mask = sub != BLACK
    if not bool(mask.any()):
        return None

    colors = np.unique(sub[mask])
    assert len(colors) == 1, "room objects must be monochrome"
    assert _is_single_component(mask), "room object must be connected"

    coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in np.argwhere(mask)]
    room_h, room_w = sub.shape
    for rr, cc in np.argwhere(mask):
        assert 0 < int(rr) < room_h - 1 and 0 < int(cc) < room_w - 1, "object must not touch the walls"
    return {"color": int(colors[0]), "coords": coords}


def _other_room(door: dict, room_id: tuple[int, int]) -> tuple[int, int]:
    room_a, room_b = door["rooms"]
    return room_b if room_a == room_id else room_a


def _mirror_cell(r: int, c: int, door: dict) -> tuple[int, int]:
    gap_r, gap_c = map(int, door["gap"])
    if door["orientation"] == "vertical":
        return int(r), int(2 * gap_c - c)
    return int(2 * gap_r - r), int(c)


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    wall_rows, wall_cols = _find_wall_lines(g)
    rooms, row_bands, col_bands = _room_bboxes(wall_rows, wall_cols)
    _doors, room_doors = _collect_doors(g, rooms, wall_rows, wall_cols, row_bands, col_bands)

    moves: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int]] = []
    for room_id, bbox in rooms.items():
        obj = _extract_room_object(g, bbox)
        if obj is None:
            continue

        matching = [door for door in room_doors[room_id] if int(door["color"]) == int(obj["color"])]
        assert len(matching) == 1, "an object's color must match exactly one door in its room"

        door = matching[0]
        target_room = _other_room(door, room_id)
        tr0, tc0, tr1, tc1 = rooms[target_room]

        projected: list[tuple[int, int]] = []
        for r, c in obj["coords"]:
            nr, nc = _mirror_cell(int(r), int(c), door)
            if tr0 <= nr <= tr1 and tc0 <= nc <= tc1:
                projected.append((int(nr), int(nc)))
        moves.append((obj["coords"], projected, int(obj["color"])))

    out = g.copy()
    for source_cells, _target_cells, _color in moves:
        for r, c in source_cells:
            out[int(r), int(c)] = BLACK
    for _source_cells, target_cells, color in moves:
        for r, c in target_cells:
            out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each room's colored object through the matching door into the adjacent room with a mirrored offset.

    Input structure:
    - The background is black.
    - Blue wall rows and columns form a rectangular room lattice with a blue outer border.
    - Each internal wall segment has one one-cell gap, flanked by two same-colored doorframe pixels on the wall.
    - Every room contains at most one small monochrome connected object whose color matches one of that room's doors, and the object does not touch the walls.

    Rule:
    - Identify the rooms from the wall lattice and detect every door from its colored frame and black gap.
    - For each room object, find the unique door in that room whose frame color matches the object color.
    - Erase the object from its source room, then mirror each object pixel across the matched door gap along the axis perpendicular to the wall, placing the result in the adjacent room.
    - Keep blue walls and colored doorframes unchanged, and discard mirrored pixels that would land outside the target room.

    Key insight:
    - The object color is a routing key that selects a specific door, and the transfer is not a plain translation: the object keeps its offsets from the door except for a sign flip across the wall.
    """
    return _apply_rule(grid)


def _wall_positions(sizes: list[int]) -> list[int]:
    lines = [0]
    pos = 0
    for size in sizes:
        pos += int(size) + 1
        lines.append(int(pos))
    return lines


def _build_base_grid(
    room_heights: list[int], room_widths: list[int], rng: random.Random
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]], list[dict]]:
    wall_rows = _wall_positions(room_heights)
    wall_cols = _wall_positions(room_widths)
    h = int(wall_rows[-1] + 1)
    w = int(wall_cols[-1] + 1)
    assert h <= 30 and w <= 30, "generator must stay within ARC size limits"

    grid = np.zeros((h, w), dtype=int)
    for r in wall_rows:
        grid[int(r), :] = BLUE
    for c in wall_cols:
        grid[:, int(c)] = BLUE

    row_bands = _bands(wall_rows)
    col_bands = _bands(wall_cols)

    for wall_idx in range(1, len(wall_cols) - 1):
        c = int(wall_cols[wall_idx])
        color = int(VERTICAL_DOOR_COLORS[(wall_idx - 1) % len(VERTICAL_DOOR_COLORS)])
        for r0, r1 in row_bands:
            gap_r = int(rng.randint(r0 + 1, r1 - 1))
            grid[gap_r, c] = BLACK
            grid[gap_r - 1, c] = color
            grid[gap_r + 1, c] = color

    for wall_idx in range(1, len(wall_rows) - 1):
        r = int(wall_rows[wall_idx])
        color = int(HORIZONTAL_DOOR_COLORS[(wall_idx - 1) % len(HORIZONTAL_DOOR_COLORS)])
        for c0, c1 in col_bands:
            gap_c = int(rng.randint(c0 + 1, c1 - 1))
            grid[r, gap_c] = BLACK
            grid[r, gap_c - 1] = color
            grid[r, gap_c + 1] = color

    rooms, row_bands, col_bands = _room_bboxes(wall_rows, wall_cols)
    doors, _room_doors = _collect_doors(grid, rooms, wall_rows, wall_cols, row_bands, col_bands)
    return grid, rooms, doors


def _random_shape(rng: random.Random, max_h: int, max_w: int) -> tuple[list[tuple[int, int]], int, int]:
    for _ in range(SHAPE_RETRIES):
        box_h = int(rng.randint(1, max_h))
        box_w = int(rng.randint(1, max_w))
        target = int(rng.randint(1, min(5, box_h * box_w)))

        start = (int(rng.randrange(box_h)), int(rng.randrange(box_w)))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        coords = sorted((int(r - r0), int(c - c0)) for r, c in cells)
        return coords, int(r1 - r0 + 1), int(c1 - c0 + 1)
    return [(0, 0)], 1, 1


def _project_shape(
    coords: list[tuple[int, int]], door: dict, target_bbox: tuple[int, int, int, int]
) -> tuple[list[tuple[int, int]], bool]:
    tr0, tc0, tr1, tc1 = target_bbox
    projected: list[tuple[int, int]] = []
    clipped = False
    for r, c in coords:
        nr, nc = _mirror_cell(int(r), int(c), door)
        if tr0 <= nr <= tr1 and tc0 <= nc <= tc1:
            projected.append((int(nr), int(nc)))
        else:
            clipped = True
    return projected, bool(clipped)


def _place_objects(
    base_grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    selections: list[tuple[tuple[int, int], dict]],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.asarray(base_grid, dtype=int).copy()
    output_occupied: set[tuple[int, int]] = set()

    for room_id, door in selections:
        bbox = rooms[room_id]
        target_bbox = rooms[_other_room(door, room_id)]
        r0, c0, r1, c1 = bbox
        usable_h = int(r1 - r0 - 1)
        usable_w = int(c1 - c0 - 1)
        if usable_h <= 0 or usable_w <= 0:
            return None

        prefer_clipped = bool(rng.getrandbits(1))
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            shape, shape_h, shape_w = _random_shape(rng, min(3, usable_h), min(3, usable_w))
            candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]], bool]] = []

            top_min = int(r0 + 1)
            top_max = int(r1 - shape_h)
            left_min = int(c0 + 1)
            left_max = int(c1 - shape_w)
            for top in range(top_min, top_max + 1):
                for left in range(left_min, left_max + 1):
                    abs_coords = [(int(top + rr), int(left + cc)) for rr, cc in shape]
                    projected, clipped = _project_shape(abs_coords, door, target_bbox)
                    if not projected:
                        continue
                    if any((int(r), int(c)) in output_occupied for r, c in projected):
                        continue
                    candidates.append((abs_coords, projected, bool(clipped)))

            if not candidates:
                continue

            preferred = [cand for cand in candidates if bool(cand[2]) == prefer_clipped]
            source_cells, target_cells, _clipped = rng.choice(preferred or candidates)
            for r, c in source_cells:
                grid[int(r), int(c)] = int(door["color"])
            output_occupied.update((int(r), int(c)) for r, c in target_cells)
            placed = True
            break

        if not placed:
            return None

    return grid


def _choose_layout(rng: random.Random) -> tuple[str, int, int]:
    kind = rng.choice(("vertical", "horizontal", "grid"))
    if kind == "vertical":
        return kind, 1, int(rng.randint(2, 3))
    if kind == "horizontal":
        return kind, int(rng.randint(2, 3)), 1
    return kind, int(rng.randint(2, 3)), int(rng.randint(2, 3))


def _select_sources(
    rng: random.Random,
    doors: list[dict],
    room_count: int,
    kind: str,
) -> list[tuple[tuple[int, int], dict]] | None:
    door_by_id = {int(door["id"]): door for door in doors}
    oriented = []
    for door in doors:
        for room_id in door["rooms"]:
            oriented.append((room_id, int(door["id"])))

    min_count = 2 if kind == "grid" else 1
    max_count = min(4, room_count, len(doors))
    if max_count < min_count:
        return None
    target_count = int(rng.randint(min_count, max_count))

    picks: list[tuple[tuple[int, int], dict]] = []
    used_rooms: set[tuple[int, int]] = set()
    used_doors: set[int] = set()

    def pick_one(orientation: str | None = None) -> bool:
        options = []
        for room_id, door_id in oriented:
            if room_id in used_rooms or door_id in used_doors:
                continue
            door = door_by_id[door_id]
            if orientation is not None and door["orientation"] != orientation:
                continue
            options.append((room_id, door))
        if not options:
            return False
        room_id, door = rng.choice(options)
        picks.append((room_id, door))
        used_rooms.add(room_id)
        used_doors.add(int(door["id"]))
        return True

    if kind == "grid" and target_count >= 2:
        if not pick_one("vertical"):
            return None
        if not pick_one("horizontal"):
            return None

    while len(picks) < target_count:
        if not pick_one():
            break

    return picks if len(picks) >= min_count else None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    kind, room_rows, room_cols = _choose_layout(rng)
    room_heights = [int(rng.randint(4, 8)) for _ in range(room_rows)]
    room_widths = [int(rng.randint(4, 8)) for _ in range(room_cols)]

    base_grid, rooms, doors = _build_base_grid(room_heights, room_widths, rng)
    selections = _select_sources(rng, doors, len(rooms), kind)
    if not selections:
        return None

    grid = _place_objects(base_grid, rooms, selections, rng)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door transport puzzle")
