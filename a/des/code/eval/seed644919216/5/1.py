import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GREY = 5
COLOR_POOL = tuple(range(10))


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_same_color(grid: np.ndarray, mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _door_direction(door: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    r, c = door
    if r == r0:
        return -1, 0
    if r == r1:
        return 1, 0
    if c == c0:
        return 0, -1
    if c == c1:
        return 0, 1
    raise AssertionError("door must lie on the room border")


def _door_inside_anchor(door: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return door[0] - dr, door[1] - dc


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    rooms: list[dict] = []
    for wall_color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        wall_mask = grid == wall_color
        for comp in _components_same_color(grid, wall_mask):
            coords = comp["coords"]
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            comp_set = set(coords)
            border = _border_positions(r0, c0, r1, c1)
            border_set = set(border)
            if len(comp_set) != len(border_set) - 1:
                continue
            if any((r, c) not in border_set for r, c in comp_set):
                continue
            missing = list(border_set - comp_set)
            if len(missing) != 1:
                continue
            door = missing[0]
            if door in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                continue
            door_color = int(grid[door])
            if door_color in {bg, wall_color}:
                continue
            if not (r0 + 1 <= r1 - 1 and c0 + 1 <= c1 - 1):
                continue
            if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == wall_color):
                continue

            interior_mask = np.zeros((h, w), dtype=bool)
            interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True
            interior_mask &= grid != bg
            interior_mask &= grid != wall_color
            pieces = _components_same_color(grid, interior_mask)
            key_pieces = [piece for piece in pieces if piece["color"] == door_color]
            if len(key_pieces) != 1:
                continue

            rooms.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "wall_color": wall_color,
                    "door": door,
                    "door_color": door_color,
                    "direction": _door_direction(door, (r0, c0, r1, c1)),
                    "pieces": pieces,
                    "key_piece": key_pieces[0],
                }
            )
    rooms.sort(key=lambda room: (room["bbox"], room["door"]))
    return rooms


def _piece_order_key(room: dict, piece: dict) -> tuple[int, int]:
    dr, dc = room["direction"]
    anchor_r, anchor_c = _door_inside_anchor(room["door"], room["direction"])
    coords = piece["coords"]
    if dr == -1:
        primary = min(r for r, _ in coords) - anchor_r
        secondary = min(c for _, c in coords)
    elif dr == 1:
        primary = anchor_r - max(r for r, _ in coords)
        secondary = min(c for _, c in coords)
    elif dc == -1:
        primary = min(c for _, c in coords) - anchor_c
        secondary = min(r for r, _ in coords)
    else:
        primary = anchor_c - max(c for _, c in coords)
        secondary = min(r for r, _ in coords)
    return primary, secondary


def _slide_component(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    bg: int,
) -> None:
    h, w = out.shape
    for r, c in coords:
        out[r, c] = bg

    dr, dc = direction
    shift = 0
    while True:
        next_cells = [(r + (shift + 1) * dr, c + (shift + 1) * dc) for r, c in coords]
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in next_cells):
            break
        if any(int(out[rr, cc]) != bg for rr, cc in next_cells):
            break
        shift += 1

    for r, c in coords:
        out[r + shift * dr, c + shift * dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each door color to the same-colored key piece inside the room,
    consume that key, then slide the remaining room pieces through the opened
    door and stack them along the outward channel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    rooms = _find_rooms(g, bg)
    assert rooms, "expected at least one valid room"

    out = g.copy()
    for room in rooms:
        out[room["door"]] = bg
        for r, c in room["key_piece"]["coords"]:
            out[r, c] = bg

    for room in rooms:
        movables = [piece for piece in room["pieces"] if piece is not room["key_piece"]]
        movables.sort(key=lambda piece: _piece_order_key(room, piece))
        for piece in movables:
            _slide_component(out, piece["coords"], piece["color"], room["direction"], bg)

    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
ROOM_COUNT_MIN = 1
ROOM_COUNT_MAX = 3
MAX_ATTEMPTS = 400
ROOM_PLACE_ATTEMPTS = 200


def _sample_segments_in_lane(lane_len: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    for _ in range(80):
        count = int(rng.integers(1, 4))
        lengths = [int(rng.integers(1, 4)) for _ in range(count)]
        needed = sum(lengths) + max(0, count - 1)
        if needed > lane_len:
            continue
        extra = lane_len - needed
        gaps = [0] + [1] * max(0, count - 1) + [0]
        for _ in range(extra):
            gaps[int(rng.integers(0, len(gaps)))] += 1
        pos = gaps[0]
        segments: list[tuple[int, int]] = []
        for i, length in enumerate(lengths):
            segments.append((pos, length))
            pos += length + gaps[i + 1]
        return segments
    return None


def _shape_from_pool(
    pool: set[tuple[int, int]], size: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    if not pool:
        return None
    pool_list = list(pool)
    for _ in range(80):
        start = pool_list[int(rng.integers(0, len(pool_list)))]
        chosen = [start]
        chosen_set = {start}
        frontier = []
        for dr, dc in DIRS4:
            cand = (start[0] + dr, start[1] + dc)
            if cand in pool and cand not in chosen_set:
                frontier.append(cand)
        while len(chosen) < size and frontier:
            idx = int(rng.integers(0, len(frontier)))
            cell = frontier.pop(idx)
            if cell in chosen_set:
                continue
            chosen.append(cell)
            chosen_set.add(cell)
            for dr, dc in DIRS4:
                cand = (cell[0] + dr, cell[1] + dc)
                if cand in pool and cand not in chosen_set:
                    frontier.append(cand)
        if len(chosen) == size:
            return chosen
    return None


def _mark_reserved(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _room_geometry(direction: int, rng: np.random.Generator) -> tuple[int, int]:
    if direction in {0, 1}:
        return int(rng.integers(4, 8)), int(rng.integers(3, 6))
    return int(rng.integers(3, 6)), int(rng.integers(4, 8))


def _candidate_room(
    h: int, w: int, rng: np.random.Generator
) -> dict | None:
    direction = int(rng.integers(0, 4))
    interior_h, interior_w = _room_geometry(direction, rng)
    room_h = interior_h + 2
    room_w = interior_w + 2
    lane_len = interior_h if direction in {0, 1} else interior_w
    segments = _sample_segments_in_lane(lane_len, rng)
    if segments is None:
        return None

    total_movable_cells = sum(length for _, length in segments)
    stop_with_wall = bool(rng.integers(0, 2))
    extra_space = int(rng.integers(0, 4))
    required_space = total_movable_cells + extra_space + (1 if stop_with_wall else 0)

    if direction == 0:
        r0_low, r0_high = required_space, h - room_h - 1
        c0_low, c0_high = 1, w - room_w - 1
    elif direction == 1:
        r0_low, r0_high = 1, h - room_h - required_space
        c0_low, c0_high = 1, w - room_w - 1
    elif direction == 2:
        r0_low, r0_high = 1, h - room_h - 1
        c0_low, c0_high = required_space, w - room_w - 1
    else:
        r0_low, r0_high = 1, h - room_h - 1
        c0_low, c0_high = 1, w - room_w - required_space

    if r0_low > r0_high or c0_low > c0_high:
        return None

    r0 = int(rng.integers(r0_low, r0_high + 1))
    c0 = int(rng.integers(c0_low, c0_high + 1))
    r1 = r0 + room_h - 1
    c1 = c0 + room_w - 1

    if direction == 0:
        lane_col = c0 + int(rng.integers(1, interior_w + 1))
        door = (r0, lane_col)
        edge_space = r0
        channel_len = int(rng.integers(total_movable_cells, edge_space)) if stop_with_wall else edge_space
        stopper = (r0 - channel_len - 1, lane_col) if stop_with_wall else None
        channel_cells = {(r0 - step, lane_col) for step in range(1, channel_len + 1)}
        anchor = (r0 + 1, lane_col)
    elif direction == 1:
        lane_col = c0 + int(rng.integers(1, interior_w + 1))
        door = (r1, lane_col)
        edge_space = h - 1 - r1
        channel_len = int(rng.integers(total_movable_cells, edge_space)) if stop_with_wall else edge_space
        stopper = (r1 + channel_len + 1, lane_col) if stop_with_wall else None
        channel_cells = {(r1 + step, lane_col) for step in range(1, channel_len + 1)}
        anchor = (r1 - 1, lane_col)
    elif direction == 2:
        lane_row = r0 + int(rng.integers(1, interior_h + 1))
        door = (lane_row, c0)
        edge_space = c0
        channel_len = int(rng.integers(total_movable_cells, edge_space)) if stop_with_wall else edge_space
        stopper = (lane_row, c0 - channel_len - 1) if stop_with_wall else None
        channel_cells = {(lane_row, c0 - step) for step in range(1, channel_len + 1)}
        anchor = (lane_row, c0 + 1)
    else:
        lane_row = r0 + int(rng.integers(1, interior_h + 1))
        door = (lane_row, c1)
        edge_space = w - 1 - c1
        channel_len = int(rng.integers(total_movable_cells, edge_space)) if stop_with_wall else edge_space
        stopper = (lane_row, c1 + channel_len + 1) if stop_with_wall else None
        channel_cells = {(lane_row, c1 + step) for step in range(1, channel_len + 1)}
        anchor = (lane_row, c1 - 1)

    movable_cells: list[list[tuple[int, int]]] = []
    for start, length in segments:
        cells: list[tuple[int, int]] = []
        for offset in range(length):
            if direction == 0:
                cells.append((anchor[0] + start + offset, anchor[1]))
            elif direction == 1:
                cells.append((anchor[0] - start - offset, anchor[1]))
            elif direction == 2:
                cells.append((anchor[0], anchor[1] + start + offset))
            else:
                cells.append((anchor[0], anchor[1] - start - offset))
        movable_cells.append(cells)

    interior_cells = {
        (rr, cc)
        for rr in range(r0 + 1, r1)
        for cc in range(c0 + 1, c1)
    }
    movable_set = {cell for cells in movable_cells for cell in cells}

    candidate_key_cells = {
        cell
        for cell in interior_cells
        if cell not in movable_set
        and all(
            (cell[0] + dr, cell[1] + dc) not in movable_set
            for dr, dc in DIRS4
        )
    }
    fallback_key_cells = interior_cells - movable_set
    key_size_order = [int(rng.integers(1, 4)), 3, 2, 1]
    key_cells = None
    for pool in (candidate_key_cells, fallback_key_cells):
        for size in key_size_order:
            key_cells = _shape_from_pool(pool, size, rng)
            if key_cells is not None:
                break
        if key_cells is not None:
            break
    if key_cells is None:
        return None

    footprint = set(channel_cells)
    if stopper is not None:
        footprint.add(stopper)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            footprint.add((rr, cc))

    return {
        "direction": direction,
        "bbox": (r0, c0, r1, c1),
        "door": door,
        "channel_cells": channel_cells,
        "stopper": stopper,
        "movable_cells": movable_cells,
        "key_cells": key_cells,
        "footprint": footprint,
    }


def _draw_room(
    grid: np.ndarray,
    reserved: np.ndarray,
    spec: dict,
    bg: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w) for r, c in spec["footprint"]):
        return False
    if any(reserved[r, c] for r, c in spec["footprint"]):
        return False

    r0, c0, r1, c1 = spec["bbox"]
    room_border = set(_border_positions(r0, c0, r1, c1))
    room_border.remove(spec["door"])

    colors = [c for c in COLOR_POOL if c not in {bg, GREY}]
    color_count = 1 + len(spec["movable_cells"])
    if color_count > len(colors):
        return False
    chosen = list(rng.choice(colors, size=color_count, replace=False))
    door_color = int(chosen[0])
    movable_colors = [int(c) for c in chosen[1:]]

    for r, c in room_border:
        grid[r, c] = GREY
    grid[spec["door"]] = door_color
    for r, c in spec["key_cells"]:
        grid[r, c] = door_color
    for cells, color in zip(spec["movable_cells"], movable_colors):
        for r, c in cells:
            grid[r, c] = color
    if spec["stopper"] is not None:
        grid[spec["stopper"]] = GREY

    _mark_reserved(reserved, spec["footprint"])
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    bg_choices = [c for c in COLOR_POOL if c != GREY]
    bg = int(bg_choices[int(rng.integers(0, len(bg_choices)))])

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    room_count = int(rng.integers(ROOM_COUNT_MIN, ROOM_COUNT_MAX + 1))
    for _ in range(room_count):
        placed = False
        for _ in range(ROOM_PLACE_ATTEMPTS):
            spec = _candidate_room(h, w, rng)
            if spec is None:
                continue
            if _draw_room(grid, reserved, spec, bg, rng):
                placed = True
                break
        if not placed:
            return None

    if _bg_color(grid) != bg:
        return None

    try:
        rooms = _find_rooms(grid, bg)
        if len(rooms) != room_count:
            return None
        _ = main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance")
