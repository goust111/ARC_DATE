from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DEFAULT_WALL = 5
FILL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
ROOM_H_RANGE = (5, 8)
ROOM_W_RANGE = (5, 8)
EXTRA_ROOM_RANGE = (0, 2)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _door_direction(
    door: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> str | None:
    r, c = door
    if r == r0 and c0 < c < c1:
        return "up"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _room_from_component(
    g: np.ndarray, comp: list[tuple[int, int]], wall_color: int
) -> dict[str, object] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    boundary = _boundary_cells(r0, c0, r1, c1)
    comp_set = set(comp)
    if not comp_set.issubset(boundary):
        return None

    missing = list(boundary - comp_set)
    if len(missing) != 1:
        return None
    door = missing[0]
    if int(g[door]) != BG:
        return None

    direction = _door_direction(door, r0, c0, r1, c1)
    if direction is None:
        return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == int(wall_color)):
        return None

    fill_colors = sorted(set(map(int, interior.ravel())) - {BG, int(wall_color)})
    if len(fill_colors) != 1:
        return None
    fill_color = int(fill_colors[0])
    fill_count = int(np.count_nonzero(interior == fill_color))
    if fill_count <= 0:
        return None

    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "door": door,
        "direction": direction,
        "color": fill_color,
        "count": fill_count,
    }


def _detect_rooms(g: np.ndarray) -> list[dict[str, object]]:
    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    if not colors:
        return []
    wall_order = [DEFAULT_WALL] + [c for c in colors if c != DEFAULT_WALL]

    best: list[dict[str, object]] = []
    for wall_color in wall_order:
        if wall_color not in colors:
            continue
        rooms: list[dict[str, object]] = []
        for comp in _components(g == wall_color):
            room = _room_from_component(g, comp, wall_color)
            if room is not None:
                rooms.append(room)
        if len(rooms) > len(best):
            best = rooms
    best.sort(key=lambda room: (int(room["r0"]), int(room["c0"])))
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each grey rectangular room with a one-pixel doorway.
    Count the room's interior fill pixels, then extend a same-colored path
    outward from the doorway for that many steps, stopping early at obstacles.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _detect_rooms(g):
        dr, dc = DIRECTIONS[str(room["direction"])]
        r, c = room["door"]
        steps = int(room["count"])
        rr = r + dr
        cc = c + dc
        for _ in range(steps):
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                break
            if int(out[rr, cc]) != BG:
                break
            out[rr, cc] = int(room["color"])
            rr += dr
            cc += dc
    return out


def _room_size() -> tuple[int, int]:
    return random.randint(*ROOM_H_RANGE), random.randint(*ROOM_W_RANGE)


def _door_offset(side: str, h: int, w: int) -> int:
    if side in ("up", "down"):
        return random.randint(1, w - 2)
    return random.randint(1, h - 2)


def _door_cell(room: dict[str, object]) -> tuple[int, int]:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    h = int(room["h"])
    w = int(room["w"])
    side = str(room["door_side"])
    off = int(room["door_offset"])
    if side == "up":
        return (r0, c0 + off)
    if side == "down":
        return (r0 + h - 1, c0 + off)
    if side == "left":
        return (r0 + off, c0)
    return (r0 + off, c0 + w - 1)


def _bbox_cells(room: dict[str, object]) -> set[tuple[int, int]]:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    h = int(room["h"])
    w = int(room["w"])
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _expanded_bbox_cells(
    room: dict[str, object], h: int, w: int, pad: int = 1
) -> set[tuple[int, int]]:
    r0 = max(0, int(room["r0"]) - pad)
    c0 = max(0, int(room["c0"]) - pad)
    r1 = min(h - 1, int(room["r0"]) + int(room["h"]) - 1 + pad)
    c1 = min(w - 1, int(room["c0"]) + int(room["w"]) - 1 + pad)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _shell_cells(room: dict[str, object]) -> list[tuple[int, int]]:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    h = int(room["h"])
    w = int(room["w"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    door = _door_cell(room)
    cells = []
    for r, c in _boundary_cells(r0, c0, r1, c1):
        if (r, c) != door:
            cells.append((r, c))
    return cells


def _interior_cells(room: dict[str, object]) -> list[tuple[int, int]]:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    h = int(room["h"])
    w = int(room["w"])
    return [
        (r, c)
        for r in range(r0 + 1, r0 + h - 1)
        for c in range(c0 + 1, c0 + w - 1)
    ]


def _corridor_cells(room: dict[str, object], h: int, w: int) -> list[tuple[int, int]]:
    r, c = _door_cell(room)
    dr, dc = DIRECTIONS[str(room["door_side"])]
    rr = r + dr
    cc = c + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _clear_run(room: dict[str, object], grid: np.ndarray) -> tuple[int, bool]:
    count = 0
    for r, c in _corridor_cells(room, *grid.shape):
        if int(grid[r, c]) != BG:
            return count, True
        count += 1
    return count, False


def _paint_shells(grid: np.ndarray, rooms: list[dict[str, object]]) -> None:
    for room in rooms:
        for r, c in _shell_cells(room):
            grid[r, c] = DEFAULT_WALL


def _reserve_path(
    room: dict[str, object], length: int, h: int, w: int
) -> set[tuple[int, int]]:
    cells = _corridor_cells(room, h, w)
    return set(cells[:length])


def _pair_right(h: int, w: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES):
        hs, ws = _room_size()
        hb, wb = _room_size()
        src_off = _door_offset("right", hs, ws)
        hit_off = random.randint(1, hb - 2)
        block_off = _door_offset("right", hb, wb)
        lane_min = max(1 + src_off, 1 + hit_off)
        lane_max = min(h - hs - 1 + src_off, h - hb - 1 + hit_off)
        if lane_min > lane_max:
            continue
        lane = random.randint(lane_min, lane_max)
        gap = random.randint(2, 5)
        max_cs = w - (ws + gap + wb) - 1
        if max_cs < 1:
            continue
        cs = random.randint(1, max_cs)
        cb = cs + ws + gap
        rooms = [
            {"r0": lane - src_off, "c0": cs, "h": hs, "w": ws, "door_side": "right", "door_offset": src_off},
            {"r0": lane - hit_off, "c0": cb, "h": hb, "w": wb, "door_side": "right", "door_offset": block_off},
        ]
        reserved = set()
        reserved |= _reserve_path(rooms[0], gap, h, w)
        reserved |= _reserve_path(rooms[1], len(_corridor_cells(rooms[1], h, w)), h, w)
        return rooms, reserved
    return None


def _pair_left(h: int, w: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES):
        hs, ws = _room_size()
        hb, wb = _room_size()
        src_off = _door_offset("left", hs, ws)
        hit_off = random.randint(1, hb - 2)
        block_off = _door_offset("left", hb, wb)
        lane_min = max(1 + src_off, 1 + hit_off)
        lane_max = min(h - hs - 1 + src_off, h - hb - 1 + hit_off)
        if lane_min > lane_max:
            continue
        lane = random.randint(lane_min, lane_max)
        gap = random.randint(2, 5)
        max_cb = w - (wb + gap + ws) - 1
        if max_cb < 1:
            continue
        cb = random.randint(1, max_cb)
        cs = cb + wb + gap
        rooms = [
            {"r0": lane - hit_off, "c0": cb, "h": hb, "w": wb, "door_side": "left", "door_offset": block_off},
            {"r0": lane - src_off, "c0": cs, "h": hs, "w": ws, "door_side": "left", "door_offset": src_off},
        ]
        reserved = set()
        reserved |= _reserve_path(rooms[1], gap, h, w)
        reserved |= _reserve_path(rooms[0], len(_corridor_cells(rooms[0], h, w)), h, w)
        return rooms, reserved
    return None


def _pair_down(h: int, w: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES):
        hs, ws = _room_size()
        hb, wb = _room_size()
        src_off = _door_offset("down", hs, ws)
        hit_off = random.randint(1, wb - 2)
        block_off = _door_offset("down", hb, wb)
        lane_min = max(1 + src_off, 1 + hit_off)
        lane_max = min(w - ws - 1 + src_off, w - wb - 1 + hit_off)
        if lane_min > lane_max:
            continue
        lane = random.randint(lane_min, lane_max)
        gap = random.randint(2, 5)
        max_rs = h - (hs + gap + hb) - 1
        if max_rs < 1:
            continue
        rs = random.randint(1, max_rs)
        rb = rs + hs + gap
        rooms = [
            {"r0": rs, "c0": lane - src_off, "h": hs, "w": ws, "door_side": "down", "door_offset": src_off},
            {"r0": rb, "c0": lane - hit_off, "h": hb, "w": wb, "door_side": "down", "door_offset": block_off},
        ]
        reserved = set()
        reserved |= _reserve_path(rooms[0], gap, h, w)
        reserved |= _reserve_path(rooms[1], len(_corridor_cells(rooms[1], h, w)), h, w)
        return rooms, reserved
    return None


def _pair_up(h: int, w: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES):
        hs, ws = _room_size()
        hb, wb = _room_size()
        src_off = _door_offset("up", hs, ws)
        hit_off = random.randint(1, wb - 2)
        block_off = _door_offset("up", hb, wb)
        lane_min = max(1 + src_off, 1 + hit_off)
        lane_max = min(w - ws - 1 + src_off, w - wb - 1 + hit_off)
        if lane_min > lane_max:
            continue
        lane = random.randint(lane_min, lane_max)
        gap = random.randint(2, 5)
        max_rb = h - (hb + gap + hs) - 1
        if max_rb < 1:
            continue
        rb = random.randint(1, max_rb)
        rs = rb + hb + gap
        rooms = [
            {"r0": rb, "c0": lane - hit_off, "h": hb, "w": wb, "door_side": "up", "door_offset": block_off},
            {"r0": rs, "c0": lane - src_off, "h": hs, "w": ws, "door_side": "up", "door_offset": src_off},
        ]
        reserved = set()
        reserved |= _reserve_path(rooms[1], gap, h, w)
        reserved |= _reserve_path(rooms[0], len(_corridor_cells(rooms[0], h, w)), h, w)
        return rooms, reserved
    return None


def _place_blocked_pair(
    h: int, w: int
) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    builders = [_pair_up, _pair_down, _pair_left, _pair_right]
    random.shuffle(builders)
    for builder in builders:
        built = builder(h, w)
        if built is not None:
            return built
    return None


def _try_extra_room(
    h: int,
    w: int,
    bbox_occupied: set[tuple[int, int]],
    buffer_occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(PLACEMENT_RETRIES):
        room_h, room_w = _room_size()
        side = random.choice(["up", "down", "left", "right"])
        room = {
            "r0": random.randint(1, h - room_h - 1),
            "c0": random.randint(1, w - room_w - 1),
            "h": room_h,
            "w": room_w,
            "door_side": side,
            "door_offset": _door_offset(side, room_h, room_w),
        }
        bbox = _bbox_cells(room)
        if bbox & buffer_occupied:
            continue
        if bbox & reserved:
            continue
        corridor = _corridor_cells(room, h, w)
        if not corridor:
            continue
        if any(cell in bbox_occupied or cell in reserved for cell in corridor):
            continue
        return room
    return None


def _expected_from_layout(grid: np.ndarray, rooms: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for room in sorted(rooms, key=lambda item: (int(item["r0"]), int(item["c0"]))):
        color = int(room["color"])
        count = int(room["fill_count"])
        clear_run = int(room["clear_run"])
        for r, c in _corridor_cells(room, *out.shape)[: min(count, clear_run)]:
            out[r, c] = color
    return out


def _make_layout() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    pair = _place_blocked_pair(h, w)
    if pair is None:
        return None
    rooms, reserved = pair
    bbox_occupied: set[tuple[int, int]] = set()
    buffer_occupied: set[tuple[int, int]] = set()
    for room in rooms:
        bbox_occupied |= _bbox_cells(room)
        buffer_occupied |= _expanded_bbox_cells(room, h, w)

    for _ in range(random.randint(*EXTRA_ROOM_RANGE)):
        extra = _try_extra_room(h, w, bbox_occupied, buffer_occupied, reserved)
        if extra is None:
            break
        rooms.append(extra)
        bbox_occupied |= _bbox_cells(extra)
        buffer_occupied |= _expanded_bbox_cells(extra, h, w)
        reserved |= set(_corridor_cells(extra, h, w))

    if len(rooms) > len(FILL_COLORS):
        return None

    grid = np.full((h, w), BG, dtype=int)
    _paint_shells(grid, rooms)

    blocked_seen = False
    clear_seen = False
    room_colors = random.sample(FILL_COLORS, len(rooms))
    for room, color in zip(rooms, room_colors):
        room["color"] = int(color)
        clear_run, blocked = _clear_run(room, grid)
        interior_area = len(_interior_cells(room))
        max_fill = interior_area - 1
        if max_fill < 1 or clear_run < 1:
            return None

        if blocked:
            blocked_seen = True
            low = clear_run + 1
            high = max_fill
            if low > high:
                return None
            fill_count = random.randint(low, high)
        else:
            clear_seen = True
            high = min(clear_run, max_fill)
            if high < 1:
                return None
            fill_count = random.randint(1, high)

        room["clear_run"] = clear_run
        room["blocked"] = blocked
        room["fill_count"] = fill_count

        fill_cells = random.sample(_interior_cells(room), fill_count)
        for r, c in fill_cells:
            grid[r, c] = int(color)

    if not blocked_seen or not clear_seen:
        return None

    expected = _expected_from_layout(grid, rooms)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        made = _make_layout()
        if made is None:
            continue
        grid, expected = made
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_RETRIES} attempts")
