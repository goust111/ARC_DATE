from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
ROOM_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
ROOM_COUNT_CHOICES = (2, 3, 4)
ROOM_COUNT_WEIGHTS = (2, 3, 3)
INNER_H_RANGE = (2, 5)
INNER_W_RANGE = (2, 5)
MAX_INTERIOR_AREA = 16
MARGIN = 1

CLEAR_ROOM_TRIES = 120
PAIR_TRIES = 160
GENERATE_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _boundary_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _room_nonbg_cells(room: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    cells = _boundary_coords(r0, c0, r1, c1)
    cells.append((room["key_r"], room["key_c"]))
    return cells


def _bboxes_have_gap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 < br0 - 1
        or br1 < ar0 - 1
        or ac1 < bc0 - 1
        or bc1 < ac0 - 1
    )


def _ray_cells_to_edge(room: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = room["direction"]
    rr = room["door_r"] + dr
    cc = room["door_c"] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _paint_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for r, c in _boundary_coords(r0, c0, r1, c1):
        grid[r, c] = GRAY
    grid[room["door_r"], room["door_c"]] = int(room["color"])
    grid[room["key_r"], room["key_c"]] = int(room["color"])


def _sample_room_spec(side: str, color: int) -> dict:
    for _ in range(100):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        if inner_h * inner_w > MAX_INTERIOR_AREA:
            continue
        key_r = random.randint(1, inner_h)
        key_c = random.randint(1, inner_w)
        door_offset = random.randint(1, inner_w if side in ("top", "bottom") else inner_h)
        return {
            "side": side,
            "color": int(color),
            "inner_h": int(inner_h),
            "inner_w": int(inner_w),
            "door_offset": int(door_offset),
            "key_r_off": int(key_r),
            "key_c_off": int(key_c),
        }
    raise AssertionError("failed to sample a room spec")


def _instantiate_room(spec: dict, r0: int, c0: int) -> dict:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    side = spec["side"]
    door_offset = int(spec["door_offset"])
    if side == "top":
        door_r, door_c = r0, c0 + door_offset
        direction = (-1, 0)
    elif side == "bottom":
        door_r, door_c = r1, c0 + door_offset
        direction = (1, 0)
    elif side == "left":
        door_r, door_c = r0 + door_offset, c0
        direction = (0, -1)
    else:
        assert side == "right"
        door_r, door_c = r0 + door_offset, c1
        direction = (0, 1)

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "color": int(spec["color"]),
        "door_r": int(door_r),
        "door_c": int(door_c),
        "direction": direction,
        "key_r": int(r0 + spec["key_r_off"]),
        "key_c": int(c0 + spec["key_c_off"]),
        "black_count": int(inner_h * inner_w - 1),
    }


def _cells_are_open(
    grid: np.ndarray, reserved: np.ndarray, cells: list[tuple[int, int]]
) -> bool:
    for r, c in cells:
        if reserved[r, c] or int(grid[r, c]) != BLACK:
            return False
    return True


def _room_position_ok(
    room: dict,
    bboxes: list[tuple[int, int, int, int]],
    reserved: np.ndarray,
) -> bool:
    for bbox in bboxes:
        if not _bboxes_have_gap(room["bbox"], bbox):
            return False
    for r, c in _room_nonbg_cells(room):
        if reserved[r, c]:
            return False
    return True


def _parse_room(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    h, w = sub.shape
    assert h >= 3 and w >= 3, "room bbox must contain an interior"

    boundary_size = 2 * (h + w) - 4
    assert int(coords.shape[0]) == boundary_size - 1, "gray wall should miss exactly one door pixel"

    door_candidates: list[tuple[int, int, int]] = []
    gray_count = 0
    for rr in range(h):
        for cc in range(w):
            on_boundary = rr in (0, h - 1) or cc in (0, w - 1)
            if not on_boundary:
                continue
            value = int(sub[rr, cc])
            if value == GRAY:
                gray_count += 1
            else:
                assert value != BLACK, "boundary must contain only gray wall and one colored door"
                door_candidates.append((rr, cc, value))

    assert gray_count == boundary_size - 1, "room boundary must be gray except at the door"
    assert len(door_candidates) == 1, "expected exactly one door per room"
    door_rr, door_cc, color = door_candidates[0]

    if door_rr == 0 and 0 < door_cc < w - 1:
        direction = (-1, 0)
    elif door_rr == h - 1 and 0 < door_cc < w - 1:
        direction = (1, 0)
    elif door_cc == 0 and 0 < door_rr < h - 1:
        direction = (0, -1)
    elif door_cc == w - 1 and 0 < door_rr < h - 1:
        direction = (0, 1)
    else:
        raise AssertionError("door must lie on exactly one wall, not on a corner")

    interior = sub[1:-1, 1:-1]
    assert interior.size > 0, "room must have an interior"
    assert int(np.count_nonzero(interior == color)) == 1, "room must contain exactly one same-colored key"
    assert bool(np.all((interior == BLACK) | (interior == color))), "interior may only contain black cells and its key"

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(color),
        "door_r": int(r0 + door_rr),
        "door_c": int(c0 + door_cc),
        "direction": direction,
        "black_count": int(np.count_nonzero(interior == BLACK)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each room's key color outward from its matching door by the room's black-cell count.

    Input structure:
    - The background is black.
    - Several axis-aligned rectangular rooms are enclosed by one-pixel-thick gray walls.
    - Each room has exactly one colored door pixel on one wall and exactly one matching colored key pixel inside.

    Rule:
    - For every gray-wall connected component, recover its rectangular room and identify the door wall.
    - Count the black interior cells of that room, excluding the gray wall and the colored key.
    - Starting just outside the door, draw a straight ray outward in the door's normal direction for that many cells, stopping before any non-black pixel.

    Key insight:
    - The door's wall determines the ray direction, while the ray length is not geometric distance but the counted number of empty black cells inside the same room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms = [_parse_room(g, coords) for coords in _components(g == GRAY)]
    assert rooms, "expected at least one gray-walled room"

    out = g.copy()
    for room in sorted(rooms, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        dr, dc = room["direction"]
        rr = room["door_r"]
        cc = room["door_c"]
        for step in range(1, room["black_count"] + 1):
            nr = rr + step * dr
            nc = cc + step * dc
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                break
            if int(g[nr, nc]) != BLACK or int(out[nr, nc]) != BLACK:
                break
            out[nr, nc] = int(room["color"])

    return out


def _place_clear_room(
    grid: np.ndarray,
    reserved: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    side: str,
    color: int,
    require_full: bool,
) -> tuple[dict, list[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(CLEAR_ROOM_TRIES):
        spec = _sample_room_spec(side, color)
        outer_h = int(spec["inner_h"]) + 2
        outer_w = int(spec["inner_w"]) + 2
        max_r0 = h - outer_h - MARGIN
        max_c0 = w - outer_w - MARGIN
        if max_r0 < MARGIN or max_c0 < MARGIN:
            continue
        r0 = random.randint(MARGIN, max_r0)
        c0 = random.randint(MARGIN, max_c0)
        room = _instantiate_room(spec, r0, c0)
        if not _room_position_ok(room, bboxes, reserved):
            continue
        if not _cells_are_open(grid, reserved, _room_nonbg_cells(room)):
            continue

        full_path = _ray_cells_to_edge(room, grid.shape)
        if require_full and len(full_path) < room["black_count"]:
            continue
        visible_len = min(len(full_path), room["black_count"])
        if visible_len == 0:
            continue
        visible_path = full_path[:visible_len]
        if not _cells_are_open(grid, reserved, visible_path):
            continue
        return room, visible_path
    return None


def _construct_blocked_pair(
    grid_shape: tuple[int, int], spec_a: dict, spec_b: dict, distance: int
) -> tuple[dict, dict] | None:
    h, w = grid_shape
    outer_ha = int(spec_a["inner_h"]) + 2
    outer_wa = int(spec_a["inner_w"]) + 2
    outer_hb = int(spec_b["inner_h"]) + 2
    outer_wb = int(spec_b["inner_w"]) + 2
    side_a = spec_a["side"]

    if side_a == "right":
        max_c0a = w - outer_wa - outer_wb - distance
        max_r0a = h - outer_ha - MARGIN
        if max_c0a < MARGIN or max_r0a < MARGIN:
            return None
        r0a = random.randint(MARGIN, max_r0a)
        c0a = random.randint(MARGIN, max_c0a)
        room_a = _instantiate_room(spec_a, r0a, c0a)
        c0b = room_a["door_c"] + distance
        lo = max(MARGIN, room_a["door_r"] - (outer_hb - 2))
        hi = min(h - outer_hb - MARGIN, room_a["door_r"] - 1)
        if lo > hi:
            return None
        r0b = random.randint(lo, hi)
        room_b = _instantiate_room(spec_b, r0b, c0b)
    elif side_a == "left":
        min_c0a = MARGIN + distance + outer_wb - 1
        max_r0a = h - outer_ha - MARGIN
        max_c0a = w - outer_wa - MARGIN
        if min_c0a > max_c0a or max_r0a < MARGIN:
            return None
        r0a = random.randint(MARGIN, max_r0a)
        c0a = random.randint(min_c0a, max_c0a)
        room_a = _instantiate_room(spec_a, r0a, c0a)
        c0b = room_a["door_c"] - distance - outer_wb + 1
        lo = max(MARGIN, room_a["door_r"] - (outer_hb - 2))
        hi = min(h - outer_hb - MARGIN, room_a["door_r"] - 1)
        if lo > hi or c0b < MARGIN:
            return None
        r0b = random.randint(lo, hi)
        room_b = _instantiate_room(spec_b, r0b, c0b)
    elif side_a == "bottom":
        max_r0a = h - outer_ha - outer_hb - distance
        max_c0a = w - outer_wa - MARGIN
        if max_r0a < MARGIN or max_c0a < MARGIN:
            return None
        r0a = random.randint(MARGIN, max_r0a)
        c0a = random.randint(MARGIN, max_c0a)
        room_a = _instantiate_room(spec_a, r0a, c0a)
        r0b = room_a["door_r"] + distance
        lo = max(MARGIN, room_a["door_c"] - (outer_wb - 2))
        hi = min(w - outer_wb - MARGIN, room_a["door_c"] - 1)
        if lo > hi:
            return None
        c0b = random.randint(lo, hi)
        room_b = _instantiate_room(spec_b, r0b, c0b)
    else:
        assert side_a == "top"
        min_r0a = MARGIN + distance + outer_hb - 1
        max_r0a = h - outer_ha - MARGIN
        max_c0a = w - outer_wa - MARGIN
        if min_r0a > max_r0a or max_c0a < MARGIN:
            return None
        r0a = random.randint(min_r0a, max_r0a)
        c0a = random.randint(MARGIN, max_c0a)
        room_a = _instantiate_room(spec_a, r0a, c0a)
        r0b = room_a["door_r"] - distance - outer_hb + 1
        lo = max(MARGIN, room_a["door_c"] - (outer_wb - 2))
        hi = min(w - outer_wb - MARGIN, room_a["door_c"] - 1)
        if lo > hi or r0b < MARGIN:
            return None
        c0b = random.randint(lo, hi)
        room_b = _instantiate_room(spec_b, r0b, c0b)

    br0, bc0, br1, bc1 = room_b["bbox"]
    if br1 > h - MARGIN - 1 or bc1 > w - MARGIN - 1:
        return None
    if not _bboxes_have_gap(room_a["bbox"], room_b["bbox"]):
        return None
    return room_a, room_b


def _place_blocked_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    side_a: str,
    color_a: int,
    side_b: str,
    color_b: int,
) -> list[tuple[dict, list[tuple[int, int]]]] | None:
    for _ in range(PAIR_TRIES):
        spec_a = _sample_room_spec(side_a, color_a)
        if int(spec_a["inner_h"]) * int(spec_a["inner_w"]) <= 2:
            continue
        spec_b = _sample_room_spec(side_b, color_b)
        room_a_template = _instantiate_room(spec_a, 0, 0)
        max_distance = min(room_a_template["black_count"], 6)
        if max_distance < 2:
            continue
        distance = random.randint(2, max_distance)
        pair = _construct_blocked_pair(grid.shape, spec_a, spec_b, distance)
        if pair is None:
            continue
        room_a, room_b = pair
        if not _room_position_ok(room_a, [], reserved):
            continue
        if not _room_position_ok(room_b, [room_a["bbox"]], reserved):
            continue

        room_a_cells = _room_nonbg_cells(room_a)
        room_b_cells = _room_nonbg_cells(room_b)
        if not _cells_are_open(grid, reserved, room_a_cells):
            continue
        if not _cells_are_open(grid, reserved, room_b_cells):
            continue

        path_a = _ray_cells_to_edge(room_a, grid.shape)
        if len(path_a) < distance:
            continue
        prefix_a = path_a[: distance - 1]
        blocker = path_a[distance - 1]
        if blocker not in set(room_b_cells):
            continue
        if not _cells_are_open(grid, reserved, prefix_a):
            continue

        temp_grid = grid.copy()
        _paint_room(temp_grid, room_a)
        _paint_room(temp_grid, room_b)
        temp_reserved = reserved.copy()
        for r, c in prefix_a:
            temp_reserved[r, c] = True

        full_path_b = _ray_cells_to_edge(room_b, grid.shape)
        if len(full_path_b) < room_b["black_count"]:
            continue
        visible_b = full_path_b[: room_b["black_count"]]
        if not _cells_are_open(temp_grid, temp_reserved, visible_b):
            continue

        return [(room_a, prefix_a), (room_b, visible_b)]
    return None


def _ray_outcome(room: dict, grid: np.ndarray) -> tuple[str, int]:
    path = _ray_cells_to_edge(room, grid.shape)
    drawn = 0
    for r, c in path[: room["black_count"]]:
        if int(grid[r, c]) != BLACK:
            return "blocked", int(drawn)
        drawn += 1
    if drawn < room["black_count"]:
        return "edge", int(drawn)
    return "full", int(drawn)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        room_count = random.choices(ROOM_COUNT_CHOICES, weights=ROOM_COUNT_WEIGHTS, k=1)[0]
        colors = random.sample(list(ROOM_COLORS), room_count)
        sides = random.sample(list(SIDES), room_count)

        grid = np.full((h, w), BLACK, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        bboxes: list[tuple[int, int, int, int]] = []
        rooms: list[dict] = []

        blocked_mode = room_count >= 2 and random.random() < 0.5
        next_index = 0

        if blocked_mode:
            pair = _place_blocked_pair(
                grid,
                reserved,
                sides[0],
                colors[0],
                sides[1],
                colors[1],
            )
            if pair is None:
                continue
            for room, ray_cells in pair:
                _paint_room(grid, room)
                for r, c in ray_cells:
                    reserved[r, c] = True
                rooms.append(room)
                bboxes.append(room["bbox"])
            next_index = 2

        while next_index < room_count:
            require_full = next_index == len(rooms)
            placed = _place_clear_room(
                grid,
                reserved,
                bboxes,
                sides[next_index],
                colors[next_index],
                require_full=require_full,
            )
            if placed is None:
                break
            room, ray_cells = placed
            _paint_room(grid, room)
            for r, c in ray_cells:
                reserved[r, c] = True
            rooms.append(room)
            bboxes.append(room["bbox"])
            next_index += 1

        if next_index != room_count:
            continue

        outcomes = [_ray_outcome(room, grid) for room in rooms]
        if not any(kind == "full" for kind, _drawn in outcomes):
            continue
        if blocked_mode and not any(kind == "blocked" for kind, _drawn in outcomes):
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
