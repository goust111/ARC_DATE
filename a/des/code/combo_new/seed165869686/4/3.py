from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
BULB_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
ROOM_H_RANGE = (4, 8)
ROOM_W_RANGE = (4, 8)
GRID_RANGE = (16, 28)
GEN_RETRIES = 400


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    ps = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    ps[1:, 1:] = mask.astype(int)
    return ps.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _door_step(side: str) -> tuple[int, int]:
    if side == "top":
        return -1, 0
    if side == "bottom":
        return 1, 0
    if side == "left":
        return 0, -1
    if side == "right":
        return 0, 1
    raise ValueError(f"unknown side: {side}")


def _room_from_bbox(
    bbox: tuple[int, int, int, int], side: str, door_offset: int, color: int
) -> dict[str, object]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        door = (r0, c0 + door_offset)
    elif side == "bottom":
        door = (r1, c0 + door_offset)
    elif side == "left":
        door = (r0 + door_offset, c0)
    else:
        door = (r0 + door_offset, c1)

    interior_rows = list(range(r0 + 1, r1))
    interior_cols = list(range(c0 + 1, c1))
    if side in ("left", "right") and random.random() < 0.5:
        bulb_r = door[0]
    else:
        bulb_r = random.choice(interior_rows)
    if side in ("top", "bottom") and random.random() < 0.5:
        bulb_c = door[1]
    else:
        bulb_c = random.choice(interior_cols)

    return {
        "bbox": bbox,
        "side": side,
        "door": door,
        "bulb": (bulb_r, bulb_c),
        "color": int(color),
    }


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int
) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _bbox_hits_cells(bbox: tuple[int, int, int, int], cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in cells:
                return True
    return False


def _can_place_bbox(
    bbox: tuple[int, int, int, int],
    used: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    if _bbox_hits_cells(bbox, forbidden):
        return False
    return all(_boxes_separated(bbox, other, pad=1) for other in used)


def _sample_room_size() -> tuple[int, int]:
    return random.randint(*ROOM_H_RANGE), random.randint(*ROOM_W_RANGE)


def _place_random_room(
    height: int,
    width: int,
    color: int,
    used: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(200):
        room_h, room_w = _sample_room_size()
        if height - room_h - 2 < 1 or width - room_w - 2 < 1:
            continue
        side = random.choice(SIDES)
        if side in ("left", "right"):
            door_offset = random.randint(1, room_h - 2)
        else:
            door_offset = random.randint(1, room_w - 2)
        r0 = random.randint(1, height - room_h - 1)
        c0 = random.randint(1, width - room_w - 1)
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        if not _can_place_bbox(bbox, used, forbidden):
            continue
        return _room_from_bbox(bbox, side, door_offset, color)
    return None


def _place_horizontal_pair(
    height: int, width: int, colors: tuple[int, int]
) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    min_gap = 3
    for _ in range(200):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        door1 = random.randint(1, h1 - 2)
        door2 = random.randint(1, h2 - 2)
        row_lo = max(door1 + 1, door2 + 1)
        row_hi = min(height - h1 - 1 + door1, height - h2 - 1 + door2)
        if row_lo > row_hi:
            continue
        y = random.randint(row_lo, row_hi)
        r01 = y - door1
        r02 = y - door2
        left_max = width - w1 - w2 - min_gap - 2
        if left_max < 1:
            continue
        c01 = random.randint(1, left_max)
        c02_lo = c01 + w1 + min_gap
        c02_hi = width - w2 - 1
        if c02_lo > c02_hi:
            continue
        c02 = random.randint(c02_lo, c02_hi)

        bbox1 = (r01, c01, r01 + h1 - 1, c01 + w1 - 1)
        bbox2 = (r02, c02, r02 + h2 - 1, c02 + w2 - 1)
        if not _boxes_separated(bbox1, bbox2, pad=1):
            continue

        room1 = _room_from_bbox(bbox1, "right", door1, colors[0])
        room2 = _room_from_bbox(bbox2, "left", door2, colors[1])
        dc1 = int(room1["door"][1])
        dc2 = int(room2["door"][1])
        protected = {(y, c) for c in range(dc1, dc2 + 1)}
        return [room1, room2], protected
    return None


def _place_vertical_pair(
    height: int, width: int, colors: tuple[int, int]
) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    min_gap = 3
    for _ in range(200):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        door1 = random.randint(1, w1 - 2)
        door2 = random.randint(1, w2 - 2)
        col_lo = max(door1 + 1, door2 + 1)
        col_hi = min(width - w1 - 1 + door1, width - w2 - 1 + door2)
        if col_lo > col_hi:
            continue
        x = random.randint(col_lo, col_hi)
        c01 = x - door1
        c02 = x - door2
        top_max = height - h1 - h2 - min_gap - 2
        if top_max < 1:
            continue
        r01 = random.randint(1, top_max)
        r02_lo = r01 + h1 + min_gap
        r02_hi = height - h2 - 1
        if r02_lo > r02_hi:
            continue
        r02 = random.randint(r02_lo, r02_hi)

        bbox1 = (r01, c01, r01 + h1 - 1, c01 + w1 - 1)
        bbox2 = (r02, c02, r02 + h2 - 1, c02 + w2 - 1)
        if not _boxes_separated(bbox1, bbox2, pad=1):
            continue

        room1 = _room_from_bbox(bbox1, "bottom", door1, colors[0])
        room2 = _room_from_bbox(bbox2, "top", door2, colors[1])
        dr1 = int(room1["door"][0])
        dr2 = int(room2["door"][0])
        protected = {(r, x) for r in range(dr1, dr2 + 1)}
        return [room1, room2], protected
    return None


def _place_cross(
    height: int, width: int, colors: tuple[int, int]
) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    for _ in range(200):
        horiz_side = random.choice(("left", "right"))
        vert_side = random.choice(("top", "bottom"))

        hh, hw = _sample_room_size()
        vh, vw = _sample_room_size()
        horiz_offset = random.randint(1, hh - 2)
        vert_offset = random.randint(1, vw - 2)

        y_lo = horiz_offset + 1
        y_hi = height - hh - 1 + horiz_offset
        x_lo = vert_offset + 1
        x_hi = width - vw - 1 + vert_offset
        if y_lo > y_hi or x_lo > x_hi:
            continue
        y = random.randint(y_lo, y_hi)
        x = random.randint(x_lo, x_hi)

        hr0 = y - horiz_offset
        vc0 = x - vert_offset
        if horiz_side == "right":
            hc0_hi = x - hw - 1
            if hc0_hi < 1:
                continue
            hc0 = random.randint(1, hc0_hi)
        else:
            hc0_lo = x + 2
            hc0_hi = width - hw - 1
            if hc0_lo > hc0_hi:
                continue
            hc0 = random.randint(hc0_lo, hc0_hi)

        if vert_side == "bottom":
            vr0_hi = y - vh - 1
            if vr0_hi < 1:
                continue
            vr0 = random.randint(1, vr0_hi)
        else:
            vr0_lo = y + 2
            vr0_hi = height - vh - 1
            if vr0_lo > vr0_hi:
                continue
            vr0 = random.randint(vr0_lo, vr0_hi)

        hbbox = (hr0, hc0, hr0 + hh - 1, hc0 + hw - 1)
        vbbox = (vr0, vc0, vr0 + vh - 1, vc0 + vw - 1)
        if not _boxes_separated(hbbox, vbbox, pad=1):
            continue

        hroom = _room_from_bbox(hbbox, horiz_side, horiz_offset, colors[0])
        vroom = _room_from_bbox(vbbox, vert_side, vert_offset, colors[1])
        hdoor = hroom["door"]
        vdoor = vroom["door"]
        protected = set()
        for c in range(min(int(hdoor[1]), x), max(int(hdoor[1]), x) + 1):
            protected.add((y, c))
        for r in range(min(int(vdoor[0]), y), max(int(vdoor[0]), y) + 1):
            protected.add((r, x))
        return [hroom, vroom], protected
    return None


def _render_rooms(height: int, width: int, rooms: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        door_r, door_c = room["door"]
        bulb_r, bulb_c = room["bulb"]
        grid[door_r, door_c] = BG
        grid[bulb_r, bulb_c] = int(room["color"])
    return grid


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    bulb_positions = [tuple(map(int, rc)) for rc in np.argwhere((g != BG) & (g != WALL))]
    assert bulb_positions, "expected at least one bulb"

    wall_ps = _prefix_sum(g == WALL)
    bulb_ps = _prefix_sum((g != BG) & (g != WALL))
    best_by_bulb: dict[tuple[int, int], dict[str, object]] = {}

    for r0 in range(height - 2):
        for r1 in range(r0 + 2, height):
            box_h = r1 - r0 + 1
            for c0 in range(width - 2):
                for c1 in range(c0 + 2, width):
                    box_w = c1 - c0 + 1
                    if _rect_sum(wall_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if _rect_sum(bulb_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 1:
                        continue

                    top = _rect_sum(wall_ps, r0, c0, r0, c1)
                    bottom = _rect_sum(wall_ps, r1, c0, r1, c1)
                    left = _rect_sum(wall_ps, r0, c0, r1, c0)
                    right = _rect_sum(wall_ps, r0, c1, r1, c1)

                    side = None
                    if top == box_w - 1 and bottom == box_w and left == box_h and right == box_h:
                        side = "top"
                    elif bottom == box_w - 1 and top == box_w and left == box_h and right == box_h:
                        side = "bottom"
                    elif left == box_h - 1 and top == box_w and bottom == box_w and right == box_h:
                        side = "left"
                    elif right == box_h - 1 and top == box_w and bottom == box_w and left == box_h:
                        side = "right"
                    if side is None:
                        continue

                    if side == "top":
                        door_cols = [c for c in range(c0 + 1, c1) if int(g[r0, c]) != WALL]
                        if len(door_cols) != 1 or int(g[r0, door_cols[0]]) != BG:
                            continue
                        door = (r0, door_cols[0])
                    elif side == "bottom":
                        door_cols = [c for c in range(c0 + 1, c1) if int(g[r1, c]) != WALL]
                        if len(door_cols) != 1 or int(g[r1, door_cols[0]]) != BG:
                            continue
                        door = (r1, door_cols[0])
                    elif side == "left":
                        door_rows = [r for r in range(r0 + 1, r1) if int(g[r, c0]) != WALL]
                        if len(door_rows) != 1 or int(g[door_rows[0], c0]) != BG:
                            continue
                        door = (door_rows[0], c0)
                    else:
                        door_rows = [r for r in range(r0 + 1, r1) if int(g[r, c1]) != WALL]
                        if len(door_rows) != 1 or int(g[door_rows[0], c1]) != BG:
                            continue
                        door = (door_rows[0], c1)

                    bulb = None
                    for br, bc in bulb_positions:
                        if r0 < br < r1 and c0 < bc < c1:
                            bulb = (br, bc)
                            break
                    if bulb is None:
                        continue

                    candidate = {
                        "bbox": (r0, c0, r1, c1),
                        "side": side,
                        "door": door,
                        "bulb": bulb,
                        "color": int(g[bulb]),
                    }
                    prev = best_by_bulb.get(bulb)
                    if prev is None or box_h * box_w < (prev["bbox"][2] - prev["bbox"][0] + 1) * (
                        prev["bbox"][3] - prev["bbox"][1] + 1
                    ):
                        best_by_bulb[bulb] = candidate

    rooms = [best_by_bulb[bulb] for bulb in sorted(best_by_bulb)]
    assert len(rooms) == len(bulb_positions), "each bulb must belong to one rectangular room"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each room's bulb color outward through that room's single doorway and along the connected corridor.

    Input structure:
    - Black cells are corridor space.
    - Grey cells are walls that form axis-aligned rectangular rooms; each room wall has exactly one 1-cell doorway on its top, bottom, left, or right side.
    - Each room interior contains exactly one non-black, non-grey bulb pixel, and bulb colors are distinct.

    Rule:
    - Detect every rectangular room, identify its unique doorway side, and read the bulb color inside it.
    - Process bulbs in row-major scan order. Starting at the doorway cell, cast a straight ray outward in the doorway direction, recoloring corridor cells with the bulb color until the ray would hit a grey wall, a room interior, or leave the grid.
    - Keep every room interior pixel and every wall pixel unchanged; if a later ray overlaps an earlier ray, the later one overwrites that corridor segment.

    Key insight:
    - The hard part is structural: the doorway direction is not marked directly, but is the unique missing wall pixel on an otherwise rectangular room border, and that direction alone determines the ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms = _find_rooms(g)
    interior_mask = np.zeros_like(g, dtype=bool)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    blocked = (g == WALL) | interior_mask
    out = g.copy()
    for room in sorted(rooms, key=lambda item: item["bulb"]):
        color = int(room["color"])
        side = str(room["side"])
        r, c = room["door"]
        dr, dc = _door_step(side)
        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and not blocked[r, c]:
            out[r, c] = color
            r += dr
            c += dc
    return out


def generate_input() -> np.ndarray:
    modes = ("random", "hpair", "vpair", "cross")
    for _ in range(GEN_RETRIES):
        mode = random.choice(modes)
        if mode == "random":
            height = random.randint(14, 28)
            width = random.randint(14, 28)
            total_rooms = random.randint(2, 6)
        else:
            height = random.randint(18, 28)
            width = random.randint(18, 28)
            total_rooms = random.randint(2, 5)

        colors = random.sample(BULB_COLORS, total_rooms)
        rooms: list[dict[str, object]] = []
        used: list[tuple[int, int, int, int]] = []
        protected: set[tuple[int, int]] = set()

        if mode == "hpair":
            placed = _place_horizontal_pair(height, width, (colors[0], colors[1]))
            if placed is None:
                continue
            forced_rooms, protected = placed
            rooms.extend(forced_rooms)
            used.extend(room["bbox"] for room in forced_rooms)
            next_color = 2
        elif mode == "vpair":
            placed = _place_vertical_pair(height, width, (colors[0], colors[1]))
            if placed is None:
                continue
            forced_rooms, protected = placed
            rooms.extend(forced_rooms)
            used.extend(room["bbox"] for room in forced_rooms)
            next_color = 2
        elif mode == "cross":
            placed = _place_cross(height, width, (colors[0], colors[1]))
            if placed is None:
                continue
            forced_rooms, protected = placed
            rooms.extend(forced_rooms)
            used.extend(room["bbox"] for room in forced_rooms)
            next_color = 2
        else:
            next_color = 0

        while len(rooms) < total_rooms:
            room = _place_random_room(height, width, colors[next_color], used, protected)
            if room is None:
                break
            rooms.append(room)
            used.append(room["bbox"])
            next_color += 1

        if len(rooms) < 2:
            continue

        grid = _render_rooms(height, width, rooms)
        try:
            detected = _find_rooms(grid)
            output = main(grid)
        except AssertionError:
            continue

        if len(detected) != len(rooms):
            continue
        if np.array_equal(output, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid room-and-corridor puzzle")
