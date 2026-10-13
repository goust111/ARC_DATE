from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
WALL = 5
OBJECT = 9
PORTAL_COLORS = (1, 2, 3, 4, 6, 7, 8)

TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- shared helpers ---------------


def _rotate_vec(dr2: int, dc2: int, turns: int) -> tuple[int, int]:
    turns %= 4
    for _ in range(turns):
        dr2, dc2 = dc2, -dr2
    return dr2, dc2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _room_contains(room: dict, r: int, c: int) -> bool:
    return room["r0"] <= r <= room["r1"] and room["c0"] <= c <= room["c1"]


def _wall_indices(g: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = g.shape
    wall_rows = [
        r
        for r in range(h)
        if np.any(g[r, :] == WALL)
        and not np.any(g[r, :] == OBJECT)
        and np.all(g[r, :] != BG)
    ]
    wall_cols = [
        c
        for c in range(w)
        if np.any(g[:, c] == WALL)
        and not np.any(g[:, c] == OBJECT)
        and np.all(g[:, c] != BG)
    ]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2
    assert wall_rows[0] == 0 and wall_rows[-1] == h - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1
    return wall_rows, wall_cols


def _build_rooms_from_grid(g: np.ndarray) -> list[dict]:
    wall_rows, wall_cols = _wall_indices(g)
    row_blocks = [
        {
            "ri": ri,
            "top": wall_rows[ri],
            "bottom": wall_rows[ri + 1],
            "r0": wall_rows[ri] + 1,
            "r1": wall_rows[ri + 1] - 1,
        }
        for ri in range(len(wall_rows) - 1)
        if wall_rows[ri + 1] - wall_rows[ri] > 1
    ]
    col_blocks = [
        {
            "ci": ci,
            "left": wall_cols[ci],
            "right": wall_cols[ci + 1],
            "c0": wall_cols[ci] + 1,
            "c1": wall_cols[ci + 1] - 1,
        }
        for ci in range(len(wall_cols) - 1)
        if wall_cols[ci + 1] - wall_cols[ci] > 1
    ]
    assert row_blocks and col_blocks

    room_rows = len(row_blocks)
    room_cols = len(col_blocks)
    rooms: list[dict] = []
    room_id = 0
    for row in row_blocks:
        for col in col_blocks:
            available_sides = []
            if row["ri"] == 0:
                available_sides.append(TOP)
            if col["ci"] == room_cols - 1:
                available_sides.append(RIGHT)
            if row["ri"] == room_rows - 1:
                available_sides.append(BOTTOM)
            if col["ci"] == 0:
                available_sides.append(LEFT)
            rooms.append(
                {
                    "id": room_id,
                    "ri": row["ri"],
                    "ci": col["ci"],
                    "top": row["top"],
                    "bottom": row["bottom"],
                    "left": col["left"],
                    "right": col["right"],
                    "r0": row["r0"],
                    "r1": row["r1"],
                    "c0": col["c0"],
                    "c1": col["c1"],
                    "height": row["r1"] - row["r0"] + 1,
                    "width": col["c1"] - col["c0"] + 1,
                    "available_sides": tuple(available_sides),
                }
            )
            room_id += 1
    return rooms


def _scan_runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    runs = []
    i = 0
    while i < len(values):
        if int(values[i]) == WALL:
            i += 1
            continue
        color = int(values[i])
        j = i
        while j + 1 < len(values) and int(values[j + 1]) == color:
            j += 1
        runs.append((i, j, color))
        i = j + 1
    return runs


def _extract_room_portals(g: np.ndarray, room: dict) -> list[dict]:
    portals: list[dict] = []
    side_specs = [
        (TOP, room["top"], g[room["top"], room["c0"] : room["c1"] + 1]),
        (BOTTOM, room["bottom"], g[room["bottom"], room["c0"] : room["c1"] + 1]),
        (LEFT, room["left"], g[room["r0"] : room["r1"] + 1, room["left"]]),
        (RIGHT, room["right"], g[room["r0"] : room["r1"] + 1, room["right"]]),
    ]
    for side, fixed, line in side_specs:
        for start, end, color in _scan_runs(line):
            assert color not in (BG, WALL, OBJECT)
            assert end - start + 1 == 2
            if side in (TOP, BOTTOM):
                abs_c = room["c0"] + start
                cells = ((fixed, abs_c), (fixed, abs_c + 1))
                center2 = (2 * fixed, 2 * abs_c + 1)
            else:
                abs_r = room["r0"] + start
                cells = ((abs_r, fixed), (abs_r + 1, fixed))
                center2 = (2 * abs_r + 1, 2 * fixed)
            portals.append(
                {
                    "room_id": room["id"],
                    "side": side,
                    "color": color,
                    "cells": cells,
                    "center2": center2,
                }
            )
    assert 1 <= len(portals) <= 2
    return portals


def _portal_map(rooms: list[dict]) -> dict[int, list[dict]]:
    mapping: dict[int, list[dict]] = {}
    for room in rooms:
        mapping[room["id"]] = room["portals"]
    return mapping


def _move_object(
    grid: np.ndarray,
    object_coords: list[tuple[int, int]],
    src_room: dict,
    src_portal: dict,
    dst_room: dict,
    dst_portal: dict,
) -> np.ndarray:
    turns = (dst_portal["side"] - src_portal["side"]) % 4
    out = np.array(grid, copy=True)
    for r, c in object_coords:
        out[r, c] = BG

    dst_coords: list[tuple[int, int]] = []
    seen = set()
    for r, c in object_coords:
        dr2 = 2 * int(r) - int(src_portal["center2"][0])
        dc2 = 2 * int(c) - int(src_portal["center2"][1])
        rr2, cc2 = _rotate_vec(dr2, dc2, turns)
        nr2 = int(dst_portal["center2"][0]) + rr2
        nc2 = int(dst_portal["center2"][1]) + cc2
        assert nr2 % 2 == 0 and nc2 % 2 == 0
        nr, nc = nr2 // 2, nc2 // 2
        assert _room_contains(dst_room, nr, nc)
        assert (nr, nc) not in seen
        seen.add((nr, nc))
        dst_coords.append((nr, nc))

    for nr, nc in dst_coords:
        assert out[nr, nc] == BG
        out[nr, nc] = OBJECT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Teleport the maroon object between matching portals, rotating by wall orientation."""
    g = np.asarray(grid, dtype=int)
    rooms = _build_rooms_from_grid(g)
    room_by_id = {room["id"]: room for room in rooms}

    all_portals: list[dict] = []
    for room in rooms:
        room["portals"] = _extract_room_portals(g, room)
        all_portals.extend(room["portals"])

    portals_by_color: dict[int, list[dict]] = {}
    for portal in all_portals:
        portals_by_color.setdefault(int(portal["color"]), []).append(portal)
    assert portals_by_color
    for color, entries in portals_by_color.items():
        assert color not in (BG, WALL, OBJECT)
        assert len(entries) == 2
        assert entries[0]["room_id"] != entries[1]["room_id"]

    object_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == OBJECT)]
    assert object_coords
    object_mask = g == OBJECT
    assert len(_components(object_mask)) == 1

    source_room = None
    first_r, first_c = object_coords[0]
    for room in rooms:
        if _room_contains(room, first_r, first_c):
            source_room = room
            break
    assert source_room is not None
    assert all(_room_contains(source_room, r, c) for r, c in object_coords)
    assert len(source_room["portals"]) == 1

    src_portal = source_room["portals"][0]
    matches = [
        portal
        for portal in portals_by_color[int(src_portal["color"])]
        if portal["room_id"] != source_room["id"]
    ]
    assert len(matches) == 1
    dst_portal = matches[0]
    dst_room = room_by_id[int(dst_portal["room_id"])]
    assert dst_room["id"] != source_room["id"]

    return _move_object(g, object_coords, source_room, src_portal, dst_room, dst_portal)


# --------------- generator ---------------

ROOM_LAYOUTS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (3, 1),
    (4, 1),
    (2, 2),
    (2, 3),
    (3, 2),
)
ROOM_SIZE_MIN = 5
ROOM_SIZE_MAX = 6
OBJECT_BBOX_MAX = 3
OBJECT_AREA_MIN = 2
OBJECT_AREA_MAX = 5
GENERATION_RETRIES = 400


def _build_empty_layout(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[dict]]:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    g = np.full((height, width), BG, dtype=int)

    wall_rows = [0]
    acc = 0
    for room_h in row_heights:
        acc += room_h + 1
        wall_rows.append(acc)
    wall_cols = [0]
    acc = 0
    for room_w in col_widths:
        acc += room_w + 1
        wall_cols.append(acc)

    g[wall_rows, :] = WALL
    g[:, wall_cols] = WALL

    rooms: list[dict] = []
    room_id = 0
    room_rows = len(row_heights)
    room_cols = len(col_widths)
    for ri in range(room_rows):
        for ci in range(room_cols):
            available_sides = []
            if ri == 0:
                available_sides.append(TOP)
            if ci == room_cols - 1:
                available_sides.append(RIGHT)
            if ri == room_rows - 1:
                available_sides.append(BOTTOM)
            if ci == 0:
                available_sides.append(LEFT)
            rooms.append(
                {
                    "id": room_id,
                    "ri": ri,
                    "ci": ci,
                    "top": wall_rows[ri],
                    "bottom": wall_rows[ri + 1],
                    "left": wall_cols[ci],
                    "right": wall_cols[ci + 1],
                    "r0": wall_rows[ri] + 1,
                    "r1": wall_rows[ri + 1] - 1,
                    "c0": wall_cols[ci] + 1,
                    "c1": wall_cols[ci + 1] - 1,
                    "height": row_heights[ri],
                    "width": col_widths[ci],
                    "available_sides": tuple(available_sides),
                }
            )
            room_id += 1
    return g, rooms


def _slot_candidates(room: dict) -> list[dict]:
    slots: list[dict] = []
    for side in room["available_sides"]:
        if side == TOP:
            row = room["top"]
            for c in range(room["c0"], room["c1"]):
                slots.append(
                    {
                        "room_id": room["id"],
                        "side": side,
                        "anchor": c,
                        "cells": ((row, c), (row, c + 1)),
                        "center2": (2 * row, 2 * c + 1),
                    }
                )
        elif side == BOTTOM:
            row = room["bottom"]
            for c in range(room["c0"], room["c1"]):
                slots.append(
                    {
                        "room_id": room["id"],
                        "side": side,
                        "anchor": c,
                        "cells": ((row, c), (row, c + 1)),
                        "center2": (2 * row, 2 * c + 1),
                    }
                )
        elif side == LEFT:
            col = room["left"]
            for r in range(room["r0"], room["r1"]):
                slots.append(
                    {
                        "room_id": room["id"],
                        "side": side,
                        "anchor": r,
                        "cells": ((r, col), (r + 1, col)),
                        "center2": (2 * r + 1, 2 * col),
                    }
                )
        else:
            col = room["right"]
            for r in range(room["r0"], room["r1"]):
                slots.append(
                    {
                        "room_id": room["id"],
                        "side": side,
                        "anchor": r,
                        "cells": ((r, col), (r + 1, col)),
                        "center2": (2 * r + 1, 2 * col),
                    }
                )
    return slots


def _slots_compatible(a: dict, b: dict) -> bool:
    if a["side"] != b["side"]:
        return True
    return abs(int(a["anchor"]) - int(b["anchor"])) >= 3


def _choose_room_slots(room: dict, degree: int, forced_sides: list[int]) -> list[dict] | None:
    side_requirements = Counter(forced_sides)
    candidates = _slot_candidates(room)
    valid = []
    for combo in combinations(candidates, degree):
        if any(not _slots_compatible(combo[i], combo[j]) for i in range(degree) for j in range(i + 1, degree)):
            continue
        counts = Counter(int(slot["side"]) for slot in combo)
        if any(counts[side] < need for side, need in side_requirements.items()):
            continue
        valid.append(combo)
    if not valid:
        return None
    picked = random.choice(valid)
    return [dict(slot) for slot in picked]


def _pair_remaining_slots(slots: list[dict]) -> list[tuple[dict, dict]] | None:
    if not slots:
        return []
    first = slots[0]
    candidate_indices = list(range(1, len(slots)))
    random.shuffle(candidate_indices)
    for idx in candidate_indices:
        second = slots[idx]
        if int(first["room_id"]) == int(second["room_id"]):
            continue
        rest = slots[1:idx] + slots[idx + 1 :]
        sub = _pair_remaining_slots(rest)
        if sub is not None:
            return [(first, second), *sub]
    return None


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier
        cells.add(random.choice(frontier))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _choose_object_positions(src_room: dict, dst_room: dict, src_slot: dict, dst_slot: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    turns = (int(dst_slot["side"]) - int(src_slot["side"])) % 4
    max_h = min(OBJECT_BBOX_MAX, int(src_room["height"]), int(dst_room["height"]))
    max_w = min(OBJECT_BBOX_MAX, int(src_room["width"]), int(dst_room["width"]))
    if max_h < 1 or max_w < 1:
        return None

    for _ in range(300):
        bbox_h = random.randint(1, max_h)
        bbox_w = random.randint(1, max_w)
        if bbox_h * bbox_w < OBJECT_AREA_MIN:
            continue
        area = random.randint(OBJECT_AREA_MIN, min(OBJECT_AREA_MAX, bbox_h * bbox_w))
        mask = _random_connected_mask(bbox_h, bbox_w, area)
        h, w = mask.shape
        src_top = random.randint(int(src_room["r0"]), int(src_room["r1"]) - h + 1)
        src_left = random.randint(int(src_room["c0"]), int(src_room["c1"]) - w + 1)
        src_coords = [
            (src_top + int(dr), src_left + int(dc))
            for dr, dc in np.argwhere(mask)
        ]

        dst_coords = []
        seen = set()
        ok = True
        for r, c in src_coords:
            dr2 = 2 * int(r) - int(src_slot["center2"][0])
            dc2 = 2 * int(c) - int(src_slot["center2"][1])
            rr2, cc2 = _rotate_vec(dr2, dc2, turns)
            nr2 = int(dst_slot["center2"][0]) + rr2
            nc2 = int(dst_slot["center2"][1]) + cc2
            if nr2 % 2 or nc2 % 2:
                ok = False
                break
            nr, nc = nr2 // 2, nc2 // 2
            if not _room_contains(dst_room, nr, nc) or (nr, nc) in seen:
                ok = False
                break
            seen.add((nr, nc))
            dst_coords.append((nr, nc))
        if ok:
            return src_coords, dst_coords
    return None


def _paint_portal(grid: np.ndarray, slot: dict, color: int) -> None:
    for r, c in slot["cells"]:
        grid[int(r), int(c)] = int(color)


def _feasible_main_pairs(rooms: list[dict], turns: int) -> list[tuple[int, int, int, int]]:
    choices = []
    for src_room in rooms:
        for src_side in src_room["available_sides"]:
            dst_side = (int(src_side) + turns) % 4
            for dst_room in rooms:
                if dst_room["id"] == src_room["id"]:
                    continue
                if dst_side in dst_room["available_sides"]:
                    choices.append((src_room["id"], int(src_side), dst_room["id"], dst_side))
    return choices


def _sample_degrees(room_count: int, src_room_id: int) -> list[int] | None:
    for _ in range(100):
        degrees = [random.randint(1, 2) for _ in range(room_count)]
        degrees[src_room_id] = 1
        if sum(degrees) % 2 == 1:
            candidates = [idx for idx in range(room_count) if idx != src_room_id]
            if not candidates:
                continue
            chosen = random.choice(candidates)
            degrees[chosen] = 3 - degrees[chosen]
        if sum(degrees) // 2 <= len(PORTAL_COLORS):
            return degrees
    return None


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_LAYOUTS)
    row_heights = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_rows)]
    col_widths = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_cols)]
    base, rooms = _build_empty_layout(row_heights, col_widths)
    room_by_id = {room["id"]: room for room in rooms}

    turns = random.randrange(4)
    main_pair_choices = _feasible_main_pairs(rooms, turns)
    if not main_pair_choices:
        return None
    src_room_id, src_side, dst_room_id, dst_side = random.choice(main_pair_choices)

    degrees = _sample_degrees(len(rooms), src_room_id)
    if degrees is None:
        return None

    room_slots: dict[int, list[dict]] = {}
    for room in rooms:
        forced = [src_side] if room["id"] == src_room_id else []
        if room["id"] == dst_room_id:
            forced.append(dst_side)
        chosen = _choose_room_slots(room, degrees[room["id"]], forced)
        if chosen is None:
            return None
        room_slots[room["id"]] = chosen

    src_candidates = [slot for slot in room_slots[src_room_id] if int(slot["side"]) == src_side]
    dst_candidates = [slot for slot in room_slots[dst_room_id] if int(slot["side"]) == dst_side]
    if not src_candidates or not dst_candidates:
        return None
    src_slot = random.choice(src_candidates)
    dst_slot = random.choice(dst_candidates)

    free_slots = []
    for room_id, slots in room_slots.items():
        used_src = False
        used_dst = False
        for slot in slots:
            if room_id == src_room_id and not used_src and slot["cells"] == src_slot["cells"]:
                used_src = True
                continue
            if room_id == dst_room_id and not used_dst and slot["cells"] == dst_slot["cells"]:
                used_dst = True
                continue
            free_slots.append(slot)

    other_pairs = _pair_remaining_slots(free_slots)
    if other_pairs is None:
        return None
    all_pairs = [(src_slot, dst_slot), *other_pairs]
    if len(all_pairs) > len(PORTAL_COLORS):
        return None

    colors = random.sample(PORTAL_COLORS, len(all_pairs))
    for color, (a, b) in zip(colors, all_pairs):
        _paint_portal(base, a, color)
        _paint_portal(base, b, color)
        a["color"] = color
        b["color"] = color

    src_room = room_by_id[src_room_id]
    dst_room = room_by_id[dst_room_id]
    obj = _choose_object_positions(src_room, dst_room, src_slot, dst_slot)
    if obj is None:
        return None
    src_coords, dst_coords = obj

    inp = np.array(base, copy=True)
    out = np.array(base, copy=True)
    for r, c in src_coords:
        inp[int(r), int(c)] = OBJECT
    for r, c in dst_coords:
        out[int(r), int(c)] = OBJECT

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-room input")
