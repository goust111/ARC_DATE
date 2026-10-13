from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
DIR_TO_ENTRY_SIDE = {
    (-1, 0): "S",
    (1, 0): "N",
    (0, -1): "E",
    (0, 1): "W",
}
DOOR_COLORS = [BLUE, RED, GREEN, PINK, ORANGE, MAROON]
SPECIAL_INPUT_COLORS = {BLACK, YELLOW, GREY, TEAL}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _marker_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    yellow_cells = np.argwhere(grid == YELLOW)
    teal_cells = np.argwhere(grid == TEAL)
    assert yellow_cells.shape == (1, 2), "expected exactly one yellow head cell"
    assert teal_cells.shape == (1, 2), "expected exactly one teal tail cell"
    head = tuple(map(int, yellow_cells[0]))
    tail = tuple(map(int, teal_cells[0]))
    dr = head[0] - tail[0]
    dc = head[1] - tail[1]
    assert abs(dr) + abs(dc) == 1, "marker cells must be orthogonally adjacent"
    return head, (dr, dc)


def _door_outward_direction(
    grid: np.ndarray, room_mask: np.ndarray, door: tuple[int, int]
) -> tuple[int, int]:
    r, c = door
    inside_dirs = []
    outside_dirs = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            continue
        if room_mask[nr, nc]:
            inside_dirs.append((dr, dc))
        elif grid[nr, nc] == BLACK:
            outside_dirs.append((dr, dc))
    assert len(inside_dirs) == 1, "door must touch exactly one room-interior cell"
    assert len(outside_dirs) == 1, "door must expose exactly one corridor direction"
    return outside_dirs[0]


def _rooms(grid: np.ndarray) -> list[dict]:
    enclosed = []
    for coords in _component_coords(grid == BLACK):
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (
            np.any(rows == 0)
            or np.any(rows == grid.shape[0] - 1)
            or np.any(cols == 0)
            or np.any(cols == grid.shape[1] - 1)
        ):
            continue
        room_mask = np.zeros(grid.shape, dtype=bool)
        room_mask[rows, cols] = True
        doors = set()
        colors = set()
        for r, c in coords:
            for dr, dc in DIRS4:
                nr, nc = int(r) + dr, int(c) + dc
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                value = int(grid[nr, nc])
                if value not in SPECIAL_INPUT_COLORS:
                    doors.add((nr, nc))
                    colors.add(value)
        assert len(doors) == 2, "each enclosed room must have exactly two doors"
        assert len(colors) == 1, "each room's two doors must share one color"
        enclosed.append(
            {
                "cells": coords,
                "mask": room_mask,
                "door_color": int(next(iter(colors))),
                "doors": sorted(doors),
            }
        )
    assert enclosed, "expected at least one enclosed room"
    return enclosed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the direction marker launch a door-to-door chain that floods only the rooms it reaches.

    Input structure:
    - Grey wall loops on a black field enclose several disconnected black rooms.
    - Each room boundary contains exactly two same-colored door pixels, and different rooms use different door colors.
    - One corridor contains a two-cell direction marker made of a teal tail and an adjacent yellow head.

    Rule:
    - Fire a straight ray from the yellow head in the direction from tail to head, reading the original black corridor geometry.
    - Whenever the ray hits a door, flood that room's enclosed black interior with the door color, then restart the ray from the room's other door in the outward direction of that second door.
    - Paint every traversed corridor cell yellow, keep walls and doors unchanged, and stop when the active ray exits the grid.

    Key insight:
    - Rooms are not colored independently; a room changes only if the geometric door chain reaches it, and each activated room chooses the next ray direction through its unused door.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    head, direction = _marker_and_direction(g)
    rooms = _rooms(g)
    door_to_room = {}
    for idx, room in enumerate(rooms):
        for door in room["doors"]:
            assert door not in door_to_room, "doors must belong to only one room"
            door_to_room[door] = idx

    out = g.copy()
    current = head
    visited_rooms = set()

    while True:
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            break
        pos = (nr, nc)
        value = int(g[pos])
        if value == BLACK:
            out[pos] = YELLOW
            current = pos
            continue

        assert pos in door_to_room, "ray may only strike a door or leave the grid"
        room_idx = door_to_room[pos]
        assert room_idx not in visited_rooms, "door chain must not revisit a room"
        visited_rooms.add(room_idx)
        room = rooms[room_idx]
        out[room["cells"][:, 0], room["cells"][:, 1]] = room["door_color"]
        exit_door = room["doors"][1] if room["doors"][0] == pos else room["doors"][0]
        direction = _door_outward_direction(g, room["mask"], exit_door)
        current = exit_door

    return out


def _door_coord(box: tuple[int, int, int, int], side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "N":
        return r0, c0 + offset
    if side == "S":
        return r1, c0 + offset
    if side == "W":
        return r0 + offset, c0
    return r0 + offset, c1


def _box_from_entry(
    entry: tuple[int, int], side: str, height: int, width: int, offset: int
) -> tuple[int, int, int, int]:
    r, c = entry
    if side == "N":
        r0, c0 = r, c - offset
    elif side == "S":
        r0, c0 = r - height + 1, c - offset
    elif side == "W":
        r0, c0 = r - offset, c
    else:
        r0, c0 = r - offset, c - width + 1
    return r0, c0, r0 + height - 1, c0 + width - 1


def _iter_line_exclusive(
    start: tuple[int, int], direction: tuple[int, int], steps: int
) -> list[tuple[int, int]]:
    cells = []
    r, c = start
    for k in range(1, steps):
        cells.append((r + direction[0] * k, c + direction[1] * k))
    return cells


def _box_ok(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    occupied: np.ndarray,
    padded: np.ndarray,
    reserved: np.ndarray,
) -> bool:
    r0, c0, r1, c1 = box
    if not (0 <= r0 < r1 < shape[0] and 0 <= c0 < c1 < shape[1]):
        return False
    if np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    if np.any(reserved[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    if np.any(padded[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    return True


def _mark_box(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 0) -> None:
    r0, c0, r1, c1 = box
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(mask.shape[0] - 1, r1 + pad)
    c1 = min(mask.shape[1] - 1, c1 + pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _render_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["box"]
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    for door in room["doors"]:
        grid[door] = room["color"]


def _place_chain(
    shape: tuple[int, int],
    rng: random.Random,
    chain_colors: list[int],
    distractor_colors: list[int],
) -> list[dict] | None:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    padded = np.zeros(shape, dtype=bool)
    reserved = np.zeros(shape, dtype=bool)

    chain_len = len(chain_colors)
    must_turn = chain_len >= 3 and rng.random() < 0.7
    initial_dir = rng.choice(list(DIR_TO_ENTRY_SIDE.keys()))

    for _ in range(50):
        head = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
        tail = (head[0] - initial_dir[0], head[1] - initial_dir[1])
        if _in_bounds(shape, tail[0], tail[1]):
            break
    else:
        return None

    reserved[head] = True
    reserved[tail] = True

    rooms = []
    current = head
    direction = initial_dir
    saw_turn = False

    for idx in range(chain_len):
        entry_side = DIR_TO_ENTRY_SIDE[direction]
        success = False
        for _ in range(80):
            box_h = rng.randint(5, 8)
            box_w = rng.randint(5, 8)
            gap = rng.randint(2, 6)
            entry = (current[0] + direction[0] * gap, current[1] + direction[1] * gap)
            if not _in_bounds(shape, entry[0], entry[1]):
                continue
            offset_hi = box_h - 2 if entry_side in ("W", "E") else box_w - 2
            offset = rng.randint(1, offset_hi)
            box = _box_from_entry(entry, entry_side, box_h, box_w, offset)
            path = _iter_line_exclusive(current, direction, gap)
            if not path or any(not _in_bounds(shape, r, c) for r, c in path):
                continue
            if not _box_ok(box, shape, occupied, padded, reserved):
                continue
            if any(occupied[r, c] or reserved[r, c] for r, c in path):
                continue

            straight = [side for side in "NSEW" if side != entry_side and SIDE_TO_DIR[side] == direction]
            turns = [side for side in "NSEW" if side != entry_side and SIDE_TO_DIR[side] != direction]
            rng.shuffle(straight)
            rng.shuffle(turns)
            exit_sides = turns + straight if rng.random() < 0.7 else straight + turns
            chosen = None
            for exit_side in exit_sides:
                exit_offset_hi = box_h - 2 if exit_side in ("W", "E") else box_w - 2
                exit_offset = rng.randint(1, exit_offset_hi)
                exit_door = _door_coord(box, exit_side, exit_offset)
                if exit_door == entry:
                    continue
                out_dir = SIDE_TO_DIR[exit_side]
                outside = (exit_door[0] + out_dir[0], exit_door[1] + out_dir[1])
                if not _in_bounds(shape, outside[0], outside[1]):
                    continue
                if occupied[outside] or reserved[outside]:
                    continue
                chosen = (exit_door, exit_side, out_dir)
                break
            if chosen is None:
                continue

            exit_door, exit_side, next_dir = chosen
            room = {
                "box": box,
                "doors": [entry, exit_door],
                "color": chain_colors[idx],
                "chain": True,
            }
            rooms.append(room)
            _mark_box(occupied, box)
            _mark_box(padded, box, pad=1)
            for r, c in path:
                reserved[r, c] = True
            current = exit_door
            saw_turn |= next_dir != direction
            direction = next_dir
            success = True
            break
        if not success:
            return None

    exit_cells = []
    r, c = current
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(shape, r, c):
            break
        if occupied[r, c] or reserved[r, c]:
            return None
        exit_cells.append((r, c))
    if len(exit_cells) < 2:
        return None
    for cell in exit_cells:
        reserved[cell] = True

    if must_turn and not saw_turn:
        return None

    for color in distractor_colors:
        placed = False
        for _ in range(80):
            box_h = rng.randint(5, 8)
            box_w = rng.randint(5, 8)
            r0 = rng.randint(0, h - box_h)
            c0 = rng.randint(0, w - box_w)
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if not _box_ok(box, shape, occupied, padded, reserved):
                continue
            side1, side2 = rng.sample(list("NSEW"), 2)
            off1_hi = box_h - 2 if side1 in ("W", "E") else box_w - 2
            off2_hi = box_h - 2 if side2 in ("W", "E") else box_w - 2
            door1 = _door_coord(box, side1, rng.randint(1, off1_hi))
            door2 = _door_coord(box, side2, rng.randint(1, off2_hi))
            if door1 == door2:
                continue
            out1 = SIDE_TO_DIR[side1]
            out2 = SIDE_TO_DIR[side2]
            ext1 = (door1[0] + out1[0], door1[1] + out1[1])
            ext2 = (door2[0] + out2[0], door2[1] + out2[1])
            if not (
                _in_bounds(shape, ext1[0], ext1[1])
                and _in_bounds(shape, ext2[0], ext2[1])
                and not occupied[ext1]
                and not occupied[ext2]
                and not reserved[ext1]
                and not reserved[ext2]
            ):
                continue
            room = {
                "box": box,
                "doors": [door1, door2],
                "color": color,
                "chain": False,
            }
            rooms.append(room)
            _mark_box(occupied, box)
            _mark_box(padded, box, pad=1)
            placed = True
            break
        if not placed:
            return None

    marker = {"head": head, "tail": tail}
    rooms.append({"marker": marker})
    return rooms


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(22, 29), rng.randint(22, 29))
    chain_len = rng.choices([2, 3, 4], weights=[2, 3, 2])[0]
    distractor_count = rng.randint(1, min(2, len(DOOR_COLORS) - chain_len))
    colors = rng.sample(DOOR_COLORS, chain_len + distractor_count)
    spec = _place_chain(shape, rng, colors[:chain_len], colors[chain_len:])
    if spec is None:
        return None

    grid = np.zeros(shape, dtype=int)
    marker = spec[-1]["marker"]
    rooms = spec[:-1]
    if len(rooms) != chain_len + distractor_count:
        return None
    for room in rooms:
        _render_room(grid, room)
    grid[marker["tail"]] = TEAL
    grid[marker["head"]] = YELLOW

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    changed_room_colors = {
        room["color"]
        for room in rooms
        if np.any(out[room["box"][0] + 1 : room["box"][2], room["box"][1] + 1 : room["box"][3]] == room["color"])
    }
    if len(changed_room_colors) < 2:
        return None
    if len(changed_room_colors) >= len(rooms):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(600):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-chain flood scene")
