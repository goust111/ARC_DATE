from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ROOM_SIZES = (3, 5, 7)
GEN_RETRIES = 400
CLEARANCE = 1


@dataclass(frozen=True)
class Room:
    r0: int
    c0: int
    size: int

    @property
    def r1(self) -> int:
        return self.r0 + self.size - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.size - 1

    @property
    def center_r(self) -> int:
        return self.r0 + self.size // 2

    @property
    def center_c(self) -> int:
        return self.c0 + self.size // 2


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_source_square(grid: np.ndarray) -> tuple[int, int]:
    red = np.argwhere(grid == RED)
    assert red.shape == (9, 2), "expected exactly one solid 3x3 red square"
    r0 = int(red[:, 0].min())
    c0 = int(red[:, 1].min())
    r1 = int(red[:, 0].max())
    c1 = int(red[:, 1].max())
    assert (r1 - r0, c1 - c0) == (2, 2), "red source must be 3x3"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RED), "red source must be solid"
    return r0 + 1, c0 + 1


def _find_rooms(grid: np.ndarray) -> list[Room]:
    rooms: list[Room] = []
    for comp in _find_components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        size = r1 - r0 + 1
        assert size >= 3 and size % 2 == 1 and size == c1 - c0 + 1, "room must be an odd square"
        assert len(comp) == 4 * size - 4, "room frame must be one cell thick"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == GREY), "room border must be grey"
        assert np.all(patch[~border] == BG), "room interior must be black"
        rooms.append(Room(r0, c0, size))
    return rooms


def _room_wall_lookup(rooms: list[Room]) -> dict[tuple[int, int], Room]:
    lookup: dict[tuple[int, int], Room] = {}
    for room in rooms:
        for r in range(room.r0, room.r1 + 1):
            for c in range(room.c0, room.c1 + 1):
                if r in (room.r0, room.r1) or c in (room.c0, room.c1):
                    lookup[(r, c)] = room
    return lookup


def _trace_ray(
    grid: np.ndarray,
    out: np.ndarray,
    start: tuple[int, int],
    direction: str,
    room_lookup: dict[tuple[int, int], Room],
) -> None:
    dr, dc = DIRS[direction]
    r, c = start
    while _in_bounds(grid.shape, r, c):
        val = int(grid[r, c])
        if val == BG:
            out[r, c] = RED
            r += dr
            c += dc
            continue
        if val == BLUE:
            return

        assert val == GREY, "rays may only encounter background, blue blockers, or grey room walls"
        room = room_lookup[(r, c)]
        aligned = (dr == 0 and r == room.center_r) or (dc == 0 and c == room.center_c)
        if not aligned:
            return

        out[room.r0 + 1 : room.r1, room.c0 + 1 : room.c1] = RED
        if direction == "up":
            r, c = room.r0 - 1, room.center_c
        elif direction == "down":
            r, c = room.r1 + 1, room.center_c
        elif direction == "left":
            r, c = room.center_r, room.c0 - 1
        else:
            r, c = room.center_r, room.c1 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four red rays from the 3x3 source square, lighting only center-aligned framed rooms.

    Input structure:
    - The background is black.
    - One solid 3x3 red square is the unique ray source.
    - Grey odd-sized hollow square rooms and isolated blue blocker pixels are scattered around it.

    Rule:
    - Launch one one-cell-thick red ray outward from each side-center of the red square.
    - A ray paints black cells red until the grid edge, but stops immediately before a blue blocker.
    - If a ray hits a grey room wall on that room's center row or center column, fill the room interior red and resume from the aligned cell just outside the opposite wall; if the hit is off-axis, stop before the wall and do not fill the room.

    Key insight:
    - Grey rooms are penetrable only through their center axis, so the task is ray tracing with selective pass-through and chained room activation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(np.unique(g)).issubset({BG, BLUE, RED, GREY}), "unexpected colors present"

    center_r, center_c = _find_source_square(g)
    rooms = _find_rooms(g)
    room_lookup = _room_wall_lookup(rooms)

    out = g.copy()
    starts = {
        "up": (center_r - 2, center_c),
        "down": (center_r + 2, center_c),
        "left": (center_r, center_c - 2),
        "right": (center_r, center_c + 2),
    }
    for direction, start in starts.items():
        _trace_ray(g, out, start, direction, room_lookup)
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = CLEARANCE) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    return not separated


def _can_place_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    reserved: list[tuple[int, int, int, int]],
) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= shape[0] or c1 >= shape[1]:
        return False
    return all(not _boxes_too_close(box, other) for other in reserved)


def _paint_room(grid: np.ndarray, room: Room) -> None:
    grid[room.r0, room.c0 : room.c1 + 1] = GREY
    grid[room.r1, room.c0 : room.c1 + 1] = GREY
    grid[room.r0 : room.r1 + 1, room.c0] = GREY
    grid[room.r0 : room.r1 + 1, room.c1] = GREY


def _try_place_room(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    room: Room,
) -> bool:
    box = (room.r0, room.c0, room.r1, room.c1)
    if not _can_place_box(box, grid.shape, reserved):
        return False
    if np.any(grid[room.r0 : room.r1 + 1, room.c0 : room.c1 + 1] != BG):
        return False
    _paint_room(grid, room)
    reserved.append(box)
    return True


def _try_place_blocker(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    r: int,
    c: int,
) -> bool:
    box = (r, c, r, c)
    if not _can_place_box(box, grid.shape, reserved):
        return False
    if grid[r, c] != BG:
        return False
    grid[r, c] = BLUE
    reserved.append(box)
    return True


def _feature_box(
    direction: str,
    cursor: int,
    axis: int,
    size: int,
    offset: int,
) -> tuple[Room, int]:
    half = size // 2
    if direction == "up":
        bottom = cursor - 1
        top = bottom - size + 1
        room = Room(top, axis - half + offset, size)
        return room, top - 1
    if direction == "down":
        top = cursor + 1
        room = Room(top, axis - half + offset, size)
        return room, room.r1 + 1
    if direction == "left":
        right = cursor - 1
        left = right - size + 1
        room = Room(axis - half + offset, left, size)
        return room, left - 1
    left = cursor + 1
    room = Room(axis - half + offset, left, size)
    return room, room.c1 + 1


def _candidate_aligned_rooms(
    direction: str,
    cursor: int,
    axis: int,
) -> list[tuple[Room, int]]:
    candidates: list[tuple[Room, int]] = []
    for gap in (2, 3, 4):
        stepped_cursor = cursor
        if direction in ("up", "left"):
            stepped_cursor -= gap - 1
        else:
            stepped_cursor += gap - 1
        for size in ROOM_SIZES:
            candidates.append(_feature_box(direction, stepped_cursor, axis, size, 0))
    random.shuffle(candidates)
    return candidates


def _candidate_misaligned_rooms(
    direction: str,
    cursor: int,
    axis: int,
) -> list[tuple[Room, int]]:
    candidates: list[tuple[Room, int]] = []
    for gap in (2, 3, 4):
        stepped_cursor = cursor
        if direction in ("up", "left"):
            stepped_cursor -= gap - 1
        else:
            stepped_cursor += gap - 1
        for size in ROOM_SIZES:
            half = size // 2
            for offset in range(-half, half + 1):
                if offset == 0:
                    continue
                candidates.append(_feature_box(direction, stepped_cursor, axis, size, offset))
    random.shuffle(candidates)
    return candidates


def _candidate_blockers(direction: str, cursor: int, axis: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for gap in (2, 3, 4, 5):
        if direction == "up":
            candidates.append((cursor - gap + 1, axis))
        elif direction == "down":
            candidates.append((cursor + gap - 1, axis))
        elif direction == "left":
            candidates.append((axis, cursor - gap + 1))
        else:
            candidates.append((axis, cursor + gap - 1))
    random.shuffle(candidates)
    return candidates


def _place_one_room(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    candidates: list[tuple[Room, int]],
) -> int | None:
    for room, new_cursor in candidates:
        if _try_place_room(grid, reserved, room):
            return new_cursor
    return None


def _place_role(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    center_r: int,
    center_c: int,
    direction: str,
    role: str,
) -> bool:
    axis = center_c if direction in ("up", "down") else center_r
    cursor = center_r - 2 if direction == "up" else center_r + 2 if direction == "down" else center_c - 2 if direction == "left" else center_c + 2

    if role == "chain":
        count = random.choice((2, 2, 3))
        for _ in range(count):
            new_cursor = _place_one_room(grid, reserved, _candidate_aligned_rooms(direction, cursor, axis))
            if new_cursor is None:
                return False
            cursor = new_cursor
        return True

    if role == "blocker":
        if random.random() < 0.7:
            new_cursor = _place_one_room(grid, reserved, _candidate_aligned_rooms(direction, cursor, axis))
            if new_cursor is None:
                return False
            cursor = new_cursor
        for r, c in _candidate_blockers(direction, cursor, axis):
            if _in_bounds(grid.shape, r, c) and _try_place_blocker(grid, reserved, r, c):
                return True
        return False

    if role == "misaligned":
        if random.random() < 0.6:
            new_cursor = _place_one_room(grid, reserved, _candidate_aligned_rooms(direction, cursor, axis))
            if new_cursor is None:
                return False
            cursor = new_cursor
        new_cursor = _place_one_room(grid, reserved, _candidate_misaligned_rooms(direction, cursor, axis))
        return new_cursor is not None

    assert role == "clear"
    if random.random() < 0.65:
        new_cursor = _place_one_room(grid, reserved, _candidate_aligned_rooms(direction, cursor, axis))
        if new_cursor is None:
            return False
    return True


def _place_extra_rooms(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    center_r: int,
    center_c: int,
) -> None:
    for _ in range(random.randint(0, 2)):
        candidates: list[Room] = []
        for size in ROOM_SIZES:
            for r0 in range(1, grid.shape[0] - size):
                for c0 in range(1, grid.shape[1] - size):
                    room = Room(r0, c0, size)
                    if room.r0 <= center_r <= room.r1 or room.c0 <= center_c <= room.c1:
                        continue
                    candidates.append(room)
        random.shuffle(candidates)
        for room in candidates[:80]:
            if _try_place_room(grid, reserved, room):
                break


def _place_extra_blockers(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    center_r: int,
    center_c: int,
) -> None:
    cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if r != center_r and c != center_c]
    random.shuffle(cells)
    placed = 0
    target = random.randint(1, 4)
    for r, c in cells:
        if _try_place_blocker(grid, reserved, r, c):
            placed += 1
            if placed == target:
                return


def _build_input() -> np.ndarray | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)

    center_r = random.randint(9, h - 10)
    center_c = random.randint(9, w - 10)
    grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = RED
    reserved: list[tuple[int, int, int, int]] = [(center_r - 1, center_c - 1, center_r + 1, center_c + 1)]

    roles = ["chain", "blocker", "misaligned", "clear"]
    random.shuffle(roles)
    for direction, role in zip(("up", "down", "left", "right"), roles):
        if not _place_role(grid, reserved, center_r, center_c, direction, role):
            return None

    _place_extra_rooms(grid, reserved, center_r, center_c)
    _place_extra_blockers(grid, reserved, center_r, center_c)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        if np.count_nonzero(grid == GREY) == 0 or np.count_nonzero(grid == BLUE) == 0:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for the ray-through-rooms task")
