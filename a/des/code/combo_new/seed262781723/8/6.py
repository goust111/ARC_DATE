from dataclasses import dataclass
import random

import numpy as np


BG = 0
WALL = 5
ROOM_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 16
GRID_MAX = 30
ROOT_MIN = 14
MIN_ROOM_SPAN = 7
MAX_ROOMS = 6
MAX_DOORS = 4
GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Room:
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


@dataclass(frozen=True)
class Adjacency:
    orient: str
    wall: int
    start: int
    end: int
    first: int
    second: int


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange neighboring room colors onto the inner sides of shared-wall doors.

    Input structure:
    - The background is black.
    - Grey one-pixel walls form axis-aligned rectangular rooms, with some rooms sharing a full wall segment.
    - Every room interior is filled with one unique non-grey color.
    - A door is a single black gap cut out of a shared grey wall between exactly two colored rooms.

    Rule:
    - Detect every door by finding a black wall-gap whose opposite sides are two different room colors.
    - For a vertical door, paint the two interior cells directly above and below the doorway on each side of the wall.
    - For a horizontal door, paint the two interior cells directly left and right of the doorway on each side of the wall.
    - Each room-side stripe takes the fill color of the room on the opposite side of the door.
    - Leave all walls, door gaps, and all other cells unchanged.

    Key insight:
    - The black gap matters only as a connector between two colored rooms; the output is a local color swap painted onto the interior wall-side cells flanking that doorway.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BG:
                continue

            if _is_vertical_door(g, r, c):
                left_color = int(g[r, c - 1])
                right_color = int(g[r, c + 1])
                for rr, cc in ((r - 1, c - 1), (r + 1, c - 1)):
                    out[rr, cc] = right_color
                for rr, cc in ((r - 1, c + 1), (r + 1, c + 1)):
                    out[rr, cc] = left_color
            elif _is_horizontal_door(g, r, c):
                top_color = int(g[r - 1, c])
                bottom_color = int(g[r + 1, c])
                for rr, cc in ((r - 1, c - 1), (r - 1, c + 1)):
                    out[rr, cc] = bottom_color
                for rr, cc in ((r + 1, c - 1), (r + 1, c + 1)):
                    out[rr, cc] = top_color

    return out


def _is_vertical_door(g: np.ndarray, r: int, c: int) -> bool:
    if int(g[r - 1, c]) != WALL or int(g[r + 1, c]) != WALL:
        return False
    left_color = int(g[r, c - 1])
    right_color = int(g[r, c + 1])
    if not _is_room_color(left_color) or not _is_room_color(right_color) or left_color == right_color:
        return False
    return (
        int(g[r - 1, c - 1]) == left_color
        and int(g[r + 1, c - 1]) == left_color
        and int(g[r - 1, c + 1]) == right_color
        and int(g[r + 1, c + 1]) == right_color
    )


def _is_horizontal_door(g: np.ndarray, r: int, c: int) -> bool:
    if int(g[r, c - 1]) != WALL or int(g[r, c + 1]) != WALL:
        return False
    top_color = int(g[r - 1, c])
    bottom_color = int(g[r + 1, c])
    if not _is_room_color(top_color) or not _is_room_color(bottom_color) or top_color == bottom_color:
        return False
    return (
        int(g[r - 1, c - 1]) == top_color
        and int(g[r - 1, c + 1]) == top_color
        and int(g[r + 1, c - 1]) == bottom_color
        and int(g[r + 1, c + 1]) == bottom_color
    )


def _is_room_color(value: int) -> bool:
    return value not in (BG, WALL)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle")


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    rect_h = rng.randint(ROOT_MIN, h - 2)
    rect_w = rng.randint(ROOT_MIN, w - 2)
    top = rng.randint(1, h - rect_h - 1)
    left = rng.randint(1, w - rect_w - 1)
    root = Room(top, top + rect_h - 1, left, left + rect_w - 1)

    target_rooms = rng.randint(2, min(MAX_ROOMS, len(ROOM_COLORS)))
    rooms = _split_rooms(root, target_rooms, rng)
    if rooms is None:
        return None

    adjacencies = _find_adjacencies(rooms)
    if not adjacencies:
        return None

    room_colors = rng.sample(ROOM_COLORS, len(rooms))
    grid = np.zeros((h, w), dtype=int)
    _draw_rooms(grid, rooms, room_colors)

    doors = _choose_doors(grid, rooms, room_colors, adjacencies, rng)
    if not doors:
        return None

    for r, c in doors:
        grid[r, c] = BG

    if len(_detect_doors(grid)) != len(doors):
        return None

    out = main(grid)
    if np.count_nonzero(out != grid) != 4 * len(doors):
        return None

    return grid


def _split_rooms(root: Room, target_rooms: int, rng: random.Random) -> list[Room] | None:
    rooms = [root]
    while len(rooms) < target_rooms:
        candidates: list[tuple[int, list[str]]] = []
        for idx, room in enumerate(rooms):
            options = []
            if room.height >= 2 * MIN_ROOM_SPAN - 1:
                options.append("horizontal")
            if room.width >= 2 * MIN_ROOM_SPAN - 1:
                options.append("vertical")
            if options:
                candidates.append((idx, options))
        if not candidates:
            return None

        idx, options = rng.choice(candidates)
        room = rooms.pop(idx)
        orient = rng.choice(options)

        if orient == "vertical":
            split = rng.randint(room.c0 + MIN_ROOM_SPAN - 1, room.c1 - MIN_ROOM_SPAN + 1)
            rooms.append(Room(room.r0, room.r1, room.c0, split))
            rooms.append(Room(room.r0, room.r1, split, room.c1))
        else:
            split = rng.randint(room.r0 + MIN_ROOM_SPAN - 1, room.r1 - MIN_ROOM_SPAN + 1)
            rooms.append(Room(room.r0, split, room.c0, room.c1))
            rooms.append(Room(split, room.r1, room.c0, room.c1))

    return rooms


def _find_adjacencies(rooms: list[Room]) -> list[Adjacency]:
    result: list[Adjacency] = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]

            if a.c1 == b.c0 or b.c1 == a.c0:
                wall = a.c1 if a.c1 == b.c0 else b.c1
                start = max(a.r0, b.r0)
                end = min(a.r1, b.r1)
                if end - start + 1 >= 5:
                    left_idx, right_idx = (i, j) if a.c0 < b.c0 else (j, i)
                    result.append(Adjacency("vertical", wall, start, end, left_idx, right_idx))

            if a.r1 == b.r0 or b.r1 == a.r0:
                wall = a.r1 if a.r1 == b.r0 else b.r1
                start = max(a.c0, b.c0)
                end = min(a.c1, b.c1)
                if end - start + 1 >= 5:
                    top_idx, bottom_idx = (i, j) if a.r0 < b.r0 else (j, i)
                    result.append(Adjacency("horizontal", wall, start, end, top_idx, bottom_idx))

    return result


def _draw_rooms(grid: np.ndarray, rooms: list[Room], room_colors: list[int]) -> None:
    for room, color in zip(rooms, room_colors):
        grid[room.r0 + 1 : room.r1, room.c0 + 1 : room.c1] = color
        grid[room.r0, room.c0 : room.c1 + 1] = WALL
        grid[room.r1, room.c0 : room.c1 + 1] = WALL
        grid[room.r0 : room.r1 + 1, room.c0] = WALL
        grid[room.r0 : room.r1 + 1, room.c1] = WALL


def _choose_doors(
    grid: np.ndarray,
    rooms: list[Room],
    room_colors: list[int],
    adjacencies: list[Adjacency],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    target = rng.randint(1, min(MAX_DOORS, len(adjacencies)))
    vertical = [adj for adj in adjacencies if adj.orient == "vertical"]
    horizontal = [adj for adj in adjacencies if adj.orient == "horizontal"]

    chosen: list[tuple[int, int]] = []
    used_stripes: set[tuple[int, int]] = set()
    used_adjs: set[Adjacency] = set()

    if target >= 2 and vertical and horizontal and rng.random() < 0.5:
        for bucket in (vertical, horizontal):
            adj = rng.choice(bucket)
            door = _pick_door_for_adjacency(grid, room_colors, adj, used_stripes, rng)
            if door is None:
                return None
            chosen.append(door[:2])
            used_stripes.update(door[2])
            used_adjs.add(adj)

    remaining = [adj for adj in adjacencies if adj not in used_adjs]
    rng.shuffle(remaining)
    for adj in remaining:
        if len(chosen) >= target:
            break
        door = _pick_door_for_adjacency(grid, room_colors, adj, used_stripes, rng)
        if door is None:
            continue
        chosen.append(door[:2])
        used_stripes.update(door[2])

    return chosen if len(chosen) == target else None


def _pick_door_for_adjacency(
    grid: np.ndarray,
    room_colors: list[int],
    adj: Adjacency,
    used_stripes: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
    positions = list(range(adj.start + 2, adj.end - 1))
    rng.shuffle(positions)

    first_color = room_colors[adj.first]
    second_color = room_colors[adj.second]

    for pos in positions:
        if adj.orient == "vertical":
            r, c = pos, adj.wall
            stripes = ((r - 1, c - 1), (r + 1, c - 1), (r - 1, c + 1), (r + 1, c + 1))
            if any(cell in used_stripes for cell in stripes):
                continue
            if (
                int(grid[r, c]) == WALL
                and int(grid[r, c - 1]) == first_color
                and int(grid[r, c + 1]) == second_color
                and int(grid[r - 1, c]) == WALL
                and int(grid[r + 1, c]) == WALL
                and int(grid[r - 1, c - 1]) == first_color
                and int(grid[r + 1, c - 1]) == first_color
                and int(grid[r - 1, c + 1]) == second_color
                and int(grid[r + 1, c + 1]) == second_color
            ):
                return r, c, stripes
        else:
            r, c = adj.wall, pos
            stripes = ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
            if any(cell in used_stripes for cell in stripes):
                continue
            if (
                int(grid[r, c]) == WALL
                and int(grid[r - 1, c]) == first_color
                and int(grid[r + 1, c]) == second_color
                and int(grid[r, c - 1]) == WALL
                and int(grid[r, c + 1]) == WALL
                and int(grid[r - 1, c - 1]) == first_color
                and int(grid[r - 1, c + 1]) == first_color
                and int(grid[r + 1, c - 1]) == second_color
                and int(grid[r + 1, c + 1]) == second_color
            ):
                return r, c, stripes

    return None


def _detect_doors(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    doors = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BG:
                continue
            if _is_vertical_door(g, r, c) or _is_horizontal_door(g, r, c):
                doors.append((r, c))
    return doors
