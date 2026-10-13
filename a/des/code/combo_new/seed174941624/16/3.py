from __future__ import annotations

import random
from collections import deque

import numpy as np

BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
WALL = 5
ORANGE = 7
AZURE = 8
MAROON = 9

PALETTE = [BLUE, AZURE, GREEN, YELLOW, ORANGE, RED, MAROON]

MIN_ROOM_SIDE = 3
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_enclosed_rooms(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    passable = (grid == BG) | (grid == RED)
    seen = np.zeros((h, w), dtype=bool)
    rooms: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                rooms.append(np.array(coords, dtype=int))

    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed room by the rank of its red-marker count.

    Input structure:
    - The background color is black.
    - A connected grey wall structure surrounds several enclosed rooms; a black
      region touching the grid border is exterior background and is not a room.
    - Every room is a 4-connected area of black cells that may also contain
      isolated single-cell red markers.

    Rule:
    - Find every enclosed room separated by grey walls and count its red marker
      pixels.
    - Sort the rooms by marker count from fewest to most and map them to the
      fixed palette blue, azure, green, yellow, orange, red, maroon.
    - Recolor each room's black cells to its assigned palette color, recolor its
      red marker cells to grey, and leave the wall structure plus any exterior
      black background unchanged.

    Key insight:
    - The red pixels are only counting cues: the transformation is a component
      fill over enclosed wall-separated rooms, and the markers disappear into
      the grey walls after their counts determine the room-color ranking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms = _find_enclosed_rooms(g)
    assert 1 <= len(rooms) <= len(PALETTE), "room count must fit the palette"

    marker_counts: list[int] = []
    for coords in rooms:
        values = g[coords[:, 0], coords[:, 1]]
        assert np.all((values == BG) | (values == RED)), "rooms may only contain black cells and red markers"
        count = int(np.count_nonzero(values == RED))
        assert count > 0, "every room must contain at least one marker"
        marker_counts.append(count)

    unique_counts = sorted(set(marker_counts))
    assert len(unique_counts) == len(marker_counts), "marker counts must be distinct across rooms"
    color_for_count = {count: PALETTE[i] for i, count in enumerate(unique_counts)}

    out = g.copy()
    for coords, count in zip(rooms, marker_counts):
        rr = coords[:, 0]
        cc = coords[:, 1]
        values = g[rr, cc]
        room_color = color_for_count[count]
        out[rr[values == BG], cc[values == BG]] = room_color
        out[rr[values == RED], cc[values == RED]] = WALL

    return out


Room = tuple[int, int, int, int]
WallSeg = tuple[str, int, int, int]


def _room_area(room: Room) -> int:
    r0, c0, r1, c1 = room
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _split_choices(room: Room) -> list[str]:
    r0, c0, r1, c1 = room
    choices: list[str] = []
    if c1 - c0 + 1 >= 2 * MIN_ROOM_SIDE + 1:
        choices.append("v")
    if r1 - r0 + 1 >= 2 * MIN_ROOM_SIDE + 1:
        choices.append("h")
    return choices


def _split_room(room: Room) -> tuple[Room, Room, WallSeg] | None:
    choices = _split_choices(room)
    if not choices:
        return None

    r0, c0, r1, c1 = room
    orient = random.choice(choices)
    if orient == "v":
        wall_c = random.randint(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE)
        left = (r0, c0, r1, wall_c - 1)
        right = (r0, wall_c + 1, r1, c1)
        return left, right, ("v", wall_c, r0, r1)

    wall_r = random.randint(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE)
    top = (r0, c0, wall_r - 1, c1)
    bottom = (wall_r + 1, c0, r1, c1)
    return top, bottom, ("h", wall_r, c0, c1)


def _build_room_layout(room_count: int) -> tuple[list[Room], list[WallSeg], int, int] | None:
    inner_h = random.randint(max(8, room_count + 3), 24)
    inner_w = random.randint(max(8, room_count + 3), 24)
    rooms: list[Room] = [(1, 1, inner_h, inner_w)]
    walls: list[WallSeg] = []

    while len(rooms) < room_count:
        viable = [i for i, room in enumerate(rooms) if _split_choices(room)]
        if not viable:
            return None
        weights = [_room_area(rooms[i]) for i in viable]
        idx = random.choices(viable, weights=weights, k=1)[0]
        result = _split_room(rooms[idx])
        if result is None:
            return None
        first, second, wall = result
        rooms.pop(idx)
        rooms.extend([first, second])
        walls.append(wall)

    return rooms, walls, inner_h + 2, inner_w + 2


def _room_candidates(room: Room) -> dict[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = room
    candidates: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for parity in (0, 1):
        inner: list[tuple[int, int]] = []
        border: list[tuple[int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r + c) % 2 != parity:
                    continue
                if r0 < r < r1 and c0 < c < c1:
                    inner.append((r, c))
                else:
                    border.append((r, c))
        random.shuffle(inner)
        random.shuffle(border)
        candidates[parity] = inner + border
    return candidates


def _assign_marker_counts(rooms: list[Room]) -> list[int] | None:
    n = len(rooms)
    counts = list(range(1, n + 1))
    capacities = [max(len(v) for v in _room_candidates(room).values()) for room in rooms]
    order = sorted(range(n), key=lambda i: capacities[i])
    assignment = [0] * n

    def search(i: int, remaining: list[int]) -> bool:
        if i == n:
            return True
        room_idx = order[i]
        feasible = [count for count in remaining if count <= capacities[room_idx]]
        random.shuffle(feasible)
        for count in feasible:
            assignment[room_idx] = count
            next_remaining = [value for value in remaining if value != count]
            if search(i + 1, next_remaining):
                return True
        assignment[room_idx] = 0
        return False

    if not search(0, counts):
        return None
    return assignment


def _draw_layout(rooms: list[Room], walls: list[WallSeg], shape: tuple[int, int], counts: list[int]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    for orient, line, start, end in walls:
        if orient == "v":
            grid[start : end + 1, line] = WALL
        else:
            grid[line, start : end + 1] = WALL

    for room, count in zip(rooms, counts):
        candidates = _room_candidates(room)
        choices = [cells for cells in candidates.values() if len(cells) >= count]
        if not choices:
            return None
        selected = random.choice(choices)[:count]
        for r, c in selected:
            grid[r, c] = RED

    return grid


def _paste_compound(compound: np.ndarray) -> np.ndarray | None:
    h, w = compound.shape
    inset = random.random() < 0.5
    if not inset:
        return compound

    if h > 28 or w > 28:
        return None

    top = random.randint(1, min(3, 29 - h))
    bottom_max = min(3, 30 - h - top)
    if bottom_max < 1:
        return None
    bottom = random.randint(1, bottom_max)

    left = random.randint(1, min(3, 29 - w))
    right_max = min(3, 30 - w - left)
    if right_max < 1:
        return None
    right = random.randint(1, right_max)

    out = np.full((h + top + bottom, w + left + right), BG, dtype=int)
    out[top : top + h, left : left + w] = compound
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(2, len(PALETTE))
    layout = _build_room_layout(room_count)
    if layout is None:
        return None

    rooms, walls, h, w = layout
    counts = _assign_marker_counts(rooms)
    if counts is None:
        return None

    compound = _draw_layout(rooms, walls, (h, w), counts)
    if compound is None:
        return None

    grid = _paste_compound(compound)
    if grid is None or max(grid.shape) > 30:
        return None

    rooms_found = _find_enclosed_rooms(grid)
    if len(rooms_found) != room_count:
        return None
    found_counts = sorted(int(np.count_nonzero(grid[coords[:, 0], coords[:, 1]] == RED)) for coords in rooms_found)
    if found_counts != list(range(1, room_count + 1)):
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-ranking ARC input")
