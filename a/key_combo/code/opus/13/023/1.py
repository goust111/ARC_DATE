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
MOVABLE_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
MIN_ROOM_SIZE = 5
GEN_RETRIES = 400
ROOM_RETRIES = 160


_MISSING_TO_GRAVITY = {
    (0, 0): (1, 1),
    (0, 1): (1, -1),
    (1, 0): (-1, 1),
    (1, 1): (-1, -1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rooms(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    mask = grid != WALL
    rooms: list[set[tuple[int, int]]] = []
    for comp in _component_coords(mask):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp):
            continue
        rooms.append(set(comp))
    rooms.sort(key=lambda room: _bbox(list(room)))
    return rooms


def _decode_marker(room: set[tuple[int, int]], grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    reds = [(r, c) for r, c in room if int(grid[r, c]) == RED]
    assert len(reds) == 3, "each room must contain exactly one 3-cell red L"
    r0, c0, r1, c1 = _bbox(reds)
    assert (r1 - r0, c1 - c0) == (1, 1), "marker must occupy a 2x2 box"
    local = {(r - r0, c - c0) for r, c in reds}
    missing = next((rr, cc) for rr in (0, 1) for cc in (0, 1) if (rr, cc) not in local)
    return set(reds), _MISSING_TO_GRAVITY[missing]


def _sort_key(coord: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    r, c = coord
    return dr * r, dc * c


def _slide_records(
    room: set[tuple[int, int]],
    movers: list[tuple[int, int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int, int, bool]]:
    occupied = {(r, c) for r, c, _ in movers}
    placed: list[tuple[int, int, int, bool]] = []
    for r, c, color in sorted(movers, key=lambda item: _sort_key((item[0], item[1]), dr, dc)):
        occupied.remove((r, c))
        nr, nc = r, c
        while (nr + dr, nc) in room and (nr + dr, nc) not in occupied:
            nr += dr
        while (nr, nc + dc) in room and (nr, nc + dc) not in occupied:
            nc += dc
        occupied.add((nr, nc))
        placed.append((nr, nc, color, (nr, nc) != (r, c)))
    return placed


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each red L marker as a room-local diagonal gravity vector and slide the loose pixels into that corner.

    Input structure:
    - The grid has black background with grey 1-cell walls forming several closed rectangular rooms.
    - Each room contains exactly one red 3-cell L marker inside a 2x2 box.
    - Each room also contains 1-3 non-red single pixels that are separated from the marker.

    Rule:
    - Find every enclosed room bounded by grey walls.
    - In each room, read the red L by its missing corner cell; the gravity direction is the opposite diagonal.
    - Erase the red marker, then slide each loose pixel first along the vertical gravity component and then along the horizontal component, stopping at room boundaries or previously placed pixels, with the farthest pixels processed first.
    - Keep the grey walls unchanged.

    Key insight:
    - The L does not mark a destination cell directly; its open corner encodes the opposite diagonal corner that acts as gravity for that room only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for room in _find_rooms(g):
        marker_cells, (dr, dc) = _decode_marker(room, g)
        movers = [
            (r, c, int(g[r, c]))
            for r, c in room
            if int(g[r, c]) not in (BG, WALL, RED)
        ]

        for r, c in marker_cells:
            out[r, c] = BG
        for r, c, _ in movers:
            out[r, c] = BG

        for r, c, color, _ in _slide_records(room, movers, dr, dc):
            out[r, c] = color

    return out


def _split_rooms(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    target_count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    while len(rooms) < target_count:
        options: list[tuple[int, str]] = []
        for idx, (r0, c0, r1, c1) in enumerate(rooms):
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if width >= 2 * MIN_ROOM_SIZE + 1:
                options.append((idx, "v"))
            if height >= 2 * MIN_ROOM_SIZE + 1:
                options.append((idx, "h"))
        if not options:
            return None
        idx, orient = rng.choice(options)
        r0, c0, r1, c1 = rooms.pop(idx)
        if orient == "v":
            wall_c = rng.randint(c0 + MIN_ROOM_SIZE, c1 - MIN_ROOM_SIZE)
            grid[r0 - 1 : r1 + 2, wall_c] = WALL
            rooms.append((r0, c0, r1, wall_c - 1))
            rooms.append((r0, wall_c + 1, r1, c1))
        else:
            wall_r = rng.randint(r0 + MIN_ROOM_SIZE, r1 - MIN_ROOM_SIZE)
            grid[wall_r, c0 - 1 : c1 + 2] = WALL
            rooms.append((r0, c0, wall_r - 1, c1))
            rooms.append((wall_r + 1, c0, r1, c1))
    rooms.sort()
    return rooms


def _marker_cells(
    top: int,
    left: int,
    missing: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = {(top + rr, left + cc) for rr in (0, 1) for cc in (0, 1)}
    cells.remove((top + missing[0], left + missing[1]))
    return cells


def _marker_positions_for_room(
    room_box: tuple[int, int, int, int],
    gravity: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_box
    dr, dc = gravity
    rows = list(range(r0, r1))
    cols = list(range(c0, c1))
    if dr == -1 and len(rows) > 1:
        rows = [r for r in rows if r > r0]
    if dr == 1 and len(rows) > 1:
        rows = [r for r in rows if r < r1 - 1]
    if dc == -1 and len(cols) > 1:
        cols = [c for c in cols if c > c0]
    if dc == 1 and len(cols) > 1:
        cols = [c for c in cols if c < c1 - 1]
    return [(r, c) for r in rows for c in cols]


def _can_place_pixel(
    pos: tuple[int, int],
    marker_cells: set[tuple[int, int]],
    chosen: list[tuple[int, int]],
) -> bool:
    if pos in marker_cells:
        return False
    if any(pos == nbr for cell in marker_cells for nbr in _neighbors8(*cell)):
        return False
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) > 1 for r, c in chosen)


def _populate_room(
    grid: np.ndarray,
    room_box: tuple[int, int, int, int],
    missing: tuple[int, int],
    rng: random.Random,
) -> bool:
    room_cells = {
        (r, c)
        for r in range(room_box[0], room_box[2] + 1)
        for c in range(room_box[1], room_box[3] + 1)
    }
    gravity = _MISSING_TO_GRAVITY[missing]
    marker_tops = _marker_positions_for_room(room_box, gravity)
    rng.shuffle(marker_tops)

    for top, left in marker_tops:
        marker_cells = _marker_cells(top, left, missing)
        candidates = [cell for cell in sorted(room_cells) if _can_place_pixel(cell, marker_cells, [])]
        if len(candidates) < 1:
            continue

        max_pixels = min(3, len(MOVABLE_COLORS), len(candidates))
        pixel_count = rng.randint(1, max_pixels)
        colors = rng.sample(MOVABLE_COLORS, pixel_count)

        for _ in range(ROOM_RETRIES):
            rng.shuffle(candidates)
            chosen: list[tuple[int, int]] = []
            for cell in candidates:
                if _can_place_pixel(cell, marker_cells, chosen):
                    chosen.append(cell)
                    if len(chosen) == pixel_count:
                        break
            if len(chosen) != pixel_count:
                continue

            movers = [(r, c, color) for (r, c), color in zip(chosen, colors)]
            placed = _slide_records(room_cells, movers, gravity[0], gravity[1])
            if not all(moved for _, _, _, moved in placed):
                continue

            for r, c in marker_cells:
                grid[r, c] = RED
            for r, c, color in movers:
                grid[r, c] = color
            return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 19)
    w = rng.randint(13, 19)
    room_target = rng.choice([3, 4])

    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    rooms = _split_rooms(grid, [(1, 1, h - 2, w - 2)], room_target, rng)
    if rooms is None:
        return None

    missing_choices = list(_MISSING_TO_GRAVITY)
    if room_target == 4:
        room_missing = rng.sample(missing_choices, 4)
    else:
        room_missing = rng.sample(missing_choices, 3)

    for room_box, missing in zip(rooms, room_missing):
        if not _populate_room(grid, room_box, missing, rng):
            return None

    # Structural validation before returning.
    found_rooms = _find_rooms(grid)
    if len(found_rooms) != room_target:
        return None
    orientations = set()
    for room in found_rooms:
        _, gravity = _decode_marker(room, grid)
        orientations.add(gravity)
        movers = [(r, c, int(grid[r, c])) for r, c in room if int(grid[r, c]) in MOVABLE_COLORS]
        if not movers:
            return None
        if not all(moved for _, _, _, moved in _slide_records(room, movers, gravity[0], gravity[1])):
            return None
    if len(orientations) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for the diagonal-room-gravity task")
