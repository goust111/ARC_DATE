import random
from collections import deque

import numpy as np


WALL = 5
BG = 0
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
MIN_ROOM_H = 4
MIN_ROOM_W = 4
ROOM_COUNT_OPTIONS = (2, 3)
GEN_RETRY_LIMIT = 400


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rooms(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rooms = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == WALL:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != WALL:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
                raise ValueError("room interior must be a solid rectangle")
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                raise ValueError("room interior must be enclosed by walls")
            if not np.all(grid[r0 - 1, c0 - 1 : c1 + 2] == WALL):
                raise ValueError("room top wall missing")
            if not np.all(grid[r1 + 1, c0 - 1 : c1 + 2] == WALL):
                raise ValueError("room bottom wall missing")
            if not np.all(grid[r0 - 1 : r1 + 2, c0 - 1] == WALL):
                raise ValueError("room left wall missing")
            if not np.all(grid[r0 - 1 : r1 + 2, c1 + 1] == WALL):
                raise ValueError("room right wall missing")
            rooms.append((r0, c0, r1, c1))
    return sorted(rooms)


def _room_markers(grid, room):
    r0, c0, r1, c1 = room
    markers = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(grid[r, c])
            if color not in (BG, WALL):
                markers.append((r, c, color))
    return markers


def _settle_markers(room, markers):
    _, _, bottom, _ = room
    occupied = set()
    settled = []
    for r, c, color in sorted(markers, key=lambda item: (-item[0], item[1])):
        nr = r
        while nr < bottom and (nr + 1, c) not in occupied:
            nr += 1
        occupied.add((nr, c))
        settled.append((nr, c, color))
    return settled


def _palette_from_markers(markers, mode):
    if mode == "settled_columns":
        ordered = sorted(markers, key=lambda item: (item[1], item[0]))
    elif mode == "reading_order":
        ordered = sorted(markers, key=lambda item: (item[0], item[1]))
    else:
        raise ValueError(f"unknown palette mode: {mode}")
    return [color for _, _, color in ordered]


def _fill_room(out, room, palette):
    r0, c0, r1, c1 = room
    palette_len = len(palette)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            dist = min(r - r0 + 1, r1 - r + 1, c - c0 + 1, c1 - c + 1)
            out[r, c] = palette[dist % palette_len]


def _transform(grid, *, palette_mode="settled_columns", strict=False):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    rooms = _find_rooms(g)
    if strict and len(rooms) not in (2, 3):
        raise ValueError("expected 2 or 3 rooms")

    out = g.copy()
    for room in rooms:
        markers = _room_markers(g, room)
        if strict:
            if not 2 <= len(markers) <= 4:
                raise ValueError("each room must contain 2-4 markers")
            colors = [color for _, _, color in markers]
            if len(set(colors)) != len(colors):
                raise ValueError("room marker colors must be distinct")
            if any(color in (BG, WALL) for color in colors):
                raise ValueError("marker colors cannot use background or wall")
        settled = _settle_markers(room, markers)
        palette_source = settled if palette_mode == "settled_columns" else markers
        palette = _palette_from_markers(palette_source, palette_mode)
        if not palette:
            raise ValueError("room palette cannot be empty")
        _fill_room(out, room, palette)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the room markers to define a palette, then repaint each room as cyclic concentric Chebyshev bands.

    Input structure:
    - A black background is partitioned by one-cell-thick grey walls into 2 or 3 rectangular rooms.
    - Each room interior contains 2-4 isolated marker pixels with distinct non-black, non-grey colors.
    - A room interior is otherwise black, and the walls fully enclose each room.

    Rule:
    - Identify each rectangular room enclosed by grey walls.
    - Within a room, simulate downward gravity on the marker pixels, processing lower markers first so each pixel
      stops just above the floor or a previously settled marker.
    - Sort the settled markers from left to right to obtain the room's cyclic palette.
    - For every non-wall cell in the room, compute the Chebyshev distance to the nearest wall pixel of that room and
      paint the cell with palette[distance mod palette_length].
    - Keep the grey walls unchanged, and overwrite the original marker positions with the band fill.

    Key insight:
    - Gravity does not directly create the output picture; it converts scattered marker clues into an ordered color
      palette, and that palette then drives the concentric rectangular band pattern inside each room.
    """
    return _transform(grid, palette_mode="settled_columns", strict=True)


def _split_options(room):
    r0, c0, r1, c1 = room
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options = []
    if width >= 2 * MIN_ROOM_W + 1:
        for wall_c in range(c0 + MIN_ROOM_W, c1 - MIN_ROOM_W + 1):
            options.append(("v", wall_c))
    if height >= 2 * MIN_ROOM_H + 1:
        for wall_r in range(r0 + MIN_ROOM_H, r1 - MIN_ROOM_H + 1):
            options.append(("h", wall_r))
    return options


def _partition_rooms(rng):
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    rooms = [(1, 1, h - 2, w - 2)]
    target_room_count = rng.choice(ROOM_COUNT_OPTIONS)
    while len(rooms) < target_room_count:
        candidates = [(idx, _split_options(room)) for idx, room in enumerate(rooms)]
        candidates = [(idx, opts) for idx, opts in candidates if opts]
        if not candidates:
            return None, None
        idx, options = rng.choice(candidates)
        orientation, wall_pos = rng.choice(options)
        r0, c0, r1, c1 = rooms.pop(idx)
        if orientation == "v":
            grid[r0 - 1 : r1 + 2, wall_pos] = WALL
            left = (r0, c0, r1, wall_pos - 1)
            right = (r0, wall_pos + 1, r1, c1)
            rooms.extend([left, right])
        else:
            grid[wall_pos, c0 - 1 : c1 + 2] = WALL
            top = (r0, c0, wall_pos - 1, c1)
            bottom = (wall_pos + 1, c0, r1, c1)
            rooms.extend([top, bottom])
    return grid, sorted(rooms)


def _max_markers(room):
    r0, c0, r1, c1 = room
    return min(4, r1 - r0 + 1, c1 - c0 + 1)


def _choose_palette(rng, count, forbidden_sets):
    for _ in range(40):
        colors = rng.sample(MARKER_COLORS, count)
        if tuple(sorted(colors)) not in forbidden_sets:
            rng.shuffle(colors)
            return colors
    colors = rng.sample(MARKER_COLORS, count)
    rng.shuffle(colors)
    return colors


def _place_room_markers(rng, grid, room, palette):
    r0, c0, r1, c1 = room
    count = len(palette)
    cols = sorted(rng.sample(range(c0, c1 + 1), count))
    rows = sorted(rng.sample(range(r0, r1 + 1), count))

    for _ in range(50):
        row_order = rows[:]
        rng.shuffle(row_order)
        if row_order != sorted(row_order):
            break
    else:
        return False

    for row, col, color in zip(row_order, cols, palette):
        grid[row, col] = color
    return True


def _assign_room_palettes(rng, rooms):
    room_palettes = [None] * len(rooms)
    shared_done = False
    if len(rooms) >= 2 and rng.random() < 0.65:
        eligible_pairs = []
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                common_max = min(_max_markers(rooms[i]), _max_markers(rooms[j]))
                if common_max >= 2:
                    eligible_pairs.append((i, j, common_max))
        if eligible_pairs:
            i, j, common_max = rng.choice(eligible_pairs)
            count = rng.randint(2, common_max)
            base = rng.sample(MARKER_COLORS, count)
            palette_a = base[:]
            rng.shuffle(palette_a)
            palette_b = base[:]
            for _ in range(20):
                rng.shuffle(palette_b)
                if palette_b != palette_a:
                    break
            room_palettes[i] = palette_a
            room_palettes[j] = palette_b
            shared_done = True

    used_sets = set()
    for idx, palette in enumerate(room_palettes):
        if palette is not None:
            used_sets.add(tuple(sorted(palette)))
            continue
        max_markers = _max_markers(rooms[idx])
        count = rng.randint(2, max_markers)
        if shared_done and rng.random() < 0.25:
            count = min(count + 1, max_markers)
        room_palettes[idx] = _choose_palette(rng, count, used_sets)
        used_sets.add(tuple(sorted(room_palettes[idx])))
    return room_palettes


def _validate_generated(grid):
    try:
        rooms = _find_rooms(grid)
        if len(rooms) not in (2, 3):
            return False
        inversion_rooms = 0
        for room in rooms:
            markers = _room_markers(grid, room)
            if not 2 <= len(markers) <= 4:
                return False
            colors = [color for _, _, color in markers]
            cols = [c for _, c, _ in markers]
            if len(set(colors)) != len(colors):
                return False
            if len(set(cols)) != len(cols):
                return False
            settled = _settle_markers(room, markers)
            if _palette_from_markers(settled, "settled_columns") != _palette_from_markers(markers, "reading_order"):
                inversion_rooms += 1
            if min(room[2] - room[0] + 1, room[3] - room[1] + 1) < 4:
                return False
        if inversion_rooms == 0:
            return False

        actual = _transform(grid, palette_mode="settled_columns", strict=True)
        wrong_reading = _transform(grid, palette_mode="reading_order", strict=False)
        if np.array_equal(actual, wrong_reading):
            return False
        if np.array_equal(actual, grid):
            return False
        if len(np.unique(actual)) < 3:
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        rng = random.Random()
        grid, rooms = _partition_rooms(rng)
        if grid is None:
            continue

        palettes = _assign_room_palettes(rng, rooms)
        ok = True
        for room, palette in zip(rooms, palettes):
            if not _place_room_markers(rng, grid, room, palette):
                ok = False
                break
        if not ok:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
