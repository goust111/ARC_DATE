import random
from collections import Counter

import numpy as np


BG = 0
WALL = 3

UP, RIGHT, DOWN, LEFT = range(4)
ROOM_COUNT_RANGE = (3, 6)
ROOM_SIDE_OPTIONS = (5, 7)
GENERATION_RETRIES = 400

FURNITURE_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_positions(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    rooms = []
    for coords in _find_color_components(grid, WALL):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        border = _border_positions(r0, c0, r1, c1)
        wall_cells = {tuple(map(int, cell)) for cell in coords}
        if not wall_cells.issubset(border):
            continue

        missing = sorted(border - wall_cells)
        if len(missing) != 1:
            continue

        mr, mc = missing[0]
        if mr == r0 and c0 < mc < c1:
            entrance = UP
        elif mc == c1 and r0 < mr < r1:
            entrance = RIGHT
        elif mr == r1 and c0 < mc < c1:
            entrance = DOWN
        elif mc == c0 and r0 < mr < r1:
            entrance = LEFT
        else:
            continue

        if len(wall_cells) != len(border) - 1:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        if interior.shape[0] != interior.shape[1] or interior.shape[0] % 2 == 0:
            continue

        furniture_mask = (interior != BG) & (interior != WALL)
        if not bool(furniture_mask.any()):
            continue

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "entrance": entrance,
                "interior": interior,
            }
        )

    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    assert rooms, "expected at least one framed room"

    side = int(rooms[0]["interior"].shape[0])
    assert all(room["interior"].shape == (side, side) for room in rooms), "rooms must share one interior size"
    return rooms


def _rotation_index(base: np.ndarray, patch: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(base, k), patch)]
    assert len(matches) == 1, "all rooms must contain the same non-symmetric furniture up to rotation"
    return int(matches[0])


def _front_direction(interior: np.ndarray) -> int:
    mask = (interior != BG) & (interior != WALL)
    coords = np.argwhere(mask)
    assert coords.size > 0

    center = interior.shape[0] // 2
    projections = [
        [center - int(r) for r, c in coords],
        [int(c) - center for r, c in coords],
        [int(r) - center for r, c in coords],
        [center - int(c) for r, c in coords],
    ]
    signatures = [tuple(sorted(values, reverse=True)) for values in projections]
    best = max(range(4), key=lambda idx: signatures[idx])
    assert sum(sig == signatures[best] for sig in signatures) == 1, "front direction must be unique"
    return int(best)


def _rotate_interior_to_entrance(interior: np.ndarray, entrance: int) -> np.ndarray:
    current_front = _front_direction(interior)
    return np.rot90(interior, (current_front - entrance) % 4)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each room's furniture so its front points toward that room's wall opening.

    Input structure:
    - The background is black and each room is a green rectangular frame with a one-cell entrance gap on exactly one wall.
    - All rooms have the same odd-sized square interior and contain the same multi-cell furniture sprite, possibly rotated.
    - Furniture colors are inside the rooms; the green walls and black entrance gaps stay on the outer room border.

    Rule:
    - Detect every room and read the side of its missing wall cell to get the entrance direction.
    - Compare room interiors under 90-degree rotations; the already-correct rooms agree on one shared mapping between furniture orientation and entrance direction.
    - Rotate each room interior by 0, 90, 180, or 270 degrees so the furniture's front faces that room's own entrance, while leaving walls and gaps unchanged.

    Key insight:
    - The same furniture appears in every room, so relative rotations across rooms reveal which orientation is the "facing the entrance" one; each room must then be solved relative to its own doorway, not by one absolute global direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _extract_rooms(g)
    out = g.copy()

    base = rooms[0]["interior"]
    relative_rotations = [_rotation_index(base, room["interior"]) for room in rooms]
    shift_counts = Counter((room["entrance"] + rot) % 4 for room, rot in zip(rooms, relative_rotations))
    shift, count = shift_counts.most_common(1)[0]

    if count >= 2:
        assert sum(v == count for v in shift_counts.values()) == 1, "aligned reference orientation must be unique"
        for room, rot in zip(rooms, relative_rotations):
            target_orientation = (shift - room["entrance"]) % 4
            k = (target_orientation - rot) % 4
            r0, c0, r1, c1 = room["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(room["interior"], k)
        return out

    aligned_found = False
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        rotated = _rotate_interior_to_entrance(room["interior"], room["entrance"])
        aligned_found |= np.array_equal(rotated, room["interior"])
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated
    assert aligned_found, "expected at least one room already facing its entrance"
    return out


def _distinct_rotations(patch: np.ndarray) -> bool:
    return len({np.rot90(patch, k).tobytes() for k in range(4)}) == 4


def _make_furniture(side: int) -> np.ndarray:
    center = side // 2
    front_len = 2 if side == 5 else random.choice((2, 3))
    wing = 1 if front_len == 2 else random.choice((1, 2))
    tail_len = 1 if front_len == 2 else random.choice((1, 2))
    foot_side = random.choice((-1, 1))
    foot_offset = 1 if wing == 1 else random.choice((1, 2))
    palette = random.sample(FURNITURE_COLORS, random.randint(2, 3))
    tip_color = int(palette[0])
    body_color = int(palette[1])
    accent_color = int(palette[2] if len(palette) == 3 else palette[0])

    patch = np.full((side, side), BG, dtype=int)
    tip = (center - front_len, center)
    patch[tip] = tip_color

    for d in range(-front_len + 1, tail_len + 1):
        patch[center + d, center] = body_color

    shoulder_row = center - front_len + 1
    for dc in range(-wing, wing + 1):
        patch[shoulder_row, center + dc] = body_color

    patch[center + tail_len, center + foot_side * foot_offset] = accent_color
    if side == 7 and random.random() < 0.6:
        patch[center + max(1, tail_len - 1), center + foot_side] = accent_color

    assert _front_direction(patch) == UP
    assert _distinct_rotations(patch)
    return patch


def _valid_layouts(room_count: int, outer_side: int) -> list[tuple[int, int, int, int, int, int]]:
    layouts = []
    for rows in range(1, 4):
        for cols in range(1, 4):
            if rows * cols < room_count:
                continue
            for row_gap in range(1, 4):
                core_h = rows * outer_side + (rows - 1) * row_gap
                if core_h > 30:
                    continue
                for col_gap in range(1, 4):
                    core_w = cols * outer_side + (cols - 1) * col_gap
                    if core_w > 30:
                        continue
                    extra_h = 30 - core_h
                    extra_w = 30 - core_w
                    if extra_h < 0 or extra_w < 0:
                        continue
                    layouts.append((rows, cols, row_gap, col_gap, extra_h, extra_w))
    return layouts


def _sample_room_positions(room_count: int, outer_side: int) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]] | None:
    layouts = _valid_layouts(room_count, outer_side)
    if not layouts:
        return None

    rows, cols, row_gap, col_gap, extra_h, extra_w = random.choice(layouts)

    top_margin = random.randint(0, min(3, extra_h))
    bottom_margin = random.randint(0, min(3, extra_h - top_margin))
    left_margin = random.randint(0, min(3, extra_w))
    right_margin = random.randint(0, min(3, extra_w - left_margin))

    height = rows * outer_side + (rows - 1) * row_gap + top_margin + bottom_margin
    width = cols * outer_side + (cols - 1) * col_gap + left_margin + right_margin

    slots = []
    for rr in range(rows):
        for cc in range(cols):
            top = top_margin + rr * (outer_side + row_gap)
            left = left_margin + cc * (outer_side + col_gap)
            slots.append((top, left))

    chosen = tuple(random.sample(slots, room_count))
    return (height, width), chosen


def _draw_room(grid: np.ndarray, top: int, left: int, interior: np.ndarray, entrance: int, opening: int) -> None:
    side = int(interior.shape[0])
    outer = side + 2

    grid[top, left : left + outer] = WALL
    grid[top + outer - 1, left : left + outer] = WALL
    grid[top : top + outer, left] = WALL
    grid[top : top + outer, left + outer - 1] = WALL

    if entrance == UP:
        grid[top, left + opening] = BG
    elif entrance == RIGHT:
        grid[top + opening, left + outer - 1] = BG
    elif entrance == DOWN:
        grid[top + outer - 1, left + opening] = BG
    else:
        grid[top + opening, left] = BG

    grid[top + 1 : top + outer - 1, left + 1 : left + outer - 1] = interior


def _aligned_rotation_for_entrance(entrance: int) -> int:
    return (-entrance) % 4


def _choose_entrances(room_count: int) -> list[int]:
    for _ in range(100):
        entrances = [random.randrange(4) for _ in range(room_count)]
        if len(set(entrances)) >= 2:
            return entrances
    raise RuntimeError("failed to sample diverse entrances")


def _expected_output(shape: tuple[int, int], rooms: list[dict], furniture: np.ndarray) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for room in rooms:
        aligned = np.rot90(furniture, _aligned_rotation_for_entrance(room["entrance"]))
        _draw_room(out, room["top"], room["left"], aligned, room["entrance"], room["opening"])
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(*ROOM_COUNT_RANGE)
    side = random.choice(ROOM_SIDE_OPTIONS)
    outer_side = side + 2
    layout = _sample_room_positions(room_count, outer_side)
    if layout is None:
        return None

    shape, positions = layout
    furniture = _make_furniture(side)
    entrances = _choose_entrances(room_count)

    aligned_count = random.randint(room_count // 2 + 1, room_count - 1)
    aligned_indices = set(random.sample(range(room_count), aligned_count))

    rooms = []
    inp = np.full(shape, BG, dtype=int)
    for idx, (top, left) in enumerate(positions):
        entrance = entrances[idx]
        aligned_rot = _aligned_rotation_for_entrance(entrance)
        if idx in aligned_indices:
            current_rot = aligned_rot
        else:
            current_rot = random.choice([rot for rot in range(4) if rot != aligned_rot])

        opening = random.randint(1, side)
        interior = np.rot90(furniture, current_rot)
        _draw_room(inp, top, left, interior, entrance, opening)
        rooms.append(
            {
                "top": int(top),
                "left": int(left),
                "entrance": int(entrance),
                "opening": int(opening),
            }
        )

    expected = _expected_output(shape, rooms, furniture)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-orientation puzzle")
