from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_IDS = tuple(range(10))
PALETTE_COLORS = tuple(color for color in COLOR_IDS if color not in (BLACK, GREY))

GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
TARGET_ROOM_COUNT_RANGE = (3, 7)
MIN_ROOM_HEIGHT = 4
MIN_ROOM_WIDTH = 4
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _room_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    assert grid.shape[0] >= 2, "expected a palette row and at least one room row"
    passable = grid[1:] != GREY
    rooms: list[tuple[int, int, int, int]] = []

    for coords in _mask_components(passable):
        coords = coords.copy()
        coords[:, 0] += 1
        r0, c0, r1, c1 = _bbox(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(coords) != area:
            continue
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            continue
        rooms.append((r0, c0, r1, c1))

    assert rooms, "expected at least one rectangular room"
    rooms.sort()
    return rooms


def _count_monochrome_objects(window: np.ndarray) -> int:
    height, width = window.shape
    seen = np.zeros_like(window, dtype=bool)
    count = 0

    for r in range(height):
        for c in range(width):
            color = int(window[r, c])
            if color in (BLACK, GREY) or bool(seen[r, c]):
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if bool(seen[nr, nc]) or int(window[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each fenced room's black floor according to how many furniture objects it contains.

    Input structure:
    - Black is the room-floor background.
    - Grey one-pixel horizontal and vertical fence segments partition the area below row 0 into enclosed rectangular rooms.
    - Each room contains 0 to 4 small furniture objects, where each furniture object is a 4-connected monochrome region in a non-black, non-grey color.
    - Row 0 is not part of any room and contains a left-to-right sequence of five distinct one-pixel colors.

    Rule:
    - Read the five palette colors from row 0 in left-to-right order.
    - Identify every room separated by grey fences.
    - Count the number N of furniture objects inside each room.
    - Repaint every black floor pixel in that room with the palette color at index N, while leaving furniture, fences, and the palette row unchanged.

    Key insight:
    - The task is a composition of three deductions: use the grey fence layout to recover room boundaries, count interior objects room by room, then look up the fill color from the top palette row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = [int(g[0, c]) for c in range(g.shape[1]) if int(g[0, c]) not in (BLACK, GREY)]
    assert len(palette) == 5, "expected exactly five palette colors on row 0"
    assert len(set(palette)) == 5, "palette colors must be distinct"

    out = g.copy()
    for r0, c0, r1, c1 in _room_bboxes(g):
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        furniture_count = _count_monochrome_objects(room)
        assert 0 <= furniture_count <= 4, "room furniture count must be in 0..4"
        room_out = out[r0 : r1 + 1, c0 : c1 + 1]
        room_out[room == BLACK] = palette[furniture_count]

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_transforms(base: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    current = list(base)
    variants = set()
    for flip in (False, True):
        transformed = [(-r, c) if flip else (r, c) for r, c in current]
        for _ in range(4):
            transformed = [(c, -r) for r, c in transformed]
            variants.add(_normalize_shape(set(transformed)))
    return sorted(variants)


BASE_SHAPES = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]
SHAPE_VARIANTS = [variant for base in BASE_SHAPES for variant in _shape_transforms(base)]


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _room_height(room: tuple[int, int, int, int]) -> int:
    return int(room[2] - room[0] + 1)


def _room_width(room: tuple[int, int, int, int]) -> int:
    return int(room[3] - room[1] + 1)


def _splittable_orientations(room: tuple[int, int, int, int]) -> list[str]:
    options = []
    if _room_height(room) >= 2 * MIN_ROOM_HEIGHT + 1:
        options.append("h")
    if _room_width(room) >= 2 * MIN_ROOM_WIDTH + 1:
        options.append("v")
    return options


def _split_room(
    room: tuple[int, int, int, int],
    orientation: str,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = room
    if orientation == "h":
        wall_r = random.randint(r0 + MIN_ROOM_HEIGHT, r1 - MIN_ROOM_HEIGHT)
        children = [(r0, c0, wall_r - 1, c1), (wall_r + 1, c0, r1, c1)]
        wall = (wall_r, c0 - 1, wall_r, c1 + 1)
        return children, wall

    wall_c = random.randint(c0 + MIN_ROOM_WIDTH, c1 - MIN_ROOM_WIDTH)
    children = [(r0, c0, r1, wall_c - 1), (r0, wall_c + 1, r1, c1)]
    wall = (r0 - 1, wall_c, r1 + 1, wall_c)
    return children, wall


def _draw_wall(grid: np.ndarray, wall: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = wall
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY


def _partition_rooms(
    height: int,
    width: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[1, :] = GREY
    grid[-1, :] = GREY
    grid[1:, 0] = GREY
    grid[1:, -1] = GREY

    rooms = [(2, 1, height - 2, width - 2)]
    target = random.randint(*TARGET_ROOM_COUNT_RANGE)
    while len(rooms) < target:
        candidates = [
            (idx, _splittable_orientations(room))
            for idx, room in enumerate(rooms)
            if _splittable_orientations(room)
        ]
        if not candidates:
            break
        idx, orientations = random.choice(candidates)
        children, wall = _split_room(rooms[idx], random.choice(orientations))
        _draw_wall(grid, wall)
        rooms[idx : idx + 1] = children

    if len(rooms) < TARGET_ROOM_COUNT_RANGE[0]:
        return None
    if any(_room_height(room) < MIN_ROOM_HEIGHT or _room_width(room) < MIN_ROOM_WIDTH for room in rooms):
        return None
    return grid, rooms


def _max_singleton_capacity(room: tuple[int, int, int, int]) -> int:
    return min(4, ((_room_height(room) + 1) // 2) * ((_room_width(room) + 1) // 2))


def _assign_counts(rooms: list[tuple[int, int, int, int]]) -> list[int] | None:
    capacities = [_max_singleton_capacity(room) for room in rooms]
    counts = [0] * len(rooms)

    if len(rooms) >= 5 and random.random() < 0.7:
        available = list(range(len(rooms)))
        random.shuffle(available)
        for value in range(5):
            candidates = [idx for idx in available if capacities[idx] >= value]
            if not candidates:
                return None
            idx = random.choice(candidates)
            counts[idx] = value
            available.remove(idx)
        for idx in available:
            counts[idx] = random.randint(0, capacities[idx])
    else:
        for idx, cap in enumerate(capacities):
            counts[idx] = random.randint(0, cap)
        if len(rooms) > 1 and len(set(counts)) == 1:
            choices = [idx for idx, cap in enumerate(capacities) if cap > 0]
            if choices:
                idx = random.choice(choices)
                alternatives = [value for value in range(capacities[idx] + 1) if value != counts[idx]]
                if alternatives:
                    counts[idx] = random.choice(alternatives)

    return counts


def _halo(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _candidate_placements(
    room_h: int,
    room_w: int,
    blocked: set[tuple[int, int]],
    remaining_objects: int,
) -> list[tuple[set[tuple[int, int]], int]]:
    if remaining_objects >= 3 or min(room_h, room_w) <= 4:
        max_area = 2
    elif remaining_objects == 2:
        max_area = 3
    else:
        max_area = 4

    placements: list[tuple[set[tuple[int, int]], int]] = []
    for shape in SHAPE_VARIANTS:
        area = len(shape)
        if area > max_area:
            continue
        shape_h, shape_w = _shape_size(shape)
        if shape_h > room_h or shape_w > room_w:
            continue
        for top in range(room_h - shape_h + 1):
            for left in range(room_w - shape_w + 1):
                cells = {(top + r, left + c) for r, c in shape}
                if cells & blocked:
                    continue
                placements.append((cells, area))

    if not placements and max_area > 1:
        for shape in SHAPE_VARIANTS:
            if len(shape) != 1:
                continue
            for top in range(room_h):
                for left in range(room_w):
                    cells = {(top, left)}
                    if cells & blocked:
                        continue
                    placements.append((cells, 1))
    return placements


def _place_furniture(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    count: int,
    furniture_colors: tuple[int, ...],
) -> bool:
    if count == 0:
        return True

    r0, c0, r1, c1 = room
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    room_area = room_h * room_w

    for _ in range(120):
        placed: list[tuple[set[tuple[int, int]], int]] = []
        blocked: set[tuple[int, int]] = set()
        occupied_cells: set[tuple[int, int]] = set()

        success = True
        for obj_index in range(count):
            remaining_objects = count - obj_index
            candidates = _candidate_placements(room_h, room_w, blocked, remaining_objects)
            if not candidates:
                success = False
                break

            candidates.sort(key=lambda item: (item[1], random.random()))
            sample = candidates[: min(80, len(candidates))]
            weights = [6 - area for _cells, area in sample]
            cells, _area = random.choices(sample, weights=weights, k=1)[0]
            placed.append((cells, random.choice(furniture_colors)))
            occupied_cells |= cells
            blocked |= _halo(cells, room_h, room_w)

        if not success or len(occupied_cells) >= room_area:
            continue

        for cells, color in placed:
            for rr, cc in cells:
                grid[r0 + rr, c0 + cc] = color
        return True

    return False


def _paint_expected(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    counts: list[int],
    palette: list[int],
) -> np.ndarray:
    out = grid.copy()
    for room, count in zip(rooms, counts):
        r0, c0, r1, c1 = room
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        room_out = out[r0 : r1 + 1, c0 : c1 + 1]
        room_out[window == BLACK] = palette[count]
    return out


def _palette_columns(width: int) -> list[int]:
    cols = sorted(random.sample(range(width), 5))
    return [int(col) for col in cols]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    if width < 5 or height < 8:
        return None

    partition = _partition_rooms(height, width)
    if partition is None:
        return None
    grid, rooms = partition

    counts = _assign_counts(rooms)
    if counts is None:
        return None

    palette = random.sample(list(PALETTE_COLORS), 5)
    for col, color in zip(_palette_columns(width), palette):
        grid[0, col] = int(color)

    furniture_colors = tuple(color for color in PALETTE_COLORS if color not in palette)
    if len(furniture_colors) < 2:
        return None

    room_order = list(range(len(rooms)))
    random.shuffle(room_order)
    for idx in room_order:
        if not _place_furniture(grid, rooms[idx], counts[idx], furniture_colors):
            return None

    expected = _paint_expected(grid, rooms, counts, palette)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(set(counts)) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-room puzzle")
