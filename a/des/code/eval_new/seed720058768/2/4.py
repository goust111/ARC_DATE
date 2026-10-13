import random
from collections import deque

import numpy as np


WALL_COLOR = 9
SIDES = ("top", "bottom", "left", "right")
COLOR_IDS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _edge_coords(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _room_pixels(bbox: tuple[int, int, int, int], object_coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for side in SIDES:
        pixels.update(_edge_coords(bbox, side))
    pixels.update(object_coords)
    return pixels


def _reflect_cell(r: int, c: int, bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return 2 * r0 - r, c
    if side == "bottom":
        return 2 * r1 - r, c
    if side == "left":
        return r, 2 * c0 - c
    return r, 2 * c1 - c


def _is_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    cells = set(coords)
    queue = deque([coords[0]])
    seen = {coords[0]}
    while queue:
        r, c = queue.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == len(cells)


def _extract_object(
    interior: np.ndarray,
    bg: int,
    mirror_color: int,
    bbox: tuple[int, int, int, int],
) -> tuple[int, list[tuple[int, int]]] | None:
    non_bg = np.argwhere(interior != bg)
    if non_bg.size == 0 or len(non_bg) > 4:
        return None
    colors = {int(interior[r, c]) for r, c in non_bg}
    if len(colors) != 1:
        return None
    object_color = next(iter(colors))
    if object_color in {bg, WALL_COLOR, mirror_color}:
        return None

    ih, iw = interior.shape
    if np.any(non_bg[:, 0] == 0) or np.any(non_bg[:, 0] == ih - 1):
        return None
    if np.any(non_bg[:, 1] == 0) or np.any(non_bg[:, 1] == iw - 1):
        return None

    r0, c0, _, _ = bbox
    object_coords = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in non_bg]
    if not _is_connected(object_coords):
        return None
    return object_color, object_coords


def _match_room(
    grid: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
) -> dict | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    side_checks = {
        "top": (
            grid[r0, c0 : c1 + 1],
            [
                grid[r1, c0 : c1 + 1],
                grid[r0 + 1 : r1 + 1, c0],
                grid[r0 + 1 : r1 + 1, c1],
            ],
        ),
        "bottom": (
            grid[r1, c0 : c1 + 1],
            [
                grid[r0, c0 : c1 + 1],
                grid[r0:r1, c0],
                grid[r0:r1, c1],
            ],
        ),
        "left": (
            grid[r0 : r1 + 1, c0],
            [
                grid[r0 : r1 + 1, c1],
                grid[r0, c0 + 1 : c1 + 1],
                grid[r1, c0 + 1 : c1 + 1],
            ],
        ),
        "right": (
            grid[r0 : r1 + 1, c1],
            [
                grid[r0 : r1 + 1, c0],
                grid[r0, c0:c1],
                grid[r1, c0:c1],
            ],
        ),
    }

    for side, (mirror_edge, wall_edges) in side_checks.items():
        mirror_values = {int(v) for v in np.asarray(mirror_edge).ravel()}
        if len(mirror_values) != 1:
            continue
        mirror_color = next(iter(mirror_values))
        if mirror_color in {bg, WALL_COLOR}:
            continue
        if any(not np.all(edge == WALL_COLOR) for edge in wall_edges):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        extracted = _extract_object(interior, bg, mirror_color, bbox)
        if extracted is None:
            continue
        object_color, object_coords = extracted
        return {
            "bbox": bbox,
            "side": side,
            "mirror_color": mirror_color,
            "object_color": object_color,
            "object_coords": object_coords,
        }
    return None


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    H, W = grid.shape
    rooms: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0 in range(H - 4):
        for c0 in range(W - 4):
            if int(grid[r0, c0]) == bg:
                continue
            for r1 in range(r0 + 4, H):
                for c1 in range(c0 + 4, W):
                    room = _match_room(grid, bg, (r0, c0, r1, c1))
                    if room is None:
                        continue
                    bbox = room["bbox"]
                    if bbox in seen:
                        continue
                    seen.add(bbox)
                    rooms.append(room)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each room's interior object across its unique colored wall and repaint the reflection with the wall color.

    Input structure:
    - The background is the most common color.
    - One or more rectangular rooms are bounded by one-cell-thick walls that are maroon on three sides.
    - In each room, the fourth wall is a different non-background color, and the room interior contains one small 4-connected object separated from every wall by at least one background cell.

    Rule:
    - Detect the uniquely colored wall of each room and treat it as the mirror.
    - Reflect every object cell across that wall so the copy lands outside the room at the symmetric positions.
    - Keep the original room, walls, and object unchanged, and draw only reflected cells whose destination is background in the input; every reflected cell takes the mirror wall's color.

    Key insight:
    - The single non-maroon wall encodes both the reflection axis and the output color, so solving the task requires reading geometry and recoloring information from the same structure.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    rooms = _find_rooms(g, bg)

    out = g.copy()
    H, W = g.shape
    for room in rooms:
        bbox = room["bbox"]
        side = room["side"]
        mirror_color = int(room["mirror_color"])
        for r, c in room["object_coords"]:
            rr, cc = _reflect_cell(int(r), int(c), bbox, side)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == bg:
                out[rr, cc] = mirror_color
    return out


_BASE_SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
]

GRID_SIDE_RANGE = (20, 28)
ROOM_COUNT_CHOICES = (3, 4)
ROOM_SIZE_RANGE = (5, 9)
GENERATION_ATTEMPTS = 500
ROOM_PLACE_ATTEMPTS = 250
ROOM_GAP = 1
COLLISION_PROBABILITY = 0.5


def _normalize_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _transform_shape(
    coords: tuple[tuple[int, int], ...],
    rotations: int,
    flip: bool,
) -> tuple[tuple[int, int], ...]:
    pts = np.asarray(coords, dtype=int)
    for _ in range(rotations):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    if flip:
        pts[:, 1] *= -1
    return _normalize_shape(tuple((int(r), int(c)) for r, c in pts))


def _all_shapes() -> list[tuple[tuple[int, int], ...]]:
    shapes = set()
    for base in _BASE_SHAPES:
        for rotations in range(4):
            for flip in (False, True):
                shapes.add(_transform_shape(base, rotations, flip))
    return sorted(shapes)


_SHAPES = _all_shapes()


def _bbox_clear(mask: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(mask.shape[0] - 1, r1 + margin)
    cc1 = min(mask.shape[1] - 1, c1 + margin)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _paint_room(grid: np.ndarray, spec: dict) -> None:
    bbox = spec["bbox"]
    for side in SIDES:
        for r, c in _edge_coords(bbox, side):
            grid[r, c] = WALL_COLOR
    for r, c in _edge_coords(bbox, spec["side"]):
        grid[r, c] = int(spec["mirror_color"])
    for r, c in spec["object_coords"]:
        grid[r, c] = int(spec["object_color"])


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    bg = _most_common_color(grid)
    out = grid.copy()
    H, W = grid.shape
    for spec in specs:
        bbox = spec["bbox"]
        side = spec["side"]
        mirror_color = int(spec["mirror_color"])
        for r, c in spec["object_coords"]:
            rr, cc = _reflect_cell(int(r), int(c), bbox, side)
            if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == bg:
                out[rr, cc] = mirror_color
    return out


def _sample_room_spec(
    H: int,
    W: int,
    side: str,
    mirror_color: int,
    object_color: int,
    occupied: np.ndarray,
    footprints: np.ndarray,
    reserved: np.ndarray,
    require_collision: bool,
) -> dict | None:
    for _ in range(ROOM_PLACE_ATTEMPTS):
        h = random.randint(*ROOM_SIZE_RANGE)
        w = random.randint(*ROOM_SIZE_RANGE)
        bbox = (
            random.randint(0, H - h),
            random.randint(0, W - w),
            0,
            0,
        )
        r0, c0 = bbox[0], bbox[1]
        r1, c1 = r0 + h - 1, c0 + w - 1
        bbox = (r0, c0, r1, c1)

        if not _bbox_clear(footprints, bbox, ROOM_GAP):
            continue
        if not _bbox_clear(reserved, bbox, 0):
            continue

        inner_h = h - 4
        inner_w = w - 4
        fitting_shapes = [shape for shape in _SHAPES if _shape_dims(shape)[0] <= inner_h and _shape_dims(shape)[1] <= inner_w]
        if not fitting_shapes:
            continue
        shape = random.choice(fitting_shapes)
        shape_h, shape_w = _shape_dims(shape)
        dr = random.randint(0, inner_h - shape_h)
        dc = random.randint(0, inner_w - shape_w)
        object_coords = [(r0 + 2 + dr + r, c0 + 2 + dc + c) for r, c in shape]

        room_pixel_list = list(_room_pixels(bbox, object_coords))
        if any(reserved[r, c] for r, c in room_pixel_list):
            continue

        reflection_coords = [_reflect_cell(r, c, bbox, side) for r, c in object_coords]
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in reflection_coords):
            continue

        free_targets = [(rr, cc) for rr, cc in reflection_coords if not occupied[rr, cc]]
        collision_targets = [(rr, cc) for rr, cc in reflection_coords if occupied[rr, cc]]
        if not free_targets:
            continue
        if require_collision and not collision_targets:
            continue
        if not require_collision and collision_targets:
            continue
        if any(footprints[rr, cc] for rr, cc in free_targets):
            continue
        if any(reserved[rr, cc] for rr, cc in free_targets):
            continue

        return {
            "bbox": bbox,
            "side": side,
            "mirror_color": mirror_color,
            "object_color": object_color,
            "object_coords": object_coords,
            "reflection_coords": reflection_coords,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    bg = random.choice([c for c in COLOR_IDS if c != WALL_COLOR])
    n_rooms = random.choice(ROOM_COUNT_CHOICES)
    collision_mode = random.random() < COLLISION_PROBABILITY

    palette = [c for c in COLOR_IDS if c not in {bg, WALL_COLOR}]
    random.shuffle(palette)
    mirror_colors = palette[:n_rooms]
    object_colors = palette[n_rooms : 2 * n_rooms]
    sides = random.sample(SIDES, n_rooms)

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    footprints = np.zeros((H, W), dtype=bool)
    reserved = np.zeros((H, W), dtype=bool)
    specs: list[dict] = []

    for idx in range(n_rooms):
        require_collision = collision_mode and idx == n_rooms - 1
        spec = _sample_room_spec(
            H,
            W,
            sides[idx],
            mirror_colors[idx],
            object_colors[idx],
            occupied,
            footprints,
            reserved,
            require_collision,
        )
        if spec is None:
            return None

        _paint_room(grid, spec)
        for r, c in _room_pixels(spec["bbox"], spec["object_coords"]):
            occupied[r, c] = True
        r0, c0, r1, c1 = spec["bbox"]
        footprints[r0 : r1 + 1, c0 : c1 + 1] = True
        for rr, cc in spec["reflection_coords"]:
            if not occupied[rr, cc]:
                reserved[rr, cc] = True
        specs.append(spec)

    expected = _expected_output(grid, specs)
    detected = _find_rooms(grid, bg)
    if len(detected) != n_rooms:
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for mirror-room task")
