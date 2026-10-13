import random

import numpy as np


BLACK = 0
BLUE = 1
OBJECT_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
VERTICAL_SIDES = ("top", "bottom")
HORIZONTAL_SIDES = ("left", "right")

GRID_MAX = 30
ROOM_COUNT_RANGE = (2, 4)
GRID_BASE_MIN = 16
GRID_PER_ROOM = 2
OBJECT_SIZE_RANGE = (1, 3)
INTERIOR_SIZE_MAX = 6
PLACEMENT_RETRIES = 200
GENERATE_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _analyze_room(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3, "room bbox must contain an interior"

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    blue_mask = window == BLUE
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    assert not np.any(blue_mask & ~border_mask), "blue room pixels must lie on the rectangle border"
    missing = np.argwhere(border_mask & ~blue_mask)
    assert missing.shape[0] == 1, "expected exactly one border gap per room"
    gap_r_local, gap_c_local = map(int, missing[0])
    on_top = gap_r_local == 0
    on_bottom = gap_r_local == height - 1
    on_left = gap_c_local == 0
    on_right = gap_c_local == width - 1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "gap must lie on exactly one side"

    if on_top:
        side = "top"
        dr, dc = -1, 0
    elif on_bottom:
        side = "bottom"
        dr, dc = 1, 0
    elif on_left:
        side = "left"
        dr, dc = 0, -1
    else:
        side = "right"
        dr, dc = 0, 1

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0, "room must have a non-empty interior"
    object_colors = sorted(set(map(int, np.unique(interior))) - {BLACK, BLUE})
    assert len(object_colors) == 1, "each room interior must contain exactly one object color"
    color = int(object_colors[0])
    object_mask = interior == color
    object_coords_local = np.argwhere(object_mask)
    obj_r0, obj_c0 = object_coords_local.min(axis=0)
    obj_r1, obj_c1 = object_coords_local.max(axis=0)
    obj_r0, obj_c0, obj_r1, obj_c1 = map(int, (obj_r0, obj_c0, obj_r1, obj_c1))
    object_height = obj_r1 - obj_r0 + 1
    object_width = obj_c1 - obj_c0 + 1
    object_area = int(object_mask.sum())
    assert object_area == object_height * object_width, "room object must be a solid rectangle"
    assert np.all(interior[obj_r0 : obj_r1 + 1, obj_c0 : obj_c1 + 1] == color)

    span = object_height if side in VERTICAL_SIDES else object_width
    gap_r = r0 + gap_r_local
    gap_c = c0 + gap_c_local
    ray = [(gap_r + dr * step, gap_c + dc * step) for step in range(1, span + 1)]

    object_coords = object_coords_local + np.array([r0 + 1, c0 + 1], dtype=int)
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (gap_r, gap_c),
        "side": side,
        "color": color,
        "object_coords": object_coords,
        "ray": ray,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each room's interior object with an outward ray whose length is encoded by the object's cross-door span.

    Input structure:
    - The background is black.
    - One or more blue one-pixel-thick rectangular room borders appear on the grid, and each border has exactly one one-pixel gap on one side.
    - Each room contains exactly one solid non-black, non-blue rectangular object, and different rooms use different object colors.

    Rule:
    - Detect which side contains the gap: top, bottom, left, or right.
    - Measure the object's span perpendicular to that side: use object height for top/bottom gaps and object width for left/right gaps.
    - Remove the interior object.
    - Starting from the cell immediately outside the gap, draw a one-pixel-wide straight ray in the gap direction with that measured length and the object's color.
    - Leave every blue border and every gap cell unchanged.

    Key insight:
    - The room structure supplies the ray direction, while the object shape supplies the ray length through its orthogonal extent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms = _components(g == BLUE)
    assert rooms, "expected at least one blue room"
    infos = [_analyze_room(g, room) for room in rooms]
    infos.sort(key=lambda info: tuple(info["bbox"]))  # deterministic traversal

    out = g.copy()
    for info in infos:
        object_coords = np.asarray(info["object_coords"], dtype=int)
        out[object_coords[:, 0], object_coords[:, 1]] = BLACK

    drawn = np.zeros_like(g, dtype=bool)
    height, width = g.shape
    for info in infos:
        color = int(info["color"])
        for r, c in info["ray"]:
            assert 0 <= r < height and 0 <= c < width, "ray must stay within the grid"
            assert g[r, c] == BLACK, "ray path must start on exterior background"
            assert not drawn[r, c], "rays must not overlap each other"
            drawn[r, c] = True
            out[r, c] = color
    return out


def _choose_sides(room_count: int) -> list[str]:
    sides = [random.choice(VERTICAL_SIDES), random.choice(HORIZONTAL_SIDES)]
    if room_count > 2:
        remaining = [side for side in SIDES if side not in sides]
        sides.extend(random.sample(remaining, room_count - 2))
    random.shuffle(sides)
    return sides


def _footprint_mask(
    shape: tuple[int, int],
    r0: int,
    c0: int,
    height: int,
    width: int,
    ray: list[tuple[int, int]],
) -> np.ndarray:
    grid_h, grid_w = shape
    mask = np.zeros(shape, dtype=bool)
    rr0 = max(0, r0 - 1)
    rr1 = min(grid_h, r0 + height + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(grid_w, c0 + width + 1)
    mask[rr0:rr1, cc0:cc1] = True
    for r, c in ray:
        mask[max(0, r - 1) : min(grid_h, r + 2), max(0, c - 1) : min(grid_w, c + 2)] = True
    return mask


def _place_room(grid: np.ndarray, reserved: np.ndarray, color: int, side: str) -> bool:
    grid_h, grid_w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        obj_h = random.randint(*OBJECT_SIZE_RANGE)
        obj_w = random.randint(*OBJECT_SIZE_RANGE)
        interior_h = random.randint(obj_h, min(INTERIOR_SIZE_MAX, obj_h + 3))
        interior_w = random.randint(obj_w, min(INTERIOR_SIZE_MAX, obj_w + 3))
        room_h = interior_h + 2
        room_w = interior_w + 2
        span = obj_h if side in VERTICAL_SIDES else obj_w

        r0_min = 1
        c0_min = 1
        r0_max = grid_h - room_h - 1
        c0_max = grid_w - room_w - 1
        if side == "top":
            r0_min = max(r0_min, span)
        elif side == "bottom":
            r0_max = min(r0_max, grid_h - room_h - span)
        elif side == "left":
            c0_min = max(c0_min, span)
        else:
            c0_max = min(c0_max, grid_w - room_w - span)

        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1

        if side == "top":
            gap = (r0, c0 + random.randint(1, room_w - 2))
            dr, dc = -1, 0
        elif side == "bottom":
            gap = (r1, c0 + random.randint(1, room_w - 2))
            dr, dc = 1, 0
        elif side == "left":
            gap = (r0 + random.randint(1, room_h - 2), c0)
            dr, dc = 0, -1
        else:
            gap = (r0 + random.randint(1, room_h - 2), c1)
            dr, dc = 0, 1

        ray = [(gap[0] + dr * step, gap[1] + dc * step) for step in range(1, span + 1)]
        if not all(0 <= rr < grid_h and 0 <= cc < grid_w for rr, cc in ray):
            continue

        footprint = _footprint_mask(grid.shape, r0, c0, room_h, room_w, ray)
        if np.any(footprint & reserved):
            continue
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
            continue
        if any(grid[rr, cc] != BLACK for rr, cc in ray):
            continue

        obj_r = r0 + 1 + random.randint(0, interior_h - obj_h)
        obj_c = c0 + 1 + random.randint(0, interior_w - obj_w)

        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        grid[gap] = BLACK
        grid[obj_r : obj_r + obj_h, obj_c : obj_c + obj_w] = color
        reserved |= footprint
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(*ROOM_COUNT_RANGE)
    min_dim = min(GRID_MAX, GRID_BASE_MIN + room_count * GRID_PER_ROOM)
    grid_h = random.randint(min_dim, GRID_MAX)
    grid_w = random.randint(min_dim, GRID_MAX)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    colors = random.sample(OBJECT_COLORS, room_count)
    sides = _choose_sides(room_count)
    for color, side in zip(colors, sides):
        if not _place_room(grid, reserved, color, side):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
