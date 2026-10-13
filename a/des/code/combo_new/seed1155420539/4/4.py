from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
BLOCK_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 18
GRID_MAX = 30
ROOM_H_RANGE = (3, 6)
ROOM_W_RANGE = (3, 6)
BLOCK_H_RANGE = (1, 3)
BLOCK_W_RANGE = (1, 4)
GAP_RANGE = (1, 4)
PLACEMENT_RETRIES = 250
GENERATE_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _touches_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _arrow_cells(root: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    dr, dc = DIRS[direction]
    pr, pc = dc, -dr
    rr, rc = root
    cells = {
        (rr, rc),
        (rr + dr, rc + dc),
        (rr + 2 * dr, rc + 2 * dc),
        (rr + dr + pr, rc + dc + pc),
        (rr + dr - pr, rc + dc - pc),
    }
    return cells


def _room_arrow_candidates(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[tuple[str, tuple[int, int], set[tuple[int, int]]]]:
    r0, c0, r1, c1 = bbox
    wall_top, wall_left, wall_bottom, wall_right = r0 - 1, c0 - 1, r1 + 1, c1 + 1
    candidates: list[tuple[str, tuple[int, int], set[tuple[int, int]]]] = []

    for c in range(c0, c1 + 1):
        for direction, root in (("up", (wall_top, c)), ("down", (wall_bottom, c))):
            cells = _arrow_cells(root, direction)
            if all(_in_bounds(shape, rr, cc) for rr, cc in cells):
                candidates.append((direction, root, cells))

    for r in range(r0, r1 + 1):
        for direction, root in (("left", (r, wall_left)), ("right", (r, wall_right))):
            cells = _arrow_cells(root, direction)
            if all(_in_bounds(shape, rr, cc) for rr, cc in cells):
                candidates.append((direction, root, cells))

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room interior with the color of the first target block hit by its outward-pointing wall arrow.

    Input structure:
    - The background is black.
    - One-pixel-thick grey rectangular room walls enclose black rectangular interiors.
    - Each room has exactly one red five-pixel arrow attached to one wall and pointing outward.
    - Away from the rooms, every non-black/non-grey/non-red foreground component is a solid monochrome rectangle.

    Rule:
    - Find every enclosed black rectangular room interior.
    - Match the nearby red arrow shape to that room and infer its direction from the arrowhead.
    - Cast a ray outward from the arrow tip and find the first target rectangle on that ray.
    - Fill the room interior with that rectangle's color while leaving the grey walls, red arrows, and target blocks unchanged.

    Key insight:
    - The red marker contributes two independent clues: its asymmetric five-pixel shape reveals the direction, and the first rectangle along that direction supplies the room's fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    room_interiors: list[dict[str, object]] = []
    for comp in _components(g == BG):
        if _touches_edge(comp["coords"], g.shape):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1), "room interior must be a full rectangle"
        room_interiors.append(comp)
    assert room_interiors, "expected at least one enclosed room"

    red_components = _components(g == RED)
    assert red_components, "expected red arrow markers"
    for comp in red_components:
        assert comp["area"] == 5, "each arrow must be a five-pixel component"

    block_components: list[dict[str, object]] = []
    block_lookup: dict[tuple[int, int], int] = {}
    for color in map(int, np.unique(g)):
        if color in {BG, RED, GREY}:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = comp["bbox"]
            assert comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1), "target block must be a solid rectangle"
            idx = len(block_components)
            comp = dict(comp)
            comp["color"] = color
            block_components.append(comp)
            for cell in comp["coords"]:
                block_lookup[cell] = idx
    assert block_components, "expected at least one target block"

    out = g.copy()
    used_red: set[int] = set()

    for room in room_interiors:
        bbox = room["bbox"]
        matches: list[tuple[int, str, tuple[int, int]]] = []
        for red_idx, red_comp in enumerate(red_components):
            red_cells = red_comp["coord_set"]
            for direction, root, cells in _room_arrow_candidates(bbox, g.shape):
                if red_cells == cells:
                    matches.append((red_idx, direction, root))
        assert len(matches) == 1, "each room must match exactly one arrow"

        red_idx, direction, root = matches[0]
        used_red.add(red_idx)
        dr, dc = DIRS[direction]
        rr, rc = root
        ray_r, ray_c = rr + 3 * dr, rc + 3 * dc
        fill_color: int | None = None

        while _in_bounds(g.shape, ray_r, ray_c):
            value = int(g[ray_r, ray_c])
            if value == BG:
                ray_r += dr
                ray_c += dc
                continue
            if (ray_r, ray_c) in block_lookup:
                fill_color = int(block_components[block_lookup[(ray_r, ray_c)]]["color"])
                break
            raise AssertionError("ray hit a non-target object before reaching a target block")

        assert fill_color is not None, "every arrow must hit a target block"
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    assert len(used_red) == len(red_components) == len(room_interiors), "rooms and arrows must pair one-to-one"
    return out


def _expanded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(shape[0] - 1, r1 + pad),
        min(shape[1] - 1, c1 + pad),
    )


def _sample_plan(shape: tuple[int, int]) -> dict[str, object] | None:
    h, w = shape
    direction = random.choice(tuple(DIRS))
    interior_h = random.randint(*ROOM_H_RANGE)
    interior_w = random.randint(*ROOM_W_RANGE)
    block_h = random.randint(*BLOCK_H_RANGE)
    block_w = random.randint(*BLOCK_W_RANGE)
    gap = random.randint(*GAP_RANGE)
    color = random.choice(BLOCK_COLORS)

    room_top = random.randint(0, h - interior_h - 2)
    room_left = random.randint(0, w - interior_w - 2)
    room_bottom = room_top + interior_h + 1
    room_right = room_left + interior_w + 1
    inner_bbox = (room_top + 1, room_left + 1, room_bottom - 1, room_right - 1)

    if direction in {"up", "down"}:
        root_col = room_left + 1 + random.randint(1, interior_w - 2)
        root = (room_top, root_col) if direction == "up" else (room_bottom, root_col)
        axis_offset = random.randint(0, block_w - 1)
        block_left = root_col - axis_offset
        block_right = block_left + block_w - 1
        if block_left < 0 or block_right >= w:
            return None
        if direction == "up":
            block_bottom = room_top - gap - 3
            block_top = block_bottom - block_h + 1
        else:
            block_top = room_bottom + gap + 3
            block_bottom = block_top + block_h - 1
    else:
        root_row = room_top + 1 + random.randint(1, interior_h - 2)
        root = (root_row, room_left) if direction == "left" else (root_row, room_right)
        axis_offset = random.randint(0, block_h - 1)
        block_top = root_row - axis_offset
        block_bottom = block_top + block_h - 1
        if block_top < 0 or block_bottom >= h:
            return None
        if direction == "left":
            block_right = room_left - gap - 3
            block_left = block_right - block_w + 1
        else:
            block_left = room_right + gap + 3
            block_right = block_left + block_w - 1

    if block_top < 0 or block_left < 0 or block_bottom >= h or block_right >= w:
        return None

    arrow = _arrow_cells(root, direction)
    if not all(_in_bounds(shape, r, c) for r, c in arrow):
        return None

    object_cells = set(arrow)
    for r in range(room_top, room_bottom + 1):
        for c in range(room_left, room_right + 1):
            object_cells.add((r, c))
    for r in range(block_top, block_bottom + 1):
        for c in range(block_left, block_right + 1):
            object_cells.add((r, c))

    rows = [r for r, _ in object_cells]
    cols = [c for _, c in object_cells]
    footprint_bbox = (min(rows), min(cols), max(rows), max(cols))

    return {
        "direction": direction,
        "color": color,
        "room_outer": (room_top, room_left, room_bottom, room_right),
        "room_inner": inner_bbox,
        "block_bbox": (block_top, block_left, block_bottom, block_right),
        "root": root,
        "arrow": arrow,
        "footprint_bbox": footprint_bbox,
    }


def _can_place(reserved: np.ndarray, plan: dict[str, object]) -> bool:
    r0, c0, r1, c1 = _expanded_bbox(plan["footprint_bbox"], reserved.shape, pad=1)
    return not np.any(reserved[r0 : r1 + 1, c0 : c1 + 1])


def _paint_plan(grid: np.ndarray, expected: np.ndarray, reserved: np.ndarray, plan: dict[str, object]) -> None:
    room_top, room_left, room_bottom, room_right = plan["room_outer"]
    inner_top, inner_left, inner_bottom, inner_right = plan["room_inner"]
    block_top, block_left, block_bottom, block_right = plan["block_bbox"]
    color = int(plan["color"])

    for canvas in (grid, expected):
        canvas[room_top : room_bottom + 1, room_left] = GREY
        canvas[room_top : room_bottom + 1, room_right] = GREY
        canvas[room_top, room_left : room_right + 1] = GREY
        canvas[room_bottom, room_left : room_right + 1] = GREY
        canvas[block_top : block_bottom + 1, block_left : block_right + 1] = color
    for r, c in plan["arrow"]:
        grid[r, c] = RED
        expected[r, c] = RED
    expected[inner_top : inner_bottom + 1, inner_left : inner_right + 1] = color

    r0, c0, r1, c1 = _expanded_bbox(plan["footprint_bbox"], reserved.shape, pad=1)
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    area = h * w
    max_rooms = max(1, min(4, area // 180))
    room_count = random.randint(1, max_rooms)

    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()
    reserved = np.zeros((h, w), dtype=bool)
    placed = 0

    for _ in range(room_count):
        plan = None
        for _ in range(PLACEMENT_RETRIES):
            candidate = _sample_plan((h, w))
            if candidate is None or not _can_place(reserved, candidate):
                continue
            plan = candidate
            break
        if plan is None:
            return None
        _paint_plan(grid, expected, reserved, plan)
        placed += 1

    if placed == 0:
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-room puzzle")
