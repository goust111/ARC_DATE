from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
CORNERS = ("TL", "TR", "BL", "BR")

MIN_ROOM_SIZE = 5
MAX_ROOM_SIZE = 8
ROOM_COUNT_RANGE = (2, 4)
DOOR_PROBABILITY = 0.5
MAX_GENERATION_ATTEMPTS = 400

PATTERNS = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1], [0], [1]], dtype=bool),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
)


def _adjacent_wall_indices(g: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(g.shape[0]) if np.any((g[r, :-1] == WALL) & (g[r, 1:] == WALL))]
    return [c for c in range(g.shape[1]) if np.any((g[:-1, c] == WALL) & (g[1:, c] == WALL))]


def _build_rooms(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], dict[str, object]]:
    rooms: dict[tuple[int, int], dict[str, object]] = {}
    for ri in range(len(wall_rows) - 1):
        for cj in range(len(wall_cols) - 1):
            r0, r1 = int(wall_rows[ri]), int(wall_rows[ri + 1])
            c0, c1 = int(wall_cols[cj]), int(wall_cols[cj + 1])
            assert r1 - r0 >= 2 and c1 - c0 >= 2, "each room needs at least one interior cell"
            rooms[(ri, cj)] = {
                "index": (ri, cj),
                "walls": (r0, c0, r1, c1),
                "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
    return rooms


def _extract_seed(g: np.ndarray, room: dict[str, object]) -> dict[str, object]:
    ir0, ic0, ir1, ic1 = room["inner"]
    region = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    mask = (region != BG) & (region != WALL)
    assert np.any(mask), "every room must contain one seed pattern"
    colors = {int(v) for v in region[mask]}
    assert len(colors) == 1, "seed pattern must be monochrome"
    color = colors.pop()

    coords = np.argwhere(mask)
    sr0, sc0 = map(int, coords.min(axis=0))
    sr1, sc1 = map(int, coords.max(axis=0))
    patch_mask = mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
    rel_coords = [tuple(map(int, rc)) for rc in np.argwhere(patch_mask)]
    patch_h = sr1 - sr0 + 1
    patch_w = sc1 - sc0 + 1
    assert patch_h <= 3 and patch_w <= 3, "seed pattern bbox must fit within 3x3"

    top_aligned = sr0 == 0
    bottom_aligned = sr1 == region.shape[0] - 1
    left_aligned = sc0 == 0
    right_aligned = sc1 == region.shape[1] - 1
    assert (top_aligned or bottom_aligned) and (left_aligned or right_aligned), "seed must occupy a room corner"

    abs_coords = {(ir0 + int(r), ic0 + int(c)) for r, c in coords}
    return {
        "color": color,
        "shape": (patch_h, patch_w),
        "coords": rel_coords,
        "protected": abs_coords,
    }


def _room_doors(
    g: np.ndarray,
    room: dict[str, object],
    room_rows: int,
    room_cols: int,
) -> list[tuple[int, int, str, tuple[int, int]]]:
    ri, cj = room["index"]
    r0, c0, r1, c1 = room["walls"]
    doors: list[tuple[int, int, str, tuple[int, int]]] = []

    if ri > 0:
        for c in range(c0 + 1, c1):
            assert int(g[r0, c]) in {BG, WALL}
            if int(g[r0, c]) == BG:
                doors.append((r0, c, "top", (ri - 1, cj)))
    if ri + 1 < room_rows:
        for c in range(c0 + 1, c1):
            assert int(g[r1, c]) in {BG, WALL}
            if int(g[r1, c]) == BG:
                doors.append((r1, c, "bottom", (ri + 1, cj)))
    if cj > 0:
        for r in range(r0 + 1, r1):
            assert int(g[r, c0]) in {BG, WALL}
            if int(g[r, c0]) == BG:
                doors.append((r, c0, "left", (ri, cj - 1)))
    if cj + 1 < room_cols:
        for r in range(r0 + 1, r1):
            assert int(g[r, c1]) in {BG, WALL}
            if int(g[r, c1]) == BG:
                doors.append((r, c1, "right", (ri, cj + 1)))

    doors.sort(key=lambda item: (item[0], item[1]))
    return doors


def _stamp_projection(
    out: np.ndarray,
    room: dict[str, object],
    side: str,
    door_r: int,
    door_c: int,
    seed: dict[str, object],
    protected: set[tuple[int, int]],
) -> None:
    ir0, ic0, ir1, ic1 = room["inner"]
    patch_h, patch_w = seed["shape"]
    center_r = (patch_h - 1) // 2
    center_c = (patch_w - 1) // 2

    if side == "top":
        base_r = ir0
        base_c = door_c - center_c
    elif side == "bottom":
        base_r = ir1 - patch_h + 1
        base_c = door_c - center_c
    elif side == "left":
        base_r = door_r - center_r
        base_c = ic0
    else:
        base_r = door_r - center_r
        base_c = ic1 - patch_w + 1

    assert ir0 <= base_r <= ir1 - patch_h + 1, "projected pattern must fit vertically inside the room"
    assert ic0 <= base_c <= ic1 - patch_w + 1, "projected pattern must fit horizontally inside the room"

    color = int(seed["color"])
    for dr, dc in seed["coords"]:
        r = base_r + int(dr)
        c = base_c + int(dc)
        if (r, c) not in protected:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each neighboring room's seed pattern through wall doorways into the current room.

    Input structure:
    - The background is black.
    - Azure one-cell-thick horizontal and vertical walls split the grid into rectangular rooms.
    - Shared wall segments may contain black single-cell door holes.
    - Each room contains one monochrome seed pattern whose tight bbox is at most 3x3 and sits in a room corner.

    Rule:
    - Detect the wall rows and wall columns, then treat each rectangle between consecutive walls as one room.
    - For every room, inspect its four wall segments for door holes and identify the adjacent room across each hole.
    - Copy the adjacent room's seed pattern into the current room so it touches the doorway wall from the inside, with the pattern center aligned to the doorway along that wall.
    - If several doorway projections overlap, apply them in doorway order from top to bottom, then left to right; the room's original seed cells stay unchanged.

    Key insight:
    - A doorway is an inlet: it does not send the current room's seed outward, it pulls the neighbor's seed inward with wall-aligned placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_rows = _adjacent_wall_indices(g, axis=0)
    wall_cols = _adjacent_wall_indices(g, axis=1)
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected at least one room"
    assert wall_rows[0] == 0 and wall_rows[-1] == g.shape[0] - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == g.shape[1] - 1

    rooms = _build_rooms(wall_rows, wall_cols)
    seeds = {index: _extract_seed(g, room) for index, room in rooms.items()}
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1

    out = g.copy()
    for index, room in rooms.items():
        protected = seeds[index]["protected"]
        for door_r, door_c, side, neighbor in _room_doors(g, room, room_rows, room_cols):
            _stamp_projection(out, room, side, door_r, door_c, seeds[neighbor], protected)
    return out


def _sample_axis_sizes() -> list[int]:
    count = random.randint(*ROOM_COUNT_RANGE)
    sizes = [MIN_ROOM_SIZE] * count
    extra_budget = random.randint(0, count * (MAX_ROOM_SIZE - MIN_ROOM_SIZE))
    while extra_budget > 0:
        choices = [idx for idx, size in enumerate(sizes) if size < MAX_ROOM_SIZE]
        if not choices:
            break
        idx = random.choice(choices)
        sizes[idx] += 1
        extra_budget -= 1
    return sizes


def _wall_positions(sizes: list[int]) -> list[int]:
    walls = [0]
    for size in sizes:
        walls.append(walls[-1] + size + 1)
    return walls


def _corner_anchor(room: dict[str, object], patch: np.ndarray, corner: str) -> tuple[int, int]:
    ir0, ic0, ir1, ic1 = room["inner"]
    patch_h, patch_w = patch.shape
    if corner == "TL":
        return ir0, ic0
    if corner == "TR":
        return ir0, ic1 - patch_w + 1
    if corner == "BL":
        return ir1 - patch_h + 1, ic0
    if corner == "BR":
        return ir1 - patch_h + 1, ic1 - patch_w + 1
    raise AssertionError(corner)


def _paint_seed(grid: np.ndarray, room: dict[str, object], patch: np.ndarray, color: int, corner: str) -> None:
    base_r, base_c = _corner_anchor(room, patch, corner)
    for dr, dc in np.argwhere(patch):
        grid[base_r + int(dr), base_c + int(dc)] = int(color)


def _valid_center_interval(low: int, high: int, span: int) -> tuple[int, int]:
    center = (span - 1) // 2
    tail = span - 1 - center
    return low + center, high - tail


def _horizontal_candidates(
    room: dict[str, object],
    upper_width: int,
    lower_width: int,
) -> list[int]:
    _ir0, ic0, _ir1, ic1 = room["inner"]
    a0, a1 = _valid_center_interval(ic0, ic1, upper_width)
    b0, b1 = _valid_center_interval(ic0, ic1, lower_width)
    start = max(a0, b0)
    end = min(a1, b1)
    return list(range(start, end + 1)) if start <= end else []


def _vertical_candidates(
    room: dict[str, object],
    left_height: int,
    right_height: int,
) -> list[int]:
    ir0, _ic0, ir1, _ic1 = room["inner"]
    a0, a1 = _valid_center_interval(ir0, ir1, left_height)
    b0, b1 = _valid_center_interval(ir0, ir1, right_height)
    start = max(a0, b0)
    end = min(a1, b1)
    return list(range(start, end + 1)) if start <= end else []


def _assign_colors(room_rows: int, room_cols: int) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    for ri in range(room_rows):
        for cj in range(room_cols):
            blocked = set()
            if ri > 0:
                blocked.add(colors[(ri - 1, cj)])
            if cj > 0:
                blocked.add(colors[(ri, cj - 1)])
            choices = [color for color in PALETTE if color not in blocked]
            colors[(ri, cj)] = random.choice(choices)
    return colors


def _build_input() -> np.ndarray | None:
    row_sizes = _sample_axis_sizes()
    col_sizes = _sample_axis_sizes()
    wall_rows = _wall_positions(row_sizes)
    wall_cols = _wall_positions(col_sizes)
    height = wall_rows[-1] + 1
    width = wall_cols[-1] + 1
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    rooms = _build_rooms(wall_rows, wall_cols)
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    colors = _assign_colors(room_rows, room_cols)

    room_specs: dict[tuple[int, int], dict[str, object]] = {}
    for index, room in rooms.items():
        patch = random.choice(PATTERNS)
        corner = random.choice(CORNERS)
        color = colors[index]
        room_specs[index] = {"patch": patch, "corner": corner, "color": color}
        _paint_seed(grid, room, patch, color, corner)

    candidate_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    chosen_doors: list[tuple[int, int]] = []

    for ri in range(room_rows - 1):
        for cj in range(room_cols):
            top_patch = room_specs[(ri, cj)]["patch"]
            bottom_patch = room_specs[(ri + 1, cj)]["patch"]
            candidates = _horizontal_candidates(rooms[(ri, cj)], top_patch.shape[1], bottom_patch.shape[1])
            if not candidates:
                continue
            candidate_segments.append(((wall_rows[ri + 1], -1), (ri, cj)))
            if random.random() < DOOR_PROBABILITY:
                chosen_doors.append((wall_rows[ri + 1], random.choice(candidates)))

    for ri in range(room_rows):
        for cj in range(room_cols - 1):
            left_patch = room_specs[(ri, cj)]["patch"]
            right_patch = room_specs[(ri, cj + 1)]["patch"]
            candidates = _vertical_candidates(rooms[(ri, cj)], left_patch.shape[0], right_patch.shape[0])
            if not candidates:
                continue
            candidate_segments.append(((-1, wall_cols[cj + 1]), (ri, cj)))
            if random.random() < DOOR_PROBABILITY:
                chosen_doors.append((random.choice(candidates), wall_cols[cj + 1]))

    if not chosen_doors:
        fallback_options: list[tuple[int, int]] = []
        for ri in range(room_rows - 1):
            for cj in range(room_cols):
                top_patch = room_specs[(ri, cj)]["patch"]
                bottom_patch = room_specs[(ri + 1, cj)]["patch"]
                for col in _horizontal_candidates(rooms[(ri, cj)], top_patch.shape[1], bottom_patch.shape[1]):
                    fallback_options.append((wall_rows[ri + 1], col))
        for ri in range(room_rows):
            for cj in range(room_cols - 1):
                left_patch = room_specs[(ri, cj)]["patch"]
                right_patch = room_specs[(ri, cj + 1)]["patch"]
                for row in _vertical_candidates(rooms[(ri, cj)], left_patch.shape[0], right_patch.shape[0]):
                    fallback_options.append((row, wall_cols[cj + 1]))
        if not fallback_options:
            return None
        chosen_doors.append(random.choice(fallback_options))

    for r, c in chosen_doors:
        grid[r, c] = BG

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid doorway-projection ARC input")
