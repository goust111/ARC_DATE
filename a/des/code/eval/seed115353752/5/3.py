import random
from collections import Counter

import numpy as np


DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in CARDINALS:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _room_from_component(
    grid: np.ndarray,
    color: int,
    coords: list[tuple[int, int]],
    bg: int,
):
    if len(coords) < 7:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coord_set = set(coords)
    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if len(coord_set) != len(perimeter) - 1 or not coord_set.issubset(perimeter):
        return None

    missing = list(perimeter - coord_set)
    if len(missing) != 1:
        return None
    door_r, door_c = missing[0]
    if door_r == r0 and c0 < door_c < c1:
        direction = "up"
    elif door_r == r1 and c0 < door_c < c1:
        direction = "down"
    elif door_c == c0 and r0 < door_r < r1:
        direction = "left"
    elif door_c == c1 and r0 < door_r < r1:
        direction = "right"
    else:
        return None

    furniture: list[tuple[int, int, int]] = []
    inside_colors = set()
    empty_count = 0
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            val = int(grid[r, c])
            if val == bg:
                empty_count += 1
            else:
                if val == color:
                    return None
                furniture.append((r, c, val))
                inside_colors.add(val)

    if not furniture or len(inside_colors) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_color": color,
        "direction": direction,
        "door": (door_r, door_c),
        "distance": empty_count,
        "furniture": furniture,
    }


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rooms = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _cc4(g == color):
            room = _room_from_component(g, color, comp, bg)
            if room is not None:
                rooms.append(room)
    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1], room["border_color"]))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for room in _find_rooms(g):
        dr, dc = DIR_VECS[room["direction"]]
        distance = int(room["distance"])
        for r, c, val in room["furniture"]:
            rr = r + dr * distance
            cc = c + dc * distance
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = val
    return out


INTERIOR_OPTIONS = [
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
]
ROOM_COUNT_RANGE = (1, 3)
GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
ROOM_BUILD_ATTEMPTS = 80
ROOM_PLACE_ATTEMPTS = 120
GEN_ATTEMPTS = 400
CLEARANCE = 1
FG_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _accessible_cells(direction: str, ih: int, iw: int, distance: int) -> list[tuple[int, int]]:
    cells = []
    if direction == "up":
        max_row = min(ih - 1, distance - 2)
        for r in range(max_row + 1):
            for c in range(iw):
                cells.append((r, c))
    elif direction == "down":
        min_row = max(0, ih + 1 - distance)
        for r in range(min_row, ih):
            for c in range(iw):
                cells.append((r, c))
    elif direction == "left":
        max_col = min(iw - 1, distance - 2)
        for r in range(ih):
            for c in range(max_col + 1):
                cells.append((r, c))
    else:
        min_col = max(0, iw + 1 - distance)
        for r in range(ih):
            for c in range(min_col, iw):
                cells.append((r, c))
    return cells


def _random_connected_subset(
    rng: random.Random,
    allowed_cells: list[tuple[int, int]],
    size: int,
):
    allowed = set(allowed_cells)
    if len(allowed) < size:
        return None
    allowed_list = list(allowed)
    for _ in range(60):
        start = rng.choice(allowed_list)
        shape = {start}
        while len(shape) < size:
            frontier = set()
            for r, c in shape:
                for dr, dc in CARDINALS:
                    nxt = (r + dr, c + dc)
                    if nxt in allowed and nxt not in shape:
                        frontier.add(nxt)
            if not frontier:
                break
            shape.add(rng.choice(list(frontier)))
        if len(shape) == size:
            return sorted(shape)
    return None


def _build_room_spec(
    rng: random.Random,
    direction: str,
    border_color: int,
    furniture_color: int,
):
    for _ in range(ROOM_BUILD_ATTEMPTS):
        ih, iw = rng.choice(INTERIOR_OPTIONS)
        area = ih * iw
        feasible = []
        max_cells = min(5, area - 2)
        for size in range(2, max_cells + 1):
            distance = area - size
            if distance < 2:
                continue
            allowed = _accessible_cells(direction, ih, iw, distance)
            if len(allowed) >= size:
                feasible.append((size, distance, allowed))
        if not feasible:
            continue

        size, distance, allowed = rng.choice(feasible)
        interior_cells = _random_connected_subset(rng, allowed, size)
        if interior_cells is None:
            continue

        outer_h = ih + 2
        outer_w = iw + 2
        if direction == "up":
            door = (0, rng.randrange(1, outer_w - 1))
        elif direction == "down":
            door = (outer_h - 1, rng.randrange(1, outer_w - 1))
        elif direction == "left":
            door = (rng.randrange(1, outer_h - 1), 0)
        else:
            door = (rng.randrange(1, outer_h - 1), outer_w - 1)

        border_cells = list(_perimeter_cells(0, 0, outer_h - 1, outer_w - 1) - {door})
        furniture_cells = [(1 + r, 1 + c) for r, c in interior_cells]
        dr, dc = DIR_VECS[direction]
        projection_cells = [(r + dr * distance, c + dc * distance) for r, c in furniture_cells]
        track_cells = set()
        for r, c in furniture_cells:
            for step in range(1, distance + 1):
                track_cells.add((r + dr * step, c + dc * step))

        return {
            "direction": direction,
            "distance": distance,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "border_color": border_color,
            "furniture_color": furniture_color,
            "border_cells": border_cells,
            "furniture_cells": furniture_cells,
            "projection_cells": projection_cells,
            "track_cells": sorted(track_cells),
        }
    return None


def _reservation_cells(spec: dict, top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, top - CLEARANCE), min(h, top + spec["outer_h"] + CLEARANCE)):
        for c in range(max(0, left - CLEARANCE), min(w, left + spec["outer_w"] + CLEARANCE)):
            cells.add((r, c))
    for rel_r, rel_c in spec["track_cells"]:
        abs_r = top + rel_r
        abs_c = left + rel_c
        for r in range(max(0, abs_r - CLEARANCE), min(h, abs_r + CLEARANCE + 1)):
            for c in range(max(0, abs_c - CLEARANCE), min(w, abs_c + CLEARANCE + 1)):
                cells.add((r, c))
    return cells


def _can_place(spec: dict, top: int, left: int, reserved: np.ndarray) -> bool:
    h, w = reserved.shape
    if top < 0 or left < 0 or top + spec["outer_h"] > h or left + spec["outer_w"] > w:
        return False

    for rel_r, rel_c in spec["track_cells"]:
        abs_r = top + rel_r
        abs_c = left + rel_c
        if not (0 <= abs_r < h and 0 <= abs_c < w):
            return False

    for r, c in _reservation_cells(spec, top, left, h, w):
        if reserved[r, c]:
            return False
    return True


def _draw_room(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    for rel_r, rel_c in spec["border_cells"]:
        grid[top + rel_r, left + rel_c] = spec["border_color"]
    for rel_r, rel_c in spec["furniture_cells"]:
        grid[top + rel_r, left + rel_c] = spec["furniture_color"]


def _mark_reserved(reserved: np.ndarray, spec: dict, top: int, left: int) -> None:
    for r, c in _reservation_cells(spec, top, left, *reserved.shape):
        reserved[r, c] = True


def _expected_output(grid: np.ndarray, placements: list[tuple[dict, int, int]]) -> np.ndarray:
    out = grid.copy()
    for spec, top, left in placements:
        dr, dc = DIR_VECS[spec["direction"]]
        for rel_r, rel_c in spec["furniture_cells"]:
            rr = top + rel_r + dr * spec["distance"]
            cc = left + rel_c + dc * spec["distance"]
            out[rr, cc] = spec["furniture_color"]
    return out


def _generate_once(rng: random.Random):
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    room_count = rng.randint(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1])
    directions = rng.sample(list(DIR_VECS), k=4)[:room_count]
    colors = rng.sample(FG_COLORS, k=2 * room_count)

    placements: list[tuple[dict, int, int]] = []
    for idx in range(room_count):
        spec = _build_room_spec(
            rng,
            directions[idx],
            colors[2 * idx],
            colors[2 * idx + 1],
        )
        if spec is None:
            return None

        rel_rows = [0, spec["outer_h"] - 1] + [r for r, _ in spec["track_cells"]]
        rel_cols = [0, spec["outer_w"] - 1] + [c for _, c in spec["track_cells"]]
        min_r = min(rel_rows)
        max_r = max(rel_rows)
        min_c = min(rel_cols)
        max_c = max(rel_cols)
        top_min = -min_r
        top_max = h - 1 - max_r
        left_min = -min_c
        left_max = w - 1 - max_c
        if top_min > top_max or left_min > left_max:
            return None

        placed = False
        for _ in range(ROOM_PLACE_ATTEMPTS):
            top = rng.randint(top_min, top_max)
            left = rng.randint(left_min, left_max)
            if not _can_place(spec, top, left, reserved):
                continue
            _draw_room(grid, spec, top, left)
            _mark_reserved(reserved, spec, top, left)
            placements.append((spec, top, left))
            placed = True
            break
        if not placed:
            return None

    rooms = _find_rooms(grid)
    if len(rooms) != room_count:
        return None

    expected = _expected_output(grid, placements)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid instance")
