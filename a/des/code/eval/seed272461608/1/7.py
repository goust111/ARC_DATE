from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
FURNITURE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_GRID_DIM = 30
MIN_GRID_DIM = 14
TRUNK_MARGIN = 3
GENERATION_ATTEMPTS = 500

ROOM_COUNT_CHOICES = [2, 3, 4]
ROOM_H_RANGE = (4, 7)
ROOM_W_RANGE = (4, 8)
SMALL_ROOM_H_RANGE = (4, 5)
SMALL_ROOM_W_RANGE = (4, 6)
TRUNK_SIDES = ("left", "right", "top", "bottom")

FURNITURE_PATTERNS = {
    "up": {
        "direction": (-1, 0),
        "cells": {(0, 0), (0, 1), (1, 0)},
        "tip": (0, 0),
    },
    "right": {
        "direction": (0, 1),
        "cells": {(0, 0), (0, 1), (1, 1)},
        "tip": (0, 1),
    },
    "down": {
        "direction": (1, 0),
        "cells": {(0, 1), (1, 0), (1, 1)},
        "tip": (1, 1),
    },
    "left": {
        "direction": (0, -1),
        "cells": {(0, 0), (1, 0), (1, 1)},
        "tip": (1, 0),
    },
}
TIP_TO_DIRECTION = {
    (0, 0): (-1, 0),
    (0, 1): (0, 1),
    (1, 1): (1, 0),
    (1, 0): (0, -1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        if len(comps) != 1:
            continue
        if _touches_boundary(comps[0], (h, w)):
            candidates.append((int(comps[0].shape[0]), color))

    assert candidates, "expected a unique wall color connected to the grid edge"
    best_area = max(area for area, _color in candidates)
    best_colors = [color for area, color in candidates if area == best_area]
    assert len(best_colors) == 1, "wall color must dominate all other edge-touching colors"
    return int(best_colors[0])


def _decode_furniture(local_coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    assert local_coords.shape[0] == 3, "furniture must have exactly three pixels"
    r0, c0, r1, c1 = _bbox(local_coords)
    assert (r1 - r0, c1 - c0) == (1, 1), "furniture must fit inside a 2x2 box"

    rel = {(int(r - r0), int(c - c0)) for r, c in local_coords}
    missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - rel
    assert len(missing) == 1, "furniture must be a 2x2 box missing one corner"

    missing_r, missing_c = next(iter(missing))
    tip_rel = (1 - missing_r, 1 - missing_c)
    tip = (r0 + tip_rel[0], c0 + tip_rel[1])
    return (int(tip[0]), int(tip[1])), TIP_TO_DIRECTION[tip_rel]


def _find_rooms(grid: np.ndarray) -> tuple[list[dict], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    wall = _detect_wall_color(g, bg)
    h, w = g.shape
    rooms: list[dict] = []

    for comp in _components(g == bg):
        if _touches_boundary(comp, (h, w)):
            continue

        r0, c0, r1, c1 = _bbox(comp)
        comp_set = set(map(tuple, comp.tolist()))

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if int(g[rr, cc]) == bg:
                    assert (rr, cc) in comp_set, "room background must fill a rectangle except for furniture"

        assert r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1
        assert np.all(g[r0 - 1, c0 : c1 + 1] == wall)
        assert np.all(g[r1 + 1, c0 : c1 + 1] == wall)
        assert np.all(g[r0 : r1 + 1, c0 - 1] == wall)
        assert np.all(g[r0 : r1 + 1, c1 + 1] == wall)

        furniture_mask = np.zeros_like(g, dtype=bool)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if int(g[rr, cc]) not in {bg, wall}:
                    furniture_mask[rr, cc] = True

        furniture_components = _components(furniture_mask)
        assert len(furniture_components) == 1, "each room must contain exactly one furniture component"
        tip, direction = _decode_furniture(furniture_components[0])
        rooms.append({"bbox": (r0, c0, r1, c1), "tip": tip, "direction": direction})

    return rooms, bg, wall


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open one wall pixel for each enclosed room based on the room's 3-cell L furniture.

    The room is the non-boundary background rectangle surrounded by the connected wall
    network. The furniture is a 2x2 L missing one corner. The occupied corner that is
    diagonally opposite the missing cell acts as the furniture tip. Its corner position
    around the 2x2 box maps clockwise to the door direction: top-left -> up,
    top-right -> right, bottom-right -> down, bottom-left -> left.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rooms, bg, wall = _find_rooms(g)
    h, w = g.shape

    for room in rooms:
        rr, cc = room["tip"]
        dr, dc = room["direction"]
        while True:
            rr += dr
            cc += dc
            assert 0 <= rr < h and 0 <= cc < w, "door ray must hit a wall before leaving the grid"
            if int(g[rr, cc]) == wall:
                out[rr, cc] = bg
                break
            assert int(g[rr, cc]) == bg, "door ray must cross only room background"

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _split_with_min(total: int, parts: int, min_each: int) -> list[int] | None:
    extra = total - parts * min_each
    if extra < 0:
        return None
    return [min_each + value for value in _random_partition(extra, parts)]


def _sample_room_sizes(count: int) -> list[tuple[int, int]]:
    h_range = SMALL_ROOM_H_RANGE if count == 4 else ROOM_H_RANGE
    w_range = SMALL_ROOM_W_RANGE if count == 4 else ROOM_W_RANGE
    return [(random.randint(*h_range), random.randint(*w_range)) for _ in range(count)]


def _place_rooms(trunk_side: str) -> tuple[int, int, list[dict]] | None:
    count = random.choice(ROOM_COUNT_CHOICES)
    sizes = _sample_room_sizes(count)
    box_heights = [room_h + 2 for room_h, _room_w in sizes]
    box_widths = [room_w + 2 for _room_h, room_w in sizes]

    if trunk_side in {"left", "right"}:
        min_h = sum(box_heights) + count + 1
        if min_h > MAX_GRID_DIM:
            return None
        grid_h = random.randint(max(MIN_GRID_DIM, min_h), min(MAX_GRID_DIM, min_h + 8))
        gaps = _split_with_min(grid_h - sum(box_heights), count + 1, 1)
        if gaps is None:
            return None

        min_w = TRUNK_MARGIN + max(box_widths) + 2
        if min_w > MAX_GRID_DIM:
            return None
        grid_w = random.randint(max(MIN_GRID_DIM, min_w), min(MAX_GRID_DIM, min_w + 8))

        rooms: list[dict] = []
        row = gaps[0]
        for idx, (room_h, room_w) in enumerate(sizes):
            box_h = room_h + 2
            box_w = room_w + 2
            col_min = TRUNK_MARGIN if trunk_side == "left" else 2
            col_max = grid_w - box_w - 2 if trunk_side == "left" else grid_w - box_w - TRUNK_MARGIN
            if col_min > col_max:
                return None
            col = random.randint(col_min, col_max)
            rooms.append(
                {
                    "top": row,
                    "left": col,
                    "bottom": row + box_h - 1,
                    "right": col + box_w - 1,
                    "interior_h": room_h,
                    "interior_w": room_w,
                }
            )
            row += box_h + gaps[idx + 1]
        return grid_h, grid_w, rooms

    min_w = sum(box_widths) + count + 1
    if min_w > MAX_GRID_DIM:
        return None
    grid_w = random.randint(max(MIN_GRID_DIM, min_w), min(MAX_GRID_DIM, min_w + 8))
    gaps = _split_with_min(grid_w - sum(box_widths), count + 1, 1)
    if gaps is None:
        return None

    min_h = TRUNK_MARGIN + max(box_heights) + 2
    if min_h > MAX_GRID_DIM:
        return None
    grid_h = random.randint(max(MIN_GRID_DIM, min_h), min(MAX_GRID_DIM, min_h + 8))

    rooms = []
    col = gaps[0]
    for idx, (room_h, room_w) in enumerate(sizes):
        box_h = room_h + 2
        box_w = room_w + 2
        row_min = TRUNK_MARGIN if trunk_side == "top" else 2
        row_max = grid_h - box_h - 2 if trunk_side == "top" else grid_h - box_h - TRUNK_MARGIN
        if row_min > row_max:
            return None
        row = random.randint(row_min, row_max)
        rooms.append(
            {
                "top": row,
                "left": col,
                "bottom": row + box_h - 1,
                "right": col + box_w - 1,
                "interior_h": room_h,
                "interior_w": room_w,
            }
        )
        col += box_w + gaps[idx + 1]
    return grid_h, grid_w, rooms


def _planned_directions(count: int) -> list[str]:
    names = list(FURNITURE_PATTERNS)
    random.shuffle(names)
    chosen = names[: min(count, len(names))]
    while len(chosen) < count:
        chosen.append(random.choice(names))
    random.shuffle(chosen)
    return chosen


def _choose_furniture_position(room: dict, direction_name: str) -> tuple[int, int] | None:
    interior_h = int(room["interior_h"])
    interior_w = int(room["interior_w"])
    positions: list[tuple[int, int]] = []

    for box_r in range(1, interior_h - 2):
        for box_c in range(1, interior_w - 2):
            positions.append((box_r, box_c))

    if not positions:
        return None

    return random.choice(positions)


def _choose_connector_axis(
    room: dict,
    trunk_side: str,
    direction_name: str,
    furniture_box: tuple[int, int],
) -> int | None:
    box_r, box_c = furniture_box
    tip_r, tip_c = FURNITURE_PATTERNS[direction_name]["tip"]
    tip_row = box_r + tip_r
    tip_col = box_c + tip_c
    interior_h = int(room["interior_h"])
    interior_w = int(room["interior_w"])
    dr, dc = FURNITURE_PATTERNS[direction_name]["direction"]

    if trunk_side in {"left", "right"}:
        options = list(range(interior_h))
        if (trunk_side == "left" and dc == -1) or (trunk_side == "right" and dc == 1):
            options = [row for row in options if row != tip_row]
    else:
        options = list(range(interior_w))
        if (trunk_side == "top" and dr == -1) or (trunk_side == "bottom" and dr == 1):
            options = [col for col in options if col != tip_col]

    if not options:
        return None

    return random.choice(options)


def _paint_room(grid: np.ndarray, room: dict) -> None:
    top = int(room["top"])
    left = int(room["left"])
    bottom = int(room["bottom"])
    right = int(room["right"])
    grid[top, left : right + 1] = WALL_COLOR
    grid[bottom, left : right + 1] = WALL_COLOR
    grid[top : bottom + 1, left] = WALL_COLOR
    grid[top : bottom + 1, right] = WALL_COLOR


def _paint_connector(grid: np.ndarray, room: dict, trunk_side: str, axis: int) -> None:
    top = int(room["top"])
    left = int(room["left"])
    bottom = int(room["bottom"])
    right = int(room["right"])

    if trunk_side == "left":
        rr = top + 1 + axis
        grid[rr, 0 : left + 1] = WALL_COLOR
    elif trunk_side == "right":
        rr = top + 1 + axis
        grid[rr, right : grid.shape[1]] = WALL_COLOR
    elif trunk_side == "top":
        cc = left + 1 + axis
        grid[0 : top + 1, cc] = WALL_COLOR
    else:
        cc = left + 1 + axis
        grid[bottom : grid.shape[0], cc] = WALL_COLOR


def _paint_furniture(
    grid: np.ndarray,
    room: dict,
    direction_name: str,
    furniture_box: tuple[int, int],
    color: int,
) -> None:
    top = int(room["top"]) + 1
    left = int(room["left"]) + 1
    box_r, box_c = furniture_box
    for dr, dc in FURNITURE_PATTERNS[direction_name]["cells"]:
        grid[top + box_r + dr, left + box_c + dc] = int(color)


def _valid_generated_input(grid: np.ndarray, expected_room_count: int) -> bool:
    if _mode_color(grid) != BG_COLOR:
        return False

    try:
        rooms, bg, wall = _find_rooms(grid)
        out = main(grid)
    except Exception:
        return False

    if len(rooms) != expected_room_count:
        return False

    diff = np.argwhere(grid != out)
    if diff.shape[0] != expected_room_count:
        return False
    if not np.all(grid[diff[:, 0], diff[:, 1]] == wall):
        return False
    if not np.all(out[diff[:, 0], diff[:, 1]] == bg):
        return False

    boundary_bg = np.zeros_like(out, dtype=bool)
    for comp in _components(out == bg):
        if _touches_boundary(comp, out.shape):
            boundary_bg[comp[:, 0], comp[:, 1]] = True
    if not all(boundary_bg[r, c] for r, c in map(tuple, diff.tolist())):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    trunk_side = random.choice(TRUNK_SIDES)
    placed = _place_rooms(trunk_side)
    if placed is None:
        return None

    grid_h, grid_w, rooms = placed
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    if trunk_side == "left":
        grid[:, 0] = WALL_COLOR
    elif trunk_side == "right":
        grid[:, -1] = WALL_COLOR
    elif trunk_side == "top":
        grid[0, :] = WALL_COLOR
    else:
        grid[-1, :] = WALL_COLOR

    colors = random.sample(FURNITURE_COLORS, len(rooms))
    directions = _planned_directions(len(rooms))

    for room, color, direction_name in zip(rooms, colors, directions):
        _paint_room(grid, room)
        furniture_box = _choose_furniture_position(room, direction_name)
        if furniture_box is None:
            return None
        connector_axis = _choose_connector_axis(room, trunk_side, direction_name, furniture_box)
        if connector_axis is None:
            return None
        _paint_connector(grid, room, trunk_side, connector_axis)
        _paint_furniture(grid, room, direction_name, furniture_box, color)

    if not _valid_generated_input(grid, len(rooms)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_ATTEMPTS} attempts")
