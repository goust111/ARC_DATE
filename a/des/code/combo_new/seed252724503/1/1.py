from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
WALL = 5

DIR_COLOR_TO_VEC = {
    2: (0, 1),   # red -> right
    1: (1, 0),   # blue -> down
    3: (0, -1),  # green -> left
    4: (-1, 0),  # yellow -> up
}
VEC_TO_DIR_COLOR = {vec: color for color, vec in DIR_COLOR_TO_VEC.items()}
SEED_COLORS = (6, 7, 8, 9)

MODE_CHOICES = ("horizontal_pair", "vertical_pair", "ring_2x2")
MODE_WEIGHTS = (1, 1, 2)
ROOM_SIZE_RANGE = (2, 5)
SIDE_MARGIN_RANGE = (1, 3)
LEGEND_EXTRA_MARGIN_RANGE = (1, 3)
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_label = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
            comps.append(coords)
            next_label += 1

    return labels, comps


def _extract_rooms(
    grid: np.ndarray,
    bg: int,
    wall: int,
) -> tuple[np.ndarray, list[dict], dict[int, int]]:
    passable = grid != wall
    labels, comps = _components(passable)
    rooms: list[dict] = []
    room_by_label: dict[int, int] = {}

    for label, coords in enumerate(comps):
        if _touches_boundary(coords, grid.shape):
            continue

        seed_cells = [(r, c) for r, c in coords if int(grid[r, c]) != bg]
        assert len(seed_cells) == 1, "each enclosed room must contain exactly one seed pixel"
        seed_r, seed_c = seed_cells[0]
        seed_color = int(grid[seed_r, seed_c])

        r0, c0, r1, c1 = _bbox(coords)
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "room interior must be a solid rectangle"

        bg_cells = [(r, c) for r, c in coords if int(grid[r, c]) == bg]
        room_index = len(rooms)
        rooms.append(
            {
                "index": room_index,
                "label": label,
                "bbox": (r0, c0, r1, c1),
                "seed_pos": (seed_r, seed_c),
                "seed_color": seed_color,
                "bg_cells": bg_cells,
            }
        )
        room_by_label[label] = room_index

    assert rooms, "expected at least one enclosed room"
    return labels, rooms, room_by_label


def _legend_mapping(grid: np.ndarray, bg: int, wall: int, seed_colors: set[int]) -> dict[int, tuple[int, int]]:
    fg_mask = (grid != bg) & (grid != wall)
    _, comps = _components(fg_mask)
    mapping: dict[int, tuple[int, int]] = {}

    for coords in comps:
        if not _touches_boundary(coords, grid.shape):
            continue

        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) != 2 or r0 != r1 or c1 - c0 != 1:
            continue

        colors = [int(grid[r, c]) for r, c in sorted(coords, key=lambda cell: cell[1])]
        candidates = [color for color in colors if color in seed_colors]
        directions = [color for color in colors if color in DIR_COLOR_TO_VEC]
        if len(candidates) != 1 or len(directions) != 1:
            continue

        seed_color = candidates[0]
        dir_color = directions[0]
        mapping[seed_color] = DIR_COLOR_TO_VEC[dir_color]

    assert set(mapping) == seed_colors, "legend must define one direction for every room seed color"
    return mapping


def _door_and_target(room: dict, direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    seed_r, seed_c = room["seed_pos"]
    r0, c0, r1, c1 = room["bbox"]

    if (dr, dc) == (0, 1):
        return (seed_r, c1 + 1), (seed_r, c1 + 2)
    if (dr, dc) == (0, -1):
        return (seed_r, c0 - 1), (seed_r, c0 - 2)
    if (dr, dc) == (1, 0):
        return (r1 + 1, seed_c), (r1 + 2, seed_c)
    if (dr, dc) == (-1, 0):
        return (r0 - 1, seed_c), (r0 - 2, seed_c)
    raise AssertionError("legend direction must be cardinal")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open a wall door for each seeded room using the corner legend, then flood the neighboring room with the seed color.

    Input structure:
    - The most common color is the black background, and one other dominant color forms one-cell-thick grey walls.
    - The walls partition the scene into enclosed rectangular room interiors; each interior contains exactly one distinct colored seed pixel.
    - A corner legend consists of several horizontal 1x2 color pairs touching the grid boundary, where one cell is a seed color and the other is a direction color.

    Rule:
    - Read the legend to map each seed color to a direction: red means open to the right, blue down, green left, yellow up.
    - For each room, locate the wall cell on the indicated side that is aligned with the seed row or seed column, and erase that wall cell to background.
    - Look one more step beyond that opened wall in the original grid, identify the neighboring enclosed room there, and recolor every background cell in that neighboring room with the source seed color.
    - Keep all seed pixels and the legend unchanged except for the removed door cells.

    Key insight:
    - The corner color pairs are a legend from seed color to door direction, and the fill applies to the adjacent room reached beyond that specific aligned wall segment, not to the seed's own room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected walls, seeds, and legend pixels"
    wall = _most_common_color(non_bg)

    labels, rooms, room_by_label = _extract_rooms(g, bg, wall)
    seed_colors = {int(room["seed_color"]) for room in rooms}
    legend = _legend_mapping(g, bg, wall, seed_colors)

    h, w = g.shape
    door_cells: set[tuple[int, int]] = set()
    paint_plan: dict[tuple[int, int], int] = {}

    for room in rooms:
        door, target_start = _door_and_target(room, legend[room["seed_color"]])
        dr, dc = door
        tr, tc = target_start

        assert 0 <= dr < h and 0 <= dc < w and int(g[dr, dc]) == wall, "door must remove a wall cell"
        assert 0 <= tr < h and 0 <= tc < w, "door must point to an in-bounds neighboring room"

        target_label = int(labels[tr, tc])
        assert target_label in room_by_label, "door must point into another enclosed room"
        target_room = rooms[room_by_label[target_label]]
        assert target_room["index"] != room["index"], "door must lead to a different room"

        door_cells.add(door)
        for cell in target_room["bg_cells"]:
            prev = paint_plan.get(cell)
            if prev is None:
                paint_plan[cell] = int(room["seed_color"])
            else:
                assert prev == int(room["seed_color"]), "different seeds cannot fill the same target background cell"

    out = g.copy()
    for r, c in door_cells:
        out[r, c] = bg
    for (r, c), color in paint_plan.items():
        out[r, c] = color
    return out


def _mode_spec() -> tuple[str, int, int, list[tuple[int, int]]]:
    mode = random.choices(MODE_CHOICES, weights=MODE_WEIGHTS, k=1)[0]
    if mode == "horizontal_pair":
        return mode, 1, 2, [(0, 1), (0, -1)]
    if mode == "vertical_pair":
        return mode, 2, 1, [(1, 0), (-1, 0)]
    clockwise = random.random() < 0.5
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)] if clockwise else [(1, 0), (0, -1), (0, 1), (-1, 0)]
    return mode, 2, 2, directions


def _corner_margins(legend_h: int, corner: str) -> tuple[int, int, int, int]:
    top = random.randint(*SIDE_MARGIN_RANGE)
    bottom = random.randint(*SIDE_MARGIN_RANGE)
    left = random.randint(*SIDE_MARGIN_RANGE)
    right = random.randint(*SIDE_MARGIN_RANGE)

    extra = random.randint(*LEGEND_EXTRA_MARGIN_RANGE)
    if corner in ("top_left", "top_right"):
        top = legend_h + extra
    else:
        bottom = legend_h + extra
    return top, bottom, left, right


def _room_bounds(start: int, sizes: list[int]) -> list[int]:
    bounds = [start]
    pos = start
    for size in sizes:
        pos += size + 1
        bounds.append(pos)
    return bounds


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    _, rows, cols, directions = _mode_spec()
    n_rooms = rows * cols
    seed_colors = random.sample(list(SEED_COLORS), n_rooms)
    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(cols)]

    chamber_h = sum(room_heights) + rows + 1
    chamber_w = sum(room_widths) + cols + 1
    legend_h = 2 * n_rooms - 1
    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    top_m, bottom_m, left_m, right_m = _corner_margins(legend_h, corner)

    height = top_m + chamber_h + bottom_m
    width = left_m + chamber_w + right_m
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    row_bounds = _room_bounds(top_m, room_heights)
    col_bounds = _room_bounds(left_m, room_widths)

    r0, r1 = row_bounds[0], row_bounds[-1]
    c0, c1 = col_bounds[0], col_bounds[-1]
    grid[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
    grid[row_bounds, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, col_bounds] = WALL

    rooms: list[dict] = []
    for rr in range(rows):
        for cc in range(cols):
            idx = rr * cols + cc
            ir0 = row_bounds[rr] + 1
            ir1 = row_bounds[rr + 1] - 1
            ic0 = col_bounds[cc] + 1
            ic1 = col_bounds[cc + 1] - 1
            seed_r = random.randint(ir0, ir1)
            seed_c = random.randint(ic0, ic1)
            seed_color = seed_colors[idx]
            grid[seed_r, seed_c] = seed_color

            bg_cells = [
                (r, c)
                for r in range(ir0, ir1 + 1)
                for c in range(ic0, ic1 + 1)
                if (r, c) != (seed_r, seed_c)
            ]
            rooms.append(
                {
                    "index": idx,
                    "grid_pos": (rr, cc),
                    "bbox": (ir0, ic0, ir1, ic1),
                    "seed_pos": (seed_r, seed_c),
                    "seed_color": seed_color,
                    "direction": directions[idx],
                    "bg_cells": bg_cells,
                }
            )

    legend_order = list(range(n_rooms))
    random.shuffle(legend_order)
    legend_row0 = 0 if corner in ("top_left", "top_right") else height - legend_h
    legend_col0 = 0 if corner in ("top_left", "bottom_left") else width - 2
    for slot, room_idx in enumerate(legend_order):
        room = rooms[room_idx]
        rr = legend_row0 + 2 * slot
        grid[rr, legend_col0] = room["seed_color"]
        grid[rr, legend_col0 + 1] = VEC_TO_DIR_COLOR[room["direction"]]

    expected = grid.copy()
    for room in rooms:
        door, _ = _door_and_target(room, room["direction"])
        expected[door] = BACKGROUND
    for room in rooms:
        dr, dc = room["direction"]
        rr, cc = room["grid_pos"]
        target_idx = (rr + dr) * cols + (cc + dc)
        target = rooms[target_idx]
        for r, c in target["bg_cells"]:
            expected[r, c] = room["seed_color"]

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_scene()
        if built is None:
            continue
        grid, expected = built
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
