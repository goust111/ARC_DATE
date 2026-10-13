from __future__ import annotations

import random
from collections import deque

import numpy as np


WALL = 9
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components_bool(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mono_components(g: np.ndarray, bg: int, wall: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in (bg, wall):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": tuple(cells)})
    return comps


def _door_axis(g: np.ndarray, cells: tuple[tuple[int, int], ...], wall: int) -> str | None:
    h, w = g.shape

    def is_wall(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and int(g[r, c]) == wall

    if len(cells) == 1:
        r, c = cells[0]
        horizontal = is_wall(r, c - 1) and is_wall(r, c + 1)
        vertical = is_wall(r - 1, c) and is_wall(r + 1, c)
        if horizontal ^ vertical:
            return "h" if horizontal else "v"
        return None

    if len(cells) != 2:
        return None
    (r0, c0), (r1, c1) = sorted(cells)
    if r0 == r1 and abs(c0 - c1) == 1:
        return "h" if is_wall(r0, min(c0, c1) - 1) and is_wall(r0, max(c0, c1) + 1) else None
    if c0 == c1 and abs(r0 - r1) == 1:
        return "v" if is_wall(min(r0, r1) - 1, c0) and is_wall(max(r0, r1) + 1, c0) else None
    return None


def _touches(cells: tuple[tuple[int, int], ...] | set[tuple[int, int]], region: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in region:
                return True
    return False


def _direction_from_object_to_door(
    obj_cells: tuple[tuple[int, int], ...],
    door_cells: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    obj_rs = [r for r, _ in obj_cells]
    obj_cs = [c for _, c in obj_cells]
    door_rs = [r for r, _ in door_cells]
    door_cs = [c for _, c in door_cells]
    obj_r = (min(obj_rs) + max(obj_rs)) / 2.0
    obj_c = (min(obj_cs) + max(obj_cs)) / 2.0
    door_r = sum(door_rs) / len(door_rs)
    door_c = sum(door_cs) / len(door_cs)
    if abs(door_r - obj_r) >= abs(door_c - obj_c):
        return (-1, 0) if door_r < obj_r else (1, 0)
    return (0, -1) if door_c < obj_c else (0, 1)


def _slide_distance(
    cells: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
    barrier: np.ndarray,
) -> int:
    h, w = barrier.shape
    dr, dc = direction
    steps = 0
    while True:
        nxt = steps + 1
        blocked = False
        for r, c in cells:
            nr, nc = r + dr * nxt, c + dc * nxt
            if not (0 <= nr < h and 0 <= nc < w) or bool(barrier[nr, nc]):
                blocked = True
                break
        if blocked:
            return steps
        steps = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open key-colored wall doors and slide room objects through the resulting gaps.

    Input structure:
    - bg = most common color; maroon (9) pixels form one-pixel-thick room walls.
    - Each closed room contains one monochrome object and optionally one single-pixel key whose color differs from the object.
    - Some maroon wall pixels are replaced by one- or two-pixel colored door segments; a door has the same color as its matching key.

    Rule:
    - For every room containing a key, find the wall door segment with the key's color.
    - Recolor that door segment and the key pixel to bg.
    - Move the room's object rigidly toward the opened gap, continuing in that cardinal direction until the next step would hit a maroon wall, a still-closed door, or the grid boundary.
    - Rooms with no key keep their object and walls unchanged.

    Key insight: A key color names the matching wall gap, and the gap's position relative to the object gives the slide direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background(g)
    wall = WALL

    colored = _mono_components(g, bg, wall)
    doors_by_color: dict[int, tuple[tuple[int, int], ...]] = {}
    door_sets: set[frozenset[tuple[int, int]]] = set()
    for comp in colored:
        cells = comp["cells"]
        if _door_axis(g, cells, wall) is not None and comp["color"] not in doors_by_color:
            doors_by_color[int(comp["color"])] = cells
            door_sets.add(frozenset(cells))

    content = [comp for comp in colored if frozenset(comp["cells"]) not in door_sets]
    room_regions = []
    for pts in _components_bool(g == bg):
        region = set(pts)
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in region):
            continue
        room_regions.append(region)

    records = []
    opened_doors: set[tuple[int, int]] = set()
    all_door_cells = {cell for cells in doors_by_color.values() for cell in cells}

    for region in room_regions:
        adjacent = [comp for comp in content if _touches(comp["cells"], region)]
        key_comp = None
        for comp in adjacent:
            color = int(comp["color"])
            if len(comp["cells"]) == 1 and color in doors_by_color and _touches(doors_by_color[color], region):
                key_comp = comp
                break
        if key_comp is None:
            continue

        objects = [comp for comp in adjacent if comp is not key_comp]
        if not objects:
            continue
        obj = max(objects, key=lambda comp: len(comp["cells"]))
        door_cells = doors_by_color[int(key_comp["color"])]
        direction = _direction_from_object_to_door(obj["cells"], door_cells)
        obj_values = tuple(int(g[r, c]) for r, c in obj["cells"])
        records.append(
            {
                "key": key_comp["cells"][0],
                "door": door_cells,
                "object": obj["cells"],
                "values": obj_values,
                "direction": direction,
            }
        )
        opened_doors.update(door_cells)

    barrier = g == wall
    for r, c in all_door_cells - opened_doors:
        barrier[r, c] = True

    out = g.copy()
    moves = []
    for rec in records:
        dist = _slide_distance(rec["object"], rec["direction"], barrier)
        dr, dc = rec["direction"]
        moved = tuple((r + dr * dist, c + dc * dist) for r, c in rec["object"])
        moves.append((rec, moved))

    for rec, _ in moves:
        kr, kc = rec["key"]
        out[kr, kc] = bg
        for r, c in rec["door"]:
            out[r, c] = bg
        for r, c in rec["object"]:
            out[r, c] = bg

    for rec, moved in moves:
        for (r, c), value in zip(moved, rec["values"]):
            out[r, c] = value

    return out


def _room_layout(rng: random.Random):
    nrows, ncols = rng.choice([(2, 2), (2, 3), (3, 2)])
    heights = [rng.randint(5, 8) for _ in range(nrows)]
    widths = [rng.randint(5, 8) for _ in range(ncols)]
    row_walls = [0]
    for size in heights:
        row_walls.append(row_walls[-1] + size + 1)
    col_walls = [0]
    for size in widths:
        col_walls.append(col_walls[-1] + size + 1)
    rooms = {}
    for i in range(nrows):
        for j in range(ncols):
            rooms[(i, j)] = (
                row_walls[i] + 1,
                row_walls[i + 1] - 1,
                col_walls[j] + 1,
                col_walls[j + 1] - 1,
            )
    return nrows, ncols, row_walls[-1] + 1, col_walls[-1] + 1, row_walls, col_walls, rooms


def _draw_walls(grid: np.ndarray, row_walls: list[int], col_walls: list[int]) -> None:
    for r in row_walls:
        grid[r, :] = WALL
    for c in col_walls:
        grid[:, c] = WALL


def _directed_edges(nrows: int, ncols: int):
    for i in range(nrows):
        for j in range(ncols):
            for name, (di, dj) in {
                "up": (-1, 0),
                "down": (1, 0),
                "left": (0, -1),
                "right": (0, 1),
            }.items():
                ni, nj = i + di, j + dj
                if 0 <= ni < nrows and 0 <= nj < ncols:
                    yield (i, j), (ni, nj), name


def _wall_key(src: tuple[int, int], dst: tuple[int, int]):
    return tuple(sorted((src, dst)))


def _door_cells(
    rng: random.Random,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    src: tuple[int, int],
    direction: str,
    length: int,
) -> tuple[tuple[int, int], ...] | None:
    r0, r1, c0, c1 = rooms[src]
    if direction in ("left", "right"):
        wall_c = c0 - 1 if direction == "left" else c1 + 1
        low, high = r0 + 1, r1 - length
        if high < low:
            low, high = r0, r1 - length + 1
        if high < low:
            return None
        start = rng.randint(low, high)
        return tuple((start + k, wall_c) for k in range(length))
    wall_r = r0 - 1 if direction == "up" else r1 + 1
    low, high = c0 + 1, c1 - length
    if high < low:
        low, high = c0, c1 - length + 1
    if high < low:
        return None
    start = rng.randint(low, high)
    return tuple((wall_r, start + k) for k in range(length))


def _moving_shape(rng: random.Random, direction: str, door_len: int) -> tuple[tuple[int, int], ...]:
    perp = door_len if door_len == 2 and rng.random() < 0.7 else 1
    axis = rng.randint(2, 3)
    if direction in ("left", "right"):
        height, width = perp, axis
    else:
        height, width = axis, perp
    cells = [(r, c) for r in range(height) for c in range(width)]
    if height > 1 and width > 1 and rng.random() < 0.35:
        drop = rng.choice([(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)])
        cells = [p for p in cells if p != drop]
    return tuple(cells)


def _place_moving_object(
    rng: random.Random,
    room: tuple[int, int, int, int],
    door: tuple[tuple[int, int], ...],
    direction: str,
    shape: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    r0, r1, c0, c1 = room
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    height, width = max_r + 1, max_c + 1
    door_rows = sorted(r for r, _ in door)
    door_cols = sorted(c for _, c in door)

    if direction in ("left", "right"):
        if height > len(door_rows):
            return None
        if height == len(door_rows):
            top = door_rows[0]
        else:
            top = rng.choice(door_rows)
        if direction == "right":
            high = door[0][1] - width
            low = c0
        else:
            low = door[0][1] + 1
            high = c1 - width + 1
        if not (r0 <= top and top + height - 1 <= r1 and low <= high):
            return None
        left = rng.randint(low, high)
    else:
        if width > len(door_cols):
            return None
        if width == len(door_cols):
            left = door_cols[0]
        else:
            left = rng.choice(door_cols)
        if direction == "down":
            high = door[0][0] - height
            low = r0
        else:
            low = door[0][0] + 1
            high = r1 - height + 1
        if not (c0 <= left and left + width - 1 <= c1 and low <= high):
            return None
        top = rng.randint(low, high)

    return tuple((top + r, left + c) for r, c in shape)


def _static_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    choices = [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
    ]
    return tuple(rng.choice(choices))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _can_use_cells(cells: tuple[tuple[int, int], ...], grid: np.ndarray, blocked: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != _background(grid) or (r, c) in blocked:
            return False
    return True


def _place_static_object(
    rng: random.Random,
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    color: int,
    blocked: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    bg = _background(grid)
    r0, r1, c0, c1 = room
    for _ in range(80):
        shape = _static_shape(rng)
        height, width = _shape_dims(shape)
        rr0, rr1 = (r0 + 1, r1 - height) if r1 - r0 + 1 > height + 1 else (r0, r1 - height + 1)
        cc0, cc1 = (c0 + 1, c1 - width) if c1 - c0 + 1 > width + 1 else (c0, c1 - width + 1)
        if rr1 < rr0 or cc1 < cc0:
            continue
        top, left = rng.randint(rr0, rr1), rng.randint(cc0, cc1)
        cells = tuple((top + r, left + c) for r, c in shape)
        if all(int(grid[r, c]) == bg and (r, c) not in blocked for r, c in cells):
            for r, c in cells:
                grid[r, c] = color
            return cells
    return None


def _translated_cells(
    cells: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
    barrier: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    dist = _slide_distance(cells, direction, barrier)
    dr, dc = direction
    return tuple((r + dr * dist, c + dc * dist) for r, c in cells)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    nrows, ncols, h, w, row_walls, col_walls, rooms = _room_layout(rng)
    if max(h, w) > 30:
        return None
    bg = rng.choice([c for c in range(9)])
    grid = np.full((h, w), bg, dtype=int)
    _draw_walls(grid, row_walls, col_walls)

    room_ids = list(rooms)
    room_count = len(room_ids)
    max_keys = min(3, 8 - room_count)
    if max_keys < 1:
        return None
    key_count = rng.randint(1, max_keys)

    palette = [c for c in range(10) if c not in (bg, WALL)]
    rng.shuffle(palette)
    if len(palette) < room_count + key_count:
        return None
    object_colors = {room_id: palette.pop() for room_id in room_ids}
    key_colors = [palette.pop() for _ in range(key_count)]

    edge_options = list(_directed_edges(nrows, ncols))
    rng.shuffle(edge_options)
    selected = []
    used_walls = set()
    used_sources = set()
    for src, dst, direction in edge_options:
        wk = _wall_key(src, dst)
        if wk in used_walls or src in used_sources:
            continue
        selected.append((src, dst, direction))
        used_walls.add(wk)
        used_sources.add(src)
        if len(selected) == key_count:
            break
    if len(selected) != key_count:
        return None

    plans = []
    opened_cells: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    for idx, (src, _dst, direction) in enumerate(selected):
        door_len = rng.choice([1, 2])
        door = _door_cells(rng, rooms, src, direction, door_len)
        if door is None or any(cell in occupied for cell in door):
            return None
        for r, c in door:
            grid[r, c] = key_colors[idx]
        shape = _moving_shape(rng, direction, door_len)
        obj = _place_moving_object(rng, rooms[src], door, direction, shape)
        if obj is None or any(cell in occupied for cell in obj):
            return None
        for r, c in obj:
            grid[r, c] = object_colors[src]
        occupied.update(door)
        occupied.update(obj)
        opened_cells.update(door)
        plans.append(
            {
                "room": src,
                "direction": DIRS[direction],
                "door": door,
                "object": obj,
                "object_color": object_colors[src],
                "key_color": key_colors[idx],
            }
        )

    barrier = grid == WALL
    for r, c in opened_cells:
        barrier[r, c] = False

    reserved: set[tuple[int, int]] = set()
    for plan in plans:
        cells = plan["object"]
        dr, dc = plan["direction"]
        dist = _slide_distance(cells, (dr, dc), barrier)
        if dist == 0:
            return None
        for step in range(dist + 1):
            for r, c in cells:
                reserved.add((r + dr * step, c + dc * step))

    placed_objects = {plan["room"]: plan["object"] for plan in plans}
    for room_id in room_ids:
        if room_id in placed_objects:
            continue
        cells = _place_static_object(rng, grid, rooms[room_id], object_colors[room_id], reserved | occupied)
        if cells is None:
            return None
        placed_objects[room_id] = cells
        occupied.update(cells)

    bg_count = int(np.sum(grid == bg))
    wall_count = int(np.sum(grid == WALL))
    if bg_count <= wall_count:
        return None

    for plan in plans:
        r0, r1, c0, c1 = rooms[plan["room"]]
        candidates = []
        obj_near = set()
        for r, c in plan["object"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    obj_near.add((r + dr, c + dc))
        door_near = set()
        for r, c in plan["door"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    door_near.add((r + dr, c + dc))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (
                    int(grid[r, c]) == bg
                    and (r, c) not in reserved
                    and (r, c) not in obj_near
                    and (r, c) not in door_near
                ):
                    candidates.append((r, c))
        if not candidates:
            return None
        kr, kc = rng.choice(candidates)
        grid[kr, kc] = plan["key_color"]
        occupied.add((kr, kc))

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30 or np.array_equal(out, grid):
        return None
    if _background(grid) != bg:
        return None
    for plan in plans:
        color = plan["object_color"]
        out_cells = set((int(r), int(c)) for r, c in zip(*np.where(out == color)))
        if len(out_cells) != len(plan["object"]) or out_cells == set(plan["object"]):
            return None
        if int(np.sum(out == plan["key_color"])) != 0:
            return None
    for room_id, cells in placed_objects.items():
        color = object_colors[room_id]
        if int(np.sum(out == color)) != len(cells):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(600):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid room-and-key grid")
