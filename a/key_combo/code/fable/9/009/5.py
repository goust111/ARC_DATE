from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _multi_source_bfs(passable: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in sources:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _solid_rectangle_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _identify_colors(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]]]:
    bg = _most_common_color(grid)
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    assert len(colors) == 2, "expected background, wall, and water colors only"

    water_candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for color in colors:
        comps = _components_of_color(grid, color)
        if len(comps) == 1 and _solid_rectangle_component(comps[0]):
            water_candidates.append((color, comps[0]))
    assert len(water_candidates) == 1, "expected exactly one solid rectangular reservoir"

    water, reservoir = water_candidates[0]
    wall = colors[0] if colors[1] == water else colors[1]
    return bg, wall, water, reservoir


def _side_gap_count(side: np.ndarray, bg: int, wall: int) -> int | None:
    vals = set(map(int, side))
    if not vals <= {bg, wall}:
        return None
    gap_count = int(np.count_nonzero(side == bg))
    if gap_count > 1:
        return None
    return gap_count


def _find_rooms(grid: np.ndarray, bg: int, wall: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    rooms: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r0 in range(1, h - 2):
        for c0 in range(1, w - 2):
            if int(grid[r0, c0]) != bg:
                continue
            if int(grid[r0 - 1, c0]) == bg or int(grid[r0, c0 - 1]) == bg:
                continue

            width = 0
            while c0 + width < w - 1 and int(grid[r0, c0 + width]) == bg:
                width += 1
            height = 0
            while r0 + height < h - 1 and int(grid[r0 + height, c0]) == bg:
                height += 1
            if height < 2 or width < 2:
                continue

            r1 = r0 + height - 1
            c1 = c0 + width - 1
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(block == bg):
                continue

            if (
                int(grid[r0 - 1, c0 - 1]) != wall
                or int(grid[r0 - 1, c1 + 1]) != wall
                or int(grid[r1 + 1, c0 - 1]) != wall
                or int(grid[r1 + 1, c1 + 1]) != wall
            ):
                continue

            top = _side_gap_count(grid[r0 - 1, c0 : c1 + 1], bg, wall)
            bottom = _side_gap_count(grid[r1 + 1, c0 : c1 + 1], bg, wall)
            left = _side_gap_count(grid[r0 : r1 + 1, c0 - 1], bg, wall)
            right = _side_gap_count(grid[r0 : r1 + 1, c1 + 1], bg, wall)
            if None in (top, bottom, left, right):
                continue

            door_count = int(top) + int(bottom) + int(left) + int(right)
            if door_count not in (1, 2):
                continue

            box = (r0, c0, r1, c1)
            if box not in seen:
                seen.add(box)
                rooms.append(box)

    assert rooms, "expected at least one framed room"
    return rooms


def _room_distance(dist: np.ndarray, room: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = room
    vals = dist[r0 : r1 + 1, c0 : c1 + 1]
    reachable = vals[vals >= 0]
    assert reachable.size > 0, "every room must be reachable from the reservoir"
    return int(reachable.min())


def _ordered_rooms(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    bg, wall, water, reservoir = _identify_colors(grid)
    rooms = _find_rooms(grid, bg, wall)
    dist = _multi_source_bfs(grid != wall, reservoir)

    ordered = sorted(rooms, key=lambda box: (_room_distance(dist, box), box))
    dists = [_room_distance(dist, box) for box in ordered]
    assert len(set(dists)) == len(dists), "room arrival order must be unique"
    return bg, wall, water, reservoir, ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Distribute reservoir water along the corridor chain and stop with a bottom-up partial fill.

    Input structure:
    - The grid has one dominant background, one wall color, and one water color.
    - A single solid rectangular reservoir in the water color touches one end of a wall-confined corridor chain.
    - Along that chain sit several axis-aligned framed rooms whose interiors are background and whose walls may contain one-cell door notches.

    Rule:
    - The total water budget equals the number of reservoir pixels.
    - Traverse rooms in the order they are first reached from the reservoir through non-wall cells.
    - Fill each room interior completely in that order while budget remains.
    - When the next room cannot be filled completely, fill only whole bottom rows of that room using the remaining budget divided by the room's interior width, then stop.
    - Leave the reservoir, corridor cells, and door notches unchanged.

    Key insight:
    - Reachability determines only the room sequence; the visible fill depth is controlled by a decreasing area budget measured from the reservoir's pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, _wall, water, reservoir, rooms = _ordered_rooms(g)
    remaining = len(reservoir)

    out = g.copy()
    for r0, c0, r1, c1 in rooms:
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = height * width
        if remaining >= area:
            out[r0 : r1 + 1, c0 : c1 + 1] = water
            remaining -= area
            continue

        rows = remaining // width
        assert rows * width == remaining, "partial fill must stop on a whole row"
        assert 0 < rows < height, "generator guarantees a non-degenerate partial room"
        out[r1 - rows + 1 : r1 + 1, c0 : c1 + 1] = water
        remaining = 0
        break

    return out


ROOM_COUNT_RANGE = (3, 4)
ROOM_SIDE_RANGE = (2, 4)
PRIMARY_GAP_RANGE = (3, 5)
OFFSET_CHOICES = (-3, -2, 0, 2, 3)
RESERVOIR_GAP_RANGE = (2, 4)
GRID_LIMIT = 30
GEN_RETRIES = 400


def _make_room(r0: int, c0: int, inner_h: int, inner_w: int) -> dict:
    return {
        "outer": (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1),
        "inner": (r0 + 1, c0 + 1, r0 + inner_h, c0 + inner_w),
        "h": inner_h,
        "w": inner_w,
    }


def _door(room: dict, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = room["outer"]
    if side == "west":
        return r0 + 1 + room["h"] // 2, c0
    if side == "east":
        return r0 + 1 + room["h"] // 2, c1
    if side == "north":
        return r0, c0 + 1 + room["w"] // 2
    if side == "south":
        return r1, c0 + 1 + room["w"] // 2
    raise ValueError(side)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        step = 1 if b[1] >= a[1] else -1
        return [(r, c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        c = a[1]
        step = 1 if b[0] >= a[0] else -1
        return [(r, c) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment endpoints must align")


def _connect_rooms(a: dict, b: dict, orientation: str) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        start = _door(a, "east")
        end = _door(b, "west")
        if start[0] == end[0]:
            return _segment(start, end)
        bend_c = random.randint(start[1] + 1, end[1] - 1)
        cells = _segment(start, (start[0], bend_c))
        cells += _segment((start[0] + (1 if end[0] > start[0] else -1), bend_c), (end[0], bend_c))
        cells += _segment((end[0], bend_c), end)
        return list(dict.fromkeys(cells))

    start = _door(a, "south")
    end = _door(b, "north")
    if start[1] == end[1]:
        return _segment(start, end)
    bend_r = random.randint(start[0] + 1, end[0] - 1)
    cells = _segment(start, (bend_r, start[1]))
    cells += _segment((bend_r, start[1] + (1 if end[1] > start[1] else -1)), (bend_r, end[1]))
    cells += _segment((bend_r, end[1]), end)
    return list(dict.fromkeys(cells))


def _connect_reservoir(res_bbox: tuple[int, int, int, int], room: dict, orientation: str) -> list[tuple[int, int]]:
    rr0, cc0, rr1, cc1 = res_bbox
    if orientation == "horizontal":
        row = _door(room, "west")[0]
        return _segment((row, cc1 + 1), _door(room, "west"))
    col = _door(room, "north")[1]
    return _segment((rr1 + 1, col), _door(room, "north"))


def _capacity_factor_pairs(area: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(2, 9):
        if area % h == 0:
            w = area // h
            if 2 <= w <= 8:
                dims.append((h, w))
                if h != w:
                    dims.append((w, h))
    return dims


def _build_layout() -> dict | None:
    orientation = random.choice(["horizontal", "vertical"])
    room_count = random.randint(*ROOM_COUNT_RANGE)
    room_sizes = [
        (random.randint(*ROOM_SIDE_RANGE), random.randint(*ROOM_SIDE_RANGE))
        for _ in range(room_count)
    ]

    rooms: list[dict] = [_make_room(0, 0, room_sizes[0][0], room_sizes[0][1])]
    corridor_cells: set[tuple[int, int]] = set()

    for idx in range(1, room_count):
        prev = rooms[-1]
        inner_h, inner_w = room_sizes[idx]
        gap = random.randint(*PRIMARY_GAP_RANGE)
        offset = random.choice(OFFSET_CHOICES)

        if orientation == "horizontal":
            prev_row = _door(prev, "east")[0]
            new_r0 = prev_row + offset - 1 - inner_h // 2
            new_c0 = prev["outer"][3] + gap
        else:
            prev_col = _door(prev, "south")[1]
            new_r0 = prev["outer"][2] + gap
            new_c0 = prev_col + offset - 1 - inner_w // 2

        room = _make_room(new_r0, new_c0, inner_h, inner_w)
        rooms.append(room)
        corridor_cells.update(_connect_rooms(prev, room, orientation))

    areas = [room["h"] * room["w"] for room in rooms]
    prefix_areas = [0]
    running = 0
    for area in areas:
        running += area
        prefix_areas.append(running)
    reservoir_options = []
    for partial_idx in range(1, room_count - 1):
        room = rooms[partial_idx]
        for rows in range(1, room["h"]):
            capacity = prefix_areas[partial_idx] + rows * room["w"]
            for res_h, res_w in _capacity_factor_pairs(capacity):
                reservoir_options.append((partial_idx, rows, capacity, res_h, res_w))
    if not reservoir_options:
        return None

    partial_idx, rows, capacity, res_h, res_w = random.choice(reservoir_options)
    first = rooms[0]
    res_gap = random.randint(*RESERVOIR_GAP_RANGE)
    if orientation == "horizontal":
        row = _door(first, "west")[0]
        res_r0 = row - res_h // 2
        res_c1 = first["outer"][1] - res_gap
        res_bbox = (res_r0, res_c1 - res_w + 1, res_r0 + res_h - 1, res_c1)
    else:
        col = _door(first, "north")[1]
        res_c0 = col - res_w // 2
        res_r1 = first["outer"][0] - res_gap
        res_bbox = (res_r1 - res_h + 1, res_c0, res_r1, res_c0 + res_w - 1)

    corridor_cells.update(_connect_reservoir(res_bbox, first, orientation))

    carved = set(corridor_cells)
    for room in rooms:
        r0, c0, r1, c1 = room["inner"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                carved.add((r, c))
    rr0, cc0, rr1, cc1 = res_bbox
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            carved.add((r, c))

    min_r = min(r for r, _ in carved)
    min_c = min(c for _, c in carved)
    shift_r = 1 - min_r
    shift_c = 1 - min_c

    def shift_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        r0, c0, r1, c1 = box
        return r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c

    shifted_rooms = []
    for room in rooms:
        shifted_rooms.append(
            {
                "outer": shift_box(room["outer"]),
                "inner": shift_box(room["inner"]),
                "h": room["h"],
                "w": room["w"],
            }
        )
    shifted_res = shift_box(res_bbox)
    shifted_corridor = {(r + shift_r, c + shift_c) for r, c in corridor_cells}

    carved_shifted = set(shifted_corridor)
    rr0, cc0, rr1, cc1 = shifted_res
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            carved_shifted.add((r, c))
    for room in shifted_rooms:
        r0, c0, r1, c1 = room["inner"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                carved_shifted.add((r, c))

    max_r = max(r for r, _ in carved_shifted)
    max_c = max(c for _, c in carved_shifted)
    shell_h = max_r + 2
    shell_w = max_c + 2
    if shell_h > 24 or shell_w > 24:
        return None

    room_bg_area = sum(room["h"] * room["w"] for room in shifted_rooms)
    wall_count = shell_h * shell_w - len(shifted_corridor) - room_bg_area - capacity
    min_total_area = wall_count + capacity + max(wall_count, capacity) + 1
    if GRID_LIMIT * GRID_LIMIT < min_total_area:
        return None

    dim_options = []
    for grid_h in range(shell_h + 2, GRID_LIMIT + 1):
        for grid_w in range(shell_w + 2, GRID_LIMIT + 1):
            if grid_h * grid_w >= min_total_area:
                dim_options.append((grid_h, grid_w))
    if not dim_options:
        return None

    grid_h, grid_w = random.choice(dim_options)
    spare_h = grid_h - shell_h
    spare_w = grid_w - shell_w
    margin_top = random.randint(1, spare_h - 1)
    margin_left = random.randint(1, spare_w - 1)
    margin_bottom = spare_h - margin_top
    margin_right = spare_w - margin_left

    return {
        "orientation": orientation,
        "rooms": shifted_rooms,
        "reservoir": shifted_res,
        "corridor": shifted_corridor,
        "capacity": capacity,
        "partial_idx": partial_idx,
        "partial_rows": rows,
        "grid_shape": (grid_h, grid_w),
        "shell_offset": (margin_top, margin_left),
        "shell_shape": (shell_h, shell_w),
    }


def _render_instance(layout: dict) -> tuple[np.ndarray, np.ndarray]:
    bg = 0
    wall = random.choice([5, 6, 7, 8, 9])
    water = random.choice([color for color in range(1, 10) if color != wall])

    grid_h, grid_w = layout["grid_shape"]
    off_r, off_c = layout["shell_offset"]
    shell_h, shell_w = layout["shell_shape"]
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[off_r : off_r + shell_h, off_c : off_c + shell_w] = wall

    def place_box(box: tuple[int, int, int, int], value: int) -> None:
        r0, c0, r1, c1 = box
        grid[off_r + r0 : off_r + r1 + 1, off_c + c0 : off_c + c1 + 1] = value

    for room in layout["rooms"]:
        place_box(room["inner"], bg)
    for r, c in layout["corridor"]:
        grid[off_r + r, off_c + c] = bg
    place_box(layout["reservoir"], water)

    out = grid.copy()
    for idx, room in enumerate(layout["rooms"]):
        r0, c0, r1, c1 = room["inner"]
        gr0 = off_r + r0
        gc0 = off_c + c0
        gr1 = off_r + r1
        gc1 = off_c + c1
        if idx < layout["partial_idx"]:
            out[gr0 : gr1 + 1, gc0 : gc1 + 1] = water
        elif idx == layout["partial_idx"]:
            rows = layout["partial_rows"]
            out[gr1 - rows + 1 : gr1 + 1, gc0 : gc1 + 1] = water
            break

    return grid, out


def _validate_instance(grid: np.ndarray, expected: np.ndarray, layout: dict) -> bool:
    if np.array_equal(grid, expected):
        return False
    if grid.ndim != 2 or max(grid.shape) > GRID_LIMIT:
        return False

    rooms = [room["inner"] for room in layout["rooms"]]
    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    bg, wall, water, reservoir, found_rooms = _ordered_rooms(grid)
    if bg != 0 or len(found_rooms) != len(rooms):
        return False
    if set(found_rooms) != set(
        (
            layout["shell_offset"][0] + r0,
            layout["shell_offset"][1] + c0,
            layout["shell_offset"][0] + r1,
            layout["shell_offset"][1] + c1,
        )
        for r0, c0, r1, c1 in rooms
    ):
        return False

    if len(reservoir) != layout["capacity"]:
        return False

    corridor_global = {
        (layout["shell_offset"][0] + r, layout["shell_offset"][1] + c) for r, c in layout["corridor"]
    }
    if np.any(out[tuple(zip(*corridor_global))] == water):
        return False

    for idx, (r0, c0, r1, c1) in enumerate(found_rooms):
        room = out[r0 : r1 + 1, c0 : c1 + 1]
        if idx < layout["partial_idx"]:
            if not np.all(room == water):
                return False
        elif idx == layout["partial_idx"]:
            rows = layout["partial_rows"]
            split = (r1 - r0 + 1) - rows
            if split <= 0:
                return False
            if not np.all(room[split:, :] == water):
                return False
            if not np.all(room[:split, :] == bg):
                return False
        else:
            if not np.all(room == bg):
                return False

    return set(map(int, grid.ravel())) == {bg, wall, water}


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layout = _build_layout()
        if layout is None:
            continue
        grid, expected = _render_instance(layout)
        if _validate_instance(grid, expected, layout):
            return grid
    raise RuntimeError("failed to generate a valid reservoir-corridor instance")
