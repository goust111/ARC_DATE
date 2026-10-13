from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATE_RETRIES = 400
MATCH_RETRIES = 120
DOOR_SAMPLE_RETRIES = 80


@dataclass(frozen=True)
class Room:
    row_idx: int
    col_idx: int
    r0: int
    r1: int
    c0: int
    c1: int
    arrow_color: int
    direction: str
    door_color: Optional[int]


def _wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    wall_rows = [r for r in range(h) if np.all(grid[r, :] != BLACK)]
    wall_cols = [c for c in range(w) if np.all(grid[:, c] != BLACK)]
    assert wall_rows and wall_cols, "expected enclosing wall rows and columns"
    assert wall_rows[0] == 0 and wall_rows[-1] == h - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1
    assert all(b - a >= 2 for a, b in zip(wall_rows, wall_rows[1:])), "rooms must have positive height"
    assert all(b - a >= 2 for a, b in zip(wall_cols, wall_cols[1:])), "rooms must have positive width"
    return wall_rows, wall_cols


def _door_color(segment: np.ndarray) -> Optional[int]:
    vals = {int(v) for v in np.asarray(segment).ravel()}
    assert BLACK not in vals, "wall segments cannot contain black"
    if vals == {GREY}:
        return None
    non_grey = vals - {GREY}
    assert len(non_grey) == 1, "a door edge must use a single non-grey color"
    return next(iter(non_grey))


def _infer_arrow_direction(coords: np.ndarray) -> str:
    pts = [(int(r), int(c)) for r, c in coords]
    row_map: dict[int, list[int]] = {}
    col_map: dict[int, list[int]] = {}
    for r, c in pts:
        row_map.setdefault(r, []).append(c)
        col_map.setdefault(c, []).append(r)

    if len(row_map) == 2:
        row_counts = sorted((len(cols), row) for row, cols in row_map.items())
        if row_counts[0][0] == 1 and 2 <= row_counts[1][0] <= 4 and len(pts) == row_counts[1][0] + 1:
            tip_row = row_counts[0][1]
            shaft_row = row_counts[1][1]
            shaft_cols = sorted(row_map[shaft_row])
            tip_col = row_map[tip_row][0]
            if shaft_cols == list(range(shaft_cols[0], shaft_cols[-1] + 1)) and tip_col in set(shaft_cols):
                return "up" if tip_row < shaft_row else "down"

    if len(col_map) == 2:
        col_counts = sorted((len(rows), col) for col, rows in col_map.items())
        if col_counts[0][0] == 1 and 2 <= col_counts[1][0] <= 4 and len(pts) == col_counts[1][0] + 1:
            tip_col = col_counts[0][1]
            shaft_col = col_counts[1][1]
            shaft_rows = sorted(col_map[shaft_col])
            tip_row = col_map[tip_col][0]
            if shaft_rows == list(range(shaft_rows[0], shaft_rows[-1] + 1)) and tip_row in set(shaft_rows):
                return "left" if tip_col < shaft_col else "right"

    raise AssertionError("unrecognized arrow shape")


def _extract_rooms(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], Room]:
    rooms: dict[tuple[int, int], Room] = {}
    for i, (r0, r1) in enumerate(zip(wall_rows, wall_rows[1:])):
        for j, (c0, c1) in enumerate(zip(wall_cols, wall_cols[1:])):
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            arrow_coords = np.argwhere(inner != BLACK)
            assert arrow_coords.size > 0, "each room must contain an arrow"
            arrow_colors = {int(inner[r, c]) for r, c in arrow_coords}
            assert len(arrow_colors) == 1, "each room must contain exactly one monochrome arrow"
            arrow_color = next(iter(arrow_colors))
            direction = _infer_arrow_direction(arrow_coords)

            candidates = []
            segments = (
                grid[r0, c0 + 1 : c1],
                grid[r1, c0 + 1 : c1],
                grid[r0 + 1 : r1, c0],
                grid[r0 + 1 : r1, c1],
            )
            for segment in segments:
                color = _door_color(segment)
                if color is not None:
                    candidates.append(color)
            assert len(candidates) <= 1, "each room may have at most one door-colored boundary"
            door_color = candidates[0] if candidates else None

            rooms[(i, j)] = Room(
                row_idx=i,
                col_idx=j,
                r0=r0,
                r1=r1,
                c0=c0,
                c1=c1,
                arrow_color=arrow_color,
                direction=direction,
                door_color=door_color,
            )
    return rooms


def _neighbor(room: Room, room_rows: int, room_cols: int) -> Optional[tuple[int, int]]:
    i, j = room.row_idx, room.col_idx
    if room.direction == "up":
        return (i - 1, j) if i > 0 else None
    if room.direction == "down":
        return (i + 1, j) if i + 1 < room_rows else None
    if room.direction == "left":
        return (i, j - 1) if j > 0 else None
    if room.direction == "right":
        return (i, j + 1) if j + 1 < room_cols else None
    raise AssertionError("bad direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each room's arrow into its neighboring room, then refill the source room from its door-colored edge.

    Input structure:
    - Grey one-pixel divider rows and columns split the canvas into rectangular rooms.
    - Every room has a black interior and contains exactly one monochrome arrow-like component.
    - An arrow is a 2-4 pixel straight shaft plus one tip pixel attached on one side, so its tip side
      determines one of the four cardinal directions.
    - Some room boundaries are recolored away from grey as a full door-colored edge; each room has at most one.

    Rule:
    - Infer every room and each arrow's facing direction from its tip position.
    - If the pointed side has an adjacent room across the divider, repaint that target room's black interior
      pixels with the arrow color, leaving the target arrow itself unchanged.
    - For every source room that has a door-colored boundary, repaint its full interior, including the arrow,
      with that door color.
    - Keep all divider pixels unchanged.

    Key insight:
    - The room dividers encode a hidden rectangular adjacency graph, the arrow shape encodes movement direction,
      and the non-grey boundary segment provides a second independent label that determines the source-room refill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_rows, wall_cols = _wall_lines(g)
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    rooms = _extract_rooms(g, wall_rows, wall_cols)

    incoming: dict[tuple[int, int], int] = {}
    for room in rooms.values():
        target = _neighbor(room, room_rows, room_cols)
        if target is None:
            continue
        prev = incoming.get(target)
        if prev is None:
            incoming[target] = room.arrow_color
        else:
            assert prev == room.arrow_color, "multiple different incoming colors are undefined"

    out = g.copy()
    for target, color in incoming.items():
        room = rooms[target]
        src_inner = g[room.r0 + 1 : room.r1, room.c0 + 1 : room.c1]
        dst_inner = out[room.r0 + 1 : room.r1, room.c0 + 1 : room.c1]
        dst_inner[src_inner == BLACK] = color

    for room in rooms.values():
        if room.door_color is None:
            continue
        out[room.r0 + 1 : room.r1, room.c0 + 1 : room.c1] = room.door_color

    return out


def _room_sizes(rng: random.Random) -> tuple[list[int], list[int]]:
    room_rows = rng.randint(2, 4)
    room_cols = rng.randint(2, 4)
    heights = [rng.randint(4, 6) for _ in range(room_rows)]
    widths = [rng.randint(4, 6) for _ in range(room_cols)]
    return heights, widths


def _wall_positions(sizes: list[int]) -> list[int]:
    pos = 0
    walls = [0]
    for size in sizes:
        pos += size + 1
        walls.append(pos)
    return walls


def _all_rooms(room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(room_rows) for j in range(room_cols)]


def _border_rooms(room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    return [(i, j) for i, j in _all_rooms(room_rows, room_cols) if i in (0, room_rows - 1) or j in (0, room_cols - 1)]


def _interior_rooms(room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    return [(i, j) for i, j in _all_rooms(room_rows, room_cols) if i not in (0, room_rows - 1) and j not in (0, room_cols - 1)]


def _adjacent_options(room: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[tuple[int, int], str]]:
    i, j = room
    options = []
    if i > 0:
        options.append(((i - 1, j), "up"))
    if i + 1 < room_rows:
        options.append(((i + 1, j), "down"))
    if j > 0:
        options.append(((i, j - 1), "left"))
    if j + 1 < room_cols:
        options.append(((i, j + 1), "right"))
    return options


def _outward_options(room: tuple[int, int], room_rows: int, room_cols: int) -> list[str]:
    i, j = room
    options = []
    if i == 0:
        options.append("up")
    if i == room_rows - 1:
        options.append("down")
    if j == 0:
        options.append("left")
    if j == room_cols - 1:
        options.append("right")
    return options


def _match_valid_rooms(
    valid_rooms: set[tuple[int, int]], room_rows: int, room_cols: int, rng: random.Random
) -> Optional[dict[tuple[int, int], str]]:
    edges: dict[tuple[int, int], list[tuple[tuple[int, int], str]]] = {}
    for room in valid_rooms:
        options = [(nbr, direction) for nbr, direction in _adjacent_options(room, room_rows, room_cols) if nbr in valid_rooms]
        rng.shuffle(options)
        if not options:
            return None
        edges[room] = options

    sources = list(valid_rooms)
    rng.shuffle(sources)
    target_to_source: dict[tuple[int, int], tuple[int, int]] = {}
    direction_for_source: dict[tuple[int, int], str] = {}

    def dfs(source: tuple[int, int], seen: set[tuple[int, int]]) -> bool:
        for target, direction in edges[source]:
            if target in seen:
                continue
            seen.add(target)
            prev = target_to_source.get(target)
            if prev is None or dfs(prev, seen):
                target_to_source[target] = source
                direction_for_source[source] = direction
                return True
        return False

    for source in sources:
        if not dfs(source, set()):
            return None

    return direction_for_source


def _choose_direction_map(room_rows: int, room_cols: int, rng: random.Random) -> Optional[dict[tuple[int, int], str]]:
    interior = _interior_rooms(room_rows, room_cols)
    border = _border_rooms(room_rows, room_cols)
    total = room_rows * room_cols
    outward_count = total // 2
    valid_count = total - outward_count
    extra_border = valid_count - len(interior)
    if extra_border < 0 or extra_border > len(border):
        return None

    border_pool = list(border)
    for _ in range(MATCH_RETRIES):
        chosen_border = set(rng.sample(border_pool, extra_border))
        valid_rooms = set(interior) | chosen_border
        direction_map = _match_valid_rooms(valid_rooms, room_rows, room_cols, rng)
        if direction_map is None:
            continue

        outward_rooms = [room for room in border if room not in valid_rooms]
        if len(outward_rooms) != outward_count:
            continue
        for room in outward_rooms:
            direction_map[room] = rng.choice(_outward_options(room, room_rows, room_cols))
        return direction_map

    return None


def _choose_doors(
    border_rooms: list[tuple[int, int]],
    valid_rooms: set[tuple[int, int]],
    outward_rooms: set[tuple[int, int]],
    rng: random.Random,
) -> Optional[set[tuple[int, int]]]:
    door_count = len(valid_rooms | outward_rooms) // 2
    border_set = set(border_rooms)
    valid_border = valid_rooms & border_set
    for _ in range(DOOR_SAMPLE_RETRIES):
        doors = set(rng.sample(border_rooms, door_count))
        if not doors & outward_rooms:
            continue
        if not (outward_rooms - doors):
            continue
        if valid_border and not (doors & valid_border):
            continue
        if not ((valid_rooms | outward_rooms) - doors):
            continue
        return doors
    return None


def _door_side(room: tuple[int, int], room_rows: int, room_cols: int, rng: random.Random) -> str:
    return rng.choice(_outward_options(room, room_rows, room_cols))


def _paint_door(grid: np.ndarray, room: tuple[int, int], side: str, color: int, row_walls: list[int], col_walls: list[int]) -> None:
    i, j = room
    r0, r1 = row_walls[i], row_walls[i + 1]
    c0, c1 = col_walls[j], col_walls[j + 1]
    if side == "up":
        grid[r0, c0 + 1 : c1] = color
    elif side == "down":
        grid[r1, c0 + 1 : c1] = color
    elif side == "left":
        grid[r0 + 1 : r1, c0] = color
    elif side == "right":
        grid[r0 + 1 : r1, c1] = color
    else:
        raise AssertionError("bad door side")


def _arrow_cells(height: int, width: int, direction: str, rng: random.Random) -> list[tuple[int, int]]:
    if direction in ("up", "down"):
        shaft_len = rng.randint(2, min(4, width))
        start_col = rng.randint(0, width - shaft_len)
        shaft_cols = list(range(start_col, start_col + shaft_len))
        shaft_row = rng.randint(1, height - 1) if direction == "up" else rng.randint(0, height - 2)
        tip_row = shaft_row - 1 if direction == "up" else shaft_row + 1
        tip_col = rng.choice(shaft_cols)
        return [(shaft_row, c) for c in shaft_cols] + [(tip_row, tip_col)]

    shaft_len = rng.randint(2, min(4, height))
    start_row = rng.randint(0, height - shaft_len)
    shaft_rows = list(range(start_row, start_row + shaft_len))
    shaft_col = rng.randint(1, width - 1) if direction == "left" else rng.randint(0, width - 2)
    tip_col = shaft_col - 1 if direction == "left" else shaft_col + 1
    tip_row = rng.choice(shaft_rows)
    return [(r, shaft_col) for r in shaft_rows] + [(tip_row, tip_col)]


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    heights, widths = _room_sizes(rng)
    row_walls = _wall_positions(heights)
    col_walls = _wall_positions(widths)
    h = row_walls[-1] + 1
    w = col_walls[-1] + 1
    if h > 30 or w > 30:
        return None

    room_rows = len(heights)
    room_cols = len(widths)
    direction_map = _choose_direction_map(room_rows, room_cols, rng)
    if direction_map is None:
        return None

    border_rooms = _border_rooms(room_rows, room_cols)
    interior_rooms = set(_interior_rooms(room_rows, room_cols))
    valid_rooms = {room for room in _all_rooms(room_rows, room_cols) if room not in border_rooms or room in direction_map and direction_map[room] not in _outward_options(room, room_rows, room_cols)}
    outward_rooms = {room for room in border_rooms if direction_map[room] in _outward_options(room, room_rows, room_cols)}
    if len(valid_rooms) + len(outward_rooms) != room_rows * room_cols:
        valid_rooms = set(_all_rooms(room_rows, room_cols)) - outward_rooms
    if not interior_rooms <= valid_rooms:
        return None

    door_rooms = _choose_doors(border_rooms, valid_rooms, outward_rooms, rng)
    if door_rooms is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[row_walls, :] = GREY
    grid[:, col_walls] = GREY

    door_side_by_room: dict[tuple[int, int], str] = {}
    for room in door_rooms:
        side = _door_side(room, room_rows, room_cols, rng)
        door_side_by_room[room] = side
        color = rng.choice(PALETTE)
        _paint_door(grid, room, side, color, row_walls, col_walls)

    for i in range(room_rows):
        for j in range(room_cols):
            height = heights[i]
            width = widths[j]
            direction = direction_map[(i, j)]
            color = rng.choice(PALETTE)
            if (i, j) in door_rooms and color == _door_color(
                grid[row_walls[i], col_walls[j] + 1 : col_walls[j + 1]]
                if door_side_by_room[(i, j)] == "up"
                else grid[row_walls[i + 1], col_walls[j] + 1 : col_walls[j + 1]]
                if door_side_by_room[(i, j)] == "down"
                else grid[row_walls[i] + 1 : row_walls[i + 1], col_walls[j]]
                if door_side_by_room[(i, j)] == "left"
                else grid[row_walls[i] + 1 : row_walls[i + 1], col_walls[j + 1]]
            ):
                color = rng.choice([c for c in PALETTE if c != color])
            for rr, cc in _arrow_cells(height, width, direction, rng):
                grid[row_walls[i] + 1 + rr, col_walls[j] + 1 + cc] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
