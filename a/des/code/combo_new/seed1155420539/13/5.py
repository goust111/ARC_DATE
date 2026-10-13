from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

MIN_ROOM_SIDE = 2
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp = [(r, c)]
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _find_doors(grid: np.ndarray) -> list[tuple[int, int, str]]:
    h, w = grid.shape
    doors: list[tuple[int, int, str]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            vertical = (
                int(grid[r - 1, c]) == GREY
                and int(grid[r + 1, c]) == GREY
                and int(grid[r, c - 1]) != GREY
                and int(grid[r, c + 1]) != GREY
            )
            horizontal = (
                int(grid[r, c - 1]) == GREY
                and int(grid[r, c + 1]) == GREY
                and int(grid[r - 1, c]) != GREY
                and int(grid[r + 1, c]) != GREY
            )
            if vertical ^ horizontal:
                doors.append((r, c, "v" if vertical else "h"))
    return doors


def _sealed_rooms(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, list[tuple[int, int, str]]]:
    doors = _find_doors(grid)
    sealed = np.array(grid, dtype=int, copy=True)
    for r, c, _orientation in doors:
        sealed[r, c] = GREY

    room_map = np.full(sealed.shape, -1, dtype=int)
    rooms: list[np.ndarray] = []
    for comp in _components(sealed != GREY):
        if any(r == 0 or c == 0 or r == sealed.shape[0] - 1 or c == sealed.shape[1] - 1 for r, c in comp):
            continue
        coords = np.array(comp, dtype=int)
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "each room must be a solid rectangle"
        assert np.all(sealed[r0 : r1 + 1, c0 : c1 + 1] != GREY), "room interior must be wall-free after sealing doors"
        room_id = len(rooms)
        room_map[coords[:, 0], coords[:, 1]] = room_id
        rooms.append(coords)
    assert rooms, "expected at least one enclosed room"
    return rooms, room_map, doors


def _room_graph(room_map: np.ndarray, doors: list[tuple[int, int, str]]) -> list[set[int]]:
    room_count = int(room_map.max()) + 1
    graph = [set() for _ in range(room_count)]
    for r, c, orientation in doors:
        if orientation == "v":
            a = int(room_map[r, c - 1])
            b = int(room_map[r, c + 1])
        else:
            a = int(room_map[r - 1, c])
            b = int(room_map[r + 1, c])
        if a >= 0 and b >= 0 and a != b:
            graph[a].add(b)
            graph[b].add(a)
    return graph


def _room_depths(graph: list[set[int]], root: int) -> list[int]:
    depth = [-1] * len(graph)
    depth[root] = 0
    q = deque([root])
    while q:
        room = q.popleft()
        for nxt in graph[room]:
            if depth[nxt] == -1:
                depth[nxt] = depth[room] + 1
                q.append(nxt)
    assert all(d >= 0 for d in depth), "door graph must connect all rooms to the red root"
    return depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each walled room by tree depth parity while preserving doors and marker pixels.

    Input structure:
    - The background is black, representing the exterior space outside the room complex.
    - Grey one-pixel-thick walls partition an enclosed area into rectangular black rooms.
    - A single red pixel marks the root room, and some rooms may also contain one green marker pixel.
    - Shared walls can contain a one-pixel black door opening that 4-connects exactly two neighboring rooms.

    Rule:
    - Detect door openings from their local wall-gap pattern, then temporarily seal those openings to recover the full room rectangles.
    - Identify every enclosed room and build the room-adjacency tree from the original door locations, rooted at the room containing the red marker.
    - Fill each room floor blue at even depth and yellow at odd depth, except that any room containing a green marker is filled maroon regardless of depth.
    - Keep the red and green marker pixels unchanged, keep all grey walls unchanged, and leave every door opening black.

    Key insight:
    Treat each single missing wall pixel as a deliberate door, not damage: once those gaps are sealed, the rooms become easy to isolate, and the real task is a BFS depth-parity coloring on the induced room tree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms, room_map, doors = _sealed_rooms(g)
    graph = _room_graph(room_map, doors)

    red_cells = np.argwhere(g == RED)
    assert red_cells.shape[0] == 1, "expected exactly one red root marker"
    root = int(room_map[int(red_cells[0, 0]), int(red_cells[0, 1])])
    assert root >= 0, "red marker must lie inside a room"

    green_rooms = {
        int(room_map[int(r), int(c)])
        for r, c in np.argwhere(g == GREEN)
        if int(room_map[int(r), int(c)]) >= 0
    }
    depths = _room_depths(graph, root)

    out = np.array(g, dtype=int, copy=True)
    for room_id, coords in enumerate(rooms):
        fill = MAROON if room_id in green_rooms else (BLUE if depths[room_id] % 2 == 0 else YELLOW)
        for r, c in coords:
            if int(g[r, c]) not in {RED, GREEN}:
                out[r, c] = fill
    return out


def _draw_outer_frame(grid: np.ndarray, room: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = room
    grid[r0 - 1, c0 - 1 : c1 + 2] = GREY
    grid[r1 + 1, c0 - 1 : c1 + 2] = GREY
    grid[r0 - 1 : r1 + 2, c0 - 1] = GREY
    grid[r0 - 1 : r1 + 2, c1 + 1] = GREY


def _split_candidates(
    grid: np.ndarray, rooms: list[tuple[int, int, int, int]]
) -> list[tuple[int, str, int]]:
    candidates: list[tuple[int, str, int]] = []
    for idx, (r0, c0, r1, c1) in enumerate(rooms):
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width >= 2 * MIN_ROOM_SIDE + 1:
            for wall_c in range(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE + 1):
                if int(grid[r0 - 1, wall_c]) == GREY and int(grid[r1 + 1, wall_c]) == GREY:
                    candidates.append((idx, "v", wall_c))
        if height >= 2 * MIN_ROOM_SIDE + 1:
            for wall_r in range(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE + 1):
                if int(grid[wall_r, c0 - 1]) == GREY and int(grid[wall_r, c1 + 1]) == GREY:
                    candidates.append((idx, "h", wall_r))
    return candidates


def _apply_split(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    doors: list[tuple[int, int, str]],
    room_idx: int,
    orientation: str,
    wall_pos: int,
) -> None:
    r0, c0, r1, c1 = rooms.pop(room_idx)
    if orientation == "v":
        grid[r0 - 1 : r1 + 2, wall_pos] = GREY
        door_r = random.randint(r0, r1)
        grid[door_r, wall_pos] = BLACK
        doors.append((door_r, wall_pos, "v"))
        rooms.append((r0, c0, r1, wall_pos - 1))
        rooms.append((r0, wall_pos + 1, r1, c1))
    else:
        grid[wall_pos, c0 - 1 : c1 + 2] = GREY
        door_c = random.randint(c0, c1)
        grid[wall_pos, door_c] = BLACK
        doors.append((wall_pos, door_c, "h"))
        rooms.append((r0, c0, wall_pos - 1, c1))
        rooms.append((wall_pos + 1, c0, r1, c1))


def _room_map_from_boxes(shape: tuple[int, int], rooms: list[tuple[int, int, int, int]]) -> np.ndarray:
    room_map = np.full(shape, -1, dtype=int)
    for idx, (r0, c0, r1, c1) in enumerate(rooms):
        room_map[r0 : r1 + 1, c0 : c1 + 1] = idx
    return room_map


def _expected_output(
    inp: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    doors: list[tuple[int, int, str]],
    root: int,
    green_rooms: set[int],
) -> np.ndarray:
    room_map = _room_map_from_boxes(inp.shape, rooms)
    graph = _room_graph(room_map, doors)
    assert sum(len(neigh) for neigh in graph) == 2 * (len(rooms) - 1), "generated door graph must be a tree"
    depths = _room_depths(graph, root)

    out = np.array(inp, dtype=int, copy=True)
    for idx, (r0, c0, r1, c1) in enumerate(rooms):
        fill = MAROON if idx in green_rooms else (BLUE if depths[idx] % 2 == 0 else YELLOW)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(inp[r, c]) not in {RED, GREEN}:
                    out[r, c] = fill
    return out


def _room_cells(room: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _mark_room(inp: np.ndarray, room: tuple[int, int, int, int], color: int, used: set[tuple[int, int]]) -> None:
    choices = [cell for cell in _room_cells(room) if cell not in used]
    assert choices, "room must have a free marker cell"
    cell = random.choice(choices)
    used.add(cell)
    inp[cell] = color


def _gen_attempt() -> np.ndarray | None:
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    inner_h = random.randint(8, 22)
    inner_w = random.randint(8, 22)
    grid_h = top + inner_h + bottom + 2
    grid_w = left + inner_w + right + 2
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    initial_room = (top + 1, left + 1, top + inner_h, left + inner_w)
    _draw_outer_frame(grid, initial_room)

    rooms = [initial_room]
    doors: list[tuple[int, int, str]] = []

    max_target = min(9, max(2, 1 + (inner_h // 3) * (inner_w // 3)))
    target_rooms = random.randint(2, max_target)
    while len(rooms) < target_rooms:
        candidates = _split_candidates(grid, rooms)
        if not candidates:
            break
        room_idx, orientation, wall_pos = random.choice(candidates)
        _apply_split(grid, rooms, doors, room_idx, orientation, wall_pos)

    if len(rooms) < 2:
        return None

    inp = np.array(grid, dtype=int, copy=True)
    root = random.randrange(len(rooms))
    used_cells: set[tuple[int, int]] = set()
    _mark_room(inp, rooms[root], RED, used_cells)

    if random.random() < 0.5:
        green_rooms: set[int] = set()
    else:
        max_green = min(len(rooms) - 1, 3)
        if max_green <= 0:
            green_rooms = {0}
        else:
            green_count = random.randint(1, max_green)
            green_rooms = set(random.sample(range(len(rooms)), green_count))
        for room_idx in green_rooms:
            _mark_room(inp, rooms[room_idx], GREEN, used_cells)

    expected = _expected_output(inp, rooms, doors, root, green_rooms)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-tree sample")
