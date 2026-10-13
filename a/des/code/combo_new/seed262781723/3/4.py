from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

DEPTH_TO_COLOR = {
    1: GREEN,
    2: YELLOW,
    3: ORANGE,
    4: MAROON,
}

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class _Room:
    bbox: tuple[int, int, int, int]
    doors: tuple[tuple[int, int], ...]
    depth: int


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            label = len(components)
            queue = deque([(r, c)])
            labels[r, c] = label
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            components.append(coords)

    return labels, components


def _detect_doors(grid: np.ndarray) -> list[tuple[int, int, str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    doors: list[tuple[int, int, str]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue

            vertical_wall_gap = (
                int(g[r - 1, c]) == BLUE
                and int(g[r + 1, c]) == BLUE
                and int(g[r, c - 1]) != BLUE
                and int(g[r, c + 1]) != BLUE
            )
            horizontal_wall_gap = (
                int(g[r, c - 1]) == BLUE
                and int(g[r, c + 1]) == BLUE
                and int(g[r - 1, c]) != BLUE
                and int(g[r + 1, c]) != BLUE
            )

            if vertical_wall_gap ^ horizontal_wall_gap:
                axis = "vertical" if vertical_wall_gap else "horizontal"
                doors.append((r, c, axis))

    return doors


def _room_depths(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    red_pixels = np.argwhere(g == RED)
    assert len(red_pixels) == 1, "expected a unique red entrance marker"
    red_r, red_c = map(int, red_pixels[0])

    doors = _detect_doors(g)
    assert doors, "expected at least one wall door"

    closed = g.copy()
    for r, c, _ in doors:
        closed[r, c] = BLUE

    passable = closed != BLUE
    labels, components = _label_components(passable)
    outside_label = int(labels[red_r, red_c])
    assert outside_label >= 0, "entrance must lie in the outside region"

    graph: dict[int, set[int]] = {label: set() for label in range(len(components))}
    for r, c, axis in doors:
        if axis == "vertical":
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
        else:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
        if a >= 0 and b >= 0 and a != b:
            graph[a].add(b)
            graph[b].add(a)

    depths = {outside_label: 0}
    queue = deque([outside_label])
    while queue:
        label = queue.popleft()
        for nxt in graph[label]:
            if nxt not in depths:
                depths[nxt] = depths[label] + 1
                queue.append(nxt)

    return labels, components, depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each room by how many wall-doors it is away from the red entrance.

    Input structure:
    - The background is black.
    - Blue one-pixel walls partition the grid into rectangular rooms, and some wall cells are replaced by single-pixel door gaps.
    - A unique red pixel sits just outside the outer boundary wall next to the entrance door.

    Rule:
    - Detect every door gap by finding a black pixel that interrupts an otherwise straight blue wall.
    - Temporarily close all door gaps to recover the separate room regions, then build a room-adjacency graph from the doors they reconnect.
    - Starting from the outside region containing the red marker, assign depth 1 to the first room, depth 2 to rooms reached through one more door, and so on.
    - Fill room interiors by depth: 1 -> green, 2 -> yellow, 3 -> orange, 4 and deeper -> maroon, while keeping blue walls, black door gaps, and the red entrance unchanged.

    Key insight:
    - The visible rooms are easiest to recover by closing the door gaps first; after that, the puzzle reduces to breadth-first search on the door-defined connectivity graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    labels, components, depths = _room_depths(g)
    out = g.copy()

    for label, coords in enumerate(components):
        depth = depths.get(label)
        if depth is None or depth <= 0:
            continue
        color = DEPTH_TO_COLOR[min(depth, 4)]
        for r, c in coords:
            if int(out[r, c]) == BLACK:
                out[r, c] = color

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE


def _carve_entrance(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    side = random.choice(("top", "bottom", "left", "right"))

    if side == "top":
        door = (r0, random.randint(c0 + 1, c1 - 1))
        red = (r0 - 1, door[1])
    elif side == "bottom":
        door = (r1, random.randint(c0 + 1, c1 - 1))
        red = (r1 + 1, door[1])
    elif side == "left":
        door = (random.randint(r0 + 1, r1 - 1), c0)
        red = (door[0], c0 - 1)
    else:
        door = (random.randint(r0 + 1, r1 - 1), c1)
        red = (door[0], c1 + 1)

    grid[door] = BLACK
    grid[red] = RED
    return door, red


def _split_positions(room: _Room, orientation: str) -> list[int]:
    r0, c0, r1, c1 = room.bbox
    positions: list[int] = []

    if orientation == "vertical":
        for x in range(c0 + 3, c1 - 2):
            sides = set()
            for dr, dc in room.doors:
                if dc == c0:
                    sides.add("left")
                elif dc == c1:
                    sides.add("right")
                elif dr in (r0, r1):
                    if dc == x:
                        sides.add("blocked")
                    elif dc < x:
                        sides.add("left")
                    else:
                        sides.add("right")
                else:
                    sides.add("blocked")
            if len(sides) == 1 and "blocked" not in sides:
                positions.append(x)
    else:
        for y in range(r0 + 3, r1 - 2):
            sides = set()
            for dr, dc in room.doors:
                if dr == r0:
                    sides.add("top")
                elif dr == r1:
                    sides.add("bottom")
                elif dc in (c0, c1):
                    if dr == y:
                        sides.add("blocked")
                    elif dr < y:
                        sides.add("top")
                    else:
                        sides.add("bottom")
                else:
                    sides.add("blocked")
            if len(sides) == 1 and "blocked" not in sides:
                positions.append(y)

    return positions


def _split_room(grid: np.ndarray, room: _Room, orientation: str) -> tuple[_Room, _Room] | None:
    positions = _split_positions(room, orientation)
    if not positions:
        return None

    r0, c0, r1, c1 = room.bbox

    if orientation == "vertical":
        x = random.choice(positions)
        inherited_side = "left" if all(
            (dc == c0) or (dr in (r0, r1) and dc < x) for dr, dc in room.doors
        ) else "right"
        door = (random.randint(r0 + 1, r1 - 1), x)
        for rr in range(r0, r1 + 1):
            grid[rr, x] = BLUE
        grid[door] = BLACK

        if inherited_side == "left":
            inherited = _Room((r0, c0, r1, x), room.doors + (door,), room.depth)
            child = _Room((r0, x, r1, c1), (door,), room.depth + 1)
        else:
            inherited = _Room((r0, x, r1, c1), room.doors + (door,), room.depth)
            child = _Room((r0, c0, r1, x), (door,), room.depth + 1)
    else:
        y = random.choice(positions)
        inherited_side = "top" if all(
            (dr == r0) or (dc in (c0, c1) and dr < y) for dr, dc in room.doors
        ) else "bottom"
        door = (y, random.randint(c0 + 1, c1 - 1))
        for cc in range(c0, c1 + 1):
            grid[y, cc] = BLUE
        grid[door] = BLACK

        if inherited_side == "top":
            inherited = _Room((r0, c0, y, c1), room.doors + (door,), room.depth)
            child = _Room((y, c0, r1, c1), (door,), room.depth + 1)
        else:
            inherited = _Room((y, c0, r1, c1), room.doors + (door,), room.depth)
            child = _Room((r0, c0, y, c1), (door,), room.depth + 1)

    return inherited, child


def _render_expected(inp: np.ndarray, rooms: list[_Room]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        r0, c0, r1, c1 = room.bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = DEPTH_TO_COLOR[min(room.depth, 4)]
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    grid = np.full((h, w), BLACK, dtype=int)

    box_h = random.randint(10, h - 2)
    box_w = random.randint(10, w - 2)
    r0 = random.randint(1, h - box_h - 1)
    c0 = random.randint(1, w - box_w - 1)
    outer = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)

    _draw_frame(grid, outer)
    entry, _red = _carve_entrance(grid, outer)

    rooms = [_Room(outer, (entry,), 1)]
    max_depth = random.choice((2, 3, 4))
    target_splits = random.randint(1, 6)

    for _ in range(target_splits):
        candidates: list[tuple[int, str]] = []
        for idx, room in enumerate(rooms):
            if room.depth >= max_depth:
                continue
            if _split_positions(room, "vertical"):
                candidates.append((idx, "vertical"))
            if _split_positions(room, "horizontal"):
                candidates.append((idx, "horizontal"))
        if not candidates:
            break

        idx, orientation = random.choice(candidates)
        result = _split_room(grid, rooms[idx], orientation)
        if result is None:
            continue
        inherited, child = result
        rooms[idx] = inherited
        rooms.append(child)

    if len(rooms) < 2:
        return None

    expected = _render_expected(grid, rooms)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    used_depths = {room.depth for room in rooms}
    if 1 not in used_depths:
        return None
    if any(depth > 4 for depth in used_depths):
        return None
    if max(used_depths) == 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid room-door instance")
