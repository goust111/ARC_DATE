from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 0
END_COLOR = 7
START_COLOR = 9
ROOM_COLOR_POOL = (1, 2, 3, 4, 5, 6, 8)

MIN_ROOMS = 3
MAX_ROOMS = 6
INTERIOR_MIN = 3
INTERIOR_MAX = 6
OUTER_PAD_MIN = 1
OUTER_PAD_MAX = 3
LAYOUT_TRIES = 120
GEN_TRIES = 300


@dataclass(frozen=True)
class Room:
    idx: int
    r0: int
    c0: int
    r1: int
    c1: int
    color: int
    doors: tuple[tuple[int, int], ...]


def _integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r1, c) for c in range(c0, c1 + 1)]
    coords += [(r, c0) for r in range(r0 + 1, r1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    return coords


def _scan_side(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[bool, tuple[int, int] | None]:
    gap = None
    for r, c in coords[1:-1]:
        value = int(grid[r, c])
        if value == BG:
            if gap is not None:
                return False, None
            gap = (r, c)
        elif value in (START_COLOR, END_COLOR):
            return False, None
    return True, gap


def _dominant_border_color(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int | None:
    counts: dict[int, int] = {}
    for r, c in _border_coords(r0, c0, r1, c1):
        value = int(grid[r, c])
        if value in (BG, START_COLOR, END_COLOR):
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _find_rooms(grid: np.ndarray) -> list[Room]:
    h, w = grid.shape
    wall_mask = (grid != BG) & (grid != START_COLOR) & (grid != END_COLOR)
    wall_ps = _integral(wall_mask)
    found: list[tuple[int, int, int, int, int, tuple[tuple[int, int], ...]]] = []

    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    if _rect_sum(wall_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if not (wall_mask[r0, c0] and wall_mask[r0, c1] and wall_mask[r1, c0] and wall_mask[r1, c1]):
                        continue

                    width = c1 - c0 + 1
                    height = r1 - r0 + 1
                    if _rect_sum(wall_ps, r0, c0, r0, c1) < width - 1:
                        continue
                    if _rect_sum(wall_ps, r1, c0, r1, c1) < width - 1:
                        continue
                    if _rect_sum(wall_ps, r0, c0, r1, c0) < height - 1:
                        continue
                    if _rect_sum(wall_ps, r0, c1, r1, c1) < height - 1:
                        continue

                    top = [(r0, c) for c in range(c0, c1 + 1)]
                    bottom = [(r1, c) for c in range(c0, c1 + 1)]
                    left = [(r, c0) for r in range(r0, r1 + 1)]
                    right = [(r, c1) for r in range(r0, r1 + 1)]

                    ok_top, gap_top = _scan_side(grid, top)
                    ok_bottom, gap_bottom = _scan_side(grid, bottom)
                    ok_left, gap_left = _scan_side(grid, left)
                    ok_right, gap_right = _scan_side(grid, right)
                    if not (ok_top and ok_bottom and ok_left and ok_right):
                        continue

                    color = _dominant_border_color(grid, r0, c0, r1, c1)
                    if color is None:
                        continue

                    doors = tuple(door for door in (gap_top, gap_bottom, gap_left, gap_right) if door is not None)
                    found.append((r0, c0, r1, c1, color, doors))

    found.sort()
    rooms: list[Room] = []
    for idx, (r0, c0, r1, c1, color, doors) in enumerate(found):
        rooms.append(Room(idx=idx, r0=r0, c0=c0, r1=r1, c1=c1, color=color, doors=doors))
    return rooms


def _marker_point(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert len(pts) == 1
    return int(pts[0, 0]), int(pts[0, 1])


def _room_for_point(rooms: list[Room], point: tuple[int, int]) -> int:
    r, c = point
    matches = [room.idx for room in rooms if room.r0 < r < room.r1 and room.c0 < c < room.c1]
    assert len(matches) == 1
    return matches[0]


def _build_room_graph(rooms: list[Room]) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple[int, int]]]:
    graph = {room.idx: set() for room in rooms}
    door_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    by_door: dict[tuple[int, int], list[int]] = {}

    for room in rooms:
        for door in room.doors:
            by_door.setdefault(door, []).append(room.idx)

    for door, ids in by_door.items():
        if len(ids) != 2:
            continue
        a, b = sorted(ids)
        graph[a].add(b)
        graph[b].add(a)
        door_lookup[(a, b)] = door

    return graph, door_lookup


def _path_between(graph: dict[int, set[int]], start: int, end: int) -> list[int]:
    if start == end:
        return [start]
    queue = [start]
    parent = {start: -1}
    for node in queue:
        for nxt in sorted(graph[node]):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == end:
                queue = []
                break
            queue.append(nxt)
    assert end in parent

    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _door_inner_point(room: Room, door: tuple[int, int]) -> tuple[int, int]:
    r, c = door
    if r == room.r0:
        return r + 1, c
    if r == room.r1:
        return r - 1, c
    if c == room.c0:
        return r, c + 1
    if c == room.c1:
        return r, c - 1
    raise AssertionError("door must lie on room border")


def _paint_l_path(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    step_c = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + step_c, step_c):
        if out[r0, c] == BG:
            out[r0, c] = color
    step_r = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step_r, step_r):
        if out[r, c1] == BG:
            out[r, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rooms = _find_rooms(g)
    assert rooms

    start = _marker_point(g, START_COLOR)
    end = _marker_point(g, END_COLOR)
    start_room = _room_for_point(rooms, start)
    end_room = _room_for_point(rooms, end)

    graph, door_lookup = _build_room_graph(rooms)
    path = _path_between(graph, start_room, end_room)

    out = g.copy()
    for i, room_idx in enumerate(path):
        room = rooms[room_idx]
        if i == 0:
            entry = start
        else:
            prev_idx = path[i - 1]
            entry = _door_inner_point(room, door_lookup[tuple(sorted((prev_idx, room_idx)))])

        if i == len(path) - 1:
            exit_point = end
        else:
            next_idx = path[i + 1]
            exit_point = _door_inner_point(room, door_lookup[tuple(sorted((room_idx, next_idx)))])

        _paint_l_path(out, entry, exit_point, room.color)

    return out


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad


def _shared_full_side(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    if a[0] == b[0] and a[2] == b[2] and (a[3] == b[1] or b[3] == a[1]):
        return True
    if a[1] == b[1] and a[3] == b[3] and (a[2] == b[0] or b[2] == a[0]):
        return True
    return False


def _room_interior_shape(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return rect[2] - rect[0] - 1, rect[3] - rect[1] - 1


def _propose_neighbor(rect: tuple[int, int, int, int], direction: str, rng: np.random.Generator) -> tuple[int, int, int, int]:
    height, width = _room_interior_shape(rect)
    if direction == "left":
        new_width = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
        return rect[0], rect[1] - (new_width + 1), rect[2], rect[1]
    if direction == "right":
        new_width = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
        return rect[0], rect[3], rect[2], rect[3] + new_width + 1
    if direction == "up":
        new_height = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
        return rect[0] - (new_height + 1), rect[1], rect[0], rect[3]
    if direction == "down":
        new_height = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
        return rect[2], rect[1], rect[2] + new_height + 1, rect[3]
    raise ValueError(direction)


def _placement_ok(candidate: tuple[int, int, int, int], rects: list[tuple[int, int, int, int]]) -> bool:
    prev = rects[-1]
    if not _shared_full_side(prev, candidate):
        return False
    for rect in rects[:-1]:
        if _intersects(_expanded(candidate, 1), rect):
            return False
    return True


def _generate_chain_rects(rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    room_count = int(rng.integers(MIN_ROOMS, MAX_ROOMS + 1))
    first_h = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
    first_w = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
    rects = [(0, 0, first_h + 1, first_w + 1)]

    for _ in range(room_count - 1):
        placed = None
        for direction in rng.permutation(["left", "right", "up", "down"]):
            candidate = _propose_neighbor(rects[-1], str(direction), rng)
            if _placement_ok(candidate, rects):
                placed = candidate
                break
        if placed is None:
            return None
        rects.append(placed)

    return rects


def _shift_rects(
    rects: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _draw_room_border(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _shared_edge_info(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[str, int, int, int]:
    if a[0] == b[0] and a[2] == b[2] and a[3] == b[1]:
        return "vertical", a[3], a[0], a[2]
    if a[0] == b[0] and a[2] == b[2] and b[3] == a[1]:
        return "vertical", a[1], a[0], a[2]
    if a[1] == b[1] and a[3] == b[3] and a[2] == b[0]:
        return "horizontal", a[2], a[1], a[3]
    if a[1] == b[1] and a[3] == b[3] and b[2] == a[0]:
        return "horizontal", a[0], a[1], a[3]
    raise AssertionError("rooms must share a full side")


def _paint_shared_wall(
    grid: np.ndarray,
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
    color_a: int,
    color_b: int,
    rng: np.random.Generator,
) -> None:
    orientation, fixed, lo, hi = _shared_edge_info(rect_a, rect_b)
    split_first = bool(rng.integers(2))

    if orientation == "vertical":
        door_row = int(rng.integers(lo + 2, hi - 1))
        for r in range(lo, hi + 1):
            if r == door_row:
                grid[r, fixed] = BG
            elif r < door_row:
                grid[r, fixed] = color_a if split_first else color_b
            else:
                grid[r, fixed] = color_b if split_first else color_a
    else:
        door_col = int(rng.integers(lo + 2, hi - 1))
        for c in range(lo, hi + 1):
            if c == door_col:
                grid[fixed, c] = BG
            elif c < door_col:
                grid[fixed, c] = color_a if split_first else color_b
            else:
                grid[fixed, c] = color_b if split_first else color_a


def _random_interior_point(rect: tuple[int, int, int, int], rng: np.random.Generator) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    r = int(rng.integers(r0 + 1, r1))
    c = int(rng.integers(c0 + 1, c1))
    return r, c


def _validate_generated(grid: np.ndarray, expected_rooms: int) -> bool:
    try:
        rooms = _find_rooms(grid)
        if len(rooms) != expected_rooms:
            return False
        graph, _ = _build_room_graph(rooms)
        if sum(len(neighbors) for neighbors in graph.values()) // 2 != expected_rooms - 1:
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False

        start = _marker_point(grid, START_COLOR)
        end = _marker_point(grid, END_COLOR)
        if int(out[start]) != START_COLOR or int(out[end]) != END_COLOR:
            return False
        if np.count_nonzero(out != grid) < 2:
            return False
    except Exception:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rects = None
    for _ in range(LAYOUT_TRIES):
        rects = _generate_chain_rects(rng)
        if rects is not None:
            break
    if rects is None:
        return None

    min_r = min(r0 for r0, _, _, _ in rects)
    min_c = min(c0 for _, c0, _, _ in rects)
    max_r = max(r1 for _, _, r1, _ in rects)
    max_c = max(c1 for _, _, _, c1 in rects)

    pad_top = int(rng.integers(OUTER_PAD_MIN, OUTER_PAD_MAX + 1))
    pad_left = int(rng.integers(OUTER_PAD_MIN, OUTER_PAD_MAX + 1))
    pad_bottom = int(rng.integers(OUTER_PAD_MIN, OUTER_PAD_MAX + 1))
    pad_right = int(rng.integers(OUTER_PAD_MIN, OUTER_PAD_MAX + 1))

    shifted = _shift_rects(rects, pad_top - min_r, pad_left - min_c)
    grid_h = max_r - min_r + 1 + pad_top + pad_bottom
    grid_w = max_c - min_c + 1 + pad_left + pad_right
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    colors = list(map(int, rng.choice(ROOM_COLOR_POOL, size=len(shifted), replace=False)))
    for rect, color in zip(shifted, colors):
        _draw_room_border(grid, rect, color)

    for i in range(len(shifted) - 1):
        _paint_shared_wall(grid, shifted[i], shifted[i + 1], colors[i], colors[i + 1], rng)

    start_idx, end_idx = map(int, rng.choice(len(shifted), size=2, replace=False))
    start_pt = _random_interior_point(shifted[start_idx], rng)
    end_pt = _random_interior_point(shifted[end_idx], rng)
    grid[start_pt] = START_COLOR
    grid[end_pt] = END_COLOR

    if not _validate_generated(grid, len(shifted)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_TRIES} attempts")
