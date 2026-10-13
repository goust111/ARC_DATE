from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREEN = 3
WALL = 5
DOOR = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _pair_key(a, b):
    return (a, b) if a <= b else (b, a)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _analyze_maze(
    grid: np.ndarray,
) -> tuple[
    list[list[tuple[int, int]]],
    np.ndarray,
    list[bool],
    list[set[int]],
    dict[tuple[int, int], tuple[int, int]],
    list[tuple[tuple[int, int], int]],
]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    room_mask = (g != WALL) & (g != DOOR)
    rooms = _components(room_mask)
    room_of = np.full((h, w), -1, dtype=int)
    passable: list[bool] = []

    for room_id, cells in enumerate(rooms):
        has_green = False
        for r, c in cells:
            room_of[r, c] = room_id
            if int(g[r, c]) == GREEN:
                has_green = True
        passable.append(has_green)

    edge_to_door: dict[tuple[int, int], tuple[int, int]] = {}
    border_doors: list[tuple[tuple[int, int], int]] = []
    for r, c in np.argwhere(g == DOOR):
        rr, cc = int(r), int(c)
        room_ids = sorted(
            {
                int(room_of[nr, nc])
                for dr, dc in DIRS4
                for nr, nc in [(rr + dr, cc + dc)]
                if 0 <= nr < h and 0 <= nc < w and int(room_of[nr, nc]) >= 0
            }
        )
        if len(room_ids) == 2:
            edge_to_door[_pair_key(room_ids[0], room_ids[1])] = (rr, cc)
        elif len(room_ids) == 1 and (rr in (0, h - 1) or cc in (0, w - 1)):
            border_doors.append(((rr, cc), room_ids[0]))

    adj = [set() for _ in rooms]
    for a, b in edge_to_door:
        adj[a].add(b)
        adj[b].add(a)

    border_doors.sort(key=lambda item: item[0])
    return rooms, room_of, passable, adj, edge_to_door, border_doors


def _room_path(start: int, goal: int, passable: list[bool], adj: list[set[int]]) -> list[int] | None:
    if not (0 <= start < len(passable) and 0 <= goal < len(passable)):
        return None
    if not (passable[start] and passable[goal]):
        return None

    parent = {start: -1}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in sorted(adj[cur]):
            if nxt not in parent and passable[nxt]:
                parent[nxt] = cur
                q.append(nxt)

    if goal not in parent:
        return None

    path: list[int] = []
    cur = goal
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _doors_to_keep(grid: np.ndarray) -> set[tuple[int, int]]:
    _rooms, _room_of, passable, adj, edge_to_door, border_doors = _analyze_maze(grid)
    if len(border_doors) != 2:
        return set()

    (_, start_room), (_, goal_room) = border_doors
    path = _room_path(start_room, goal_room, passable, adj)
    if path is None:
        return set()

    keep = {border_doors[0][0], border_doors[1][0]}
    for a, b in zip(path, path[1:]):
        door = edge_to_door.get(_pair_key(a, b))
        if door is not None:
            keep.add(door)
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the maroon doors on the unique entrance-to-exit route through green-marked rooms.

    Input structure:
    - Black background with gray walls partitioning the grid into rectangular rooms.
    - Maroon single-pixel gaps in walls act as doors; exactly two lie on the outer border.
    - Each room contains zero or one green pixel; green marks the room as passable.

    Rule:
    - Treat rooms with at least one green pixel as traversable.
    - Find the unique path of rooms connecting the two border doors through traversable rooms.
    - Preserve the doors on that path and recolor every other maroon door to gray.
    - Leave green markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    keep = _doors_to_keep(g)
    for r, c in np.argwhere(out == DOOR):
        rr, cc = int(r), int(c)
        if (rr, cc) not in keep:
            out[rr, cc] = WALL
    return out


# --------------- generator ---------------

ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_SIZE_RANGE = (2, 5)
GEN_ATTEMPTS = 400
BRANCH_PROB = 0.6
MAX_DISTRACTOR_DOORS = 5


def _boundaries(sizes: list[int]) -> list[int]:
    walls = [0]
    pos = 0
    for size in sizes:
        pos += int(size) + 1
        walls.append(pos)
    return walls


def _room_neighbors(room: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    r, c = room
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < room_rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < room_cols:
        out.append((r, c + 1))
    return out


def _all_room_edges(room_rows: int, room_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(room_rows):
        for c in range(room_cols):
            if r + 1 < room_rows:
                edges.append(_pair_key((r, c), (r + 1, c)))
            if c + 1 < room_cols:
                edges.append(_pair_key((r, c), (r, c + 1)))
    return edges


def _border_rooms(room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(room_rows)
        for c in range(room_cols)
        if r == 0 or c == 0 or r == room_rows - 1 or c == room_cols - 1
    ]


def _choose_terminal_rooms(
    room_rows: int, room_cols: int, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    border = _border_rooms(room_rows, room_cols)
    pairs = [(border[i], border[j]) for i in range(len(border)) for j in range(i + 1, len(border))]
    max_dist = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in pairs)
    target_dist = 3 if max_dist >= 3 else max_dist
    choices = [pair for pair in pairs if abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]) >= target_dist]
    return rng.choice(choices if choices else pairs)


def _monotone_path(
    start: tuple[int, int], goal: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = goal
    steps = []
    dr = 1 if r1 >= r0 else -1
    dc = 1 if c1 >= c0 else -1
    steps.extend([(dr, 0)] * abs(r1 - r0))
    steps.extend([(0, dc)] * abs(c1 - c0))
    rng.shuffle(steps)

    path = [start]
    cur_r, cur_c = r0, c0
    for step_r, step_c in steps:
        cur_r += step_r
        cur_c += step_c
        path.append((cur_r, cur_c))
    return path


def _build_layout(
    room_rows: int, room_cols: int, rng: random.Random
) -> tuple[np.ndarray, list[int], list[int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    row_sizes = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    col_sizes = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    row_walls = _boundaries(row_sizes)
    col_walls = _boundaries(col_sizes)
    h = row_walls[-1] + 1
    w = col_walls[-1] + 1

    grid = np.full((h, w), BG, dtype=int)
    grid[row_walls, :] = WALL
    grid[:, col_walls] = WALL

    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(room_rows):
        for c in range(room_cols):
            boxes[(r, c)] = (
                row_walls[r] + 1,
                row_walls[r + 1] - 1,
                col_walls[c] + 1,
                col_walls[c + 1] - 1,
            )
    return grid, row_walls, col_walls, boxes


def _add_green_markers(
    grid: np.ndarray,
    passable_rooms: set[tuple[int, int]],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    rng: random.Random,
) -> None:
    for room in passable_rooms:
        r0, r1, c0, c1 = boxes[room]
        grid[rng.randint(r0, r1), rng.randint(c0, c1)] = GREEN


def _internal_door_pos(
    edge: tuple[tuple[int, int], tuple[int, int]],
    row_walls: list[int],
    col_walls: list[int],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int]:
    (r0, c0), (r1, c1) = edge
    if r0 == r1:
        box = boxes[(r0, c0)]
        wall_c = col_walls[min(c0, c1) + 1]
        return rng.randint(box[0], box[1]), wall_c
    box = boxes[(r0, c0)]
    wall_r = row_walls[min(r0, r1) + 1]
    return wall_r, rng.randint(box[2], box[3])


def _border_sides(room: tuple[int, int], room_rows: int, room_cols: int) -> list[str]:
    r, c = room
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == room_rows - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == room_cols - 1:
        sides.append("right")
    return sides


def _border_door_pos(
    room: tuple[int, int],
    side: str,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    h, w = grid_shape
    r0, r1, c0, c1 = boxes[room]
    if side == "top":
        return 0, rng.randint(c0, c1)
    if side == "bottom":
        return h - 1, rng.randint(c0, c1)
    if side == "left":
        return rng.randint(r0, r1), 0
    return rng.randint(r0, r1), w - 1


def _grow_passable_tree(
    path: list[tuple[int, int]],
    room_rows: int,
    room_cols: int,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]]:
    rooms = set(path)
    edges = {_pair_key(a, b) for a, b in zip(path, path[1:])}
    free_rooms = room_rows * room_cols - len(rooms)
    if free_rooms <= 0 or rng.random() >= BRANCH_PROB:
        return rooms, edges

    extra_cap = min(free_rooms, max(1, (room_rows * room_cols) // 3))
    extra_target = rng.randint(1, extra_cap)
    for _ in range(extra_target):
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for room in rooms:
            for nxt in _room_neighbors(room, room_rows, room_cols):
                if nxt not in rooms:
                    candidates.append((room, nxt))
        if not candidates:
            break
        parent, child = rng.choice(candidates)
        rooms.add(child)
        edges.add(_pair_key(parent, child))
    return rooms, edges


def _case_valid(grid: np.ndarray) -> bool:
    if int(np.count_nonzero(grid == DOOR)) < 3:
        return False
    keep = _doors_to_keep(grid)
    if len(keep) < 2 or len(keep) >= int(np.count_nonzero(grid == DOOR)):
        return False
    out = main(grid)
    return not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(*ROOM_ROWS_RANGE)
    room_cols = rng.randint(*ROOM_COLS_RANGE)
    grid, row_walls, col_walls, boxes = _build_layout(room_rows, room_cols, rng)

    start_room, goal_room = _choose_terminal_rooms(room_rows, room_cols, rng)
    path = _monotone_path(start_room, goal_room, rng)
    if len(path) < 3:
        return None

    passable_rooms, tree_edges = _grow_passable_tree(path, room_rows, room_cols, rng)
    _add_green_markers(grid, passable_rooms, boxes, rng)

    all_edges = _all_room_edges(room_rows, room_cols)
    distractor_candidates = [
        edge
        for edge in all_edges
        if edge not in tree_edges and not (edge[0] in passable_rooms and edge[1] in passable_rooms)
    ]
    distractor_count = 0
    if distractor_candidates:
        min_count = 0 if len(tree_edges) > len(path) - 1 else 1
        max_count = min(MAX_DISTRACTOR_DOORS, len(distractor_candidates))
        if min_count <= max_count:
            distractor_count = rng.randint(min_count, max_count)
    distractor_edges = set(rng.sample(distractor_candidates, distractor_count)) if distractor_count else set()

    for edge in tree_edges | distractor_edges:
        r, c = _internal_door_pos(edge, row_walls, col_walls, boxes, rng)
        grid[r, c] = DOOR

    for terminal_room in (start_room, goal_room):
        side = rng.choice(_border_sides(terminal_room, room_rows, room_cols))
        r, c = _border_door_pos(terminal_room, side, boxes, grid.shape, rng)
        grid[r, c] = DOOR

    return grid if _case_valid(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
