from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Room:
    idx: int
    bbox: tuple[int, int, int, int]

    @property
    def shape(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0 + 1, c1 - c0 + 1)


@dataclass(frozen=True)
class Door:
    row: int
    col: int
    color: int
    orientation: str
    rooms: tuple[int, int]


def _most_common_color(grid: np.ndarray, exclude: set[int] | None = None) -> int:
    exclude = set() if exclude is None else set(exclude)
    values, counts = np.unique(grid, return_counts=True)
    best = None
    for value, count in sorted(zip(values.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0])):
        if int(value) in exclude:
            continue
        best = int(value)
        break
    assert best is not None
    return best


def _find_doors(grid: np.ndarray, bg: int, wall: int) -> np.ndarray:
    h, w = grid.shape
    colored = (grid != bg) & (grid != wall)
    door_mask = np.zeros_like(colored, dtype=bool)
    for r, c in np.argwhere(colored):
        vertical = (
            0 < r < h - 1
            and 0 < c < w - 1
            and int(grid[r - 1, c]) == wall
            and int(grid[r + 1, c]) == wall
            and int(grid[r, c - 1]) != wall
            and int(grid[r, c + 1]) != wall
        )
        horizontal = (
            0 < r < h - 1
            and 0 < c < w - 1
            and int(grid[r, c - 1]) == wall
            and int(grid[r, c + 1]) == wall
            and int(grid[r - 1, c]) != wall
            and int(grid[r + 1, c]) != wall
        )
        if vertical or horizontal:
            door_mask[r, c] = True
    return door_mask


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=int)
    components: list[np.ndarray] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            queue = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
            next_id += 1
    return labels, components


def _extract_rooms(open_mask: np.ndarray) -> tuple[list[Room], np.ndarray]:
    h, w = open_mask.shape
    _labels, components = _label_components(open_mask)
    room_of_cell = -np.ones((h, w), dtype=int)
    rooms: list[Room] = []
    next_idx = 0
    for coords in components:
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if not np.all(open_mask[r0 : r1 + 1, c0 : c1 + 1]):
            continue
        if not np.all(~open_mask[r0 - 1, c0 : c1 + 1]):
            continue
        if not np.all(~open_mask[r1 + 1, c0 : c1 + 1]):
            continue
        if not np.all(~open_mask[r0 : r1 + 1, c0 - 1]):
            continue
        if not np.all(~open_mask[r0 : r1 + 1, c1 + 1]):
            continue
        room = Room(next_idx, (r0, c0, r1, c1))
        rooms.append(room)
        room_of_cell[r0 : r1 + 1, c0 : c1 + 1] = next_idx
        next_idx += 1
    assert rooms, "expected at least one enclosed rectangular room"
    return rooms, room_of_cell


def _extract_door_list(grid: np.ndarray, wall: int, door_mask: np.ndarray, room_of_cell: np.ndarray) -> list[Door]:
    doors: list[Door] = []
    h, w = grid.shape
    for r, c in np.argwhere(door_mask):
        vertical = (
            0 < r < h - 1
            and 0 < c < w - 1
            and int(grid[r - 1, c]) == wall
            and int(grid[r + 1, c]) == wall
        )
        horizontal = (
            0 < r < h - 1
            and 0 < c < w - 1
            and int(grid[r, c - 1]) == wall
            and int(grid[r, c + 1]) == wall
        )
        if vertical:
            a = int(room_of_cell[r, c - 1])
            b = int(room_of_cell[r, c + 1])
            assert a >= 0 and b >= 0 and a != b
            orientation = "vertical"
        else:
            assert horizontal
            a = int(room_of_cell[r - 1, c])
            b = int(room_of_cell[r + 1, c])
            assert a >= 0 and b >= 0 and a != b
            orientation = "horizontal"
        doors.append(Door(int(r), int(c), int(grid[r, c]), orientation, (a, b)))
    doors.sort(key=lambda door: (door.row, door.col))
    return doors


def _source_room_and_pattern(
    grid: np.ndarray, bg: int, rooms: list[Room]
) -> tuple[int, list[np.ndarray], list[np.ndarray]]:
    occs: list[np.ndarray] = []
    paints: list[np.ndarray] = []
    source_candidates: list[int] = []
    for room in rooms:
        r0, c0, r1, c1 = room.bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        occ = patch != bg
        occs.append(occ.copy())
        paints.append(np.where(occ, patch, bg).astype(int))
        if np.any(occ):
            source_candidates.append(room.idx)
    assert len(source_candidates) == 1, "expected exactly one source room"
    return source_candidates[0], occs, paints


def _project_occupancy(src_occ: np.ndarray, src_room: Room, door: Door, dst_room: Room) -> np.ndarray:
    dst_h, dst_w = dst_room.shape
    projected = np.zeros((dst_h, dst_w), dtype=bool)
    points = np.argwhere(src_occ)
    if points.size == 0:
        return projected

    src_r0, src_c0, _src_r1, _src_c1 = src_room.bbox
    dst_r0, dst_c0, dst_r1, dst_c1 = dst_room.bbox

    abs_rows = src_r0 + points[:, 0]
    abs_cols = src_c0 + points[:, 1]
    if door.orientation == "vertical":
        new_rows = abs_rows
        new_cols = 2 * door.col - abs_cols
    else:
        new_rows = 2 * door.row - abs_rows
        new_cols = abs_cols

    keep = (
        (dst_r0 <= new_rows)
        & (new_rows <= dst_r1)
        & (dst_c0 <= new_cols)
        & (new_cols <= dst_c1)
    )
    if not np.any(keep):
        return projected

    rr = new_rows[keep] - dst_r0
    cc = new_cols[keep] - dst_c0
    projected[rr, cc] = True
    return projected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the unique room pattern across colored doors into adjacent rooms with mirrored alignment.

    Input structure:
    - The background is the dominant color, and one other dominant non-background color forms
      thin orthogonal walls.
    - The walls partition the canvas into enclosed rectangular rooms whose floors are otherwise open.
    - Exactly one room contains a non-background floor pattern, while colored single-pixel door cells
      replace individual wall pixels between adjacent rooms.

    Rule:
    - Identify the unique source room and treat every door pixel as a wall anchor rather than part of
      a room floor.
    - Starting from the source room, each filled room projects its current occupied floor pattern
      through doors into still-empty neighboring rooms.
    - Crossing a vertical wall mirrors the pattern left-right around the door pixel; crossing a
      horizontal wall mirrors it top-bottom around the door pixel. The copied pattern is recolored to
      the door's color, clipped to the destination room, and same-round arrivals into one room are
      applied in door row-major order so later doors overwrite earlier colors.

    Key insight:
    - The door pixel is both the graph edge for recursive propagation and the exact reflection anchor,
      so solving the puzzle requires combining room parsing, repeated mirror transforms, and ordered
      overwrite of multi-door arrivals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    wall = _most_common_color(g, exclude={bg})
    door_mask = _find_doors(g, bg, wall)
    open_mask = (g != wall) & (~door_mask)

    rooms, room_of_cell = _extract_rooms(open_mask)
    doors = _extract_door_list(g, wall, door_mask, room_of_cell)
    source_idx, room_occ, room_paint = _source_room_and_pattern(g, bg, rooms)
    assert np.any(room_occ[source_idx]), "source room pattern must be non-empty"

    filled = [False] * len(rooms)
    filled[source_idx] = True
    for room in rooms:
        if room.idx != source_idx:
            room_occ[room.idx] = np.zeros(room.shape, dtype=bool)
            room_paint[room.idx] = np.full(room.shape, bg, dtype=int)

    while True:
        start_filled = filled[:]
        pending_occ: dict[int, np.ndarray] = {}
        pending_paint: dict[int, np.ndarray] = {}

        for door in doors:
            a, b = door.rooms
            if start_filled[a] == start_filled[b]:
                continue
            src_idx = a if start_filled[a] else b
            dst_idx = b if start_filled[a] else a
            projected = _project_occupancy(room_occ[src_idx], rooms[src_idx], door, rooms[dst_idx])
            if not np.any(projected):
                continue
            if dst_idx not in pending_occ:
                pending_occ[dst_idx] = np.zeros(rooms[dst_idx].shape, dtype=bool)
                pending_paint[dst_idx] = np.full(rooms[dst_idx].shape, bg, dtype=int)
            pending_occ[dst_idx] |= projected
            pending_paint[dst_idx][projected] = door.color

        if not pending_occ:
            break

        for dst_idx, occ in pending_occ.items():
            room_occ[dst_idx] = occ
            room_paint[dst_idx] = pending_paint[dst_idx]
            filled[dst_idx] = True

    out = g.copy()
    for room in rooms:
        if room.idx == source_idx or not filled[room.idx]:
            continue
        r0, c0, r1, c1 = room.bbox
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[room_occ[room.idx]] = room_paint[room.idx][room_occ[room.idx]]
    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_WALL = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_MIN_ROOM_SIZE = 4
GEN_MAX_ROOM_SIZE = 7
GEN_MAX_SIDE = 30
GEN_MAX_DOORS = len(GEN_COLORS)
GEN_RETRIES = 400


def _sample_band_sizes(count: int) -> list[int] | None:
    min_total = count * GEN_MIN_ROOM_SIZE
    max_total = count * GEN_MAX_ROOM_SIZE
    cap = GEN_MAX_SIDE - (count + 1)
    max_total = min(max_total, cap)
    if min_total > max_total:
        return None
    total = random.randint(min_total, max_total)
    sizes = [GEN_MIN_ROOM_SIZE] * count
    extra = total - min_total
    capacities = [GEN_MAX_ROOM_SIZE - GEN_MIN_ROOM_SIZE] * count
    while extra > 0:
        choices = [i for i, remaining in enumerate(capacities) if remaining > 0]
        idx = random.choice(choices)
        add = random.randint(1, min(capacities[idx], extra))
        sizes[idx] += add
        capacities[idx] -= add
        extra -= add
    random.shuffle(sizes)
    return sizes


def _layout_rooms(
    row_sizes: list[int], col_sizes: list[int], top: int, left: int
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], list[int], list[int], int, int]:
    room_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    wall_rows = [top]
    cursor = top
    row_starts: list[int] = []
    for height in row_sizes:
        row_starts.append(cursor + 1)
        cursor += height + 1
        wall_rows.append(cursor)
    struct_h = wall_rows[-1] - wall_rows[0] + 1

    wall_cols = [left]
    cursor = left
    col_starts: list[int] = []
    for width in col_sizes:
        col_starts.append(cursor + 1)
        cursor += width + 1
        wall_cols.append(cursor)
    struct_w = wall_cols[-1] - wall_cols[0] + 1

    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            room_boxes[(rr, cc)] = (
                r0,
                c0,
                r0 + row_sizes[rr] - 1,
                c0 + col_sizes[cc] - 1,
            )
    return room_boxes, wall_rows, wall_cols, struct_h, struct_w


def _room_neighbors(room: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    r, c = room
    neighbors = []
    if r > 0:
        neighbors.append((r - 1, c))
    if r + 1 < n_rows:
        neighbors.append((r + 1, c))
    if c > 0:
        neighbors.append((r, c - 1))
    if c + 1 < n_cols:
        neighbors.append((r, c + 1))
    return neighbors


def _choose_source(n_rows: int, n_cols: int) -> tuple[int, int]:
    all_rooms = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    interior = [room for room in all_rooms if 0 < room[0] < n_rows - 1 and 0 < room[1] < n_cols - 1]
    edge = [room for room in all_rooms if room not in interior]
    if interior and random.random() < 0.45:
        return random.choice(interior)
    return random.choice(edge or all_rooms)


def _choose_edges(n_rows: int, n_cols: int, source: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rooms = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    dist = {room: abs(room[0] - source[0]) + abs(room[1] - source[1]) for room in rooms}
    max_reachable = min(len(rooms), GEN_MAX_DOORS + 1)
    target_reachable = random.randint(min(4, max_reachable), max_reachable)

    reachable = {source}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(reachable) < target_reachable and len(edges) < GEN_MAX_DOORS:
        frontier = [
            room
            for room in rooms
            if room not in reachable
            and any(neigh in reachable and dist[neigh] < dist[room] for neigh in _room_neighbors(room, n_rows, n_cols))
        ]
        if not frontier:
            break
        room = random.choice(frontier)
        parents = [neigh for neigh in _room_neighbors(room, n_rows, n_cols) if neigh in reachable and dist[neigh] < dist[room]]
        parent = random.choice(parents)
        edges.append((parent, room))
        reachable.add(room)

    if len(reachable) < 3 or max(dist[room] for room in reachable) < 2:
        return []

    extras = [
        (parent, child)
        for child in reachable
        for parent in _room_neighbors(child, n_rows, n_cols)
        if parent in reachable and dist[parent] + 1 == dist[child] and (parent, child) not in edges
    ]
    random.shuffle(extras)
    extra_limit = GEN_MAX_DOORS - len(edges)
    if extra_limit > 0 and extras and random.random() < 0.6:
        extra_goal = random.randint(1, min(2, extra_limit, len(extras)))
        used_children = {child for _parent, child in edges}
        added = 0
        for edge in extras:
            if added >= extra_goal:
                break
            if edge[1] in used_children or random.random() < 0.7:
                edges.append(edge)
                added += 1
        while added < extra_goal and len(edges) < GEN_MAX_DOORS:
            edge = extras[added]
            if edge not in edges:
                edges.append(edge)
            added += 1

    return edges


def _door_position(
    room_boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    edge: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    a, b = edge
    ar0, ac0, ar1, ac1 = room_boxes[a]
    br0, bc0, br1, bc1 = room_boxes[b]
    if a[0] == b[0]:
        wall_col = min(ac1, bc1) + 1
        row_lo = max(ar0, br0) + 1
        row_hi = min(ar1, br1) - 1
        assert row_lo <= row_hi
        return random.randint(row_lo, row_hi), wall_col
    wall_row = min(ar1, br1) + 1
    col_lo = max(ac0, bc0) + 1
    col_hi = min(ac1, bc1) - 1
    assert col_lo <= col_hi
    return wall_row, random.randint(col_lo, col_hi)


def _random_pattern(height: int, width: int) -> np.ndarray:
    max_area = min(9, max(3, (height * width) // 3))
    for _ in range(80):
        area = random.randint(3, max_area)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows, cols = np.where(mask)
        if len(np.unique(rows)) > 1 and len(np.unique(cols)) > 1:
            return mask
    mask = np.zeros((height, width), dtype=bool)
    mask[:2, :2] = True
    mask[2, 1] = True
    return mask


def _render_input() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    row_sizes = _sample_band_sizes(n_rows)
    col_sizes = _sample_band_sizes(n_cols)
    if row_sizes is None or col_sizes is None:
        return None

    room_boxes0, wall_rows0, wall_cols0, struct_h, struct_w = _layout_rooms(row_sizes, col_sizes, 0, 0)
    max_top = GEN_MAX_SIDE - struct_h
    max_left = GEN_MAX_SIDE - struct_w
    if max_top < 0 or max_left < 0:
        return None
    top = random.randint(0, min(2, max_top))
    left = random.randint(0, min(2, max_left))
    bottom = random.randint(0, GEN_MAX_SIDE - struct_h - top)
    right = random.randint(0, GEN_MAX_SIDE - struct_w - left)
    if top + bottom > 4:
        bottom = min(bottom, 4 - top)
    if left + right > 4:
        right = min(right, 4 - left)

    room_boxes, wall_rows, wall_cols, struct_h, struct_w = _layout_rooms(row_sizes, col_sizes, top, left)
    height = top + struct_h + bottom
    width = left + struct_w + right
    if height > GEN_MAX_SIDE or width > GEN_MAX_SIDE:
        return None

    source = _choose_source(n_rows, n_cols)
    edges = _choose_edges(n_rows, n_cols, source)
    if not edges:
        return None

    grid = np.full((height, width), GEN_BG, dtype=int)
    r_min, r_max = wall_rows[0], wall_rows[-1]
    c_min, c_max = wall_cols[0], wall_cols[-1]
    for r in wall_rows:
        grid[r, c_min : c_max + 1] = GEN_WALL
    for c in wall_cols:
        grid[r_min : r_max + 1, c] = GEN_WALL

    door_colors = random.sample(GEN_COLORS, len(edges))
    used_positions: set[tuple[int, int]] = set()
    for edge, color in zip(edges, door_colors):
        row, col = _door_position(room_boxes, edge)
        if (row, col) in used_positions:
            return None
        used_positions.add((row, col))
        grid[row, col] = int(color)

    sr0, sc0, sr1, sc1 = room_boxes[source]
    src_h = sr1 - sr0 + 1
    src_w = sc1 - sc0 + 1
    pattern = _random_pattern(src_h, src_w)
    source_color = random.choice(GEN_COLORS)
    room_patch = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
    room_patch[pattern] = int(source_color)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len(np.unique(door_colors)) != len(door_colors):
        return None
    if height > GEN_MAX_SIDE or width > GEN_MAX_SIDE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _render_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
