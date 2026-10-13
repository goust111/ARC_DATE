from __future__ import annotations

import heapq
import random

import numpy as np


Coord = tuple[int, int]
Box = tuple[int, int, int, int]

BG_COLOR = 0
WALL_COLOR = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_TO_DIR = {"top": 0, "right": 1, "bottom": 2, "left": 3}

ROOM_COUNT_OPTIONS = (2, 3, 4)
ROOM_SIDE_MIN = 4
ROOM_SIDE_MAX = 6
ACTIVE_ROOM_MIN = 2
ACTIVE_ROOM_MAX = 6
DOOR_PROBABILITY = 0.45
GEN_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    non_bg = grid[grid != bg]
    if non_bg.size == 0:
        raise ValueError("grid has no wall color")
    colors, counts = np.unique(non_bg, return_counts=True)
    return int(colors[np.argmax(counts)])


def _wall_axes(grid: np.ndarray, wall: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_hits = np.count_nonzero(grid == wall, axis=1)
    col_hits = np.count_nonzero(grid == wall, axis=0)
    wall_rows = [int(r) for r in np.where(row_hits > w // 2)[0]]
    wall_cols = [int(c) for c in np.where(col_hits > h // 2)[0]]
    if len(wall_rows) < 2 or len(wall_cols) < 2:
        raise ValueError("failed to recover room lattice")
    return wall_rows, wall_cols


def _room_boxes(grid: np.ndarray) -> tuple[int, int, list[Box]]:
    bg = _background_color(grid)
    wall = _wall_color(grid, bg)
    wall_rows, wall_cols = _wall_axes(grid, wall)

    boxes: list[Box] = []
    for i in range(len(wall_rows) - 1):
        top = wall_rows[i] + 1
        bottom = wall_rows[i + 1] - 1
        if top > bottom:
            continue
        for j in range(len(wall_cols) - 1):
            left = wall_cols[j] + 1
            right = wall_cols[j + 1] - 1
            if left > right:
                continue
            boxes.append((top, bottom, left, right))
    return bg, wall, boxes


def _room_task(grid: np.ndarray, box: Box, bg: int, wall: int) -> tuple[int, Coord, Coord] | None:
    top, bottom, left, right = box
    room = grid[top : bottom + 1, left : right + 1]
    h, w = room.shape
    marker_colors = [int(c) for c in np.unique(room) if int(c) not in (bg, wall)]

    candidates: list[tuple[int, Coord, Coord]] = []
    for color in marker_colors:
        coords = np.argwhere(room == color)
        if coords.shape[0] != 2:
            continue
        on_ring = [
            int(r) in (0, h - 1) or int(c) in (0, w - 1)
            for r, c in coords
        ]
        if sum(on_ring) != 1:
            continue
        exit_idx = 0 if on_ring[0] else 1
        start_idx = 1 - exit_idx
        sr, sc = map(int, coords[start_idx])
        er, ec = map(int, coords[exit_idx])
        candidates.append((color, (top + sr, left + sc), (top + er, left + ec)))

    if len(candidates) == 1:
        return candidates[0]
    return None


def _extract_tasks(grid: np.ndarray) -> tuple[int, int, list[tuple[Box, int, Coord, Coord]]]:
    bg, wall, boxes = _room_boxes(grid)
    tasks: list[tuple[Box, int, Coord, Coord]] = []
    for box in boxes:
        task = _room_task(grid, box, bg, wall)
        if task is not None:
            color, start, exit_cell = task
            tasks.append((box, color, start, exit_cell))
    return bg, wall, tasks


def _exit_side(box: Box, exit_cell: Coord) -> str:
    top, bottom, left, right = box
    r, c = exit_cell
    if r == top:
        return "top"
    if r == bottom:
        return "bottom"
    if c == left:
        return "left"
    if c == right:
        return "right"
    raise ValueError("exit is not on the inner wall ring")


def _dir_priority(start: Coord, exit_cell: Coord, side: str) -> list[int]:
    sr, sc = start
    er, ec = exit_cell
    vert_toward = 0 if er < sr else 2
    vert_away = 2 if vert_toward == 0 else 0
    horiz_toward = 3 if ec < sc else 1
    horiz_away = 1 if horiz_toward == 3 else 3

    if side in ("top", "bottom"):
        return [horiz_toward, vert_toward, horiz_away, vert_away]
    return [vert_toward, horiz_toward, vert_away, horiz_away]


def _min_turn_path(room: np.ndarray, start: Coord, exit_cell: Coord, bg: int, side: str) -> list[Coord]:
    h, w = room.shape
    walkable = room == bg
    walkable[start] = True
    walkable[exit_cell] = True

    desired_final_dir = SIDE_TO_DIR[side]
    priority = _dir_priority(start, exit_cell, side)
    start_state = (start[0], start[1], 4)
    best: dict[tuple[int, int, int], tuple[int, int, int]] = {start_state: (0, 0, 0)}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[int, int, int, int, int, int]] = [(0, 0, 0, start[0], start[1], 4)]
    end_state: tuple[int, int, int] | None = None

    while heap:
        turns, dist, penalty, r, c, prev_dir = heapq.heappop(heap)
        state = (r, c, prev_dir)
        if (turns, dist, penalty) != best.get(state):
            continue
        if (r, c) == exit_cell and prev_dir != 4:
            end_state = state
            break

        for new_dir in priority:
            dr, dc = DIRS[new_dir]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not walkable[nr, nc]:
                continue

            new_turns = turns + int(prev_dir != 4 and prev_dir != new_dir)
            new_dist = dist + 1
            new_penalty = penalty + int((nr, nc) == exit_cell and new_dir != desired_final_dir)
            new_state = (nr, nc, new_dir)
            new_cost = (new_turns, new_dist, new_penalty)
            if new_cost < best.get(new_state, (10**9, 10**9, 10**9)):
                best[new_state] = new_cost
                parent[new_state] = state
                heapq.heappush(heap, (new_turns, new_dist, new_penalty, nr, nc, new_dir))

    if end_state is None:
        raise ValueError("room markers are not connected by an interior path")

    path: list[Coord] = []
    state = end_state
    while True:
        r, c, _ = state
        path.append((r, c))
        if state == start_state:
            break
        state = parent[state]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each marked room with the least-turn route from its interior start to its wall-side exit.

    Input structure:
    - A black background is partitioned by a one-pixel grey wall lattice into rectangular rooms.
    - Some internal wall segments contain one-pixel door gaps, but rooms are still read from the enclosing lattice cells.
    - A task room contains exactly two same-color markers: one start pixel away from the walls and one exit pixel on the room's inner wall ring.

    Rule:
    - Process each room independently and leave empty rooms unchanged.
    - Inside a task room, walk 4-connected through that room's background pixels from the start to the exit.
    - Choose a path with the fewest turns; among those, choose the shortest one.
    - Paint every pixel on that route, including the two markers, with the marker color, while preserving the grey walls and door gaps.

    Key insight:
    - Each room interior is an obstacle-free rectangle, so the desired route is the obvious straight segment or single-bend L-shape; when both L-shapes tie, approach the wall-side exit directly from the room interior.
    """
    g = np.asarray(grid, dtype=int)
    bg, _wall, tasks = _extract_tasks(g)

    out = g.copy()
    for box, color, start_global, exit_global in tasks:
        top, bottom, left, right = box
        room = g[top : bottom + 1, left : right + 1]
        start_local = (start_global[0] - top, start_global[1] - left)
        exit_local = (exit_global[0] - top, exit_global[1] - left)
        side = _exit_side(box, exit_global)
        path = _min_turn_path(room, start_local, exit_local, bg, side)
        for r, c in path:
            out[top + r, left + c] = color
    return out


def _sample_room_side(n_rooms: int) -> int:
    max_side = min(ROOM_SIDE_MAX, (30 - (n_rooms + 1)) // n_rooms)
    return random.randint(ROOM_SIDE_MIN, max_side)


def _build_base_grid(room_rows: int, room_cols: int, room_h: int, room_w: int) -> tuple[np.ndarray, list[int], list[int], list[Box]]:
    h = room_rows * room_h + room_rows + 1
    w = room_cols * room_w + room_cols + 1
    grid = np.full((h, w), BG_COLOR, dtype=int)

    wall_rows = [i * (room_h + 1) for i in range(room_rows + 1)]
    wall_cols = [j * (room_w + 1) for j in range(room_cols + 1)]
    grid[wall_rows, :] = WALL_COLOR
    grid[:, wall_cols] = WALL_COLOR

    boxes: list[Box] = []
    for i in range(room_rows):
        for j in range(room_cols):
            top = wall_rows[i] + 1
            bottom = wall_rows[i + 1] - 1
            left = wall_cols[j] + 1
            right = wall_cols[j + 1] - 1
            boxes.append((top, bottom, left, right))
    return grid, wall_rows, wall_cols, boxes


def _balanced_turn_flags(count: int) -> list[bool]:
    turn_count = random.choice([count // 2, (count + 1) // 2])
    flags = [False] * (count - turn_count) + [True] * turn_count
    random.shuffle(flags)
    return flags


def _choose_exit(box: Box, side: str) -> Coord:
    top, bottom, left, right = box
    if side == "top":
        return (top, random.randint(left + 1, right - 1))
    if side == "bottom":
        return (bottom, random.randint(left + 1, right - 1))
    if side == "left":
        return (random.randint(top + 1, bottom - 1), left)
    return (random.randint(top + 1, bottom - 1), right)


def _choose_start(box: Box, exit_cell: Coord, turn: bool) -> Coord | None:
    top, bottom, left, right = box
    er, ec = exit_cell
    rows = list(range(top + 1, bottom))
    cols = list(range(left + 1, right))
    candidates: list[Coord] = []

    if er == top or er == bottom:
        if turn:
            candidates = [(r, c) for r in rows for c in cols if c != ec]
        else:
            candidates = [(r, ec) for r in rows]
    else:
        if turn:
            candidates = [(r, c) for r in rows for c in cols if r != er]
        else:
            candidates = [(er, c) for c in cols]

    if not candidates:
        return None
    return random.choice(candidates)


def _wall_cell_beyond_exit(exit_cell: Coord, box: Box) -> Coord:
    top, bottom, left, right = box
    r, c = exit_cell
    if r == top:
        return (r - 1, c)
    if r == bottom:
        return (r + 1, c)
    if c == left:
        return (r, c - 1)
    if c == right:
        return (r, c + 1)
    raise ValueError("bad exit placement")


def _carve_doors(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    room_rows: int,
    room_cols: int,
    forbidden: set[Coord],
) -> None:
    eligible_segments: list[list[Coord]] = []

    for i in range(1, room_rows):
        wr = wall_rows[i]
        for j in range(room_cols):
            cells = [(wr, c) for c in range(wall_cols[j] + 1, wall_cols[j + 1]) if (wr, c) not in forbidden]
            if cells:
                eligible_segments.append(cells)

    for j in range(1, room_cols):
        wc = wall_cols[j]
        for i in range(room_rows):
            cells = [(r, wc) for r in range(wall_rows[i] + 1, wall_rows[i + 1]) if (r, wc) not in forbidden]
            if cells:
                eligible_segments.append(cells)

    carved = 0
    for cells in eligible_segments:
        if random.random() < DOOR_PROBABILITY:
            rr, cc = random.choice(cells)
            grid[rr, cc] = BG_COLOR
            carved += 1

    if carved == 0 and eligible_segments:
        rr, cc = random.choice(random.choice(eligible_segments))
        grid[rr, cc] = BG_COLOR


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.choice(ROOM_COUNT_OPTIONS)
    room_cols = random.choice(ROOM_COUNT_OPTIONS)
    room_h = _sample_room_side(room_rows)
    room_w = _sample_room_side(room_cols)
    grid, wall_rows, wall_cols, boxes = _build_base_grid(room_rows, room_cols, room_h, room_w)

    total_rooms = len(boxes)
    active_count = random.randint(ACTIVE_ROOM_MIN, min(ACTIVE_ROOM_MAX, total_rooms))
    active_indices = random.sample(range(total_rooms), active_count)
    colors = random.sample(MARKER_COLORS, active_count)
    turn_flags = _balanced_turn_flags(active_count)

    forbidden_doors: set[Coord] = set()
    expected_tasks: set[tuple[int, Coord, Coord]] = set()
    sides = ["top", "bottom", "left", "right"]

    for room_idx, color, turn in zip(active_indices, colors, turn_flags):
        box = boxes[room_idx]
        random.shuffle(sides)

        placed = False
        for side in sides:
            exit_cell = _choose_exit(box, side)
            start = _choose_start(box, exit_cell, turn)
            if start is None or start == exit_cell:
                continue
            grid[start] = color
            grid[exit_cell] = color
            forbidden_doors.add(_wall_cell_beyond_exit(exit_cell, box))
            expected_tasks.add((color, start, exit_cell))
            placed = True
            break

        if not placed:
            return None

    _carve_doors(grid, wall_rows, wall_cols, room_rows, room_cols, forbidden_doors)

    try:
        parsed_tasks = _extract_tasks(grid)[2]
    except ValueError:
        return None

    parsed_signature = {(color, start, exit_cell) for _box, color, start, exit_cell in parsed_tasks}
    if parsed_signature != expected_tasks:
        return None

    output = main(grid)
    if output.shape != grid.shape:
        return None
    if not np.array_equal(output[grid == WALL_COLOR], grid[grid == WALL_COLOR]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-room input")
