from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
AZURE = 8

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")

# A red pointer is a 2x2 L with one missing corner.  The four missing corners
# are read clockwise as up, right, down, left.
L_SHAPES = {
    frozenset({(0, 1), (1, 0), (1, 1)}): "up",
    frozenset({(0, 0), (1, 0), (1, 1)}): "right",
    frozenset({(0, 0), (0, 1), (1, 0)}): "down",
    frozenset({(0, 0), (0, 1), (1, 1)}): "left",
}
POINTER_CELLS = {direction: tuple(sorted(cells)) for cells, direction in L_SHAPES.items()}


def _bands(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    bands = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            bands.append((start, prev))
            start = prev = idx
    bands.append((start, prev))
    return bands


def _room_intervals(size: int, wall_indices: list[int]) -> list[tuple[int, int]]:
    intervals = []
    start = 0
    for b0, b1 in _bands(wall_indices):
        if start <= b0 - 1:
            intervals.append((start, b0 - 1))
        start = b1 + 1
    if start <= size - 1:
        intervals.append((start, size - 1))
    return intervals


def _rooms_from_walls(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_rows = [int(r) for r in range(g.shape[0]) if bool((g[r, :] == AZURE).all())]
    wall_cols = [int(c) for c in range(g.shape[1]) if bool((g[:, c] == AZURE).all())]
    row_intervals = _room_intervals(int(g.shape[0]), wall_rows)
    col_intervals = _room_intervals(int(g.shape[1]), wall_cols)
    assert row_intervals and col_intervals, "walls must leave at least one room"
    assert len(row_intervals) * len(col_intervals) > 1, "expected multiple rooms"
    return row_intervals, col_intervals


def _pointer_direction(cells: list[tuple[int, int]]) -> str:
    assert len(cells) == 3, "each room must contain one 3-pixel red L pointer"
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    assert r1 - r0 == 1 and c1 - c0 == 1, "red pointer must occupy a 2x2 L"
    norm = frozenset((r - r0, c - c0) for r, c in cells)
    assert norm in L_SHAPES, "unknown red L pointer orientation"
    return L_SHAPES[norm]


def _room_cells(room: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _stack_order(room: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room
    if direction == "up":
        return [(r, c) for r in range(r1, r0 - 1, -1) for c in range(c0, c1 + 1)]
    if direction == "down":
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if direction == "left":
        return [(r, c) for c in range(c1, c0 - 1, -1) for r in range(r0, r1 + 1)]
    if direction == "right":
        return [(r, c) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)]
    raise AssertionError("bad direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simultaneously teleport yellow cargo between wall-separated rooms and stack it on arrival.

    Input structure:
    - Black is the floor and Azure full rows/columns are walls dividing the canvas into equal rectangular rooms.
    - Each room contains exactly one Red 3-pixel L pointer, encoded as a 2x2 L with one open corner.
    - Each room may contain scattered Yellow cargo pixels on otherwise black floor cells.

    Rule:
    - Decode each room's Red L opening as its direction: the four open corners of the 2x2 L encode up, right, down, left.
    - Remove every Yellow cargo pixel from its source room, then send that room's whole cargo count to the adjacent room in the decoded direction.
    - Cargo sent off the outer room grid disappears; all valid transfers are placed only after every source has been collected.
    - In the destination room, cargo arriving from up/down stacks by rows from the entering wall and fills left to right; cargo arriving from left/right stacks by columns and fills top to bottom.
    - Red pointers and Azure walls stay fixed.

    Key insight:
    - The small Red L is a direction code, and all rooms must be evaluated before any Yellow cargo is placed so senders and receivers are handled simultaneously.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    row_intervals, col_intervals = _rooms_from_walls(g)
    room_rows = len(row_intervals)
    room_cols = len(col_intervals)

    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    directions: dict[tuple[int, int], str] = {}
    cargo_counts: dict[tuple[int, int], int] = {}

    for ri, (r0, r1) in enumerate(row_intervals):
        for cj, (c0, c1) in enumerate(col_intervals):
            room_key = (int(ri), int(cj))
            room = (int(r0), int(c0), int(r1), int(c1))
            rooms[room_key] = room
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            red_local = np.argwhere(patch == RED)
            red_cells = [(int(r0 + r), int(c0 + c)) for r, c in red_local]
            directions[room_key] = _pointer_direction(red_cells)
            cargo_counts[room_key] = int(np.sum(patch == YELLOW))

    incoming: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for key, count in cargo_counts.items():
        if count == 0:
            continue
        direction = directions[key]
        dr, dc = DIRS[direction]
        target = (int(key[0] + dr), int(key[1] + dc))
        if 0 <= target[0] < room_rows and 0 <= target[1] < room_cols:
            incoming[target].append((direction, int(count)))

    out = g.copy()
    out[out == YELLOW] = BLACK

    for target in sorted(incoming):
        room = rooms[target]
        for direction, count in sorted(incoming[target], key=lambda item: DIR_ORDER.index(item[0])):
            placed = 0
            for r, c in _stack_order(room, direction):
                if int(out[r, c]) == BLACK:
                    out[r, c] = YELLOW
                    placed += 1
                    if placed == count:
                        break
            assert placed == count, "destination room must have enough free floor for incoming cargo"

    return out


# --------------- generator ---------------

GEN_RETRIES = 500
LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
ROOM_MIN_SIDE = 5
ROOM_MAX_SIDE = 9
WALL_THICKNESS_CHOICES = (1, 2)
MAX_CARGO_PER_ROOM = 5


def _layout_intervals(count: int, room_side: int, wall_thickness: int) -> list[tuple[int, int]]:
    intervals = []
    pos = 0
    for _ in range(count):
        intervals.append((pos, pos + room_side - 1))
        pos += room_side + wall_thickness
    return intervals


def _make_wall_grid(room_rows: int, room_cols: int, room_h: int, room_w: int, wall_t: int) -> np.ndarray:
    h = room_rows * room_h + (room_rows - 1) * wall_t
    w = room_cols * room_w + (room_cols - 1) * wall_t
    grid = np.full((h, w), BLACK, dtype=int)
    for i in range(1, room_rows):
        r0 = i * room_h + (i - 1) * wall_t
        grid[r0 : r0 + wall_t, :] = AZURE
    for j in range(1, room_cols):
        c0 = j * room_w + (j - 1) * wall_t
        grid[:, c0 : c0 + wall_t] = AZURE
    return grid


def _place_pointer(grid: np.ndarray, room: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = room
    top_min, top_max = r0 + 1, r1 - 2
    left_min, left_max = c0 + 1, c1 - 2
    if top_max < top_min:
        top_min, top_max = r0, r1 - 1
    if left_max < left_min:
        left_min, left_max = c0, c1 - 1
    top = random.randint(top_min, top_max)
    left = random.randint(left_min, left_max)
    cells = {(top + dr, left + dc) for dr, dc in POINTER_CELLS[direction]}
    for r, c in cells:
        grid[r, c] = RED
    return cells


def _place_cargo(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    pointer: set[tuple[int, int]],
    count: int,
) -> None:
    candidates = [cell for cell in _room_cells(room) if cell not in pointer]
    halo = set()
    for pr, pc in pointer:
        for rr in range(pr - 1, pr + 2):
            for cc in range(pc - 1, pc + 2):
                halo.add((rr, cc))
    clear_candidates = [cell for cell in candidates if cell not in halo]
    pool = clear_candidates if len(clear_candidates) >= count else candidates
    for r, c in random.sample(pool, count):
        grid[r, c] = YELLOW


def _shipment_summary(
    directions: dict[tuple[int, int], str],
    counts: dict[tuple[int, int], int],
    room_rows: int,
    room_cols: int,
) -> dict[str, int | bool]:
    incoming_by_room: defaultdict[tuple[int, int], int] = defaultdict(int)
    inbound = disappeared = vertical = horizontal = 0

    for key, count in counts.items():
        if count == 0:
            continue
        direction = directions[key]
        dr, dc = DIRS[direction]
        target = (key[0] + dr, key[1] + dc)
        if 0 <= target[0] < room_rows and 0 <= target[1] < room_cols:
            incoming_by_room[target] += count
            inbound += count
            if direction in ("up", "down"):
                vertical += count
            else:
                horizontal += count
        else:
            disappeared += count

    sender_and_receiver = any(counts.get(key, 0) > 0 and incoming_by_room.get(key, 0) > 0 for key in counts)
    zero_room = any(count == 0 for count in counts.values())
    return {
        "inbound": inbound,
        "disappeared": disappeared,
        "vertical": vertical,
        "horizontal": horizontal,
        "sender_and_receiver": sender_and_receiver,
        "zero_room": zero_room,
    }


def _sample_room_side(room_count: int, wall_t: int) -> int:
    max_side = min(ROOM_MAX_SIDE, (30 - (room_count - 1) * wall_t) // room_count)
    assert max_side >= ROOM_MIN_SIDE
    return random.randint(ROOM_MIN_SIDE, max_side)


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(LAYOUTS)
    wall_t = random.choice(WALL_THICKNESS_CHOICES)
    room_h = _sample_room_side(room_rows, wall_t)
    room_w = _sample_room_side(room_cols, wall_t)

    grid = _make_wall_grid(room_rows, room_cols, room_h, room_w, wall_t)
    assert max(grid.shape) <= 30

    row_intervals = _layout_intervals(room_rows, room_h, wall_t)
    col_intervals = _layout_intervals(room_cols, room_w, wall_t)

    keys = [(ri, cj) for ri in range(room_rows) for cj in range(room_cols)]
    directions = {key: random.choice(DIR_ORDER) for key in keys}
    for key, direction in zip(random.sample(keys, 4), random.sample(DIR_ORDER, 4)):
        directions[key] = direction

    counts = {key: random.randint(0, MAX_CARGO_PER_ROOM) for key in keys}
    summary = _shipment_summary(directions, counts, room_rows, room_cols)
    if not (
        summary["inbound"] > 0
        and summary["disappeared"] > 0
        and summary["vertical"] > 0
        and summary["horizontal"] > 0
        and summary["sender_and_receiver"]
        and summary["zero_room"]
    ):
        return None

    for ri, (r0, r1) in enumerate(row_intervals):
        for cj, (c0, c1) in enumerate(col_intervals):
            key = (ri, cj)
            room = (r0, c0, r1, c1)
            pointer = _place_pointer(grid, room, directions[key])
            _place_cargo(grid, room, pointer, counts[key])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate a valid cargo-room input")
