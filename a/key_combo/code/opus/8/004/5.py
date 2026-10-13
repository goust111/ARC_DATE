from __future__ import annotations

import random
from typing import Iterable

import numpy as np


OBSTACLE = 5
BACKGROUND = 0
ROOM_COLORS = (1, 2, 3, 4, 6)


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    pref = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    pref[1:, 1:] = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return pref


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(pref[r1, c1] - pref[r0, c1] - pref[r1, c0] + pref[r0, c0])


def _room_mask(room: dict) -> np.ndarray:
    r0 = int(room["r"])
    c0 = int(room["c"])
    size = int(room["size"])
    return np.s_[r0 : r0 + size, c0 : c0 + size]


def _room_cells(room: dict) -> set[tuple[int, int]]:
    r0 = int(room["r"])
    c0 = int(room["c"])
    size = int(room["size"])
    return {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}


def _find_rooms(grid: np.ndarray) -> list[dict]:
    black = grid == BACKGROUND
    pref = _prefix_sum(black)
    H, W = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for size in range(min(H, W), 1, -1):
        for r in range(H - size + 1):
            for c in range(W - size + 1):
                if _rect_sum(pref, r, c, r + size, c + size) == size * size:
                    candidates.append((size, r, c))

    occupied = np.zeros_like(grid, dtype=bool)
    rooms: list[dict] = []
    for size, r, c in candidates:
        patch = occupied[r : r + size, c : c + size]
        if patch.any():
            continue
        occupied[r : r + size, c : c + size] = True
        rooms.append({"size": int(size), "r": int(r), "c": int(c)})
    return rooms


def _in_bounds(grid: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def _is_clear_wall(grid: np.ndarray, r: int, c: int) -> bool:
    return (not _in_bounds(grid, r, c)) or int(grid[r, c]) != BACKGROUND


def _horizontal_corridor_rows(grid: np.ndarray, left: dict, right: dict) -> list[int]:
    if left["c"] + left["size"] > right["c"]:
        return []
    start = int(left["c"] + left["size"])
    end = int(right["c"] - 1)
    if start > end:
        return []

    rows = []
    row_lo = max(int(left["r"]), int(right["r"]))
    row_hi = min(int(left["r"] + left["size"] - 1), int(right["r"] + right["size"] - 1))
    for r in range(row_lo, row_hi + 1):
        ok = True
        for c in range(start, end + 1):
            if int(grid[r, c]) != BACKGROUND:
                ok = False
                break
            if not _is_clear_wall(grid, r - 1, c):
                ok = False
                break
            if not _is_clear_wall(grid, r + 1, c):
                ok = False
                break
        if ok:
            rows.append(r)
    return rows


def _vertical_corridor_cols(grid: np.ndarray, top: dict, bottom: dict) -> list[int]:
    if top["r"] + top["size"] > bottom["r"]:
        return []
    start = int(top["r"] + top["size"])
    end = int(bottom["r"] - 1)
    if start > end:
        return []

    cols = []
    col_lo = max(int(top["c"]), int(bottom["c"]))
    col_hi = min(int(top["c"] + top["size"] - 1), int(bottom["c"] + bottom["size"] - 1))
    for c in range(col_lo, col_hi + 1):
        ok = True
        for r in range(start, end + 1):
            if int(grid[r, c]) != BACKGROUND:
                ok = False
                break
            if not _is_clear_wall(grid, r, c - 1):
                ok = False
                break
            if not _is_clear_wall(grid, r, c + 1):
                ok = False
                break
        if ok:
            cols.append(c)
    return cols


def _paint_half_segment(out: np.ndarray, cells: list[tuple[int, int]], color_a: int, color_b: int) -> None:
    span = len(cells) // 2
    for r, c in cells[:span]:
        out[r, c] = int(color_a)
    for r, c in cells[len(cells) - span :]:
        out[r, c] = int(color_b)


def main(grid: np.ndarray) -> np.ndarray:
    """Greedily color the largest square rooms, then split each straight connecting corridor between the two room colors.

    Input structure:
    - A black open-space maze is embedded in gray obstacle walls.
    - Some open regions contain axis-aligned all-black square rooms of side at least 2, while the links between rooms are width-1 straight corridors.
    - Extra black dead ends or unused corridor fragments may also exist.

    Rule:
    - Enumerate every all-black square of side at least 2 and greedily keep them from largest to smallest, skipping any candidate that overlaps a previously kept square.
    - Color the kept rooms in that greedy order with blue, red, green, yellow, then magenta.
    - For every pair of kept rooms joined by exactly one straight width-1 horizontal or vertical black corridor, color half of the corridor from one end with the first room color and half from the other end with the second room color.
    - If a corridor has odd length, leave its middle cell black; leave all other untouched black cells and all gray obstacles unchanged.

    Key insight:
    - Room identity is determined by greedy square selection rather than by connected components, so corridor colors can only be assigned after the room ranking is fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _find_rooms(g)
    assert 1 <= len(rooms) <= len(ROOM_COLORS)
    for room, color in zip(rooms, ROOM_COLORS):
        room["color"] = int(color)

    out = g.copy()
    for room in rooms:
        out[_room_mask(room)] = int(room["color"])

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]

            left, right = sorted((a, b), key=lambda room: room["c"])
            rows = _horizontal_corridor_rows(g, left, right)
            if len(rows) == 1:
                r = rows[0]
                cells = [(r, c) for c in range(left["c"] + left["size"], right["c"])]
                _paint_half_segment(out, cells, left["color"], right["color"])
                continue

            top, bottom = sorted((a, b), key=lambda room: room["r"])
            cols = _vertical_corridor_cols(g, top, bottom)
            if len(cols) == 1:
                c = cols[0]
                cells = [(r, c) for r in range(top["r"] + top["size"], bottom["r"])]
                _paint_half_segment(out, cells, top["color"], bottom["color"])

    return out


def _line_cells(start: tuple[int, int], step: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = step
    return [(r0 + dr * k, c0 + dc * k) for k in range(length)]


def _neighbors4(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    yield (r - 1, c)
    yield (r + 1, c)
    yield (r, c - 1)
    yield (r, c + 1)


def _try_attach(
    rooms: list[dict],
    open_cells: set[tuple[int, int]],
    room_size: int,
) -> tuple[dict, list[tuple[int, int]]] | None:
    directions = [
        ("right", (0, 1)),
        ("left", (0, -1)),
        ("down", (1, 0)),
        ("up", (-1, 0)),
    ]
    trials = [(parent, direction) for parent in rooms for direction in directions]
    random.shuffle(trials)

    for parent, (name, _delta) in trials:
        gap = random.randint(2, 7)
        if name in ("right", "left"):
            connect = random.randint(parent["r"], parent["r"] + parent["size"] - 1)
            offset = random.randint(0, room_size - 1)
            child_r = connect - offset
            if name == "right":
                child_c = parent["c"] + parent["size"] + gap
                corridor = [(connect, c) for c in range(parent["c"] + parent["size"], child_c)]
                parent_edge = (connect, parent["c"] + parent["size"] - 1)
                child_edge = (connect, child_c)
            else:
                child_c = parent["c"] - gap - room_size
                corridor = [(connect, c) for c in range(child_c + room_size, parent["c"])]
                parent_edge = (connect, parent["c"])
                child_edge = (connect, child_c + room_size - 1)
        else:
            connect = random.randint(parent["c"], parent["c"] + parent["size"] - 1)
            offset = random.randint(0, room_size - 1)
            child_c = connect - offset
            if name == "down":
                child_r = parent["r"] + parent["size"] + gap
                corridor = [(r, connect) for r in range(parent["r"] + parent["size"], child_r)]
                parent_edge = (parent["r"] + parent["size"] - 1, connect)
                child_edge = (child_r, connect)
            else:
                child_r = parent["r"] - gap - room_size
                corridor = [(r, connect) for r in range(child_r + room_size, parent["r"])]
                parent_edge = (parent["r"], connect)
                child_edge = (child_r + room_size - 1, connect)

        child = {"r": int(child_r), "c": int(child_c), "size": int(room_size)}
        child_cells = _room_cells(child)
        if child_cells & open_cells:
            continue
        if any(nei in open_cells for cell in child_cells for nei in _neighbors4(cell)):
            continue

        corridor_set = set(corridor)
        ok = True
        for cell in corridor:
            if cell in open_cells or cell in child_cells:
                ok = False
                break
            for nei in _neighbors4(cell):
                if nei in open_cells and nei not in corridor_set and nei not in {parent_edge, child_edge}:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        return child, corridor
    return None


def _add_spurs(
    rooms: list[dict],
    open_cells: set[tuple[int, int]],
    corridors: list[list[tuple[int, int]]],
) -> None:
    directions = {
        "right": (0, 1),
        "left": (0, -1),
        "down": (1, 0),
        "up": (-1, 0),
    }
    for _ in range(random.randint(0, 2)):
        room = random.choice(rooms)
        name = random.choice(list(directions))
        length = random.randint(1, 3)
        if name in ("right", "left"):
            row = random.randint(room["r"], room["r"] + room["size"] - 1)
            if name == "right":
                start = (row, room["c"] + room["size"])
                anchor = (row, room["c"] + room["size"] - 1)
            else:
                start = (row, room["c"] - 1)
                anchor = (row, room["c"])
        else:
            col = random.randint(room["c"], room["c"] + room["size"] - 1)
            if name == "down":
                start = (room["r"] + room["size"], col)
                anchor = (room["r"] + room["size"] - 1, col)
            else:
                start = (room["r"] - 1, col)
                anchor = (room["r"], col)

        spur = _line_cells(start, directions[name], length)
        spur_set = set(spur)
        ok = True
        for cell in spur:
            if cell in open_cells:
                ok = False
                break
            for nei in _neighbors4(cell):
                if nei in open_cells and nei not in spur_set and nei != anchor:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            open_cells.update(spur)
            corridors.append(spur)


def _build_layout() -> np.ndarray | None:
    room_count = random.randint(3, 5)
    sizes = random.sample([2, 3, 4, 5, 6], room_count)
    random.shuffle(sizes)

    rooms = [{"r": 0, "c": 0, "size": int(sizes[0])}]
    open_cells = set(_room_cells(rooms[0]))
    corridors: list[list[tuple[int, int]]] = []

    for size in sizes[1:]:
        placed = None
        for _ in range(120):
            placed = _try_attach(rooms, open_cells, int(size))
            if placed is not None:
                break
        if placed is None:
            return None
        room, corridor = placed
        rooms.append(room)
        open_cells.update(_room_cells(room))
        open_cells.update(corridor)
        corridors.append(corridor)

    _add_spurs(rooms, open_cells, corridors)

    rows = [r for r, _ in open_cells]
    cols = [c for _, c in open_cells]
    margin = 1
    H = max(rows) - min(rows) + 1 + 2 * margin
    W = max(cols) - min(cols) + 1 + 2 * margin
    if H > 30 or W > 30:
        return None

    shift_r = margin - min(rows)
    shift_c = margin - min(cols)
    grid = np.full((H, W), OBSTACLE, dtype=int)
    for r, c in open_cells:
        grid[r + shift_r, c + shift_c] = BACKGROUND

    room_sizes = sorted((room["size"] for room in _find_rooms(grid)), reverse=True)
    if room_sizes != sorted(sizes, reverse=True):
        return None

    out = main(grid)
    used_colors = {int(v) for v in np.unique(out)} - {BACKGROUND, OBSTACLE}
    if len(used_colors) != room_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-corridor puzzle")
