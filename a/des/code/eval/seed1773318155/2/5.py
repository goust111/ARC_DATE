from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 1
STAMP = 2
SEP = 3

LIB_WIDTH = 2
LIB_GAP = 1
SIDE_PRIORITY = ("up", "left", "right", "down")

GEN_ROWS_RANGE = (15, 24)
GEN_COLS_RANGE = (18, 28)
GEN_ROOM_COUNTS = (1, 2, 3)
GEN_OUTER_SIZES = (5, 7, 9)
GEN_LIBRARY_SHAPE = (2, 2, 3)
GEN_MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    if arr.size == 0:
        raise AssertionError("empty symbol")
    return arr - arr.min(axis=0, keepdims=True)


def _rotate_coords(coords: np.ndarray, turns: int) -> np.ndarray:
    out = _normalize(coords)
    for _ in range(int(turns) % 4):
        height = int(out[:, 0].max()) + 1
        out = np.column_stack((out[:, 1], height - 1 - out[:, 0]))
        out = _normalize(out)
    return out


def _facing_sides(coords: np.ndarray) -> set[str]:
    shape = _normalize(coords)
    area = int(shape.shape[0])
    height = int(shape[:, 0].max()) + 1
    width = int(shape[:, 1].max()) + 1
    cells = {tuple(map(int, rc)) for rc in shape}

    if area == 1:
        return set(SIDE_PRIORITY)
    if height == 1 or width == 1:
        return {"left", "right"} if width > height else {"up", "down"}
    if area == 3 and height == 2 and width == 2:
        missing = next((r, c) for r in range(2) for c in range(2) if (r, c) not in cells)
        facing: set[str] = set()
        facing.add("up" if missing[0] == 0 else "down")
        facing.add("left" if missing[1] == 0 else "right")
        return facing
    raise AssertionError("unsupported symbol shape")


def _lower_median(values: np.ndarray) -> int:
    ordered = np.sort(np.asarray(values, dtype=int))
    return int(ordered[(len(ordered) - 1) // 2])


def _rotate_to_face(coords: np.ndarray, target_side: str) -> np.ndarray:
    base = _normalize(coords)
    if base.shape[0] == 1:
        return base
    for turns in range(4):
        rotated = _rotate_coords(base, turns)
        if target_side in _facing_sides(rotated):
            return rotated
    raise AssertionError("unable to align symbol")


def _extract_red_symbols(arr: np.ndarray) -> dict[int, np.ndarray]:
    symbols: dict[int, np.ndarray] = {}
    for comp in _components(arr == STAMP):
        area = int(comp.shape[0])
        assert area in (1, 2, 3)
        assert area not in symbols
        symbols[area] = _normalize(comp)
    assert set(symbols) == {1, 2, 3}
    return symbols


def _extract_library_symbols(grid: np.ndarray) -> dict[int, np.ndarray]:
    strip = grid[:, -LIB_WIDTH:]
    assert set(map(int, np.unique(strip))) <= {BG, STAMP, SEP}
    sep_rows = [r for r in range(strip.shape[0]) if np.all(strip[r, :] == SEP)]
    assert len(sep_rows) >= 2
    assert not np.any(grid[:, :-LIB_WIDTH] == STAMP)
    return _extract_red_symbols(strip)


def _side_has_center_gap(values: np.ndarray) -> bool:
    vals = np.asarray(values, dtype=int)
    assert vals.ndim == 1 and vals.size >= 5 and vals.size % 2 == 1
    center = vals.size // 2
    bad = [i for i, value in enumerate(vals) if int(value) != WALL]
    if not bad:
        return False
    assert bad == [center]
    assert int(vals[center]) == BG
    return True


def _main_side(doors: tuple[str, ...]) -> str:
    for side in SIDE_PRIORITY:
        if side in doors:
            return side
    raise AssertionError("room has no doors")


def _validate_room(grid: np.ndarray, top: int, left: int, bottom: int, right: int) -> dict | None:
    if bottom - top < 4 or right - left < 4:
        return None
    if (bottom - top) % 2 or (right - left) % 2:
        return None
    if not np.all(grid[top + 1 : bottom, left + 1 : right] == BG):
        return None
    try:
        doors = []
        if _side_has_center_gap(grid[top, left : right + 1]):
            doors.append("up")
        if _side_has_center_gap(grid[bottom, left : right + 1]):
            doors.append("down")
        if _side_has_center_gap(grid[top : bottom + 1, left]):
            doors.append("left")
        if _side_has_center_gap(grid[top : bottom + 1, right]):
            doors.append("right")
    except AssertionError:
        return None

    if not 1 <= len(doors) <= 3:
        return None

    return {
        "top": int(top),
        "left": int(left),
        "bottom": int(bottom),
        "right": int(right),
        "center_r": int((top + bottom) // 2),
        "center_c": int((left + right) // 2),
        "door_count": int(len(doors)),
        "doors": tuple(doors),
        "main_side": _main_side(tuple(doors)),
    }


def _is_top_left_corner(grid: np.ndarray, r: int, c: int, room_col_limit: int) -> bool:
    h, w = grid.shape
    return (
        0 <= r + 1 < h
        and 0 <= c + 1 < room_col_limit
        and int(grid[r, c]) == WALL
        and int(grid[r, c + 1]) == WALL
        and int(grid[r + 1, c]) == WALL
        and (r == 0 or int(grid[r - 1, c]) != WALL)
        and (c == 0 or int(grid[r, c - 1]) != WALL)
    )


def _is_top_right_corner(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return (
        0 <= r + 1 < h
        and 0 <= c - 1
        and int(grid[r, c]) == WALL
        and int(grid[r, c - 1]) == WALL
        and int(grid[r + 1, c]) == WALL
        and (r == 0 or int(grid[r - 1, c]) != WALL)
        and (c + 1 == w or int(grid[r, c + 1]) != WALL)
    )


def _is_bottom_left_corner(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return (
        0 <= r - 1
        and 0 <= c + 1 < w
        and int(grid[r, c]) == WALL
        and int(grid[r - 1, c]) == WALL
        and int(grid[r, c + 1]) == WALL
        and (r + 1 == h or int(grid[r + 1, c]) != WALL)
        and (c == 0 or int(grid[r, c - 1]) != WALL)
    )


def _is_bottom_right_corner(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return (
        0 <= r - 1
        and 0 <= c - 1
        and int(grid[r, c]) == WALL
        and int(grid[r - 1, c]) == WALL
        and int(grid[r, c - 1]) == WALL
        and (r + 1 == h or int(grid[r + 1, c]) != WALL)
        and (c + 1 == w or int(grid[r, c + 1]) != WALL)
    )


def _find_rooms(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    room_col_limit = w - LIB_WIDTH
    top_lefts: list[tuple[int, int]] = []
    top_rights_by_row: dict[int, list[int]] = {}
    bottom_lefts_by_col: dict[int, list[int]] = {}
    bottom_rights: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(room_col_limit):
            if _is_top_left_corner(grid, r, c, room_col_limit):
                top_lefts.append((r, c))
            if _is_top_right_corner(grid, r, c):
                top_rights_by_row.setdefault(r, []).append(c)
            if _is_bottom_left_corner(grid, r, c):
                bottom_lefts_by_col.setdefault(c, []).append(r)
            if _is_bottom_right_corner(grid, r, c):
                bottom_rights.add((r, c))

    rooms: list[dict] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for top, left in top_lefts:
        rights = [c for c in top_rights_by_row.get(top, []) if c - left >= 4 and (c - left) % 2 == 0]
        bottoms = [r for r in bottom_lefts_by_col.get(left, []) if r - top >= 4 and (r - top) % 2 == 0]
        for right in rights:
            for bottom in bottoms:
                if (bottom, right) not in bottom_rights:
                    continue
                box = (top, left, bottom, right)
                if box in seen_boxes:
                    continue
                room = _validate_room(grid, top, left, bottom, right)
                if room is not None:
                    seen_boxes.add(box)
                    rooms.append(room)

    rooms.sort(key=lambda room: (room["top"], room["left"]))
    assert rooms
    return rooms


def _stamp_room(out: np.ndarray, room: dict, coords: np.ndarray) -> None:
    shape = _normalize(coords)
    anchor_r = _lower_median(shape[:, 0])
    anchor_c = _lower_median(shape[:, 1])
    rows = room["center_r"] + shape[:, 0] - anchor_r
    cols = room["center_c"] + shape[:, 1] - anchor_c
    assert np.all((room["top"] < rows) & (rows < room["bottom"]))
    assert np.all((room["left"] < cols) & (cols < room["right"]))
    assert np.all(out[rows, cols] == BG)
    out[rows, cols] = STAMP


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find blue rectangular rooms with 1..3 centered door gaps, select the
    red library symbol whose pixel count matches the door count, rotate it
    toward the room's priority door, and stamp it into the room center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    symbols = _extract_library_symbols(g)
    rooms = _find_rooms(g)

    out = g.copy()
    for room in rooms:
        symbol = _rotate_to_face(symbols[room["door_count"]], room["main_side"])
        _stamp_room(out, room, symbol)
    return out


def _build_library_strip() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    slot1, slot2, slot3 = GEN_LIBRARY_SHAPE
    height = slot1 + slot2 + slot3 + 2
    strip = np.full((height, LIB_WIDTH), BG, dtype=int)
    sep_rows = (slot1, slot1 + 1 + slot2)
    strip[sep_rows[0], :] = SEP
    strip[sep_rows[1], :] = SEP

    dot_r = random.randrange(slot1)
    dot_c = random.randrange(LIB_WIDTH)
    strip[dot_r, dot_c] = STAMP

    row2 = slot1 + 1
    if random.random() < 0.5:
        rr = row2 + random.randrange(slot2)
        strip[rr, :] = STAMP
    else:
        cc = random.randrange(LIB_WIDTH)
        strip[row2 : row2 + slot2, cc] = STAMP

    row3 = slot1 + 1 + slot2 + 1
    cc = random.randrange(LIB_WIDTH)
    strip[row3 : row3 + slot3, cc] = STAMP

    return strip, _extract_red_symbols(strip)


def _boxes_separate(a: dict, b: dict) -> bool:
    return (
        a["right"] + 1 < b["left"] - 1
        or b["right"] + 1 < a["left"] - 1
        or a["bottom"] + 1 < b["top"] - 1
        or b["bottom"] + 1 < a["top"] - 1
    )


def _sample_rooms(height: int, width: int) -> list[dict] | None:
    max_right = width - LIB_WIDTH - LIB_GAP - 1
    if max_right < 5:
        return None

    rooms: list[dict] = []
    room_target = random.choice(GEN_ROOM_COUNTS)
    for _ in range(room_target):
        placed = False
        for _ in range(120):
            outer_h = random.choice(GEN_OUTER_SIZES)
            outer_w = random.choice(GEN_OUTER_SIZES)
            if outer_h > height - 2 or outer_w > max_right - 1:
                continue

            top_max = height - outer_h - 1
            left_max = max_right - outer_w + 1
            if top_max < 1 or left_max < 1:
                continue

            top = random.randint(1, top_max)
            left = random.randint(1, left_max)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            door_count = random.choice((1, 2, 3))
            doors = tuple(random.sample(list(SIDE_PRIORITY), door_count))
            room = {
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "center_r": (top + bottom) // 2,
                "center_c": (left + right) // 2,
                "door_count": door_count,
                "doors": doors,
                "main_side": _main_side(doors),
            }
            if all(_boxes_separate(room, other) for other in rooms):
                rooms.append(room)
                placed = True
                break
        if not placed:
            return None

    rooms.sort(key=lambda room: (room["top"], room["left"]))
    return rooms


def _draw_room(grid: np.ndarray, room: dict) -> None:
    top = room["top"]
    left = room["left"]
    bottom = room["bottom"]
    right = room["right"]
    center_r = room["center_r"]
    center_c = room["center_c"]

    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL

    if "up" in room["doors"]:
        grid[top, center_c] = BG
    if "down" in room["doors"]:
        grid[bottom, center_c] = BG
    if "left" in room["doors"]:
        grid[center_r, left] = BG
    if "right" in room["doors"]:
        grid[center_r, right] = BG


def _manual_expected(inp: np.ndarray, rooms: list[dict], symbols: dict[int, np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        symbol = _rotate_to_face(symbols[room["door_count"]], room["main_side"])
        _stamp_room(out, room, symbol)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_ROWS_RANGE)
    width = random.randint(*GEN_COLS_RANGE)
    library_strip, symbols = _build_library_strip()
    if library_strip.shape[0] > height:
        return None

    rooms = _sample_rooms(height, width)
    if rooms is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    lib_top = random.randint(0, height - library_strip.shape[0])
    grid[lib_top : lib_top + library_strip.shape[0], -LIB_WIDTH:] = library_strip
    for room in rooms:
        _draw_room(grid, room)

    expected = _manual_expected(grid, rooms, symbols)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
