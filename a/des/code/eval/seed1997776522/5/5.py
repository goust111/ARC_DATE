import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
MAGENTA = 6

SOURCE_COLORS = (1, 3, 7, 8, 9)

GRID_MIN = 18
GRID_MAX = 30
ROOM_COUNT_MIN = 1
ROOM_COUNT_MAX = 4
ROOM_PADDING = 3
OUTER_SIZE_MIN = 5
OUTER_SIZE_MAX = 11
GEN_RETRIES = 400


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _find_corners(grid: np.ndarray) -> tuple[list[tuple[int, int]], ...]:
    h, w = grid.shape
    tl: list[tuple[int, int]] = []
    tr: list[tuple[int, int]] = []
    bl: list[tuple[int, int]] = []
    br: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY:
                continue
            if (
                r + 1 < h
                and c + 1 < w
                and int(grid[r, c + 1]) == GREY
                and int(grid[r + 1, c]) == GREY
                and int(grid[r + 1, c + 1]) == BLACK
            ):
                tl.append((r, c))
            if (
                r + 1 < h
                and c - 1 >= 0
                and int(grid[r, c - 1]) == GREY
                and int(grid[r + 1, c]) == GREY
                and int(grid[r + 1, c - 1]) == BLACK
            ):
                tr.append((r, c))
            if (
                r - 1 >= 0
                and c + 1 < w
                and int(grid[r, c + 1]) == GREY
                and int(grid[r - 1, c]) == GREY
                and int(grid[r - 1, c + 1]) == BLACK
            ):
                bl.append((r, c))
            if (
                r - 1 >= 0
                and c - 1 >= 0
                and int(grid[r, c - 1]) == GREY
                and int(grid[r - 1, c]) == GREY
                and int(grid[r - 1, c - 1]) == BLACK
            ):
                br.append((r, c))
    return tl, tr, bl, br


def _door_source(
    grid: np.ndarray,
    r: int,
    c: int,
    dr_out: int,
    dc_out: int,
    dr_in: int,
    dc_in: int,
) -> int | None:
    h, w = grid.shape
    ro, co = r + dr_out, c + dc_out
    ri, ci = r + dr_in, c + dc_in
    if not (_inside(h, w, ro, co) and _inside(h, w, ri, ci)):
        return None
    source = int(grid[ro, co])
    if source in (BLACK, GREY):
        return None
    if int(grid[ri, ci]) != BLACK:
        return None
    return source


def _analyze_room(
    grid: np.ndarray,
    prefix: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict | None:
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    if _rect_sum(prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
        return None

    doors: list[dict] = []

    for c in range(c0 + 1, c1):
        v = int(grid[r0, c])
        if v == GREY:
            continue
        if v != BLACK:
            return None
        source = _door_source(grid, r0, c, -1, 0, 1, 0)
        if source is None:
            return None
        doors.append({"pos": (r0, c), "dir": (1, 0), "source": source})

    for c in range(c0 + 1, c1):
        v = int(grid[r1, c])
        if v == GREY:
            continue
        if v != BLACK:
            return None
        source = _door_source(grid, r1, c, 1, 0, -1, 0)
        if source is None:
            return None
        doors.append({"pos": (r1, c), "dir": (-1, 0), "source": source})

    for r in range(r0 + 1, r1):
        v = int(grid[r, c0])
        if v == GREY:
            continue
        if v != BLACK:
            return None
        source = _door_source(grid, r, c0, 0, -1, 0, 1)
        if source is None:
            return None
        doors.append({"pos": (r, c0), "dir": (0, 1), "source": source})

    for r in range(r0 + 1, r1):
        v = int(grid[r, c1])
        if v == GREY:
            continue
        if v != BLACK:
            return None
        source = _door_source(grid, r, c1, 0, 1, 0, -1)
        if source is None:
            return None
        doors.append({"pos": (r, c1), "dir": (0, -1), "source": source})

    if not doors:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "doors": doors,
    }


def _find_rooms(grid: np.ndarray) -> list[dict]:
    prefix = _prefix_sum(grid != BLACK)
    tl, tr, bl, br = _find_corners(grid)

    tr_by_row: dict[int, list[int]] = defaultdict(list)
    bl_by_col: dict[int, list[int]] = defaultdict(list)
    br_set = set(br)

    for r, c in tr:
        tr_by_row[r].append(c)
    for r, c in bl:
        bl_by_col[c].append(r)

    rooms: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r0, c0 in sorted(tl):
        cols = [c1 for c1 in tr_by_row.get(r0, []) if c1 >= c0 + 4]
        rows = [r1 for r1 in bl_by_col.get(c0, []) if r1 >= r0 + 4]
        for r1 in rows:
            for c1 in cols:
                bbox = (r0, c0, r1, c1)
                if bbox in seen or (r1, c1) not in br_set:
                    continue
                room = _analyze_room(grid, prefix, r0, c0, r1, c1)
                if room is not None:
                    rooms.append(room)
                    seen.add(bbox)

    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color room interiors by door-count class and mark ray intersections in magenta.

    Input structure:
    - Black background.
    - Grey rectangular rooms with one-pixel-thick walls and one or more wall gaps.
    - Each gap has a colored source pixel immediately outside it.

    Rule:
    - A 1-door room uses that source color.
    - A 2-door room uses red.
    - A 3+-door room uses yellow.
    - From every door, cast a ray inward along the wall normal across the room interior.
    - Cells hit by multiple rays become magenta.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _find_rooms(g):
        r0, c0, r1, c1 = room["bbox"]
        doors = room["doors"]
        if len(doors) == 1:
            fill = int(doors[0]["source"])
        elif len(doors) == 2:
            fill = RED
        else:
            fill = YELLOW

        hits: dict[tuple[int, int], int] = {}
        for door in doors:
            r, c = door["pos"]
            dr, dc = door["dir"]
            rr, cc = r + dr, c + dc
            while r0 < rr < r1 and c0 < cc < c1:
                hits[(rr, cc)] = hits.get((rr, cc), 0) + 1
                rr += dr
                cc += dc

        for (r, c), count in hits.items():
            out[r, c] = MAGENTA if count > 1 else fill

    return out


def _choose_door_count(rng: random.Random) -> int:
    choice = rng.choice((1, 2, 3))
    if choice < 3:
        return choice
    return rng.randint(3, 4)


def _choose_door_sides(rng: random.Random, door_count: int) -> list[str]:
    if door_count == 1:
        return [rng.choice(("top", "bottom", "left", "right"))]
    if door_count == 2:
        if rng.random() < 0.5:
            return [rng.choice(("top", "bottom")), rng.choice(("left", "right"))]
        return list(rng.choice((("top", "bottom"), ("left", "right"))))
    return rng.sample(["top", "bottom", "left", "right"], door_count)


def _choose_source_color(rng: random.Random, used: list[int]) -> int:
    if used and rng.random() < 0.35:
        return rng.choice(used)
    return rng.choice(SOURCE_COLORS)


def _make_room_spec(rng: random.Random, h: int, w: int) -> dict:
    door_count = _choose_door_count(rng)
    min_outer = 6 if door_count >= 3 else OUTER_SIZE_MIN
    outer_h = rng.randint(min_outer, min(OUTER_SIZE_MAX, h - 2 * ROOM_PADDING))
    outer_w = rng.randint(min_outer, min(OUTER_SIZE_MAX, w - 2 * ROOM_PADDING))
    return {
        "door_count": door_count,
        "outer_h": outer_h,
        "outer_w": outer_w,
    }


def _candidate_positions(
    reserve: np.ndarray, outer_h: int, outer_w: int
) -> list[tuple[int, int]]:
    h, w = reserve.shape
    positions: list[tuple[int, int]] = []
    r_stop = h - outer_h - ROOM_PADDING + 1
    c_stop = w - outer_w - ROOM_PADDING + 1
    for r0 in range(ROOM_PADDING, r_stop):
        for c0 in range(ROOM_PADDING, c_stop):
            if reserve[
                r0 - ROOM_PADDING : r0 + outer_h + ROOM_PADDING,
                c0 - ROOM_PADDING : c0 + outer_w + ROOM_PADDING,
            ].any():
                continue
            positions.append((r0, c0))
    return positions


def _choose_doors_for_room(
    rng: random.Random,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    door_count: int,
) -> list[dict]:
    sides = _choose_door_sides(rng, door_count)
    doors_by_side: dict[str, dict] = {}

    if "top" in sides and "bottom" in sides and rng.random() < 0.4:
        col = rng.randint(c0 + 1, c1 - 1)
        doors_by_side["top"] = {"pos": (r0, col), "dir": (1, 0), "src": (r0 - 1, col)}
        doors_by_side["bottom"] = {"pos": (r1, col), "dir": (-1, 0), "src": (r1 + 1, col)}
    if "left" in sides and "right" in sides and rng.random() < 0.4:
        row = rng.randint(r0 + 1, r1 - 1)
        doors_by_side["left"] = {"pos": (row, c0), "dir": (0, 1), "src": (row, c0 - 1)}
        doors_by_side["right"] = {"pos": (row, c1), "dir": (0, -1), "src": (row, c1 + 1)}

    for side in sides:
        if side in doors_by_side:
            continue
        if side == "top":
            col = rng.randint(c0 + 1, c1 - 1)
            doors_by_side[side] = {"pos": (r0, col), "dir": (1, 0), "src": (r0 - 1, col)}
        elif side == "bottom":
            col = rng.randint(c0 + 1, c1 - 1)
            doors_by_side[side] = {"pos": (r1, col), "dir": (-1, 0), "src": (r1 + 1, col)}
        elif side == "left":
            row = rng.randint(r0 + 1, r1 - 1)
            doors_by_side[side] = {"pos": (row, c0), "dir": (0, 1), "src": (row, c0 - 1)}
        else:
            row = rng.randint(r0 + 1, r1 - 1)
            doors_by_side[side] = {"pos": (row, c1), "dir": (0, -1), "src": (row, c1 + 1)}

    used_colors: list[int] = []
    doors: list[dict] = []
    for side in ("top", "bottom", "left", "right"):
        if side not in doors_by_side:
            continue
        info = doors_by_side[side]
        color = _choose_source_color(rng, used_colors)
        used_colors.append(color)
        doors.append(
            {
                "pos": info["pos"],
                "dir": info["dir"],
                "src": info["src"],
                "color": color,
            }
        )
    return doors


def _paint_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for door in room["doors"]:
        r, c = door["pos"]
        sr, sc = door["src"]
        grid[r, c] = BLACK
        grid[sr, sc] = door["color"]


def _place_rooms(rng: random.Random, grid: np.ndarray, room_count: int) -> list[dict] | None:
    h, w = grid.shape
    reserve = np.zeros((h, w), dtype=bool)
    rooms: list[dict] = []

    for _ in range(room_count):
        spec = _make_room_spec(rng, h, w)
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        positions = _candidate_positions(reserve, outer_h, outer_w)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        doors = _choose_doors_for_room(rng, r0, c0, r1, c1, spec["door_count"])
        room = {
            "bbox": (r0, c0, r1, c1),
            "doors": doors,
        }
        _paint_room(grid, room)
        reserve[
            r0 - ROOM_PADDING : r0 + outer_h + ROOM_PADDING,
            c0 - ROOM_PADDING : c0 + outer_w + ROOM_PADDING,
        ] = True
        rooms.append(room)

    return rooms


def _grid_shape_for_rooms(rng: random.Random, room_count: int) -> tuple[int, int]:
    min_side = max(GRID_MIN, 12 + 4 * room_count)
    max_side = max(min_side, GRID_MAX)
    return rng.randint(min_side, max_side), rng.randint(min_side, max_side)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.randint(ROOM_COUNT_MIN, ROOM_COUNT_MAX)
    h, w = _grid_shape_for_rooms(rng, room_count)
    grid = np.full((h, w), BLACK, dtype=int)

    rooms = _place_rooms(rng, grid, room_count)
    if rooms is None:
        return None

    detected = _find_rooms(grid)
    expected_boxes = sorted(room["bbox"] for room in rooms)
    detected_boxes = sorted(room["bbox"] for room in detected)
    if detected_boxes != expected_boxes:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid room layout")
