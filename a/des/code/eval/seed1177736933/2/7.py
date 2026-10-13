from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
PUSH_FROM_SIDE = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}
DOOR_SIDE_FROM_PUSH = {value: key for key, value in PUSH_FROM_SIDE.items()}


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_rooms(grid: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    for coords in _components(grid == BLUE, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue

        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True

        actual = grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE
        if np.any(actual & ~expected):
            continue

        missing = np.argwhere(expected & ~actual)
        if missing.shape != (1, 2):
            continue
        mr, mc = map(int, missing[0])
        if (mr, mc) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
            continue

        if mr == 0:
            door_side = "top"
        elif mr == h - 1:
            door_side = "bottom"
        elif mc == 0:
            door_side = "left"
        else:
            door_side = "right"

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "door": (r0 + mr, c0 + mc),
                "door_side": door_side,
            }
        )

    rooms.sort(key=lambda room: room["bbox"])
    assert rooms, "expected at least one blue room"
    return rooms


def _extract_furniture(grid: np.ndarray, room: dict, bg: int) -> dict:
    r0, c0, r1, c1 = room["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != bg) & (inner != BLUE)
    comps = _components(mask, connectivity=4)
    assert len(comps) == 1, "each room must contain one furniture component"

    coords = comps[0] + np.array([r0 + 1, c0 + 1], dtype=int)
    values = np.unique(grid[coords[:, 0], coords[:, 1]])
    assert len(values) == 1, "furniture must be monochrome"
    return {"coords": coords, "color": int(values[0])}


def _parse_arrow(coords: np.ndarray) -> dict | None:
    if coords.shape[0] != 4:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rel = coords - np.array([r0, c0], dtype=int)
    patch = np.zeros((h, w), dtype=bool)
    patch[rel[:, 0], rel[:, 1]] = True

    if (h, w) == (2, 3):
        row_counts = patch.sum(axis=1)
        if sorted(map(int, row_counts)) != [1, 3]:
            return None
        line_row = int(np.argmax(row_counts))
        tip_row = 1 - line_row
        tip_cols = np.flatnonzero(patch[tip_row])
        if tip_cols.size != 1:
            return None
        tip_col = int(tip_cols[0])
        if tip_col == 0:
            direction = (0, -1)
            front = (r0 + line_row, c0)
        elif tip_col == 2:
            direction = (0, 1)
            front = (r0 + line_row, c1)
        else:
            return None
        return {"coords": coords, "direction": direction, "front": front}

    if (h, w) == (3, 2):
        col_counts = patch.sum(axis=0)
        if sorted(map(int, col_counts)) != [1, 3]:
            return None
        line_col = int(np.argmax(col_counts))
        tip_col = 1 - line_col
        tip_rows = np.flatnonzero(patch[:, tip_col])
        if tip_rows.size != 1:
            return None
        tip_row = int(tip_rows[0])
        if tip_row == 0:
            direction = (-1, 0)
            front = (r0, c0 + line_col)
        elif tip_row == 2:
            direction = (1, 0)
            front = (r1, c0 + line_col)
        else:
            return None
        return {"coords": coords, "direction": direction, "front": front}

    return None


def _find_arrows(grid: np.ndarray, rooms: list[dict]) -> list[dict]:
    arrow_mask = grid == RED
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        arrow_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    arrows: list[dict] = []
    for coords in _components(arrow_mask, connectivity=8):
        arrow = _parse_arrow(coords)
        assert arrow is not None, "all outside red components must be arrows"
        arrows.append(arrow)

    arrows.sort(key=lambda arrow: (arrow["front"], arrow["direction"]))
    assert arrows, "expected at least one arrow"
    return arrows


def _match_arrow_to_room(arrow: dict, rooms: list[dict], grid: np.ndarray, bg: int) -> int:
    dr, dc = arrow["direction"]
    front_r, front_c = arrow["front"]
    expected_side = DOOR_SIDE_FROM_PUSH[(dr, dc)]
    candidates: list[tuple[int, int]] = []

    for idx, room in enumerate(rooms):
        if room["door_side"] != expected_side:
            continue
        door_r, door_c = room["door"]

        if dr != 0:
            if door_c != front_c:
                continue
            dist = (door_r - front_r) * dr
        else:
            if door_r != front_r:
                continue
            dist = (door_c - front_c) * dc
        if dist < 1:
            continue

        clear = True
        for step in range(1, dist + 1):
            rr = front_r + dr * step
            cc = front_c + dc * step
            if int(grid[rr, cc]) != bg:
                clear = False
                break
        if clear:
            candidates.append((dist, idx))

    assert candidates, "each arrow must face a room door"
    candidates.sort()
    assert len(candidates) == 1 or candidates[0][0] < candidates[1][0], "arrow-room match must be unique"
    return int(candidates[0][1])


def _translated_coords(room: dict, furniture: dict, direction: tuple[int, int]) -> np.ndarray:
    coords = furniture["coords"]
    r0, c0, r1, c1 = room["bbox"]
    dr, dc = direction

    if (dr, dc) == (0, 1):
        delta = np.array([0, (c1 - 1) - int(coords[:, 1].max())], dtype=int)
    elif (dr, dc) == (0, -1):
        delta = np.array([0, (c0 + 1) - int(coords[:, 1].min())], dtype=int)
    elif (dr, dc) == (1, 0):
        delta = np.array([(r1 - 1) - int(coords[:, 0].max()), 0], dtype=int)
    else:
        delta = np.array([(r0 + 1) - int(coords[:, 0].min()), 0], dtype=int)
    return coords + delta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rooms are blue rectangular borders with a one-pixel door gap.
    A red arrow sits outside a door and points inward.
    Move that room's furniture in the arrow direction until it touches the opposite inner wall.
    Remove the arrows and keep the rooms unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BLACK

    rooms = _find_rooms(g)
    for room in rooms:
        room["furniture"] = _extract_furniture(g, room, bg)

    arrows = _find_arrows(g, rooms)
    out = g.copy()
    used_rooms: set[int] = set()

    for arrow in arrows:
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for arrow in arrows:
        room_idx = _match_arrow_to_room(arrow, rooms, g, bg)
        assert room_idx not in used_rooms, "each room should receive at most one arrow"
        used_rooms.add(room_idx)

        room = rooms[room_idx]
        furniture = room["furniture"]
        moved = _translated_coords(room, furniture, arrow["direction"])
        out[furniture["coords"][:, 0], furniture["coords"][:, 1]] = bg
        out[moved[:, 0], moved[:, 1]] = furniture["color"]

    return out


def _crop_cells(cells: set[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    shifted = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        patch[r, c] = True
    return patch, shifted


def _random_shape() -> np.ndarray:
    area = random.randint(2, 5)
    dims = [(h, w) for h in range(1, 4) for w in range(1, 5) if h * w >= area]
    h, w = random.choice(dims)
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_shape()
        cells.add(random.choice(frontier))
    patch, _shifted = _crop_cells(cells)
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_room_size(shape: np.ndarray, side: str) -> tuple[int, int]:
    sh, sw = shape.shape
    min_h = max(5, sh + 2)
    min_w = max(5, sw + 2)

    if side in {"left", "right"}:
        min_h = max(min_h, 5)
        room_h_choices = [x for x in range(min_h, 10) if x % 2 == 1]
        room_w_choices = [x for x in range(max(min_w, sw + 3), 11)]
    else:
        room_h_choices = [x for x in range(max(min_h, sh + 3), 11)]
        room_w_choices = [x for x in range(min_w, 10) if x % 2 == 1]

    assert room_h_choices and room_w_choices
    return random.choice(room_h_choices), random.choice(room_w_choices)


def _reserve_box(bbox: tuple[int, int, int, int], pad: int = 4) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 - pad, c0 - pad, r1 + pad, c1 + pad)


def _draw_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    dr, dc = room["door"]
    grid[dr, dc] = BLACK

    fr, fc = room["furniture_origin"]
    shape = room["shape"]
    color = room["color"]
    for rr, cc in np.argwhere(shape):
        grid[fr + int(rr), fc + int(cc)] = color


def _draw_arrow(grid: np.ndarray, room: dict) -> np.ndarray:
    door_r, door_c = room["door"]
    side = room["door_side"]
    gap = room["gap"]
    tip_sign = room["tip_sign"]
    cells: list[tuple[int, int]] = []

    if side == "left":
        front_c = door_c - gap
        row = door_r
        cells.extend((row, front_c - k) for k in range(3))
        cells.append((row + tip_sign, front_c))
    elif side == "right":
        front_c = door_c + gap
        row = door_r
        cells.extend((row, front_c + k) for k in range(3))
        cells.append((row + tip_sign, front_c))
    elif side == "top":
        front_r = door_r - gap
        col = door_c
        cells.extend((front_r - k, col) for k in range(3))
        cells.append((front_r, col + tip_sign))
    else:
        front_r = door_r + gap
        col = door_c
        cells.extend((front_r + k, col) for k in range(3))
        cells.append((front_r, col + tip_sign))

    coords = np.array(sorted(set(cells)), dtype=int)
    assert coords.shape == (4, 2)
    grid[coords[:, 0], coords[:, 1]] = RED
    return coords


def _sample_room_plan(h: int, w: int, reserved: list[tuple[int, int, int, int]]) -> dict | None:
    shape = _random_shape()
    side = random.choice(["left", "right", "top", "bottom"])
    room_h, room_w = _choose_room_size(shape, side)
    sh, sw = shape.shape

    row_lo = 4
    row_hi = h - room_h - 4
    col_lo = 4
    col_hi = w - room_w - 4
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(200):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        reserve = _reserve_box(bbox)
        if reserve[0] < 0 or reserve[1] < 0 or reserve[2] >= h or reserve[3] >= w:
            continue
        if any(_boxes_overlap(reserve, other) for other in reserved):
            continue

        door = {
            "left": (r0 + room_h // 2, c0),
            "right": (r0 + room_h // 2, c0 + room_w - 1),
            "top": (r0, c0 + room_w // 2),
            "bottom": (r0 + room_h - 1, c0 + room_w // 2),
        }[side]

        inner_top = r0 + 1
        inner_left = c0 + 1
        inner_h = room_h - 2
        inner_w = room_w - 2

        if side == "left":
            fr = inner_top + random.randint(0, inner_h - sh)
            fc = inner_left
        elif side == "right":
            fr = inner_top + random.randint(0, inner_h - sh)
            fc = inner_left + inner_w - sw
        elif side == "top":
            fr = inner_top
            fc = inner_left + random.randint(0, inner_w - sw)
        else:
            fr = inner_top + inner_h - sh
            fc = inner_left + random.randint(0, inner_w - sw)

        plan = {
            "bbox": bbox,
            "door": door,
            "door_side": side,
            "shape": shape,
            "furniture_origin": (fr, fc),
            "color": random.randint(3, 9),
            "gap": random.randint(1, 2),
            "tip_sign": random.choice([-1, 1]),
            "direction": PUSH_FROM_SIDE[side],
        }
        return plan

    return None


def _build_expected(inp: np.ndarray, rooms: list[dict], arrows: list[np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for coords in arrows:
        out[coords[:, 0], coords[:, 1]] = BLACK

    for room in rooms:
        fr, fc = room["furniture_origin"]
        shape = room["shape"]
        color = room["color"]
        shape_coords = np.argwhere(shape)
        original = shape_coords + np.array([fr, fc], dtype=int)
        out[original[:, 0], original[:, 1]] = BLACK

        r0, c0, r1, c1 = room["bbox"]
        dr, dc = room["direction"]
        if (dr, dc) == (0, 1):
            delta = np.array([0, (c1 - 1) - int(original[:, 1].max())], dtype=int)
        elif (dr, dc) == (0, -1):
            delta = np.array([0, (c0 + 1) - int(original[:, 1].min())], dtype=int)
        elif (dr, dc) == (1, 0):
            delta = np.array([(r1 - 1) - int(original[:, 0].max()), 0], dtype=int)
        else:
            delta = np.array([(r0 + 1) - int(original[:, 0].min()), 0], dtype=int)
        moved = original + delta
        out[moved[:, 0], moved[:, 1]] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    grid = np.full((h, w), BLACK, dtype=int)
    room_count = random.randint(1, 3 if min(h, w) < 24 else 4)

    reserved: list[tuple[int, int, int, int]] = []
    rooms: list[dict] = []
    for _ in range(room_count):
        plan = _sample_room_plan(h, w, reserved)
        if plan is None:
            return None
        rooms.append(plan)
        reserved.append(_reserve_box(plan["bbox"]))

    arrow_coords: list[np.ndarray] = []
    for room in rooms:
        _draw_room(grid, room)
    for room in rooms:
        arrow = _draw_arrow(grid, room)
        arrow_coords.append(arrow)

    expected = _build_expected(grid, rooms, arrow_coords)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-push puzzle")
