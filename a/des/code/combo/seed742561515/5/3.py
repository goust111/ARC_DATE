from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1

DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

KEY_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]

# --------------- core helpers ---------------


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _rect_border_is_blue(g: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return bool(
        np.all(g[r0, c0 : c1 + 1] == BLUE)
        and np.all(g[r1, c0 : c1 + 1] == BLUE)
        and np.all(g[r0 : r1 + 1, c0] == BLUE)
        and np.all(g[r0 : r1 + 1, c1] == BLUE)
    )


def _sorted_unique(values: list[int]) -> list[int]:
    return sorted(set(values))


# --------------- solver ---------------


def _find_rooms(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    rooms: list[dict] = []
    for comp in _components_mask(g != BLUE):
        if _touches_edge(comp, h, w):
            continue
        ir0, ic0, ir1, ic1 = _bbox(comp)
        if len(comp) != (ir1 - ir0 + 1) * (ic1 - ic0 + 1):
            continue
        if np.any(g[ir0 : ir1 + 1, ic0 : ic1 + 1] == BLUE):
            continue
        wr0, wc0, wr1, wc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        assert wr0 >= 0 and wc0 >= 0 and wr1 < h and wc1 < w
        assert _rect_border_is_blue(g, wr0, wc0, wr1, wc1), "room wall must be a blue rectangle"
        rooms.append(
            {
                "interior_bbox": (ir0, ic0, ir1, ic1),
                "wall_bbox": (wr0, wc0, wr1, wc1),
            }
        )

    rooms.sort(key=lambda room: room["wall_bbox"])
    assert rooms, "expected at least one blue room"
    return rooms


def _longest_segment(coords: list[tuple[int, int]]) -> dict:
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r, c in coords:
        row_to_cols.setdefault(r, []).append(c)
        col_to_rows.setdefault(c, []).append(r)

    segments: list[dict] = []
    for r, cols in row_to_cols.items():
        cols = sorted(cols)
        start = 0
        while start < len(cols):
            end = start
            while end + 1 < len(cols) and cols[end + 1] == cols[end] + 1:
                end += 1
            run = cols[start : end + 1]
            if len(run) >= 2:
                segments.append(
                    {
                        "orientation": "h",
                        "coords": [(r, c) for c in run],
                    }
                )
            start = end + 1

    for c, rows in col_to_rows.items():
        rows = sorted(rows)
        start = 0
        while start < len(rows):
            end = start
            while end + 1 < len(rows) and rows[end + 1] == rows[end] + 1:
                end += 1
            run = rows[start : end + 1]
            if len(run) >= 2:
                segments.append(
                    {
                        "orientation": "v",
                        "coords": [(r, c) for r in run],
                    }
                )
            start = end + 1

    assert segments, "key must contain a straight segment"
    max_len = max(len(seg["coords"]) for seg in segments)
    candidates = [seg for seg in segments if len(seg["coords"]) == max_len]
    assert len(candidates) == 1, "key stem must be unique"
    return candidates[0]


def _analyze_key(
    key_coords: list[tuple[int, int]],
    color: int,
    room: dict,
) -> dict:
    stem = _longest_segment(key_coords)
    stem_coords = stem["coords"]
    stem_set = set(stem_coords)
    extras = [coord for coord in key_coords if coord not in stem_set]
    ir0, ic0, ir1, ic1 = room["interior_bbox"]
    wr0, wc0, wr1, wc1 = room["wall_bbox"]

    if stem["orientation"] == "h":
        row = stem_coords[0][0]
        cols = [c for _, c in stem_coords]
        cols_sorted = list(range(min(cols), max(cols) + 1))
        assert _sorted_unique(cols) == cols_sorted
        if extras:
            offsets = {er - row for er, _ in extras}
            assert len(offsets) == 1, "all teeth must point the same way"
            drow = next(iter(offsets))
            assert abs(drow) == 1
            assert all(ec in cols_sorted for _er, ec in extras)
            direction = "up" if drow < 0 else "down"
        else:
            # For straight-line keys, only accept a unique nearer perpendicular wall.
            dist_up = row - ir0
            dist_down = ir1 - row
            assert dist_up != dist_down, "straight key direction is ambiguous"
            direction = "up" if dist_up < dist_down else "down"
        wall_row = wr0 if direction == "up" else wr1
        door = [(wall_row, c) for c in cols_sorted]
    else:
        col = stem_coords[0][1]
        rows = [r for r, _ in stem_coords]
        rows_sorted = list(range(min(rows), max(rows) + 1))
        assert _sorted_unique(rows) == rows_sorted
        if extras:
            offsets = {ec - col for _, ec in extras}
            assert len(offsets) == 1, "all teeth must point the same way"
            dcol = next(iter(offsets))
            assert abs(dcol) == 1
            assert all(er in rows_sorted for er, _ec in extras)
            direction = "left" if dcol < 0 else "right"
        else:
            dist_left = col - ic0
            dist_right = ic1 - col
            assert dist_left != dist_right, "straight key direction is ambiguous"
            direction = "left" if dist_left < dist_right else "right"
        wall_col = wc0 if direction == "left" else wc1
        door = [(r, wall_col) for r in rows_sorted]

    return {
        "color": color,
        "direction": direction,
        "door": door,
    }


def _room_instruction(g: np.ndarray, room: dict) -> dict:
    ir0, ic0, ir1, ic1 = room["interior_bbox"]
    window = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    key_mask = window != BLACK
    assert key_mask.any(), "each room must contain a key"
    key_components = _components_mask(key_mask)
    assert len(key_components) == 1, "expected exactly one key per room"
    rel_coords = key_components[0]
    key_coords = [(ir0 + r, ic0 + c) for r, c in rel_coords]
    colors = np.unique(window[key_mask])
    assert colors.size == 1, "key must be monochrome"
    color = int(colors[0])
    assert color not in (BLACK, BLUE)
    return _analyze_key(key_coords, color, room)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open a door in each blue room according to the key inside it, then emit rays of the key color.

    Input structure:
    - black background outside the rooms.
    - one-pixel-thick blue rectangular room walls.
    - exactly one non-black, non-blue key shape inside each room.

    Rule:
    - find the key's longest straight segment as its stem;
    - use the perpendicular tooth direction to choose the target wall;
    - erase a same-length door aligned with the stem on that wall;
    - cast same-colored rays outward from the door opening until a blue wall or the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rooms = _find_rooms(g)
    instructions = [_room_instruction(g, room) for room in rooms]

    out = g.copy()
    all_doors = {coord for item in instructions for coord in item["door"]}
    for r, c in all_doors:
        assert out[r, c] == BLUE, "door must lie on a blue wall"
        out[r, c] = BLACK

    for item in instructions:
        color = item["color"]
        dr, dc = DIR_TO_DELTA[item["direction"]]
        for door_r, door_c in item["door"]:
            rr, cc = door_r + dr, door_c + dc
            while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] != BLUE:
                if g[rr, cc] == BLACK:
                    if out[rr, cc] in (BLACK, color):
                        out[rr, cc] = color
                    else:
                        raise AssertionError("different rays overlap on a background cell")
                rr += dr
                cc += dc

    return out


# --------------- generator ---------------

ROOM_COUNT_CHOICES = [1, 2, 3, 4]
ROOM_COUNT_WEIGHTS = [2, 4, 4, 2]

GRID_SIZE_BY_ROOM_COUNT = {
    1: ((14, 20), (14, 22)),
    2: ((16, 24), (16, 26)),
    3: ((18, 28), (20, 30)),
    4: ((22, 30), (22, 34)),
}

ROOM_H_RANGE = (6, 10)
ROOM_W_RANGE = (6, 11)
PLACEMENT_MOAT = 1
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250


def _rand_int(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def _position_allows_ray(
    h: int,
    w: int,
    top: int,
    left: int,
    room_h: int,
    room_w: int,
    direction: str,
) -> bool:
    bottom = top + room_h - 1
    right = left + room_w - 1
    if direction == "up":
        return top >= 2
    if direction == "down":
        return bottom <= h - 3
    if direction == "left":
        return left >= 2
    return right <= w - 3


def _can_place(
    occupied: np.ndarray,
    top: int,
    left: int,
    room_h: int,
    room_w: int,
) -> bool:
    h, w = occupied.shape
    bottom = top + room_h - 1
    right = left + room_w - 1
    if bottom >= h or right >= w:
        return False
    r0 = max(0, top - PLACEMENT_MOAT)
    c0 = max(0, left - PLACEMENT_MOAT)
    r1 = min(h, bottom + PLACEMENT_MOAT + 1)
    c1 = min(w, right + PLACEMENT_MOAT + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _draw_room(g: np.ndarray, top: int, left: int, room_h: int, room_w: int) -> None:
    bottom = top + room_h - 1
    right = left + room_w - 1
    g[top, left : right + 1] = BLUE
    g[bottom, left : right + 1] = BLUE
    g[top : bottom + 1, left] = BLUE
    g[top : bottom + 1, right] = BLUE


def _place_key(
    g: np.ndarray,
    top: int,
    left: int,
    room_h: int,
    room_w: int,
    direction: str,
    tooth_index: int,
    color: int,
    rng: random.Random,
) -> None:
    ir0, ic0 = top + 1, left + 1
    ir1, ic1 = top + room_h - 2, left + room_w - 2
    if direction in ("up", "down"):
        stem_c0 = rng.randint(ic0, ic1 - 2)
        if direction == "up":
            stem_r = rng.randint(ir0 + 1, ir1)
            tooth_r = stem_r - 1
        else:
            stem_r = rng.randint(ir0, ir1 - 1)
            tooth_r = stem_r + 1
        for c in range(stem_c0, stem_c0 + 3):
            g[stem_r, c] = color
        g[tooth_r, stem_c0 + tooth_index] = color
    else:
        stem_r0 = rng.randint(ir0, ir1 - 2)
        if direction == "left":
            stem_c = rng.randint(ic0 + 1, ic1)
            tooth_c = stem_c - 1
        else:
            stem_c = rng.randint(ic0, ic1 - 1)
            tooth_c = stem_c + 1
        for r in range(stem_r0, stem_r0 + 3):
            g[r, stem_c] = color
        g[stem_r0 + tooth_index, tooth_c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.choices(ROOM_COUNT_CHOICES, weights=ROOM_COUNT_WEIGHTS, k=1)[0]
    h_range, w_range = GRID_SIZE_BY_ROOM_COUNT[room_count]
    h = _rand_int(rng, h_range)
    w = _rand_int(rng, w_range)
    g = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = rng.sample(KEY_COLORS, room_count)
    specs = []
    for idx in range(room_count):
        specs.append(
            {
                "room_h": _rand_int(rng, ROOM_H_RANGE),
                "room_w": _rand_int(rng, ROOM_W_RANGE),
                "direction": rng.choice(["up", "down", "left", "right"]),
                "tooth_index": rng.choice([0, 1, 2]),
                "color": colors[idx],
            }
        )

    specs.sort(key=lambda spec: spec["room_h"] * spec["room_w"], reverse=True)
    placements = []
    for spec in specs:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, h - spec["room_h"] - 1)
            left = rng.randint(1, w - spec["room_w"] - 1)
            if not _position_allows_ray(h, w, top, left, spec["room_h"], spec["room_w"], spec["direction"]):
                continue
            if not _can_place(occupied, top, left, spec["room_h"], spec["room_w"]):
                continue
            bottom = top + spec["room_h"] - 1
            right = left + spec["room_w"] - 1
            occupied[top : bottom + 1, left : right + 1] = True
            placements.append((top, left, spec))
            placed = True
            break
        if not placed:
            return None

    for top, left, spec in placements:
        _draw_room(g, top, left, spec["room_h"], spec["room_w"])
        _place_key(
            g,
            top,
            left,
            spec["room_h"],
            spec["room_w"],
            spec["direction"],
            spec["tooth_index"],
            spec["color"],
            rng,
        )

    try:
        out = main(g)
    except AssertionError:
        return None

    painted = int(((g == BLACK) & (out != BLACK)).sum())
    if painted < room_count:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-key puzzle")
