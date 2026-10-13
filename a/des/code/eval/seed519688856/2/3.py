from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_STEPS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = tuple(SIDE_STEPS)
VERTICAL_SIDES = ("top", "bottom")
HORIZONTAL_SIDES = ("left", "right")

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
ROOM_COUNT_RANGE = (2, 4)
OUTER_SIZE_CHOICES = (5, 7, 9)
ROOM_PADDING = 1
ROOM_RETRIES = 120
DOOR_RETRIES = 160
MARKER_RETRIES = 160
GEN_RETRIES = 400


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(ps[r1, c1])
    if r0 > 0:
        total -= int(ps[r0 - 1, c1])
    if c0 > 0:
        total -= int(ps[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(ps[r0 - 1, c0 - 1])
    return total


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    inner = np.asarray(grid[1:, 1:], dtype=int)
    vals = inner[inner != bg]
    assert vals.size > 0, "expected grey room walls away from the legend bands"
    wall = int(np.bincount(vals).argmax())
    assert np.all((inner == bg) | (inner == wall)), "interior region should contain only background and wall color"
    return wall


def _edge_markers(grid: np.ndarray, bg: int, wall: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    top = [(c, int(grid[0, c])) for c in range(1, w) if int(grid[0, c]) not in (bg, wall)]
    left = [(r, int(grid[r, 0])) for r in range(1, h) if int(grid[r, 0]) not in (bg, wall)]
    assert top and left, "expected colored markers on the top row and left column"
    assert len(top) == len(left), "top and left legends must use the same number of colors"

    top_colors = [color for _pos, color in top]
    left_colors = [color for _pos, color in left]
    assert len(set(top_colors)) == len(top_colors), "top legend colors must be unique"
    assert len(set(left_colors)) == len(left_colors), "left legend colors must be unique"
    assert set(top_colors) == set(left_colors), "top and left legends must use the same color set"
    return top, left


def _nearest_marker(markers: list[tuple[int, int]], coord: float) -> int:
    distances = [abs(pos - coord) for pos, _color in markers]
    best = min(distances)
    matches = [color for (pos, color), dist in zip(markers, distances) if dist == best]
    assert len(matches) == 1, "nearest marker must be unique"
    return int(matches[0])


def _room_signature(room: dict[str, object]) -> tuple[int, int, int, int, int, int, str]:
    r0, c0, r1, c1 = room["bbox"]
    dr, dc = room["door"]
    return int(r0), int(c0), int(r1), int(c1), int(dr), int(dc), str(room["side"])


def _find_rooms(grid: np.ndarray, bg: int = BLACK, wall: int | None = None) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall = _infer_wall_color(g, bg) if wall is None else int(wall)

    wall_ps = _prefix_sum(g == wall)
    bg_ps = _prefix_sum(g == bg)
    rooms: list[dict[str, object]] = []

    for r0 in range(1, h - 2):
        for r1 in range(r0 + 2, h):
            inner_h = r1 - r0 - 1
            for c0 in range(1, w - 2):
                for c1 in range(c0 + 2, w):
                    inner_w = c1 - c0 - 1
                    interior_area = inner_h * inner_w
                    if _rect_sum(wall_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if _rect_sum(bg_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != interior_area:
                        continue

                    box_area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    perimeter = box_area - interior_area
                    if _rect_sum(wall_ps, r0, c0, r1, c1) != perimeter - 1:
                        continue
                    if _rect_sum(bg_ps, r0, c0, r1, c1) != interior_area + 1:
                        continue

                    door_cells: list[tuple[int, int]] = []
                    for c in range(c0, c1 + 1):
                        if int(g[r0, c]) == bg:
                            door_cells.append((r0, c))
                        if int(g[r1, c]) == bg:
                            door_cells.append((r1, c))
                    for r in range(r0 + 1, r1):
                        if int(g[r, c0]) == bg:
                            door_cells.append((r, c0))
                        if int(g[r, c1]) == bg:
                            door_cells.append((r, c1))
                    if len(door_cells) != 1:
                        continue

                    door = door_cells[0]
                    if door[0] == r0:
                        side = "top"
                    elif door[0] == r1:
                        side = "bottom"
                    elif door[1] == c0:
                        side = "left"
                    else:
                        side = "right"

                    if door in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                        continue

                    rooms.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "door": door,
                            "side": side,
                            "center_r": 0.5 * (r0 + r1),
                            "center_c": 0.5 * (c0 + c1),
                        }
                    )

    assert rooms, "expected at least one room"
    rooms.sort(key=_room_signature)
    return rooms


def _paint_stripes(out: np.ndarray, room: dict[str, object], fill_color: int) -> None:
    r0, c0, r1, c1 = room["bbox"]
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    interior[:] = BLACK
    if room["side"] in VERTICAL_SIDES:
        interior[:, ::2] = fill_color
    else:
        interior[::2, :] = fill_color


def _paint_ray(out: np.ndarray, source: np.ndarray, wall: int, room: dict[str, object], ray_color: int) -> None:
    h, w = source.shape
    dr, dc = SIDE_STEPS[str(room["side"])]
    r, c = map(int, room["door"])
    while 0 <= r < h and 0 <= c < w and int(source[r, c]) != wall:
        out[r, c] = ray_color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each room's fill color from the nearest top marker, orient stripes by
    the door side, and cast a door ray colored by the nearest left marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BLACK
    wall = _infer_wall_color(g, bg)
    top_markers, left_markers = _edge_markers(g, bg, wall)
    rooms = _find_rooms(g, bg, wall)

    out = g.copy()
    for room in rooms:
        fill_color = _nearest_marker(top_markers, float(room["center_c"]))
        ray_color = _nearest_marker(left_markers, float(room["center_r"]))
        _paint_stripes(out, room, fill_color)
        _paint_ray(out, g, wall, room, ray_color)
    return out


def _make_room(r0: int, c0: int, h: int, w: int) -> dict[str, int]:
    return {"r0": r0, "c0": c0, "r1": r0 + h - 1, "c1": c0 + w - 1}


def _box_clear(occupied: np.ndarray, room: dict[str, int], pad: int = ROOM_PADDING) -> bool:
    h, w = occupied.shape
    r0 = max(0, room["r0"] - pad)
    c0 = max(0, room["c0"] - pad)
    r1 = min(h - 1, room["r1"] + pad)
    c1 = min(w - 1, room["c1"] + pad)
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_box(occupied: np.ndarray, room: dict[str, int]) -> None:
    occupied[room["r0"] : room["r1"] + 1, room["c0"] : room["c1"] + 1] = True


def _sample_random_room(h: int, w: int, room_h: int, room_w: int, occupied: np.ndarray) -> dict[str, int] | None:
    max_r0 = h - room_h - 1
    max_c0 = w - room_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return None
    r0 = random.randint(1, max_r0)
    c0 = random.randint(1, max_c0)
    room = _make_room(r0, c0, room_h, room_w)
    return room if _box_clear(occupied, room) else None


def _sample_aligned_room(
    h: int,
    w: int,
    room_h: int,
    room_w: int,
    occupied: np.ndarray,
    anchors: list[dict[str, int]],
) -> dict[str, int] | None:
    for _ in range(24):
        anchor = random.choice(anchors)
        side = random.choice(SIDES)
        gap = random.randint(1, 4)
        if side in VERTICAL_SIDES:
            if side == "top":
                r1 = anchor["r0"] - gap - 1
                r0 = r1 - room_h + 1
            else:
                r0 = anchor["r1"] + gap + 1
                r1 = r0 + room_h - 1
            if r0 < 1 or r1 > h - 2:
                continue

            anchor_w = anchor["c1"] - anchor["c0"] + 1
            max_overlap = min(room_w, anchor_w)
            if max_overlap < 3:
                continue
            overlap = random.randint(3, max_overlap)
            overlap_start = random.randint(anchor["c0"], anchor["c1"] - overlap + 1)
            offset = random.randint(0, room_w - overlap)
            c0 = overlap_start - offset
            c1 = c0 + room_w - 1
            if c0 < 1 or c1 > w - 2:
                continue
        else:
            if side == "left":
                c1 = anchor["c0"] - gap - 1
                c0 = c1 - room_w + 1
            else:
                c0 = anchor["c1"] + gap + 1
                c1 = c0 + room_w - 1
            if c0 < 1 or c1 > w - 2:
                continue

            anchor_h = anchor["r1"] - anchor["r0"] + 1
            max_overlap = min(room_h, anchor_h)
            if max_overlap < 3:
                continue
            overlap = random.randint(3, max_overlap)
            overlap_start = random.randint(anchor["r0"], anchor["r1"] - overlap + 1)
            offset = random.randint(0, room_h - overlap)
            r0 = overlap_start - offset
            r1 = r0 + room_h - 1
            if r0 < 1 or r1 > h - 2:
                continue

        room = {"r0": r0, "c0": c0, "r1": r1, "c1": c1}
        if _box_clear(occupied, room):
            return room
    return None


def _room_centers(rooms: list[dict[str, int]], axis: str) -> list[float]:
    if axis == "row":
        return [0.5 * (room["r0"] + room["r1"]) for room in rooms]
    return [0.5 * (room["c0"] + room["c1"]) for room in rooms]


def _place_rooms(h: int, w: int, room_count: int) -> list[dict[str, int]] | None:
    size_choices = [size for size in OUTER_SIZE_CHOICES if size <= min(h, w) - 4]
    if not size_choices:
        return None

    shapes = [(random.choice(size_choices), random.choice(size_choices)) for _ in range(room_count)]
    shapes.sort(key=lambda shape: shape[0] * shape[1], reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    rooms: list[dict[str, int]] = []
    for room_h, room_w in shapes:
        room = None
        for _ in range(ROOM_RETRIES):
            if rooms and random.random() < 0.7:
                room = _sample_aligned_room(h, w, room_h, room_w, occupied, rooms)
            else:
                room = _sample_random_room(h, w, room_h, room_w, occupied)
            if room is not None:
                break
        if room is None:
            return None
        rooms.append(room)
        _mark_box(occupied, room)

    if len(set(_room_centers(rooms, "row"))) < 2 or len(set(_room_centers(rooms, "col"))) < 2:
        return None

    rooms.sort(key=lambda room: (room["r0"], room["c0"], room["r1"], room["c1"]))
    return rooms


def _draw_closed_rooms(h: int, w: int, rooms: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["r0"], room["c0"], room["r1"], room["c1"]
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
    return grid


def _door_positions(room: dict[str, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["r0"], room["c0"], room["r1"], room["c1"]
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _trace_ray_path(grid: np.ndarray, door: tuple[int, int], side: str) -> tuple[list[tuple[int, int]], str]:
    h, w = grid.shape
    dr, dc = SIDE_STEPS[side]
    path = [door]
    r, c = door[0] + dr, door[1] + dc
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) != GREY:
        path.append((r, c))
        r += dr
        c += dc
    stop_type = "wall" if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == GREY else "boundary"
    return path, stop_type


def _door_candidates(room: dict[str, int], grid: np.ndarray) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for side in SIDES:
        for door in _door_positions(room, side):
            path, stop_type = _trace_ray_path(grid, door, side)
            candidates.append({"door": door, "side": side, "path": path, "stop_type": stop_type})
    return candidates


def _desired_sides(room_count: int) -> list[str]:
    chosen: list[str] = []
    if room_count >= 1:
        chosen.append(random.choice(VERTICAL_SIDES))
    if room_count >= 2:
        chosen.append(random.choice(HORIZONTAL_SIDES))
    pool = list(SIDES) * max(1, room_count)
    random.shuffle(pool)
    while len(chosen) < room_count:
        chosen.append(pool.pop())
    random.shuffle(chosen)
    return chosen


def _desired_stops(room_count: int) -> list[str]:
    chosen: list[str] = []
    if room_count >= 1:
        chosen.append("boundary")
    if room_count >= 2:
        chosen.append("wall")
    pool = ["boundary", "wall"] * max(1, room_count)
    random.shuffle(pool)
    while len(chosen) < room_count:
        chosen.append(pool.pop())
    random.shuffle(chosen)
    return chosen


def _pick_candidate(
    candidates: list[dict[str, object]],
    side: str,
    stop_type: str,
) -> dict[str, object]:
    filters = (
        lambda cand: cand["side"] == side and cand["stop_type"] == stop_type,
        lambda cand: cand["side"] == side,
        lambda cand: cand["stop_type"] == stop_type,
        lambda cand: True,
    )
    for predicate in filters:
        matches = [cand for cand in candidates if predicate(cand)]
        if matches:
            return random.choice(matches)
    raise AssertionError("candidate set should never be empty")


def _inside_room_box(cell: tuple[int, int], room: dict[str, int]) -> bool:
    r, c = cell
    return room["r0"] <= r <= room["r1"] and room["c0"] <= c <= room["c1"]


def _choose_doors(
    closed_grid: np.ndarray,
    rooms: list[dict[str, int]],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    candidate_sets = [_door_candidates(room, closed_grid) for room in rooms]
    for _ in range(DOOR_RETRIES):
        sides = _desired_sides(len(rooms))
        stops = _desired_stops(len(rooms))
        picks = [_pick_candidate(cands, side, stop) for cands, side, stop in zip(candidate_sets, sides, stops)]

        carved = closed_grid.copy()
        for pick in picks:
            r, c = pick["door"]
            carved[r, c] = BLACK

        ray_cells: set[tuple[int, int]] = set()
        stop_types: set[str] = set()
        orientations = {"vertical" if pick["side"] in VERTICAL_SIDES else "horizontal" for pick in picks}
        chosen_rooms: list[dict[str, object]] = []
        valid = True

        for room, pick in zip(rooms, picks):
            path, stop_type = _trace_ray_path(carved, pick["door"], str(pick["side"]))
            for cell in path[1:]:
                if any(_inside_room_box(cell, other) for other in rooms):
                    valid = False
                    break
            if not valid:
                break
            if ray_cells & set(path):
                valid = False
                break
            ray_cells.update(path)
            stop_types.add(stop_type)
            chosen_rooms.append(
                {
                    "bbox": (room["r0"], room["c0"], room["r1"], room["c1"]),
                    "door": tuple(pick["door"]),
                    "side": str(pick["side"]),
                    "ray_path": path,
                    "stop_type": stop_type,
                    "center_r": 0.5 * (room["r0"] + room["r1"]),
                    "center_c": 0.5 * (room["c0"] + room["c1"]),
                }
            )

        if not valid:
            continue
        if len(orientations) < 2:
            continue
        if len(stop_types) < 2:
            continue
        chosen_rooms.sort(key=_room_signature)
        return carved, chosen_rooms
    return None


def _unique_nearest_indices(positions: list[int], coords: list[float]) -> list[int] | None:
    indices: list[int] = []
    for coord in coords:
        distances = [abs(pos - coord) for pos in positions]
        best = min(distances)
        winners = [idx for idx, dist in enumerate(distances) if dist == best]
        if len(winners) != 1:
            return None
        indices.append(winners[0])
    return indices


def _choose_markers(
    h: int,
    w: int,
    rooms: list[dict[str, object]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    forbidden_top = {c for room in rooms for r, c in room["ray_path"] if r == 0}
    forbidden_left = {r for room in rooms for r, c in room["ray_path"] if c == 0}
    available_top = [c for c in range(1, w) if c not in forbidden_top]
    available_left = [r for r in range(1, h) if r not in forbidden_left]

    max_k = min(4, len(COLOR_POOL), len(available_top), len(available_left))
    if max_k < 2:
        return None

    row_centers = [float(room["center_r"]) for room in rooms]
    col_centers = [float(room["center_c"]) for room in rooms]

    for _ in range(MARKER_RETRIES):
        k = random.randint(2, min(max_k, len(rooms) + 1))
        top_positions = sorted(random.sample(available_top, k))
        left_positions = sorted(random.sample(available_left, k))

        top_indices = _unique_nearest_indices(top_positions, col_centers)
        left_indices = _unique_nearest_indices(left_positions, row_centers)
        if top_indices is None or left_indices is None:
            continue
        if len(set(top_indices)) < 2 or len(set(left_indices)) < 2:
            continue

        colors = random.sample(COLOR_POOL, k)
        top_colors = colors[:]
        left_colors = colors[:]
        random.shuffle(top_colors)
        random.shuffle(left_colors)

        fill_used = {top_colors[idx] for idx in top_indices}
        ray_used = {left_colors[idx] for idx in left_indices}
        if len(fill_used) < 2 or len(ray_used) < 2:
            continue

        top_markers = list(zip(top_positions, top_colors))
        left_markers = list(zip(left_positions, left_colors))
        return top_markers, left_markers
    return None


def _build_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    room_count = random.randint(*ROOM_COUNT_RANGE)

    rooms = _place_rooms(h, w, room_count)
    if rooms is None:
        return None

    closed_grid = _draw_closed_rooms(h, w, rooms)
    door_result = _choose_doors(closed_grid, rooms)
    if door_result is None:
        return None
    grid, door_rooms = door_result

    marker_result = _choose_markers(h, w, door_rooms)
    if marker_result is None:
        return None
    top_markers, left_markers = marker_result

    for c, color in top_markers:
        grid[0, c] = color
    for r, color in left_markers:
        grid[r, 0] = color

    detected = _find_rooms(grid, BLACK, GREY)
    if [_room_signature(room) for room in detected] != [_room_signature(room) for room in door_rooms]:
        return None

    _edge_markers(grid, BLACK, GREY)
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-code sample")
