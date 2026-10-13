import random
from typing import Optional

import numpy as np


BLACK = 0
RED = 2
AZURE = 8
FLOOR_COLORS = (1, 3, 4, 5, 6, 7, 9)

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_ROOM_OUTER = 5
MAX_GENERATION_ATTEMPTS = 400
BALANCED_MODE_PROB = 0.3


def _floor_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != RED) & (grid != AZURE)


def _extract_rooms(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    mask = _floor_mask(grid)
    seen = np.zeros((h, w), dtype=bool)
    room_ids = np.full((h, w), -1, dtype=int)
    rooms: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            colors = {int(grid[r, c])}

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                colors.add(int(grid[cr, cc]))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            if len(colors) != 1:
                raise ValueError("each room must be a single-color component")

            room_id = len(rooms)
            for cr, cc in cells:
                room_ids[cr, cc] = room_id
            rooms.append(
                {
                    "id": room_id,
                    "color": colors.pop(),
                    "cells": cells,
                    "area": len(cells),
                }
            )

    return rooms, room_ids


def _find_doors(grid: np.ndarray, room_ids: np.ndarray) -> list[dict]:
    h, w = grid.shape
    doors: list[dict] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue

            matches: list[tuple[int, int]] = []

            left_id = int(room_ids[r, c - 1])
            right_id = int(room_ids[r, c + 1])
            if (
                left_id >= 0
                and right_id >= 0
                and left_id != right_id
                and int(grid[r - 1, c]) == AZURE
                and int(grid[r + 1, c]) == AZURE
            ):
                matches.append((left_id, right_id))

            up_id = int(room_ids[r - 1, c])
            down_id = int(room_ids[r + 1, c])
            if (
                up_id >= 0
                and down_id >= 0
                and up_id != down_id
                and int(grid[r, c - 1]) == AZURE
                and int(grid[r, c + 1]) == AZURE
            ):
                matches.append((up_id, down_id))

            if len(matches) == 1:
                a, b = matches[0]
                doors.append({"pos": (r, c), "rooms": tuple(sorted((a, b)))})

    return doors


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rooms, room_ids = _extract_rooms(g)
    doors = _find_doors(g, room_ids)

    out = g.copy()
    recolor_requests: dict[int, list[tuple[int, int]]] = {}
    for door in doors:
        room_a, room_b = door["rooms"]
        area_a = int(rooms[room_a]["area"])
        area_b = int(rooms[room_b]["area"])

        if area_a < area_b:
            recolor_requests.setdefault(room_a, []).append((area_b, int(rooms[room_b]["color"])))
        elif area_b < area_a:
            recolor_requests.setdefault(room_b, []).append((area_a, int(rooms[room_a]["color"])))

    for room_id, requests in recolor_requests.items():
        best_area = max(area for area, _ in requests)
        candidate_colors = {color for area, color in requests if area == best_area}
        if len(candidate_colors) != 1:
            continue
        new_color = candidate_colors.pop()
        for r, c in rooms[room_id]["cells"]:
            out[r, c] = new_color

    for door in doors:
        r, c = door["pos"]
        out[r, c] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find one-cell wall gaps between rectangular rooms, recolor the smaller room
    to match the larger connected room, and seal every door in red.
    """
    return _transform(grid)


def _rect_height(rect: tuple[int, int, int, int]) -> int:
    return rect[2] - rect[0] + 1


def _rect_width(rect: tuple[int, int, int, int]) -> int:
    return rect[3] - rect[1] + 1


def _feasible_orientations(rect: tuple[int, int, int, int]) -> list[str]:
    orientations = []
    if _rect_width(rect) >= 2 * MIN_ROOM_OUTER - 1:
        orientations.append("V")
    if _rect_height(rect) >= 2 * MIN_ROOM_OUTER - 1:
        orientations.append("H")
    return orientations


def _split_rect(
    rect: tuple[int, int, int, int],
    orientation: str,
    rng: random.Random,
    force_center: bool = False,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect
    if orientation == "V":
        left_min = c0 + MIN_ROOM_OUTER - 1
        left_max = c1 - MIN_ROOM_OUTER + 1
        if force_center:
            cut = (c0 + c1) // 2
            if not (left_min <= cut <= left_max):
                raise ValueError("center split is not feasible")
        else:
            cut = rng.randint(left_min, left_max)
        return (r0, c0, r1, cut), (r0, cut, r1, c1)

    top_min = r0 + MIN_ROOM_OUTER - 1
    top_max = r1 - MIN_ROOM_OUTER + 1
    if force_center:
        cut = (r0 + r1) // 2
        if not (top_min <= cut <= top_max):
            raise ValueError("center split is not feasible")
    else:
        cut = rng.randint(top_min, top_max)
    return (r0, c0, cut, c1), (cut, c0, r1, c1)


def _weighted_choice(rng: random.Random, items: list[tuple[int, tuple[int, int, int, int], list[str]]]) -> tuple[int, tuple[int, int, int, int], list[str]]:
    weights = []
    for _idx, rect, _oris in items:
        weights.append(max(1, (_rect_height(rect) - 2) * (_rect_width(rect) - 2)))
    total = sum(weights)
    pick = rng.randint(1, total)
    running = 0
    for item, weight in zip(items, weights):
        running += weight
        if pick <= running:
            return item
    return items[-1]


def _partition_general(
    complex_h: int,
    complex_w: int,
    target_rooms: int,
    rng: random.Random,
) -> Optional[list[tuple[int, int, int, int]]]:
    rects = [(0, 0, complex_h - 1, complex_w - 1)]
    for _ in range(target_rooms - 1):
        splittable = []
        for idx, rect in enumerate(rects):
            orientations = _feasible_orientations(rect)
            if orientations:
                splittable.append((idx, rect, orientations))
        if not splittable:
            return None

        idx, rect, orientations = _weighted_choice(rng, splittable)
        orientation = rng.choice(orientations)
        first, second = _split_rect(rect, orientation, rng)
        rects[idx : idx + 1] = [first, second]

    return rects


def _partition_balanced(
    complex_h: int,
    complex_w: int,
    rng: random.Random,
) -> Optional[list[tuple[int, int, int, int]]]:
    choices = []
    if complex_w >= 2 * MIN_ROOM_OUTER - 1 and complex_w % 2 == 1:
        choices.append("V")
    if complex_h >= 2 * MIN_ROOM_OUTER - 1 and complex_h % 2 == 1:
        choices.append("H")
    if not choices:
        return None

    orientation = rng.choice(choices)
    whole = (0, 0, complex_h - 1, complex_w - 1)
    first, second = _split_rect(whole, orientation, rng, force_center=True)
    return [first, second]


def _render_rooms(
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
    complex_h: int,
    complex_w: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    local = np.full((complex_h, complex_w), BLACK, dtype=int)
    room_ids = np.full((complex_h, complex_w), -1, dtype=int)
    rooms: list[dict] = []

    for room_id, (rect, color) in enumerate(zip(rects, colors)):
        r0, c0, r1, c1 = rect
        local[r0 : r1 + 1, c0] = AZURE
        local[r0 : r1 + 1, c1] = AZURE
        local[r0, c0 : c1 + 1] = AZURE
        local[r1, c0 : c1 + 1] = AZURE
        local[r0 + 1 : r1, c0 + 1 : c1] = color
        room_ids[r0 + 1 : r1, c0 + 1 : c1] = room_id
        rooms.append(
            {
                "id": room_id,
                "rect": rect,
                "color": int(color),
                "area": int((r1 - r0 - 1) * (c1 - c0 - 1)),
            }
        )

    return local, room_ids, rooms


def _collect_adjacencies(local: np.ndarray, room_ids: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = local.shape
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(local[r, c]) != AZURE:
                continue

            left_id = int(room_ids[r, c - 1])
            right_id = int(room_ids[r, c + 1])
            if (
                left_id >= 0
                and right_id >= 0
                and left_id != right_id
                and int(local[r - 1, c]) == AZURE
                and int(local[r + 1, c]) == AZURE
            ):
                key = tuple(sorted((left_id, right_id)))
                adjacency.setdefault(key, []).append((r, c))

            up_id = int(room_ids[r - 1, c])
            down_id = int(room_ids[r + 1, c])
            if (
                up_id >= 0
                and down_id >= 0
                and up_id != down_id
                and int(local[r, c - 1]) == AZURE
                and int(local[r, c + 1]) == AZURE
            ):
                key = tuple(sorted((up_id, down_id)))
                adjacency.setdefault(key, []).append((r, c))

    return adjacency


def _choose_doors(
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    rooms: list[dict],
    rng: random.Random,
    prefer_equal: bool,
) -> Optional[list[tuple[int, int]]]:
    keys = list(adjacency)
    if not keys:
        return None

    preferred_key = None
    if prefer_equal:
        equal_keys = [key for key in keys if rooms[key[0]]["area"] == rooms[key[1]]["area"]]
        if equal_keys:
            preferred_key = rng.choice(equal_keys)

    rng.shuffle(keys)
    if preferred_key is not None:
        keys.remove(preferred_key)
        keys.insert(0, preferred_key)

    matching: list[tuple[int, int]] = []
    used_rooms: set[int] = set()
    for room_a, room_b in keys:
        if room_a in used_rooms or room_b in used_rooms:
            continue
        matching.append((room_a, room_b))
        used_rooms.add(room_a)
        used_rooms.add(room_b)

    if not matching:
        return None

    door_count = rng.randint(1, len(matching))
    positions = []
    for key in matching[:door_count]:
        positions.append(rng.choice(adjacency[key]))
    return positions


def _build_sample(rng: random.Random) -> Optional[np.ndarray]:
    balanced = rng.random() < BALANCED_MODE_PROB
    if balanced:
        complex_h = rng.randrange(9, 22, 2)
        complex_w = rng.randrange(9, 22, 2)
        rects = _partition_balanced(complex_h, complex_w, rng)
    else:
        target_rooms = rng.randint(3, 6)
        min_size = 12 + max(0, target_rooms - 4)
        complex_h = rng.randint(min_size, 24)
        complex_w = rng.randint(min_size, 24)
        rects = _partition_general(complex_h, complex_w, target_rooms, rng)

    if rects is None:
        return None

    room_count = len(rects)
    colors = rng.sample(FLOOR_COLORS, room_count)
    local, room_ids, rooms = _render_rooms(rects, colors, complex_h, complex_w)
    adjacency = _collect_adjacencies(local, room_ids)
    door_positions = _choose_doors(adjacency, rooms, rng, prefer_equal=balanced or rng.random() < 0.2)
    if not door_positions:
        return None

    for r, c in door_positions:
        local[r, c] = BLACK

    pad_top = rng.randint(1, 4)
    pad_bottom = rng.randint(1, 4)
    pad_left = rng.randint(1, 4)
    pad_right = rng.randint(1, 4)
    grid = np.full((complex_h + pad_top + pad_bottom, complex_w + pad_left + pad_right), BLACK, dtype=int)
    grid[pad_top : pad_top + complex_h, pad_left : pad_left + complex_w] = local

    rooms_found, room_map = _extract_rooms(grid)
    doors_found = _find_doors(grid, room_map)
    if len(rooms_found) != room_count or len(doors_found) != len(door_positions):
        return None

    room_degrees: dict[int, int] = {}
    for door in doors_found:
        room_a, room_b = door["rooms"]
        room_degrees[room_a] = room_degrees.get(room_a, 0) + 1
        room_degrees[room_b] = room_degrees.get(room_b, 0) + 1
    if any(degree > 1 for degree in room_degrees.values()):
        return None

    if not balanced:
        unequal = any(
            rooms_found[a]["area"] != rooms_found[b]["area"]
            for a, b in (door["rooms"] for door in doors_found)
        )
        if not unequal:
            return None

    out = _transform(grid)
    if np.array_equal(out, grid):
        return None

    red_pixels = int(np.count_nonzero(out == RED))
    if red_pixels != len(doors_found):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_sample(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
