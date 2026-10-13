from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
FURNITURE_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[np.ndarray, bool]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((np.asarray(cells, dtype=int), touches_border))
    return comps


def _door_start(line: np.ndarray) -> int | None:
    assert np.all(np.isin(line, [GREY, YELLOW]))
    door_idx = np.flatnonzero(line == YELLOW)
    if len(door_idx) == 0:
        return None
    assert len(door_idx) == 2 and int(door_idx[1]) == int(door_idx[0]) + 1
    return int(door_idx[0])


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    walkable = (g != GREY) & (g != YELLOW)
    rooms: list[dict] = []
    for cells, touches_border in _components(walkable):
        if touches_border:
            continue

        r0, c0 = np.min(cells, axis=0)
        r1, c1 = np.max(cells, axis=0)
        if not np.all(walkable[r0 : r1 + 1, c0 : c1 + 1]):
            continue

        top, bottom = int(r0) - 1, int(r1) + 1
        left, right = int(c0) - 1, int(c1) + 1
        assert top >= 0 and left >= 0 and bottom < g.shape[0] and right < g.shape[1]
        assert all(int(g[r, c]) == GREY for r, c in ((top, left), (top, right), (bottom, left), (bottom, right)))

        assert np.all(np.isin(g[top, left : right + 1], [GREY, YELLOW]))
        assert np.all(np.isin(g[bottom, left : right + 1], [GREY, YELLOW]))
        assert np.all(np.isin(g[top : bottom + 1, left], [GREY, YELLOW]))
        assert np.all(np.isin(g[top : bottom + 1, right], [GREY, YELLOW]))

        doors: list[dict] = []

        start = _door_start(g[top, left + 1 : right])
        if start is not None:
            cells2 = ((top, left + 1 + start), (top, left + 2 + start))
            doors.append({"side": "top", "cells": tuple(sorted(cells2))})

        start = _door_start(g[bottom, left + 1 : right])
        if start is not None:
            cells2 = ((bottom, left + 1 + start), (bottom, left + 2 + start))
            doors.append({"side": "bottom", "cells": tuple(sorted(cells2))})

        start = _door_start(g[top + 1 : bottom, left])
        if start is not None:
            cells2 = ((top + 1 + start, left), (top + 2 + start, left))
            doors.append({"side": "left", "cells": tuple(sorted(cells2))})

        start = _door_start(g[top + 1 : bottom, right])
        if start is not None:
            cells2 = ((top + 1 + start, right), (top + 2 + start, right))
            doors.append({"side": "right", "cells": tuple(sorted(cells2))})

        assert len(doors) in (1, 2)
        interior = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.any((interior != BLACK) & (interior != GREY) & (interior != YELLOW))
        rooms.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "interior": interior,
                "doors": doors,
            }
        )

    assert 3 <= len(rooms) <= 5
    return rooms


def _room_graph(rooms: list[dict]) -> dict[int, list[tuple[int, str, str]]]:
    door_map: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, str]]] = defaultdict(list)
    for idx, room in enumerate(rooms):
        for door in room["doors"]:
            door_map[door["cells"]].append((idx, door["side"]))

    graph: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(rooms))}
    for room_refs in door_map.values():
        assert len(room_refs) == 2
        (a, side_a), (b, side_b) = room_refs
        graph[a].append((b, side_a, side_b))
        graph[b].append((a, side_b, side_a))

    for neighbors in graph.values():
        assert len(neighbors) in (1, 2)
    return graph


def _chain_order(rooms: list[dict], graph: dict[int, list[tuple[int, str, str]]]) -> tuple[list[int], list[str | None]]:
    endpoints = [idx for idx, neighbors in graph.items() if len(neighbors) == 1]
    assert len(endpoints) == 2
    # The description leaves the two endpoints symmetric; use the top-left endpoint
    # as the canonical start so both solver and generator agree.
    start = min(endpoints, key=lambda idx: (rooms[idx]["top"], rooms[idx]["left"], rooms[idx]["bottom"], rooms[idx]["right"]))

    order: list[int] = []
    entry_sides: list[str | None] = []
    prev = -1
    curr = start
    entry: str | None = None
    while True:
        order.append(curr)
        entry_sides.append(entry)
        next_steps = [item for item in graph[curr] if item[0] != prev]
        if not next_steps:
            break
        assert len(next_steps) == 1
        nxt, _exit_side, next_entry = next_steps[0]
        prev, curr = curr, nxt
        entry = next_entry

    assert len(order) == len(rooms)
    return order, entry_sides


def _transform_room(interior: np.ndarray, entry_side: str | None) -> np.ndarray:
    if entry_side == "top":
        return np.flipud(interior)
    if entry_side == "bottom":
        return np.fliplr(interior)
    return interior.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll a chain of rectangular rooms into a horizontal strip, flipping each
    room interior according to the side of its incoming door.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _extract_rooms(g)
    graph = _room_graph(rooms)
    order, entry_sides = _chain_order(rooms, graph)
    patches = [_transform_room(rooms[idx]["interior"], side) for idx, side in zip(order, entry_sides)]

    out_h = max(int(patch.shape[0]) for patch in patches)
    out_w = sum(int(patch.shape[1]) for patch in patches) + len(patches) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)

    col = 0
    for idx, patch in enumerate(patches):
        ph, pw = map(int, patch.shape)
        row = (out_h - ph) // 2
        out[row : row + ph, col : col + pw] = patch
        col += pw
        if idx + 1 < len(patches):
            out[:, col] = GREY
            col += 1

    return out


# --------------- generator ---------------

ROOM_COUNT_RANGE = (3, 5)
ROOM_OUTER_SIZE_RANGE = (5, 7)
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 300
ROOM_PLACEMENT_RETRIES = 100
PATCH_RETRIES = 120
OBJECT_RETRIES = 80
SHAPE_RETRIES = 80


def _norm_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    cells = _norm_cells(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_asymmetric(arr: np.ndarray) -> bool:
    return not np.array_equal(arr, np.flipud(arr)) and not np.array_equal(arr, np.fliplr(arr))


def _grow_shape(height: int, width: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return _mask_from_cells(cells)


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(SHAPE_RETRIES):
        height = random.randint(2, min(4, max_h))
        width = random.randint(2, min(4, max_w))
        max_area = min(height * width - 1, 6)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _grow_shape(height, width, area)
        if mask is None:
            continue
        mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w and _is_asymmetric(mask):
            return mask
    return None


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + mh + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + mw + 1)
    if np.any(occupied[r0:r1, c0:c1]):
        return False
    return True


def _random_patch(height: int, width: int) -> np.ndarray | None:
    max_objects = 2 if min(height, width) >= 4 and height * width >= 16 else 1
    for _ in range(PATCH_RETRIES):
        patch = np.full((height, width), BLACK, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)
        object_count = 1 if max_objects == 1 else random.randint(1, 2)
        colors = random.sample(FURNITURE_COLORS, object_count)
        ok = True
        for color in colors:
            placed = False
            for _ in range(OBJECT_RETRIES):
                mask = _random_shape(height, width)
                if mask is None:
                    continue
                mh, mw = mask.shape
                top = random.randint(0, height - mh)
                left = random.randint(0, width - mw)
                if not _can_place_mask(occupied, mask, top, left):
                    continue
                patch[top : top + mh, left : left + mw][mask] = int(color)
                occupied[top : top + mh, left : left + mw] |= mask
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and _is_asymmetric(patch):
            return patch
    return None


def _sample_path(length: int) -> list[tuple[int, int]] | None:
    path = [(0, 0)]
    used = {(0, 0)}

    def dfs() -> bool:
        if len(path) == length:
            return True
        r, c = path[-1]
        choices = [(r + dr, c + dc) for dr, dc in DIRS4]
        random.shuffle(choices)
        for cand in choices:
            if cand in used:
                continue
            neighbors = sum((cand[0] + dr, cand[1] + dc) in used for dr, dc in DIRS4)
            if neighbors != 1:
                continue
            used.add(cand)
            path.append(cand)
            if dfs():
                return True
            path.pop()
            used.remove(cand)
        return False

    if dfs():
        return path
    return None


def _axes_used(path: list[tuple[int, int]]) -> set[str]:
    axes = set()
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        axes.add("row" if r0 != r1 else "col")
    return axes


def _sample_adjacent_bbox(prev: tuple[int, int, int, int], dr: int, dc: int, height: int, width: int) -> tuple[int, int, int, int] | None:
    pr0, pr1, pc0, pc1 = prev
    if dc == 1:
        lo, hi = pr0 - height + 4, pr1 - 3
        if lo > hi:
            return None
        top = random.randint(lo, hi)
        return (top, top + height - 1, pc1, pc1 + width - 1)
    if dc == -1:
        lo, hi = pr0 - height + 4, pr1 - 3
        if lo > hi:
            return None
        top = random.randint(lo, hi)
        return (top, top + height - 1, pc0 - width + 1, pc0)
    if dr == 1:
        lo, hi = pc0 - width + 4, pc1 - 3
        if lo > hi:
            return None
        left = random.randint(lo, hi)
        return (pr1, pr1 + height - 1, left, left + width - 1)
    if dr == -1:
        lo, hi = pc0 - width + 4, pc1 - 3
        if lo > hi:
            return None
        left = random.randint(lo, hi)
        return (pr0 - height + 1, pr0, left, left + width - 1)
    return None


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[3] < b[2] - 1 or b[3] < a[2] - 1 or a[1] < b[0] - 1 or b[1] < a[0] - 1)


def _placed_rooms(path: list[tuple[int, int]]) -> list[dict] | None:
    rooms: list[dict] = []
    first_h = random.randint(*ROOM_OUTER_SIZE_RANGE)
    first_w = random.randint(*ROOM_OUTER_SIZE_RANGE)
    first_patch = _random_patch(first_h - 2, first_w - 2)
    if first_patch is None:
        return None
    rooms.append({"bbox": (0, first_h - 1, 0, first_w - 1), "patch": first_patch})

    for i in range(1, len(path)):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        dr, dc = cr - pr, cc - pc
        placed = False
        for _ in range(ROOM_PLACEMENT_RETRIES):
            height = random.randint(*ROOM_OUTER_SIZE_RANGE)
            width = random.randint(*ROOM_OUTER_SIZE_RANGE)
            patch = _random_patch(height - 2, width - 2)
            if patch is None:
                continue
            bbox = _sample_adjacent_bbox(rooms[-1]["bbox"], dr, dc, height, width)
            if bbox is None:
                continue
            if any(_too_close(bbox, room["bbox"]) for room in rooms[:-1]):
                continue
            rooms.append({"bbox": bbox, "patch": patch})
            placed = True
            break
        if not placed:
            return None
    return rooms


def _shared_door(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    if a3 == b2 or b3 == a2:
        wall_col = a3 if a3 == b2 else a2
        lo = max(a0, b0) + 1
        hi = min(a1, b1) - 2
        if lo > hi:
            return None
        start = random.randint(lo, hi)
        return ((start, wall_col), (start + 1, wall_col))
    if a1 == b0 or b1 == a0:
        wall_row = a1 if a1 == b0 else a0
        lo = max(a2, b2) + 1
        hi = min(a3, b3) - 2
        if lo > hi:
            return None
        start = random.randint(lo, hi)
        return ((wall_row, start), (wall_row, start + 1))
    return None


def _render_rooms(rooms: list[dict]) -> np.ndarray | None:
    min_r = min(room["bbox"][0] for room in rooms)
    min_c = min(room["bbox"][2] for room in rooms)
    max_r = max(room["bbox"][1] for room in rooms)
    max_c = max(room["bbox"][3] for room in rooms)
    shift_r = 1 - min_r
    shift_c = 1 - min_c

    shifted = []
    for room in rooms:
        r0, r1, c0, c1 = room["bbox"]
        shifted.append(
            {
                "bbox": (r0 + shift_r, r1 + shift_r, c0 + shift_c, c1 + shift_c),
                "patch": room["patch"],
            }
        )

    out_h = max(room["bbox"][1] for room in shifted) + 2
    out_w = max(room["bbox"][3] for room in shifted) + 2
    if max(out_h, out_w) > MAX_GRID_SIDE:
        return None

    grid = np.full((out_h, out_w), BLACK, dtype=int)
    for room in shifted:
        r0, r1, c0, c1 = room["bbox"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[r0 + 1 : r1, c0 + 1 : c1] = room["patch"]

    for left_room, right_room in zip(shifted, shifted[1:]):
        door = _shared_door(left_room["bbox"], right_room["bbox"])
        if door is None:
            return None
        for r, c in door:
            grid[r, c] = YELLOW

    return grid


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int).copy()


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(*ROOM_COUNT_RANGE)
    path = _sample_path(room_count)
    if path is None:
        return None
    # Keep straight chains sometimes, but bias toward mixed-axis scenes so all
    # entry-side branches appear regularly in random samples.
    if len(_axes_used(path)) == 1 and random.random() < 0.65:
        return None

    rooms = _placed_rooms(path)
    if rooms is None:
        return None

    grid = _render_rooms(rooms)
    if grid is None:
        return None
    grid = _random_symmetry(grid)

    try:
        if len(_extract_rooms(grid)) != room_count:
            return None
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid room-chain input")
