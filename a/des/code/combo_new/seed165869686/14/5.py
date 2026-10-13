from __future__ import annotations

import random

import numpy as np


BLACK = 0
WALL = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 600


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expanded_bbox(r0: int, c0: int, r1: int, c1: int, pad: int, h: int, w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(max(0, r0 - pad), min(h, r1 + pad + 1))
        for c in range(max(0, c0 - pad), min(w, c1 + pad + 1))
    }


def _expanded_cells(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((rr, cc))
    return out


def _room_from_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    comp_set = set(comp)
    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if not comp_set <= perimeter:
        return None

    missing = list(perimeter - comp_set)
    if len(missing) != 1 or len(comp_set) != len(perimeter) - 1:
        return None

    gap_r, gap_c = missing[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None

    sprite_mask = (inner != BLACK) & (inner != WALL)
    if int(np.count_nonzero(sprite_mask)) != 1:
        return None
    if not bool(np.all(inner[~sprite_mask] == BLACK)):
        return None

    sr_rel, sc_rel = map(int, np.argwhere(sprite_mask)[0])
    sprite = (r0 + 1 + sr_rel, c0 + 1 + sc_rel)
    color = int(grid[sprite])

    if gap_r == r0:
        direction = "up"
        if sprite[1] != gap_c:
            return None
    elif gap_r == r1:
        direction = "down"
        if sprite[1] != gap_c:
            return None
    elif gap_c == c0:
        direction = "left"
        if sprite[0] != gap_r:
            return None
    else:
        direction = "right"
        if sprite[0] != gap_r:
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "sprite": sprite,
        "color": color,
        "direction": direction,
    }


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    rooms = []
    for comp in _components4(grid == WALL):
        room = _room_from_component(grid, comp)
        if room is not None:
            rooms.append(room)
    rooms.sort(key=lambda room: tuple(room["sprite"]))  # deterministic output order
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each room's sprite in the direction encoded by that room's wall opening.

    Input structure:
    - The background is black, with grey wall pixels forming several rectangular room outlines.
    - Every room is a one-cell-thick grey rectangle with exactly one one-cell gap on one wall.
    - Each room interior contains exactly one colored single-pixel sprite, aligned with that room's gap.
    - Additional grey wall stoppers may appear in corridor space outside the rooms.

    Rule:
    - Treat the side containing the room's unique wall gap as a direction instruction: up, down, left, or right.
    - From the sprite's position, move straight along that direction through the gap and keep moving in the same line.
    - Paint every visited cell with the sprite's color, stopping just before the next grey wall pixel or at the grid edge.
    - Leave all untouched black background and all grey walls unchanged.

    Key insight:
    - The opening's location matters only as a direction cue: the sprite shoots straight along that axis rather than navigating to a nearest exit by pathfinding.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    rooms = _find_rooms(g)
    assert rooms, "expected at least one valid room"

    all_sprites = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != WALL))
    }
    room_sprites = {tuple(room["sprite"]) for room in rooms}
    assert room_sprites == all_sprites, "every colored pixel must be the unique sprite inside a room"

    for room in rooms:
        color = int(room["color"])
        r, c = map(int, room["sprite"])
        dr, dc = DIRS[str(room["direction"])]
        while True:
            out[r, c] = color
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) == WALL:
                break
            r, c = nr, nc

    return out


def _middle_band(length: int) -> tuple[int, int]:
    low = max(4, length // 3)
    high = min(length - 5, length - length // 3 - 1)
    if high < low:
        low, high = 4, length - 5
    return low, high


def _ray_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIRS[direction]
    r, c = start
    ray = [(r, c)]
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        r, c = nr, nc
        ray.append((r, c))
    return ray


def _blocker_segment(
    center: tuple[int, int],
    direction: str,
    length: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    h, w = shape
    r, c = center
    if length == 1:
        cells = {(r, c)}
    else:
        assert length == 3
        if direction in ("up", "down"):
            cells = {(r, c - 1), (r, c), (r, c + 1)}
        else:
            cells = {(r - 1, c), (r, c), (r + 1, c)}
    if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
        return None
    return cells


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sample_room(
    rng: random.Random,
    shape: tuple[int, int],
    direction: str,
    color: int,
    want_blocked: bool,
    occupied: set[tuple[int, int]],
    buffered: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    rh_max = min(6, h - 6)
    rw_max = min(6, w - 6)
    if rh_max < 4 or rw_max < 4:
        return None

    col_low, col_high = _middle_band(w)
    row_low, row_high = _middle_band(h)

    for _ in range(250):
        rh = rng.randint(4, rh_max)
        rw = rng.randint(4, rw_max)

        if direction in ("up", "down"):
            gap_col = rng.randint(col_low, col_high)
            gap_offset = rng.randint(1, rw - 2)
            c0 = gap_col - gap_offset
            c1 = c0 + rw - 1
            if c0 < 1 or c1 > w - 2:
                continue

            if direction == "up":
                low_r0, high_r0 = 2, (h // 2) - rh
            else:
                low_r0, high_r0 = h // 2, h - rh - 3
            if high_r0 < low_r0:
                continue

            r0 = rng.randint(low_r0, high_r0)
            r1 = r0 + rh - 1
            gap_cell = (r0 if direction == "up" else r1, gap_col)
            sprite = (rng.randint(r0 + 1, r1 - 1), gap_col)
        else:
            gap_row = rng.randint(row_low, row_high)
            gap_offset = rng.randint(1, rh - 2)
            r0 = gap_row - gap_offset
            r1 = r0 + rh - 1
            if r0 < 1 or r1 > h - 2:
                continue

            if direction == "left":
                low_c0, high_c0 = 2, (w // 2) - rw
            else:
                low_c0, high_c0 = w // 2, w - rw - 3
            if high_c0 < low_c0:
                continue

            c0 = rng.randint(low_c0, high_c0)
            c1 = c0 + rw - 1
            gap_cell = (gap_row, c0 if direction == "left" else c1)
            sprite = (gap_row, rng.randint(c0 + 1, c1 - 1))

        room_buffer = _expanded_bbox(r0, c0, r1, c1, pad=1, h=h, w=w)
        if room_buffer & occupied or room_buffer & buffered or room_buffer & reserved_paths:
            continue

        wall_cells = _perimeter_cells(r0, c0, r1, c1) - {gap_cell}
        ray = _ray_to_edge(shape, sprite, direction)
        gap_index = abs(sprite[0] - gap_cell[0]) + abs(sprite[1] - gap_cell[1])
        if gap_index >= len(ray) or ray[gap_index] != gap_cell:
            continue
        if len(ray) < gap_index + 2:
            continue

        path_cells = ray
        blocker_cells: set[tuple[int, int]] = set()
        blocker_buffer: set[tuple[int, int]] = set()
        stop_mode = "edge"

        if want_blocked:
            blocker_options: list[tuple[int, set[tuple[int, int]], set[tuple[int, int]]]] = []
            blocker_indices = list(range(gap_index + 2, len(ray)))
            rng.shuffle(blocker_indices)
            for idx in blocker_indices:
                lengths = [1, 3]
                rng.shuffle(lengths)
                for length in lengths:
                    seg = _blocker_segment(ray[idx], direction, length, shape)
                    if seg is None:
                        continue
                    seg_buffer = _expanded_cells(seg, shape, pad=1)
                    if seg_buffer & occupied or seg_buffer & buffered or seg_buffer & reserved_paths:
                        continue
                    if seg_buffer & room_buffer:
                        continue
                    blocker_options.append((idx, seg, seg_buffer))
                if blocker_options:
                    break
            if not blocker_options:
                continue
            idx, blocker_cells, blocker_buffer = rng.choice(blocker_options)
            path_cells = ray[:idx]
            stop_mode = "wall"

        path_set = set(path_cells)
        if path_set & occupied or path_set & buffered or path_set & reserved_paths:
            continue
        if len(path_cells) < gap_index + 2:
            continue

        return {
            "bbox": (r0, c0, r1, c1),
            "sprite": sprite,
            "color": color,
            "direction": direction,
            "wall_cells": wall_cells,
            "room_buffer": room_buffer,
            "path": path_cells,
            "blocker_cells": blocker_cells,
            "blocker_buffer": blocker_buffer,
            "stop_mode": stop_mode,
        }

    return None


def _blocked_flags(rng: random.Random, count: int) -> list[bool]:
    if count == 1:
        return [bool(rng.randint(0, 1))]
    if count % 2 == 0:
        flags = [True] * (count // 2) + [False] * (count // 2)
    elif rng.random() < 0.5:
        flags = [True] * ((count + 1) // 2) + [False] * ((count - 1) // 2)
    else:
        flags = [True] * ((count - 1) // 2) + [False] * ((count + 1) // 2)
    rng.shuffle(flags)
    return flags


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    grid = np.full((h, w), BLACK, dtype=int)

    room_count = rng.randint(2, 4)
    directions = rng.sample(list(DIRS.keys()), room_count)
    colors = rng.sample(SPRITE_COLORS, room_count)
    blocked = _blocked_flags(rng, room_count)

    occupied: set[tuple[int, int]] = set()
    buffered: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    for direction, color, want_blocked in zip(directions, colors, blocked):
        room = _sample_room(
            rng,
            (h, w),
            direction,
            color,
            want_blocked,
            occupied,
            buffered,
            reserved_paths,
        )
        if room is None:
            return None

        _paint_cells(grid, set(room["wall_cells"]), WALL)
        _paint_cells(grid, set(room["blocker_cells"]), WALL)
        sr, sc = map(int, room["sprite"])
        grid[sr, sc] = int(room["color"])

        occupied |= set(room["wall_cells"])
        occupied |= set(room["blocker_cells"])
        occupied.add((sr, sc))
        buffered |= set(room["room_buffer"])
        buffered |= set(room["blocker_buffer"])
        reserved_paths |= set(room["path"])
        placements.append(room)

    rooms = _find_rooms(grid)
    if len(rooms) != room_count:
        return None
    if {str(room["direction"]) for room in rooms} != set(directions):
        return None
    if {tuple(room["sprite"]) for room in rooms} != {tuple(room["sprite"]) for room in placements}:
        return None
    if {"edge", "wall"} - {str(room["stop_mode"]) for room in placements}:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not bool(np.any(out != grid)):
        return None

    for room in placements:
        color = int(room["color"])
        if int(np.count_nonzero(out == color)) != len(room["path"]):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
