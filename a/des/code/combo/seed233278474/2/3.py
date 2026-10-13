from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 3
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
DIRECTIONS = ("up", "right", "down", "left")
STEP = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ROTATION_K = {
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}

GRID_MIN = 18
GRID_MAX = 30
ROOM_COUNT_RANGE = (1, 3)
INTERIOR_SIDE_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 400
ROOM_PLACEMENT_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((int(r0), int(c)))
        border.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        border.add((int(r), int(c0)))
        border.add((int(r), int(c1)))
    return border


def _direction_from_gap(
    gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> str | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "up"
    if gr == r1 and c0 < gc < c1:
        return "down"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    return None


def _trace_corridor(
    grid: np.ndarray, start: tuple[int, int], direction: str
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    dr, dc = STEP[direction]
    r, c = start
    if not (0 <= r < h and 0 <= c < w):
        return None
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) != WALL:
        if int(grid[r, c]) != BG:
            return None
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells if cells else None


def _extract_room(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    pts = np.array(comp, dtype=int)
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    comp_set = {(int(r), int(c)) for r, c in comp}
    border = _border_set(r0, c0, r1, c1)
    if not comp_set.issubset(border):
        return None

    missing = list(border - comp_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if int(grid[gap]) != BG:
        return None

    direction = _direction_from_gap(gap, r0, c0, r1, c1)
    if direction is None:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or bool(np.any(interior == WALL)):
        return None

    colored = (interior != BG) & (interior != WALL)
    if not bool(np.any(colored)):
        return None

    gr, gc = gap
    if direction == "up":
        corridor_start = (gr - 1, gc)
    elif direction == "down":
        corridor_start = (gr + 1, gc)
    elif direction == "left":
        corridor_start = (gr, gc - 1)
    else:
        corridor_start = (gr, gc + 1)

    corridor = _trace_corridor(grid, corridor_start, direction)
    if corridor is None:
        return None
    if int(colored.sum()) > len(corridor):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "direction": direction,
        "corridor": corridor,
    }


def _find_rooms(grid: np.ndarray) -> list[dict]:
    wall_mask = np.asarray(grid == WALL, dtype=bool)
    rooms = []
    for comp in _components(wall_mask):
        room = _extract_room(grid, comp)
        if room is not None:
            rooms.append(room)
    assert rooms, "expected at least one valid room"
    return rooms


def _rotate_interior(interior: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(interior, int(ROTATION_K[direction]))


def _flatten_sequence(rotated: np.ndarray, direction: str) -> list[int]:
    seq: list[int] = []
    h, w = rotated.shape
    if direction == "up":
        row_iter = range(h)
        col_iter = range(w)
        for r in row_iter:
            for c in col_iter:
                value = int(rotated[r, c])
                if value not in (BG, WALL):
                    seq.append(value)
    elif direction == "down":
        for r in range(h - 1, -1, -1):
            for c in range(w):
                value = int(rotated[r, c])
                if value not in (BG, WALL):
                    seq.append(value)
    elif direction == "left":
        for c in range(w):
            for r in range(h):
                value = int(rotated[r, c])
                if value not in (BG, WALL):
                    seq.append(value)
    else:
        for c in range(w - 1, -1, -1):
            for r in range(h):
                value = int(rotated[r, c])
                if value not in (BG, WALL):
                    seq.append(value)
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each room's interior pattern toward its exit, then compress its colored pixels into
    the outward corridor as a one-dimensional sequence scanned from the exit side inward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _find_rooms(g):
        r0, c0, r1, c1 = room["bbox"]
        direction = room["direction"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        rotated = _rotate_interior(interior, direction)
        seq = _flatten_sequence(rotated, direction)

        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        for (r, c), color in zip(room["corridor"], seq):
            assert int(out[r, c]) != WALL
            out[r, c] = int(color)

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _room_layout(
    h: int,
    w: int,
    r0: int,
    c0: int,
    interior_h: int,
    interior_w: int,
    direction: str,
    gap_index: int,
) -> dict | None:
    r1 = r0 + interior_h + 1
    c1 = c0 + interior_w + 1
    if r1 >= h or c1 >= w:
        return None

    if direction == "up":
        gap = (r0, c0 + 1 + gap_index)
        corridor = [(r, gap[1]) for r in range(r0 - 1, -1, -1)]
    elif direction == "down":
        gap = (r1, c0 + 1 + gap_index)
        corridor = [(r, gap[1]) for r in range(r1 + 1, h)]
    elif direction == "left":
        gap = (r0 + 1 + gap_index, c0)
        corridor = [(gap[0], c) for c in range(c0 - 1, -1, -1)]
    else:
        gap = (r0 + 1 + gap_index, c1)
        corridor = [(gap[0], c) for c in range(c1 + 1, w)]

    if not corridor:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for rr, cc in corridor:
        mask[rr, cc] = True

    border = _border_set(r0, c0, r1, c1)
    border.remove(gap)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": gap,
        "direction": direction,
        "corridor": corridor,
        "wall_cells": border,
        "reserve": mask,
        "buffered": _dilate(mask),
    }


def _paint_shape(
    canvas: np.ndarray, color: int, cells: list[tuple[int, int]]
) -> int:
    placed = 0
    for r, c in cells:
        if canvas[r, c] == BG:
            canvas[r, c] = int(color)
            placed += 1
    return placed


def _random_shape_cells(h: int, w: int) -> list[tuple[int, int]]:
    kind = random.choice(("dot", "line", "block", "el"))
    if kind == "dot":
        return [(random.randrange(h), random.randrange(w))]

    if kind == "line":
        if random.random() < 0.5:
            row = random.randrange(h)
            length = random.randint(2, w)
            start = random.randint(0, w - length)
            return [(row, start + dc) for dc in range(length)]
        col = random.randrange(w)
        length = random.randint(2, h)
        start = random.randint(0, h - length)
        return [(start + dr, col) for dr in range(length)]

    if kind == "block":
        bh = random.randint(1, min(3, h))
        bw = random.randint(1, min(3, w))
        r0 = random.randint(0, h - bh)
        c0 = random.randint(0, w - bw)
        return [(r0 + dr, c0 + dc) for dr in range(bh) for dc in range(bw)]

    arm_h = random.randint(2, h)
    arm_w = random.randint(2, w)
    r0 = random.randint(0, h - arm_h)
    c0 = random.randint(0, w - arm_w)
    cells = [(r0 + dr, c0) for dr in range(arm_h)]
    cells.extend((r0 + arm_h - 1, c0 + dc) for dc in range(1, arm_w))
    return cells


def _make_pattern(interior_h: int, interior_w: int, corridor_len: int) -> np.ndarray | None:
    area = interior_h * interior_w
    max_colored = min(corridor_len, max(3, int(area * 0.6)))
    if max_colored < 2:
        return None
    target = random.randint(2, max_colored)

    canvas = np.zeros((interior_h, interior_w), dtype=int)
    n_colors = random.randint(1, min(3, len(PALETTE), target))
    colors = random.sample(PALETTE, n_colors)

    # Seed each chosen color once so the palette is visually recoverable.
    for color in colors:
        for _ in range(20):
            r = random.randrange(interior_h)
            c = random.randrange(interior_w)
            if canvas[r, c] == BG:
                canvas[r, c] = int(color)
                break

    while int(np.count_nonzero(canvas)) < target:
        color = random.choice(colors)
        _paint_shape(canvas, color, _random_shape_cells(interior_h, interior_w))
        if int(np.count_nonzero(canvas)) > target:
            # Trim back to the requested density while preserving at least one of every color.
            coords = [(int(r), int(c)) for r, c in np.argwhere(canvas != BG)]
            random.shuffle(coords)
            for r, c in coords:
                if int(np.count_nonzero(canvas)) <= target:
                    break
                color_here = int(canvas[r, c])
                if int(np.count_nonzero(canvas == color_here)) <= 1:
                    continue
                canvas[r, c] = BG

    if int(np.count_nonzero(canvas)) != target:
        return None
    return canvas


def _place_room(grid: np.ndarray, layout: dict, pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = layout["bbox"]
    for rr, cc in layout["wall_cells"]:
        grid[rr, cc] = WALL
    grid[r0 + 1 : r1, c0 + 1 : c1] = pattern


def _sample_room(
    grid: np.ndarray, occupied: np.ndarray, direction: str
) -> tuple[dict, np.ndarray] | None:
    h, w = grid.shape
    for _ in range(ROOM_PLACEMENT_ATTEMPTS):
        interior_h = random.randint(*INTERIOR_SIDE_RANGE)
        interior_w = random.randint(*INTERIOR_SIDE_RANGE)
        outer_h = interior_h + 2
        outer_w = interior_w + 2

        if outer_h >= h - 1 or outer_w >= w - 1:
            continue

        r_min = 1
        r_max = h - outer_h - 1
        c_min = 1
        c_max = w - outer_w - 1
        if r_min > r_max or c_min > c_max:
            continue

        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        gap_index = random.randint(0, interior_w - 1 if direction in ("up", "down") else interior_h - 1)
        layout = _room_layout(h, w, r0, c0, interior_h, interior_w, direction, gap_index)
        if layout is None:
            continue
        if bool(np.any(layout["buffered"] & occupied)):
            continue

        pattern = _make_pattern(interior_h, interior_w, len(layout["corridor"]))
        if pattern is None:
            continue
        return layout, pattern
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    room_count = random.randint(*ROOM_COUNT_RANGE)
    directions = [random.choice(DIRECTIONS) for _ in range(room_count)]
    random.shuffle(directions)

    for direction in directions:
        sampled = _sample_room(grid, occupied, direction)
        if sampled is None:
            return None
        layout, pattern = sampled
        _place_room(grid, layout, pattern)
        occupied |= layout["buffered"]

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not bool(np.any((out != grid) & (out != WALL))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC room-corridor instance")
