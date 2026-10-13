from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


GEN_ATTEMPTS = 400
GRID_H_RANGE = (19, 31)
GRID_W_RANGE = (19, 31)
ROOM_HEIGHTS = (5, 7, 9, 11)
ROOM_WIDTHS = (5, 7, 9, 11)
ROOM_PLACEMENT_TRIES = 250
ROOM_COUNT_BY_SIZE = (
    (1, 2),
    (2, 2),
    (3, 3),
    (4, 2),
)
EDGE_PAD = 1


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _gap_and_direction(r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]) -> tuple[int, int] | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return DIRS["up"]
    if gr == r1 and c0 < gc < c1:
        return DIRS["down"]
    if gc == c0 and r0 < gr < r1:
        return DIRS["left"]
    if gc == c1 and r0 < gr < r1:
        return DIRS["right"]
    return None


def _parse_room(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))

    cells = set(comp)
    if not cells <= border:
        return None

    missing = list(border - cells)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if int(grid[gap]) != BLACK:
        return None

    direction = _gap_and_direction(r0, c0, r1, c1, gap)
    if direction is None:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    dots = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(grid[r, c]) not in (BLACK, GREY)]
    if not dots:
        return None

    dot_colors = {int(grid[r, c]) for r, c in dots}
    if len(dot_colors) != 1:
        return None
    if np.any(interior == GREY):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "count": len(dots),
        "color": dot_colors.pop(),
        "dots": dots,
    }


def _find_rooms(grid: np.ndarray) -> list[dict]:
    rooms = []
    for comp in _components_of_color(grid, GREY):
        room = _parse_room(grid, comp)
        if room is not None:
            rooms.append(room)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _arrow_cells(gap: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    cells = [(gap[0] + dr * step, gap[1] + dc * step) for step in range(1, length + 1)]
    shaft_end = cells[-1]
    tip = (gap[0] + dr * (length + 1), gap[1] + dc * (length + 1))
    pdr, pdc = -dc, dr
    wings = [
        (shaft_end[0] + pdr, shaft_end[1] + pdc),
        (shaft_end[0] - pdr, shaft_end[1] - pdc),
    ]
    return cells + [tip] + wings


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find grey rectangular rooms with one border gap. The gap direction selects the
    arrow direction, the room's dot count selects shaft length, and the dot color
    selects arrow color. Dots are removed from the room interior.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _find_rooms(g):
        for r, c in room["dots"]:
            out[r, c] = BLACK
        for r, c in _arrow_cells(room["gap"], room["direction"], room["count"]):
            if _in_bounds(out.shape, (r, c)):
                out[r, c] = int(room["color"])

    return out


def _gap_for_room(r0: int, c0: int, h: int, w: int, direction_name: str) -> tuple[int, int]:
    mid_r = r0 + h // 2
    mid_c = c0 + w // 2
    if direction_name == "up":
        return (r0, mid_c)
    if direction_name == "down":
        return (r0 + h - 1, mid_c)
    if direction_name == "left":
        return (mid_r, c0)
    return (mid_r, c0 + w - 1)


def _room_bbox_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _sample_dots(
    rng: random.Random,
    r0: int,
    c0: int,
    h: int,
    w: int,
    count: int,
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) != 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return rng.sample(cells, count)


def _weighted_room_count(rng: random.Random, max_rooms: int) -> int:
    pool: list[int] = []
    for count, weight in ROOM_COUNT_BY_SIZE:
        if count <= max_rooms:
            pool.extend([count] * weight)
    return rng.choice(pool)


def _footprint_cells(
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction_name: str,
    count: int,
) -> set[tuple[int, int]]:
    gap = _gap_for_room(r0, c0, h, w, direction_name)
    direction = DIRS[direction_name]
    cells = _room_bbox_cells(r0, c0, h, w)
    cells.update(_arrow_cells(gap, direction, count))
    return cells


def _paint_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction_name: str,
    color: int,
    dots: list[tuple[int, int]],
) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    gap = _gap_for_room(r0, c0, h, w, direction_name)
    grid[gap] = BLACK
    for r, c in dots:
        grid[r, c] = color


def _validate_generated(grid: np.ndarray) -> bool:
    rooms = _find_rooms(grid)
    if not rooms:
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(interior != BLACK):
            return False
        if int(out[room["gap"]]) != BLACK:
            return False
        for cell in _arrow_cells(room["gap"], room["direction"], room["count"]):
            if not _in_bounds(grid.shape, cell):
                return False
            if int(out[cell]) != int(room["color"]):
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    max_rooms = 2
    if min(h, w) >= 23:
        max_rooms = 3
    if min(h, w) >= 28:
        max_rooms = 4
    room_count = _weighted_room_count(rng, max_rooms)

    occupied = np.zeros((h, w), dtype=bool)
    placed = 0

    while placed < room_count:
        done = False
        for _ in range(ROOM_PLACEMENT_TRIES):
            room_h = rng.choice(ROOM_HEIGHTS)
            room_w = rng.choice(ROOM_WIDTHS)
            direction_name = rng.choice(tuple(DIRS))
            dot_count = rng.randint(1, 5)
            color = rng.choice(PALETTE)

            if room_h >= h - 2 * EDGE_PAD or room_w >= w - 2 * EDGE_PAD:
                continue

            r0 = rng.randint(EDGE_PAD, h - room_h - EDGE_PAD)
            c0 = rng.randint(EDGE_PAD, w - room_w - EDGE_PAD)
            footprint = _footprint_cells(r0, c0, room_h, room_w, direction_name, dot_count)

            if not footprint:
                continue
            if any(not (EDGE_PAD <= r < h - EDGE_PAD and EDGE_PAD <= c < w - EDGE_PAD) for r, c in footprint):
                continue

            halo = _expand_cells(footprint, grid.shape, radius=1)
            if any(occupied[r, c] for r, c in halo):
                continue

            dots = _sample_dots(rng, r0, c0, room_h, room_w, dot_count)
            _paint_room(grid, r0, c0, room_h, room_w, direction_name, color, dots)
            for r, c in halo:
                occupied[r, c] = True
            placed += 1
            done = True
            break
        if not done:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
