import random

import numpy as np


BG = 0
WALL = 5
AZURE = 8
SEED_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ROOM_COUNT_RANGE = (2, 4)
GRID_SIZE_RANGE = (18, 30)
INTERIOR_SIZE_CHOICES = (
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
    (4, 4),
)
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(pts)
    return components


def _extract_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rooms: list[dict[str, object]] = []
    room_seed_positions: set[tuple[int, int]] = set()

    for cells in _components(g != WALL):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in {BG, WALL, AZURE}]
        if not seeds:
            continue
        assert len(seeds) == 1, "each room must contain exactly one seed"
        seed = seeds[0]
        room_seed_positions.add(seed)
        cell_set = set(cells)
        markers = []
        for nr, nc in _neighbors4(*seed):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in cell_set and int(g[nr, nc]) == AZURE:
                markers.append((nr, nc))
        assert len(markers) == 1, "each seed must have exactly one adjacent azure direction marker"
        marker = markers[0]
        rooms.append(
            {
                "cells": cells,
                "seed": seed,
                "seed_color": int(g[seed]),
                "marker": marker,
                "direction": (marker[0] - seed[0], marker[1] - seed[1]),
                "area": len(cells),
            }
        )

    all_seeds = {
        tuple(map(int, pos))
        for pos in np.argwhere((g != BG) & (g != WALL) & (g != AZURE))
    }
    assert room_seed_positions == all_seeds, "all colored non-wall cells must be room seeds"
    assert int(np.count_nonzero(g == AZURE)) == len(rooms), "each room must have one azure marker"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a colored ray out of each walled room in the direction indicated by its azure marker.

    Input structure:
    - The background is black.
    - Grey pixels form closed rectangular walls; each enclosed interior is a room.
    - Every room contains exactly one colored seed pixel and exactly one azure
      pixel that is 4-neighbor-adjacent to that seed.

    Rule:
    - For each room, measure the room interior area as the full count of
      non-wall cells inside it, including black cells, the seed, and the azure marker.
    - The azure marker's side relative to the seed gives the growth direction.
    - Starting from the seed, draw a straight line of the seed color in that
      direction, with length equal to the room area and excluding the seed cell itself.
    - The line may overwrite walls and other non-seed content, but colored seed pixels
      remain preserved; azure markers are removed from the final grid.

    Key insight:
    - Direction is encoded locally by the azure neighbor, while line length is encoded
      globally by the area of that seed's enclosed room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rooms = _extract_rooms(g)
    seed_positions = {room["seed"] for room in rooms}

    out = g.copy()
    out[out == AZURE] = BG

    h, w = g.shape
    for room in rooms:
        sr, sc = room["seed"]
        dr, dc = room["direction"]
        color = int(room["seed_color"])
        length = int(room["area"])
        for step in range(1, length + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if (rr, cc) in seed_positions:
                continue
            out[rr, cc] = color

    return out


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ta, la, ba, ra = box_a
    tb, lb, bb, rb = box_b
    return not (ba + pad < tb or bb + pad < ta or ra + pad < lb or rb + pad < la)


def _sample_seed_local(rng: random.Random, ih: int, iw: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        row_choices = list(range(max(1, ih - 2), ih))
        return rng.choice(row_choices), rng.randrange(iw)
    if direction == "down":
        row_choices = list(range(0, min(2, ih - 1)))
        return rng.choice(row_choices), rng.randrange(iw)
    if direction == "left":
        col_choices = list(range(max(1, iw - 2), iw))
        return rng.randrange(ih), rng.choice(col_choices)
    col_choices = list(range(0, min(2, iw - 1)))
    return rng.randrange(ih), rng.choice(col_choices)


def _top_range(height: int, ih: int, seed_local_r: int, area: int, direction: str) -> tuple[int, int]:
    top_lo = 1
    top_hi = height - ih - 3
    if direction == "up":
        top_lo = max(top_lo, area - 1 - seed_local_r)
    elif direction == "down":
        top_hi = min(top_hi, height - area - seed_local_r - 2)
    return top_lo, top_hi


def _left_range(width: int, iw: int, seed_local_c: int, area: int, direction: str) -> tuple[int, int]:
    left_lo = 1
    left_hi = width - iw - 3
    if direction == "left":
        left_lo = max(left_lo, area - 1 - seed_local_c)
    elif direction == "right":
        left_hi = min(left_hi, width - area - seed_local_c - 2)
    return left_lo, left_hi


def _room_path(seed: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIR_TO_DELTA[direction]
    sr, sc = seed
    return [(sr + dr * step, sc + dc * step) for step in range(1, length + 1)]


def _paint_room(
    grid: np.ndarray,
    top: int,
    left: int,
    ih: int,
    iw: int,
    seed_local: tuple[int, int],
    direction: str,
    color: int,
) -> dict[str, object]:
    bottom = top + ih + 1
    right = left + iw + 1
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL

    seed = (top + 1 + seed_local[0], left + 1 + seed_local[1])
    dr, dc = DIR_TO_DELTA[direction]
    marker = (seed[0] + dr, seed[1] + dc)
    grid[seed] = color
    grid[marker] = AZURE
    return {
        "bbox": (top, left, bottom, right),
        "seed": seed,
        "marker": marker,
        "path": _room_path(seed, direction, ih * iw),
    }


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIZE_RANGE)
    width = rng.randint(*GRID_SIZE_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    room_count = rng.randint(*ROOM_COUNT_RANGE)
    colors = list(SEED_COLORS)
    rng.shuffle(colors)

    directions = list(DIR_TO_DELTA)
    rng.shuffle(directions)
    while len(directions) < room_count:
        extra = list(DIR_TO_DELTA)
        rng.shuffle(extra)
        directions.extend(extra)

    room_boxes: list[tuple[int, int, int, int]] = []
    blocked_output_cells: set[tuple[int, int]] = set()
    seed_positions: set[tuple[int, int]] = set()

    for room_index in range(room_count):
        direction = directions[room_index]
        color = colors[room_index]
        placed = False

        for _ in range(PLACEMENT_RETRIES):
            ih, iw = rng.choice(INTERIOR_SIZE_CHOICES)
            area = ih * iw
            seed_local = _sample_seed_local(rng, ih, iw, direction)

            top_lo, top_hi = _top_range(height, ih, seed_local[0], area, direction)
            left_lo, left_hi = _left_range(width, iw, seed_local[1], area, direction)
            if top_lo > top_hi or left_lo > left_hi:
                continue

            top = rng.randint(top_lo, top_hi)
            left = rng.randint(left_lo, left_hi)
            bottom = top + ih + 1
            right = left + iw + 1
            bbox = (top, left, bottom, right)

            if any(_boxes_overlap(bbox, other, pad=1) for other in room_boxes):
                continue

            seed = (top + 1 + seed_local[0], left + 1 + seed_local[1])
            path = _room_path(seed, direction, area)
            if any(not (0 <= r < height and 0 <= c < width) for r, c in path):
                continue
            if seed in blocked_output_cells:
                continue
            if any(cell in seed_positions or cell in blocked_output_cells for cell in path):
                continue

            room = _paint_room(grid, top, left, ih, iw, seed_local, direction, color)
            room_boxes.append(bbox)
            seed_positions.add(room["seed"])

            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    blocked_output_cells.add((r, c))
            blocked_output_cells.update(room["path"])
            placed = True
            break

        if not placed:
            return None

    rooms = _extract_rooms(grid)
    if len(rooms) != room_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
