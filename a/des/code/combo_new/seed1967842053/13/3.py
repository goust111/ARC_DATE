import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")


def _arrow_cells(direction, r, c):
    if direction == "up":
        return ((r, c - 1), (r, c + 1), (r + 1, c), (r + 2, c), (r + 3, c))
    if direction == "down":
        return ((r, c), (r + 1, c), (r + 2, c), (r + 3, c - 1), (r + 3, c + 1))
    if direction == "left":
        return ((r - 1, c), (r + 1, c), (r, c + 1), (r, c + 2), (r, c + 3))
    return ((r, c), (r, c + 1), (r, c + 2), (r - 1, c + 3), (r + 1, c + 3))


def _room_spans(lines, limit):
    bounds = [-1] + list(lines) + [limit]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            spans.append((a + 1, b))
    return spans


def _red_components(red_pixels):
    seen = set()
    comps = []
    for start in red_pixels:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in red_pixels and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
        comps.append(frozenset(comp))
    return comps


def _find_arrow(grid, r0, r1, c0, c1):
    red_pixels = {
        (r, c)
        for r in range(r0, r1)
        for c in range(c0, c1)
        if grid[r, c] == RED
    }
    candidates = []
    for direction in DIRECTIONS:
        for r in range(r0, r1):
            for c in range(c0, c1):
                cells = frozenset(_arrow_cells(direction, r, c))
                if len(cells) != 5:
                    continue
                if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in cells):
                    if all(grid[rr, cc] == RED for rr, cc in cells):
                        candidates.append((direction, cells))
    if not candidates:
        return None, frozenset()

    components = _red_components(red_pixels)
    for direction, cells in candidates:
        if cells in components:
            return direction, cells
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Push each room's colored pixels toward the wall indicated by its red arrow.

    Input structure:
    The grid has a black background with full grey grid lines that divide it into
    equal rectangular rooms. Each room contains scattered colored pixels and one
    distinguishable five-red-pixel arrow pointing up, down, left, or right.

    Rule:
    Preserve the grey grid, remove each arrow, and collect the other non-black,
    non-grey pixels inside that room. For a left or right arrow, keep pixels in
    increasing row order, preserving their original order within each row, and
    pack each row tightly against the indicated vertical wall. For an up or down
    arrow, keep pixels in increasing column order, preserving their original
    order within each column, and pack each column tightly against the indicated
    horizontal wall. All remaining room cells become black.

    Key insight:
    The arrow is both a direction marker and the cue for which perpendicular
    coordinate controls the stable packing order.
    """
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    out[grid == GREY] = GREY
    h, w = grid.shape

    grey_rows = np.where(np.all(grid == GREY, axis=1))[0]
    grey_cols = np.where(np.all(grid == GREY, axis=0))[0]
    row_spans = _room_spans(grey_rows, h)
    col_spans = _room_spans(grey_cols, w)

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            direction, arrow = _find_arrow(grid, r0, r1, c0, c1)
            if direction is None:
                continue

            colored = []
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if (r, c) in arrow:
                        continue
                    value = grid[r, c]
                    if value != BLACK and value != GREY:
                        colored.append((r, c, value))

            if direction in ("left", "right"):
                colored.sort(key=lambda item: item[0])
                counts = {}
                for r, c, value in colored:
                    offset = counts.get(r, 0)
                    counts[r] = offset + 1
                    nc = c0 + offset if direction == "left" else r1 * 0 + c1 - 1 - offset
                    out[r, nc] = value
            else:
                colored.sort(key=lambda item: item[1])
                counts = {}
                for r, c, value in colored:
                    offset = counts.get(c, 0)
                    counts[c] = offset + 1
                    nr = r0 + offset if direction == "up" else r1 - 1 - offset
                    out[nr, c] = value

    return out


def _place_random_arrow(grid, r0, r1, c0, c1, direction):
    if direction in ("up", "down"):
        r = random.randint(r0, r1 - 4)
        c = random.randint(c0 + 1, c1 - 2)
    else:
        r = random.randint(r0 + 1, r1 - 2)
        c = random.randint(c0, c1 - 4)
    cells = _arrow_cells(direction, r, c)
    for rr, cc in cells:
        grid[rr, cc] = RED
    return set(cells)


def generate_input() -> np.ndarray:
    rooms_y = random.randint(1, 3)
    rooms_x = random.randint(1, 3)
    room_h = random.randint(6, 8)
    room_w = random.randint(6, 8)
    h = rooms_y * room_h + rooms_y + 1
    w = rooms_x * room_w + rooms_x + 1

    grid = np.zeros((h, w), dtype=int)
    for y in range(rooms_y + 1):
        grid[y * (room_h + 1), :] = GREY
    for x in range(rooms_x + 1):
        grid[:, x * (room_w + 1)] = GREY

    for ry in range(rooms_y):
        for rx in range(rooms_x):
            r0 = ry * (room_h + 1) + 1
            c0 = rx * (room_w + 1) + 1
            r1 = r0 + room_h
            c1 = c0 + room_w
            direction = random.choice(DIRECTIONS)
            blocked = _place_random_arrow(grid, r0, r1, c0, c1, direction)
            free = [
                (r, c)
                for r in range(r0, r1)
                for c in range(c0, c1)
                if (r, c) not in blocked
            ]
            random.shuffle(free)
            count = random.randint(3, min(8, len(free)))
            for r, c in free[:count]:
                grid[r, c] = random.choice(COLORS)

    return grid
