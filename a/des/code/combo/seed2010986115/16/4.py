from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
LEGEND_LEN = 4

# --------------- solver ---------------


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
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
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _legend_colors(g: np.ndarray) -> list[int]:
    cols = np.flatnonzero(g[-1] != BLACK)
    assert len(cols) == LEGEND_LEN, "expected exactly four legend pixels on the bottom row"
    return [int(g[-1, c]) for c in cols]


def _room_ring(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    ring: list[tuple[int, int]] = []
    for c in range(c0 - 1, c1 + 2):
        ring.append((r0 - 1, c))
        ring.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        ring.append((r, c0 - 1))
        ring.append((r, c1 + 1))
    return ring


def _find_rooms(g: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, g.shape)
    rooms: list[dict[str, object]] = []

    for comp in _components(g == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue

        doors: list[tuple[int, int]] = []
        valid = True
        for rr, cc in _room_ring(r0, c0, r1, c1):
            val = int(g[rr, cc])
            if val not in (GREY, MAROON):
                valid = False
                break
            if val == MAROON:
                doors.append((int(rr), int(cc)))
        if not valid:
            continue

        if 1 <= len(doors) <= LEGEND_LEN:
            rooms.append(
                {
                    "interior": (int(r0), int(c0), int(r1), int(c1)),
                    "doors": doors,
                }
            )

    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed room according to its number of maroon doors, then recolor
    the doors with the palette color mirrored around the center of the 4-pixel
    legend on the bottom row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= LEGEND_LEN

    palette = _legend_colors(g)
    rooms = _find_rooms(g)
    assert rooms, "expected at least one closed room"

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["interior"]
        doors = room["doors"]
        door_count = len(doors)
        fill_color = int(palette[door_count - 1])
        door_color = int(palette[LEGEND_LEN - door_count])
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
        for rr, cc in doors:
            out[rr, cc] = door_color
    return out


# --------------- generator ---------------

PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8)
GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (18, 30)
ROOM_COUNT_RANGE = (2, 5)
INTERIOR_H_RANGE = (2, 6)
INTERIOR_W_RANGE = (2, 7)
ROOM_PLACE_TRIES = 100
GENERATION_ATTEMPTS = 400


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _door_candidates(top: int, left: int, bottom: int, right: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(left + 1, right):
        pts.append((top, c))
        pts.append((bottom, c))
    for r in range(top + 1, bottom):
        pts.append((r, left))
        pts.append((r, right))
    return pts


def _sample_doors(top: int, left: int, bottom: int, right: int, count: int) -> list[tuple[int, int]]:
    candidates = _door_candidates(top, left, bottom, right)
    for _ in range(80):
        doors = random.sample(candidates, count)
        if all(abs(r1 - r2) + abs(c1 - c2) > 1 for i, (r1, c1) in enumerate(doors) for r2, c2 in doors[i + 1 :]):
            return doors
    return random.sample(candidates, count)


def _sample_door_counts(room_count: int) -> list[int]:
    distinct = random.randint(2, min(LEGEND_LEN, room_count))
    counts = random.sample(range(1, LEGEND_LEN + 1), distinct)
    while len(counts) < room_count:
        counts.append(random.randint(1, LEGEND_LEN))
    random.shuffle(counts)
    return counts


def _render_input(
    grid_h: int,
    grid_w: int,
    palette: list[int],
    legend_start: int,
    rooms: list[dict[str, object]],
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[-1, legend_start : legend_start + LEGEND_LEN] = np.array(palette, dtype=int)

    for room in rooms:
        top, left, bottom, right = room["outer"]
        grid[top, left : right + 1] = GREY
        grid[bottom, left : right + 1] = GREY
        grid[top : bottom + 1, left] = GREY
        grid[top : bottom + 1, right] = GREY
        for rr, cc in room["doors"]:
            grid[rr, cc] = MAROON
    return grid


def _expected_output(inp: np.ndarray, palette: list[int], rooms: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        top, left, bottom, right = room["outer"]
        door_count = int(room["door_count"])
        out[top + 1 : bottom, left + 1 : right] = int(palette[door_count - 1])
        for rr, cc in room["doors"]:
            out[rr, cc] = int(palette[LEGEND_LEN - door_count])
    return out


def _sample_rooms(grid_h: int, grid_w: int) -> list[dict[str, object]] | None:
    usable_area = (grid_h - 3) * (grid_w - 2)
    max_rooms = min(ROOM_COUNT_RANGE[1], max(ROOM_COUNT_RANGE[0], usable_area // 40))
    room_count = random.randint(ROOM_COUNT_RANGE[0], max_rooms)
    door_counts = _sample_door_counts(room_count)

    rooms: list[dict[str, object]] = []
    blocked: list[tuple[int, int, int, int]] = []

    for door_count in door_counts:
        placed = False
        for _ in range(ROOM_PLACE_TRIES):
            interior_h = random.randint(*INTERIOR_H_RANGE)
            interior_w = random.randint(*INTERIOR_W_RANGE)
            outer_h = interior_h + 2
            outer_w = interior_w + 2
            max_top = grid_h - outer_h - 2
            max_left = grid_w - outer_w - 1
            if max_top < 1 or max_left < 1:
                return None

            top = random.randint(1, max_top)
            left = random.randint(1, max_left)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            outer = (top, left, bottom, right)
            padded = (top - 1, left - 1, bottom + 1, right + 1)

            if any(_boxes_overlap(padded, other) for other in blocked):
                continue

            doors = _sample_doors(top, left, bottom, right, int(door_count))
            rooms.append(
                {
                    "outer": outer,
                    "doors": doors,
                    "door_count": int(door_count),
                }
            )
            blocked.append(padded)
            placed = True
            break
        if not placed:
            return None

    return rooms


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    palette = random.sample(PALETTE_POOL, LEGEND_LEN)
    legend_start = random.randint(0, grid_w - LEGEND_LEN)

    rooms = _sample_rooms(grid_h, grid_w)
    if rooms is None:
        return None

    inp = _render_input(grid_h, grid_w, palette, legend_start, rooms)
    expected = _expected_output(inp, palette, rooms)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door palette puzzle")
