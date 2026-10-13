import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PAYLOAD_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE)

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
CORRIDOR_COUNT_RANGE = (2, 5)
CORRIDOR_LEN_RANGE = (4, 9)
COLOR_COUNT_CAP = 5
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 300
PAD = 1


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_barrier(v: int) -> bool:
    return int(v) in (GREY, MAROON)


def _is_corridor_cell(v: int) -> bool:
    return int(v) not in (GREY, MAROON)


def _is_payload(v: int) -> bool:
    return int(v) not in (BLACK, GREY, MAROON)


def _has_horizontal_walls(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return (
        0 < r < h - 1
        and _is_barrier(int(grid[r - 1, c]))
        and _is_barrier(int(grid[r + 1, c]))
    )


def _has_vertical_walls(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return (
        0 < c < w - 1
        and _is_barrier(int(grid[r, c - 1]))
        and _is_barrier(int(grid[r, c + 1]))
    )


def _adjacent_to_maroon(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    for rr, cc in _neighbors4(r, c):
        if _inside(h, w, rr, cc) and int(grid[rr, cc]) == MAROON:
            return True
    return False


def _find_corridors(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    corridors: list[dict] = []
    used = np.zeros((h, w), dtype=bool)

    for r in range(1, h - 1):
        c = 1
        while c < w - 1:
            if (
                used[r, c]
                or not _is_corridor_cell(int(grid[r, c]))
                or not _has_horizontal_walls(grid, r, c)
                or _has_vertical_walls(grid, r, c)
            ):
                c += 1
                continue

            start = c
            cells = []
            while (
                c < w - 1
                and not used[r, c]
                and _is_corridor_cell(int(grid[r, c]))
                and _has_horizontal_walls(grid, r, c)
                and not _has_vertical_walls(grid, r, c)
            ):
                cells.append((r, c))
                c += 1
            end = c - 1

            if len(cells) < 2:
                continue
            if not (_is_barrier(int(grid[r, start - 1])) and _is_barrier(int(grid[r, end + 1]))):
                continue

            marked = [_adjacent_to_maroon(grid, *cells[0]), _adjacent_to_maroon(grid, *cells[-1])]
            if sum(marked) != 1:
                continue

            used[r, start : end + 1] = True
            corridors.append(
                {
                    "orientation": "horizontal",
                    "cells": cells,
                    "exit_at_start": bool(marked[0]),
                }
            )

    for c in range(1, w - 1):
        r = 1
        while r < h - 1:
            if (
                used[r, c]
                or not _is_corridor_cell(int(grid[r, c]))
                or not _has_vertical_walls(grid, r, c)
                or _has_horizontal_walls(grid, r, c)
            ):
                r += 1
                continue

            start = r
            cells = []
            while (
                r < h - 1
                and not used[r, c]
                and _is_corridor_cell(int(grid[r, c]))
                and _has_vertical_walls(grid, r, c)
                and not _has_horizontal_walls(grid, r, c)
            ):
                cells.append((r, c))
                r += 1
            end = r - 1

            if len(cells) < 2:
                continue
            if not (_is_barrier(int(grid[start - 1, c])) and _is_barrier(int(grid[end + 1, c]))):
                continue

            marked = [_adjacent_to_maroon(grid, *cells[0]), _adjacent_to_maroon(grid, *cells[-1])]
            if sum(marked) != 1:
                continue

            used[start : end + 1, c] = True
            corridors.append(
                {
                    "orientation": "vertical",
                    "cells": cells,
                    "exit_at_start": bool(marked[0]),
                }
            )

    return corridors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push every colored corridor marker away from the maroon-marked exit and stack it at the opposite end.

    Input structure:
    - The background is black.
    - Grey pixels form thin walls around several straight 1-cell-wide corridors; each corridor is horizontal or vertical.
    - Corridor cells are black except for scattered colored marker pixels, and exactly one corridor endpoint is adjacent to a maroon pixel that marks the exit end.

    Rule:
    - Detect each straight corridor by finding maximal non-grey runs that are bracketed by grey side walls.
    - Use the endpoint adjacent to maroon as the exit end.
    - Move every colored marker in that corridor away from the exit and compress all markers against the non-exit end.
    - Preserve the order by original distance from the non-exit end: markers already closer to that end stay deeper in the final stack.
    - Cells vacated by markers become black; grey walls and maroon markers remain unchanged.

    Key insight:
    - Maroon does not indicate where markers should gather; it marks the forbidden exit, so the actual packing happens at the opposite end of each walled corridor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for corridor in _find_corridors(g):
        cells = corridor["cells"]
        ordered = list(reversed(cells)) if corridor["exit_at_start"] else list(cells)
        colors = [int(g[r, c]) for r, c in ordered if _is_payload(int(g[r, c]))]

        for r, c in cells:
            out[r, c] = BLACK
        for (r, c), color in zip(ordered, colors):
            out[r, c] = int(color)

    return out


def _reserve(used: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = used.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(h - 1, r1 + PAD)
    cc1 = min(w - 1, c1 + PAD)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(used: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = used.shape
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(h - 1, r1 + PAD)
    cc1 = min(w - 1, c1 + PAD)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _is_already_stacked(length: int, exit_at_start: bool, positions: list[int]) -> bool:
    ordered = list(range(length - 1, -1, -1)) if exit_at_start else list(range(length))
    occupied = [idx for idx in ordered if idx in set(positions)]
    return occupied == ordered[: len(positions)]


def _paint_horizontal(
    grid: np.ndarray,
    used: np.ndarray,
    length: int,
) -> dict | None:
    h, w = grid.shape
    if w < length + 2:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r = random.randint(1, h - 2)
        c0 = random.randint(1, w - length - 1)
        c1 = c0 + length - 1
        bbox = (r - 1, c0 - 1, r + 1, c1 + 1)
        if not _can_place(used, bbox):
            continue

        grid[r - 1, c0 - 1 : c1 + 2] = GREY
        grid[r + 1, c0 - 1 : c1 + 2] = GREY
        grid[r, c0 - 1] = GREY
        grid[r, c1 + 1] = GREY
        grid[r, c0 : c1 + 1] = BLACK

        exit_at_start = random.choice([False, True])
        marker_mode = random.choice(["axis", "wall"])
        if exit_at_start:
            if marker_mode == "axis":
                grid[r, c0 - 1] = MAROON
            else:
                grid[random.choice([r - 1, r + 1]), c0] = MAROON
        else:
            if marker_mode == "axis":
                grid[r, c1 + 1] = MAROON
            else:
                grid[random.choice([r - 1, r + 1]), c1] = MAROON

        color_count = random.randint(1, min(COLOR_COUNT_CAP, length - 1))
        positions = sorted(random.sample(range(length), color_count))
        colors = random.choices(PAYLOAD_COLORS, k=color_count)
        for idx, color in zip(positions, colors):
            grid[r, c0 + idx] = int(color)

        _reserve(used, bbox)
        return {
            "length": length,
            "exit_at_start": exit_at_start,
            "positions": positions,
        }

    return None


def _paint_vertical(
    grid: np.ndarray,
    used: np.ndarray,
    length: int,
) -> dict | None:
    h, w = grid.shape
    if h < length + 2:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        c = random.randint(1, w - 2)
        r0 = random.randint(1, h - length - 1)
        r1 = r0 + length - 1
        bbox = (r0 - 1, c - 1, r1 + 1, c + 1)
        if not _can_place(used, bbox):
            continue

        grid[r0 - 1, c] = GREY
        grid[r1 + 1, c] = GREY
        grid[r0 : r1 + 1, c - 1] = GREY
        grid[r0 : r1 + 1, c + 1] = GREY
        grid[r0 : r1 + 1, c] = BLACK

        exit_at_start = random.choice([False, True])
        marker_mode = random.choice(["axis", "wall"])
        if exit_at_start:
            if marker_mode == "axis":
                grid[r0 - 1, c] = MAROON
            else:
                grid[r0, random.choice([c - 1, c + 1])] = MAROON
        else:
            if marker_mode == "axis":
                grid[r1 + 1, c] = MAROON
            else:
                grid[r1, random.choice([c - 1, c + 1])] = MAROON

        color_count = random.randint(1, min(COLOR_COUNT_CAP, length - 1))
        positions = sorted(random.sample(range(length), color_count))
        colors = random.choices(PAYLOAD_COLORS, k=color_count)
        for idx, color in zip(positions, colors):
            grid[r0 + idx, c] = int(color)

        _reserve(used, bbox)
        return {
            "length": length,
            "exit_at_start": exit_at_start,
            "positions": positions,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    corridor_count = random.randint(*CORRIDOR_COUNT_RANGE)
    orientations = [random.choice(["horizontal", "vertical"]) for _ in range(corridor_count)]
    if corridor_count >= 2 and random.random() < 0.5:
        orientations[0] = "horizontal"
        orientations[1] = "vertical"
        random.shuffle(orientations)

    move_needed = False
    for orientation in orientations:
        max_len = min(CORRIDOR_LEN_RANGE[1], w - 2) if orientation == "horizontal" else min(CORRIDOR_LEN_RANGE[1], h - 2)
        min_len = CORRIDOR_LEN_RANGE[0]
        if max_len < min_len:
            return None
        length = random.randint(min_len, max_len)

        if orientation == "horizontal":
            spec = _paint_horizontal(grid, used, length)
        else:
            spec = _paint_vertical(grid, used, length)
        if spec is None:
            return None

        if not _is_already_stacked(spec["length"], spec["exit_at_start"], spec["positions"]):
            move_needed = True

    if not move_needed:
        return None
    if len(_find_corridors(grid)) != corridor_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor puzzle input")
