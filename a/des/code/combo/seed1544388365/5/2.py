from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
FRAGMENT_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _trace_rail(grid: np.ndarray, marker: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    mr, mc = marker
    starts: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = mr + dr, mc + dc
        if _in_bounds(h, w, rr, cc) and int(grid[rr, cc]) not in (BG, RED):
            starts.append((dr, dc))

    assert len(starts) == 1, "each red marker must touch exactly one rail endpoint"
    dr, dc = starts[0]

    cells: list[tuple[int, int]] = []
    r, c = mr + dr, mc + dc
    while _in_bounds(h, w, r, c) and int(grid[r, c]) not in (BG, RED):
        cells.append((int(r), int(c)))
        r += dr
        c += dc

    assert cells, "a rail must contain at least one cell"
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored fragments to the rail end indicated by a red marker.

    For each grey horizontal/vertical rail, the red pixel adjacent to one outer
    endpoint marks the stacking end. Fragments on the rail are sorted by their
    original distance to that end (farthest first) and then packed tightly from
    the end inward. Source fragment cells revert to grey, and the red marker is
    removed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    for marker in markers:
        rail = _trace_rail(g, marker)
        fragments = [int(g[r, c]) for r, c in rail if int(g[r, c]) != GREY]

        out[marker] = BG
        for r, c in rail:
            out[r, c] = GREY
        for (r, c), color in zip(rail, reversed(fragments)):
            out[r, c] = int(color)

    return out


def _buffer_cells(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(h, w, rr, cc):
                    reserved.add((int(rr), int(cc)))
    return reserved


def _rail_cells(marker: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    mr, mc = marker
    dr, dc = direction
    return [(int(mr + dr * step), int(mc + dc * step)) for step in range(1, length + 1)]


def _random_positions(length: int, count: int) -> list[int] | None:
    for _ in range(30):
        pos = sorted(random.sample(range(length), count))
        if pos != list(range(count)):
            return pos
    return None


def _choose_marker_and_direction(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    orientation = random.choice(["h", "v"])
    if orientation == "h":
        max_len = min(10, w - 3)
        if max_len < 4:
            return None
        length = random.randint(4, max_len)
        row = random.randint(1, h - 2)
        if random.random() < 0.5:
            marker = (row, random.randint(1, w - length - 2))
            direction = (0, 1)
        else:
            marker = (row, random.randint(length + 1, w - 2))
            direction = (0, -1)
        return marker, direction, length

    max_len = min(10, h - 3)
    if max_len < 4:
        return None
    length = random.randint(4, max_len)
    col = random.randint(1, w - 2)
    if random.random() < 0.5:
        marker = (random.randint(1, h - length - 2), col)
        direction = (1, 0)
    else:
        marker = (random.randint(length + 1, h - 2), col)
        direction = (-1, 0)
    return marker, direction, length


def _place_rail(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    choice = _choose_marker_and_direction(h, w)
    if choice is None:
        return False

    marker, direction, length = choice
    rail = _rail_cells(marker, direction, length)
    occupied = [marker, *rail]
    halo = _buffer_cells(occupied, h, w)
    if any(cell in reserved for cell in halo):
        return False

    fragment_count = random.randint(2, min(5, length - 1))
    fragment_positions = _random_positions(length, fragment_count)
    if fragment_positions is None:
        return False
    fragment_colors = random.sample(FRAGMENT_COLORS, fragment_count)

    grid[marker] = RED
    for r, c in rail:
        grid[r, c] = GREY
    for idx, color in zip(fragment_positions, fragment_colors):
        r, c = rail[idx]
        grid[r, c] = int(color)

    reserved.update(halo)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 22)
    w = random.randint(10, 22)
    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()

    if h * w >= 260:
        rail_count = random.randint(2, 4)
    elif h * w >= 170:
        rail_count = random.randint(2, 3)
    else:
        rail_count = random.randint(1, 2)

    placed = 0
    for _ in range(rail_count * 40):
        if _place_rail(grid, reserved):
            placed += 1
            if placed == rail_count:
                break
    if placed != rail_count:
        return None

    if np.count_nonzero(grid == RED) != rail_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
