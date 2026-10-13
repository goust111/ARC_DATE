from __future__ import annotations

import random

import numpy as np


BG = 8
COLOR_POOL = (0, 1, 2, 3, 4, 5, 6, 7, 9)
DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
LENGTH_RANGE = (3, 6)
OBJECT_COUNT_RANGE = (2, 4)
MAX_GRID_SIDE = 20
GEN_MAX_ATTEMPTS = 300


def _background(g: np.ndarray) -> int:
    if np.any(g == BG):
        return BG
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_distance(r: int, c: int, shape: tuple[int, int]) -> int:
    h, w = shape
    return min(r, c, h - 1 - r, w - 1 - c)


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _line_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    return [(sr + step * dr, sc + step * dc) for step in range(length)]


def _arrow_cells(end: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = end
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _ordered_track(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert LENGTH_RANGE[0] <= len(coords) <= LENGTH_RANGE[1], "track length must be 3..6"
    coords = sorted(coords)
    if len({r - c for r, c in coords}) == 1:
        r0, c0 = coords[0]
        expected = [(r0 + i, c0 + i) for i in range(len(coords))]
    elif len({r + c for r, c in coords}) == 1:
        r0, c0 = coords[0]
        expected = [(r0 + i, c0 - i) for i in range(len(coords))]
    else:
        raise AssertionError("track must lie on one 45-degree diagonal")
    assert coords == expected, "track must be contiguous"
    return coords


def _track_direction(coords: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    ordered = _ordered_track(coords)
    a = ordered[0]
    b = ordered[-1]
    da = _border_distance(a[0], a[1], shape)
    db = _border_distance(b[0], b[1], shape)
    if da < db:
        start, end = a, b
    elif db < da:
        start, end = b, a
    else:
        start, end = min(a, b), max(a, b)
    direction = (_sign(end[0] - start[0]), _sign(end[1] - start[1]))
    assert abs(direction[0]) == 1 and abs(direction[1]) == 1
    return end, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each monochrome diagonal track into a 3-pixel arrowhead at the track end.

    Each non-background color forms one contiguous diagonal track of length 3..6.
    The start is the endpoint closer to the border; the arrow tip is the opposite end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    out = np.full_like(g, bg)
    painted: dict[tuple[int, int], int] = {}

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        coords = [tuple(map(int, cell)) for cell in np.argwhere(g == color)]
        assert coords, "every foreground color must have pixels"
        end, direction = _track_direction(coords, g.shape)
        for cell in _arrow_cells(end, direction):
            if cell in painted:
                raise AssertionError("arrowheads must not overlap")
            painted[cell] = color

    for (r, c), color in painted.items():
        out[r, c] = color
    return out


def _candidate_starts(shape: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    starts: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            er = r + (length - 1) * dr
            ec = c + (length - 1) * dc
            if not (0 <= er < h and 0 <= ec < w):
                continue
            if _border_distance(r, c, shape) >= _border_distance(er, ec, shape):
                continue
            starts.append((r, c))
    return starts


def _place_track(
    shape: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    blocked: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    starts = _candidate_starts(shape, direction, length)
    random.shuffle(starts)
    for start in starts:
        line = _line_cells(start, direction, length)
        arrow = _arrow_cells(line[-1], direction)
        footprint = set(line) | arrow
        if any(cell in blocked for cell in footprint):
            continue
        return line, footprint
    return None


def _sample_specs(n_objects: int) -> list[tuple[int, tuple[int, int], int]]:
    colors = random.sample(COLOR_POOL, n_objects)
    directions = random.sample(DIRS, n_objects)
    lengths = [random.randint(*LENGTH_RANGE) for _ in range(n_objects)]
    specs = list(zip(colors, directions, lengths))
    specs.sort(key=lambda item: item[2], reverse=True)
    return specs


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    min_side = 11 + n_objects
    h = random.randint(min_side, MAX_GRID_SIDE)
    w = random.randint(min_side, MAX_GRID_SIDE)
    shape = (h, w)
    grid = np.full(shape, BG, dtype=int)

    blocked: set[tuple[int, int]] = set()
    placed: list[tuple[int, list[tuple[int, int]]]] = []
    for color, direction, length in _sample_specs(n_objects):
        result = _place_track(shape, direction, length, blocked)
        if result is None:
            return None
        line, footprint = result
        placed.append((color, line))
        blocked |= _expanded(footprint, shape)

    for color, line in placed:
        for r, c in line:
            grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if int(np.sum(out != BG)) != 3 * n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
