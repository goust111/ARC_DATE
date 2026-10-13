from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))
MIN_RAY_DISTANCE = 4
PLACEMENT_PAD = 1
MAX_GENERATION_ATTEMPTS = 300


def _perp(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = tip
    dr, dc = direction
    pr, pc = _perp(direction)
    return {
        (tr, tc),
        (tr + pr, tc + pc),
        (tr - pr, tc - pc),
        (tr - dr, tc - dc),
        (tr - 2 * dr, tc - 2 * dc),
    }


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cells_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, cell) for cell in cells)


def _ray_distance(
    tip: tuple[int, int],
    anchor: tuple[int, int],
    direction: tuple[int, int],
) -> int | None:
    tr, tc = tip
    ar, ac = anchor
    dr, dc = direction

    if dr == 0:
        if ar != tr:
            return None
        delta = ac - tc
        if delta * dc <= 0:
            return None
        return abs(delta)

    if ac != tc:
        return None
    delta = ar - tr
    if delta * dr <= 0:
        return None
    return abs(delta)


def _detect_arrow(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []
    for tip in coords:
        for direction in DIRECTIONS:
            arrow = _arrow_cells(tip, direction)
            if not arrow.issubset(coords):
                continue
            extras = coords - arrow
            if len(extras) != 1:
                continue
            anchor = next(iter(extras))
            distance = _ray_distance(tip, anchor, direction)
            if distance is None:
                continue
            candidates.append((tip, direction, anchor, distance))

    assert len(candidates) == 1, "each color must encode exactly one arrow and one anchor"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for color in [int(c) for c in np.unique(g) if int(c) != BG]:
        coords = {tuple(map(int, coord)) for coord in np.argwhere(g == color)}
        tip, direction, anchor, distance = _detect_arrow(coords)
        tr, tc = tip
        dr, dc = direction

        for step in range(1, distance + 1):
            out[tr + dr * step, tc + dc * step] = color

        for r, c in _arrow_cells(anchor, (-dr, -dc)):
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            out[r, c] = color

    return out


def _reserve_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - PLACEMENT_PAD, r + PLACEMENT_PAD + 1):
            for cc in range(c - PLACEMENT_PAD, c + PLACEMENT_PAD + 1):
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    reserved.add((rr, cc))
    return reserved


def _build_structure(
    shape: tuple[int, int],
    tip: tuple[int, int],
    direction: tuple[int, int],
    distance: int,
) -> dict[str, object] | None:
    original = _arrow_cells(tip, direction)
    if not _cells_in_bounds(shape, original):
        return None

    tr, tc = tip
    dr, dc = direction
    anchor = (tr + dr * distance, tc + dc * distance)
    mirrored = _arrow_cells(anchor, (-dr, -dc))
    if not _cells_in_bounds(shape, mirrored):
        return None

    line = {(tr + dr * step, tc + dc * step) for step in range(1, distance + 1)}
    output_cells = original | line | mirrored
    input_cells = original | {anchor}
    return {
        "tip": tip,
        "direction": direction,
        "distance": distance,
        "anchor": anchor,
        "input_cells": input_cells,
        "output_cells": output_cells,
        "reserved": _reserve_cells(shape, output_cells),
    }


def _candidate_pairs(shape: tuple[int, int], direction: tuple[int, int]) -> list[tuple[tuple[int, int], int]]:
    h, w = shape
    candidates: list[tuple[tuple[int, int], int]] = []
    limit = max(h, w)
    for r in range(h):
        for c in range(w):
            tip = (r, c)
            if not _cells_in_bounds(shape, _arrow_cells(tip, direction)):
                continue
            for distance in range(MIN_RAY_DISTANCE, limit):
                struct = _build_structure(shape, tip, direction, distance)
                if struct is None:
                    break
                candidates.append((tip, distance))
    random.shuffle(candidates)
    return candidates


def _place_objects(shape: tuple[int, int], directions: list[tuple[int, int]]) -> list[dict[str, object]] | None:
    blocked: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    order = directions[:]
    random.shuffle(order)
    for direction in order:
        placed = None
        for tip, distance in _candidate_pairs(shape, direction):
            struct = _build_structure(shape, tip, direction, distance)
            assert struct is not None
            reserved = struct["reserved"]
            assert isinstance(reserved, set)
            if reserved & blocked:
                continue
            placed = struct
            blocked |= reserved
            placements.append(struct)
            break
        if placed is None:
            return None

    return placements


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(1, 4)
    side_lo = 12 + 2 * count
    side_hi = side_lo + 8
    shape = (random.randint(side_lo, side_hi), random.randint(side_lo, side_hi))
    directions = random.sample(list(DIRECTIONS), count)

    placements = _place_objects(shape, directions)
    if placements is None:
        return None

    colors = random.sample(COLOR_POOL, count)
    grid = np.zeros(shape, dtype=int)
    for struct, color in zip(placements, colors):
        input_cells = struct["input_cells"]
        assert isinstance(input_cells, set)
        for r, c in input_cells:
            assert grid[r, c] == BG
            grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    for color in colors:
        if int(np.count_nonzero(out == color)) <= int(np.count_nonzero(grid == color)):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-anchor input")
