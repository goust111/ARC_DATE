from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator bounds.
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (2, 5)
RAY_DISTANCE_RANGE = (3, 7)
DECOY_COUNT_RANGE = (0, 6)
PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r) + dr, int(c) + dc) for dr, dc in DIRS4]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r) + dr, int(c) + dc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _l_descriptor(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    if len(coords) != 3:
        return None

    cells = set(coords)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    if max(rows) - min(rows) != 1 or max(cols) - min(cols) != 1:
        return None

    degrees = {
        cell: sum((nbr in cells) for nbr in _neighbors4(*cell))
        for cell in coords
    }
    corners = [cell for cell, degree in degrees.items() if degree == 2]
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    if len(corners) != 1 or len(endpoints) != 2:
        return None

    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    square = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = tuple((square - cells).pop())
    return corners[0], endpoints, missing


def _is_isolated_singleton(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    if int(grid[r, c]) != int(color):
        return False
    for nr, nc in _neighbors8(r, c):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == int(color):
            return False
    return True


def _ray_target(
    grid: np.ndarray,
    color: int,
    endpoint: tuple[int, int],
    step: tuple[int, int],
) -> tuple[int, int] | None:
    r = int(endpoint[0] + step[0])
    c = int(endpoint[1] + step[1])
    while _in_bounds(grid.shape, r, c):
        if int(grid[r, c]) == int(color):
            if _is_isolated_singleton(grid, r, c, color):
                return (r, c)
            return None
        r += step[0]
        c += step[1]
    return None


def _plan_moves(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    moves: list[tuple[int, int, int, int, int]] = []
    used_sources: set[tuple[int, int]] = set()
    used_targets: set[tuple[int, int]] = set()

    for color in map(int, np.unique(g)):
        if color == BLACK:
            continue
        for comp in _components(g == color):
            desc = _l_descriptor(comp)
            if desc is None:
                continue

            corner, endpoints, missing = desc
            if int(g[missing]) != BLACK:
                continue

            matches: list[tuple[int, int]] = []
            for endpoint in endpoints:
                step = (int(endpoint[0] - corner[0]), int(endpoint[1] - corner[1]))
                target = _ray_target(g, color, endpoint, step)
                if target is not None:
                    matches.append(target)

            if len(matches) == 0:
                continue
            assert len(matches) == 1, "L pointer must have exactly one valid search direction"

            source = matches[0]
            assert source not in used_sources, "singleton source cannot be reused"
            assert missing not in used_targets, "two L objects cannot fill the same cell"
            used_sources.add(source)
            used_targets.add(missing)
            moves.append((source[0], source[1], missing[0], missing[1], color))

    assert moves, "expected at least one movable L pointer"
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each same-colored isolated singleton into the missing corner of its matching L shape.

    Each transformable object is a monochromatic 3-cell L occupying three cells of a
    2x2 square. One of its two arm directions is singled out by the presence of the
    first same-colored isolated pixel farther out on that ray. That pixel is removed
    and the missing 2x2 corner is filled with the same color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for sr, sc, tr, tc, color in _plan_moves(g):
        out[sr, sc] = BLACK
        out[tr, tc] = int(color)
    return out


def _perpendiculars(step: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = step
    return ((dc, -dr), (-dc, dr))


def _protected_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    protected.add((int(nr), int(nc)))
    return protected


def _object_spec(
    shape: tuple[int, int],
    color: int,
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        search = random.choice(DIRS4)
        side = random.choice(_perpendiculars(search))
        distance = random.randint(*RAY_DISTANCE_RANGE)

        rel = [
            (0, 0),
            search,
            side,
            (search[0] + side[0], search[1] + side[1]),
            (search[0] * distance, search[1] * distance),
        ]
        row_offsets = [r for r, _c in rel]
        col_offsets = [c for _r, c in rel]
        r_lo = -min(row_offsets)
        r_hi = h - 1 - max(row_offsets)
        c_lo = -min(col_offsets)
        c_hi = w - 1 - max(col_offsets)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        corner = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        l_cells = {
            corner,
            (corner[0] + search[0], corner[1] + search[1]),
            (corner[0] + side[0], corner[1] + side[1]),
        }
        missing = (corner[0] + search[0] + side[0], corner[1] + search[1] + side[1])
        singleton = (corner[0] + distance * search[0], corner[1] + distance * search[1])
        path = {
            (corner[0] + k * search[0], corner[1] + k * search[1])
            for k in range(2, distance)
        }
        important = set(l_cells) | {missing, singleton} | path
        protected = _protected_cells(important, shape)
        if protected & blocked:
            continue

        return {
            "color": int(color),
            "l_cells": l_cells,
            "missing": missing,
            "singleton": singleton,
            "protected": protected,
        }
    return None


def _decoy_shape() -> list[tuple[int, int]]:
    choice = random.choice(("single", "domino_h", "domino_v", "line_h", "line_v"))
    if choice == "single":
        return [(0, 0)]
    if choice == "domino_h":
        return [(0, 0), (0, 1)]
    if choice == "domino_v":
        return [(0, 0), (1, 0)]
    if choice == "line_h":
        return [(0, 0), (0, 1), (0, 2)]
    return [(0, 0), (1, 0), (2, 0)]


def _place_decoy(
    shape: tuple[int, int],
    color: int,
    blocked: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    rel = _decoy_shape()
    row_offsets = [r for r, _c in rel]
    col_offsets = [c for _r, c in rel]

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, h - 1 - max(row_offsets))
        c0 = random.randint(0, w - 1 - max(col_offsets))
        cells = {(r0 + dr, c0 + dc) for dr, dc in rel}
        protected = _protected_cells(cells, shape)
        if protected & blocked:
            continue
        return cells, protected
    return None


def _render_scene(
    shape: tuple[int, int],
    objects: list[dict[str, object]],
    decoys: list[tuple[int, set[tuple[int, int]]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in objects:
        color = int(spec["color"])
        for r, c in spec["l_cells"]:
            grid[r, c] = color
        sr, sc = spec["singleton"]
        grid[sr, sc] = color
    for color, cells in decoys:
        for r, c in cells:
            grid[r, c] = int(color)
    return grid


def _expected_output(inp: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in objects:
        sr, sc = spec["singleton"]
        tr, tc = spec["missing"]
        out[sr, sc] = BLACK
        out[tr, tc] = int(spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    if object_count > len(COLOR_POOL):
        object_count = len(COLOR_POOL)
    object_colors = random.sample(COLOR_POOL, object_count)

    blocked: set[tuple[int, int]] = set()
    objects: list[dict[str, object]] = []
    for color in object_colors:
        spec = _object_spec(shape, int(color), blocked)
        if spec is None:
            return None
        objects.append(spec)
        blocked |= set(spec["protected"])

    remaining_colors = [c for c in COLOR_POOL if c not in object_colors]
    decoys: list[tuple[int, set[tuple[int, int]]]] = []
    if remaining_colors:
        for _ in range(random.randint(*DECOY_COUNT_RANGE)):
            placement = _place_decoy(shape, random.choice(remaining_colors), blocked)
            if placement is None:
                continue
            cells, protected = placement
            decoys.append((random.choice(remaining_colors), cells))
            blocked |= protected

    inp = _render_scene(shape, objects, decoys)
    expected = _expected_output(inp, objects)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
