from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
OBJ = 1
MARK = 2
BAR = 3
RAY = 4
DULL = 5
STAMP = 7

GRID_H_RANGE = (14, 19)
GRID_W_RANGE = (17, 24)
EXTRA_OBJECT_CHOICES = (0, 1, 2)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
OBJECT_ATTEMPTS = 120


Coord = tuple[int, int]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_colors(grid: np.ndarray, colors: set[int]) -> list[list[Coord]]:
    mask = np.isin(grid, list(colors))
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bar_rows = [r for r in range(h) if bool(np.all(g[r, :] == BAR))]
    assert len(bar_rows) == 1, "expected exactly one full-span green bar"
    bar_r = int(bar_rows[0])

    out = g.copy()
    components = _components_of_colors(g, {OBJ, MARK})
    assert components, "expected blue/red foreground objects"

    for coords in components:
        r0, _c0, r1, _c1 = bbox_coords(coords)
        assert r1 < bar_r or r0 > bar_r, "objects must stay fully above or below the bar"
        markers = [(r, c) for r, c in coords if int(g[r, c]) == MARK]

        if len(markers) == 0:
            for r, c in coords:
                out[r, c] = BG
            continue

        if len(markers) >= 2:
            for r, c in coords:
                out[r, c] = DULL
            continue

        marker_r, marker_c = markers[0]
        step = 1 if r1 < bar_r else -1
        r = marker_r + step
        while r != bar_r:
            assert int(g[r, marker_c]) == BG, "single-marker ray corridor must be background"
            out[r, marker_c] = RAY
            r += step

        for dc in (-1, 0, 1):
            cc = marker_c + dc
            if 0 <= cc < w:
                out[bar_r, cc] = STAMP

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort embedded-marker objects into erase, ray-cast, or dull-recolor outcomes by marker count.

    Input structure:
    - The grid has black background, one full-width green horizontal bar, and
      several separated blue foreground objects above and below the bar.
    - Some object pixels are replaced by embedded red markers; blue and red
      together form each 4-connected object.
    - Objects do not touch the green bar, and single-marker objects have a clear
      vertical background corridor from the marker to the bar.

    Rule:
    - Count the red markers inside each blue/red 4-connected object.
    - Count 0: erase the whole object to black.
    - Count 1: keep the object, cast a vertical yellow ray from the unique red
      marker toward the green bar, stop at the bar, and overwrite the hit cell
      plus its horizontal neighbors on the bar with orange.
    - Count 2 or more: recolor the whole object, including markers, to gray.

    Key insight:
    - Marker count is the gate that selects one of three mutually exclusive
      object fates, and only the exactly-one branch interacts with the full-span
      bar through a vertical projection.
    """
    return _solve(grid)


def _normalize_cells(cells: set[Coord]) -> tuple[Coord, ...]:
    r0, c0, _r1, _c1 = bbox_coords(list(cells))
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _random_transform(cells: tuple[Coord, ...], rng: random.Random) -> tuple[Coord, ...]:
    r0, c0, r1, c1 = bbox_coords(list(cells))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = np.rot90(mask, k=rng.randrange(4))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return tuple(sorted(map(tuple, np.argwhere(mask))))


def _grow_blob(target_area: int, rng: random.Random) -> tuple[Coord, ...]:
    cells = {(0, 0)}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        frontier = list(frontier)
        weights = []
        for r, c in frontier:
            degree = sum((nr, nc) in cells for nr, nc in neighbors4(r, c))
            weights.append((degree + 0.35) ** 3)
        cells.add(rng.choices(frontier, weights=weights, k=1)[0])
    return _normalize_cells(cells)


def _blue_neighbor_count(cells_set: set[Coord], markers: set[Coord], cell: Coord) -> int:
    r, c = cell
    return sum((nr, nc) in cells_set and (nr, nc) not in markers for nr, nc in neighbors4(r, c))


def _valid_marker_sets(cells: tuple[Coord, ...], count: int) -> list[tuple[Coord, ...]]:
    cells_set = set(cells)
    valid: list[tuple[Coord, ...]] = []
    for combo in itertools.combinations(cells, count):
        marker_set = set(combo)
        if all(_blue_neighbor_count(cells_set, marker_set, cell) >= 2 for cell in combo):
            valid.append(tuple(sorted(combo)))
    return valid


def _open_single_sites(cells: tuple[Coord, ...], side: str) -> list[tuple[Coord, ...]]:
    cells_set = set(cells)
    valid: list[tuple[Coord, ...]] = []
    for cell in cells:
        marker_set = {cell}
        if _blue_neighbor_count(cells_set, marker_set, cell) < 2:
            continue
        r, c = cell
        blocked = any(
            cc == c and ((rr > r) if side == "above" else (rr < r))
            for rr, cc in cells
        )
        if not blocked:
            valid.append((cell,))
    return valid


def _sample_object(kind: str, side: str, rng: random.Random) -> dict | None:
    if kind == "zero":
        size_range = (4, 8)
    elif kind == "single":
        size_range = (5, 8)
    else:
        size_range = (6, 8)

    for _ in range(SHAPE_ATTEMPTS):
        area = rng.randint(*size_range)
        cells = _random_transform(_grow_blob(area, rng), rng)
        r0, c0, r1, c1 = bbox_coords(list(cells))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if len(cells) == h * w:
            continue

        if kind == "zero":
            markers: tuple[Coord, ...] = ()
        elif kind == "single":
            options = _open_single_sites(cells, side)
            if not options:
                continue
            markers = rng.choice(options)
        else:
            options = _valid_marker_sets(cells, 2) + _valid_marker_sets(cells, 3)
            if not options:
                continue
            markers = rng.choice(options)

        return {
            "kind": kind,
            "side": side,
            "cells": cells,
            "markers": tuple(sorted(markers)),
            "height": h,
            "width": w,
        }

    return None


def _touches_occupied(abs_cells: list[Coord], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in abs_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return True
    return False


def _candidate_placements(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    obj: dict,
    bar_r: int,
    used_centers: list[int],
) -> list[tuple[int, int, list[Coord], int]]:
    h, w = grid.shape
    oh = int(obj["height"])
    ow = int(obj["width"])
    cells = obj["cells"]
    markers = obj["markers"]
    side = obj["side"]

    if side == "above":
        r_start = 0
        r_end = bar_r - oh - 1
    else:
        r_start = bar_r + 2
        r_end = h - oh
    if r_start > r_end:
        return []

    candidates: list[tuple[int, int, list[Coord], int]] = []
    for r0 in range(r_start, r_end + 1):
        for c0 in range(0, w - ow + 1):
            abs_cells = [(r0 + r, c0 + c) for r, c in cells]
            if _touches_occupied(abs_cells, occupied):
                continue
            if any(reserved[r, c] for r, c in abs_cells):
                continue

            path_cells: list[Coord] = []
            hit_center = -1
            if obj["kind"] == "single":
                marker_r, marker_c = markers[0]
                hit_center = c0 + marker_c
                if any(abs(hit_center - prev) < 3 for prev in used_centers):
                    continue
                step = 1 if side == "above" else -1
                r = r0 + marker_r + step
                while r != bar_r:
                    if occupied[r, hit_center] or reserved[r, hit_center] or int(grid[r, hit_center]) != BG:
                        path_cells = []
                        break
                    path_cells.append((r, hit_center))
                    r += step
                if not path_cells:
                    continue

            candidates.append((r0, c0, path_cells, hit_center))

    return candidates


def _paint_object(grid: np.ndarray, occupied: np.ndarray, obj: dict, r0: int, c0: int) -> None:
    marker_set = set(obj["markers"])
    for r, c in obj["cells"]:
        color = MARK if (r, c) in marker_set else OBJ
        grid[r0 + r, c0 + c] = color
        occupied[r0 + r, c0 + c] = True


def _marker_classes(grid: np.ndarray) -> set[int]:
    classes: set[int] = set()
    for coords in _components_of_colors(grid, {OBJ, MARK}):
        marker_count = sum(int(grid[r, c]) == MARK for r, c in coords)
        classes.add(2 if marker_count >= 2 else marker_count)
    return classes


def _single_direction_rows(grid: np.ndarray, bar_r: int) -> set[int]:
    out = _solve(grid)
    rows = set()
    for c in range(grid.shape[1]):
        ray_rows = [r for r in range(grid.shape[0]) if int(out[r, c]) == RAY]
        if not ray_rows:
            continue
        if max(ray_rows) < bar_r:
            rows.add(-1)
        if min(ray_rows) > bar_r:
            rows.add(1)
    return rows


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bar_r = rng.randint(4, h - 5)

    grid = np.full((h, w), BG, dtype=int)
    grid[bar_r, :] = BAR
    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)
    used_centers: list[int] = []

    specs = [
        {"kind": "single", "side": "above"},
        {"kind": "single", "side": "below"},
        {"kind": "zero", "side": rng.choice(["above", "below"])},
        {"kind": "multi", "side": rng.choice(["above", "below"])},
    ]
    extra_count = rng.choice(EXTRA_OBJECT_CHOICES)
    for _ in range(extra_count):
        specs.append(
            {
                "kind": rng.choice(["zero", "single", "multi"]),
                "side": rng.choice(["above", "below"]),
            }
        )

    order = [spec for spec in specs if spec["kind"] == "single"] + [
        spec for spec in specs if spec["kind"] != "single"
    ]

    for spec in order:
        placed = False
        for _ in range(OBJECT_ATTEMPTS):
            obj = _sample_object(spec["kind"], spec["side"], rng)
            if obj is None:
                continue
            placements = _candidate_placements(grid, occupied, reserved, obj, bar_r, used_centers)
            if not placements:
                continue

            r0, c0, path_cells, hit_center = rng.choice(placements)
            _paint_object(grid, occupied, obj, r0, c0)
            if spec["kind"] == "single":
                for r, c in path_cells:
                    reserved[r, c] = True
                used_centers.append(hit_center)
            placed = True
            break
        if not placed:
            return None

    if _marker_classes(grid) != {0, 1, 2}:
        return None
    if _single_direction_rows(grid, bar_r) != {-1, 1}:
        return None

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if not np.any(out == RAY) or not np.any(out == STAMP) or not np.any(out == DULL):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
