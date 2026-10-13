from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
INDICATOR = 5
AVAILABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
PERPENDICULAR = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}
SIZE_POOL = (2, 3, 4, 5, 6)
GRID_SHAPE_RANGE = (18, 26)
OBJECT_COUNT_RANGE = (3, 5)
MIN_BOX_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _indicator_pos(spec: dict) -> tuple[int, int]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    size = int(spec["size"])
    offset = int(spec["offset"])
    direction = spec["dir"]
    if direction == "U":
        return r0 - 1, c0 + offset
    if direction == "D":
        return r0 + size, c0 + offset
    if direction == "L":
        return r0 + offset, c0 - 1
    if direction == "R":
        return r0 + offset, c0 + size
    raise ValueError(f"unknown direction {direction}")


def _square_cells(spec: dict) -> set[tuple[int, int]]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    size = int(spec["size"])
    return {
        (r, c)
        for r in range(r0, r0 + size)
        for c in range(c0, c0 + size)
    }


def _path_cells(shape: tuple[int, int], spec: dict, *, include_indicator: bool = False) -> list[tuple[int, int]]:
    start = _indicator_pos(spec)
    dr, dc = DIRS[spec["dir"]]
    cells: list[tuple[int, int]] = [start] if include_indicator else []
    r, c = start
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            break
        cells.append((r, c))
    return cells


def _enumerate_specs(
    shape: tuple[int, int], size: int, directions: Iterable[str] | None = None
) -> list[dict]:
    h, w = shape
    directions = tuple(DIRS) if directions is None else tuple(directions)
    specs: list[dict] = []
    for direction in directions:
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                for offset in range(size):
                    spec = {
                        "r0": int(r0),
                        "c0": int(c0),
                        "size": int(size),
                        "dir": direction,
                        "offset": int(offset),
                    }
                    indicator = _indicator_pos(spec)
                    if not _in_bounds(shape, *indicator):
                        continue
                    if not _path_cells(shape, spec):
                        continue
                    specs.append(spec)
    random.shuffle(specs)
    return specs


def _spec_ok(
    shape: tuple[int, int],
    spec: dict,
    placed: list[dict],
    occupied: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> bool:
    indicator = _indicator_pos(spec)
    square = _square_cells(spec)
    if indicator in occupied or indicator in reserved_paths:
        return False
    if square & occupied:
        return False
    if square & reserved_paths:
        return False

    path = set(_path_cells(shape, spec))
    if not path:
        return False
    if path & occupied:
        return False

    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    size = int(spec["size"])
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    for other in placed:
        or0 = int(other["r0"])
        oc0 = int(other["c0"])
        osize = int(other["size"])
        or1 = or0 + osize - 1
        oc1 = oc0 + osize - 1
        row_gap = _interval_gap(r0, r1, or0, or1)
        col_gap = _interval_gap(c0, c1, oc0, oc1)
        if row_gap < MIN_BOX_GAP and col_gap < MIN_BOX_GAP:
            return False
    return True


def _render(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in _square_cells(spec):
            grid[r, c] = color
        ir, ic = _indicator_pos(spec)
        grid[ir, ic] = INDICATOR
    return grid


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = sorted(int(x) for x in np.unique(g) if int(x) not in (BACKGROUND, INDICATOR))
    objects: list[dict] = []
    matched_indicators: set[tuple[int, int]] = set()
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) > 0
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        assert size_r == size_c, "foreground objects must be squares"
        size = size_r
        assert int(coords.shape[0]) == size * size, "squares must be solid"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "bbox must be monochrome"

        candidates: list[tuple[int, int, str]] = []
        if r0 - 1 >= 0:
            for c in range(c0, c1 + 1):
                if int(g[r0 - 1, c]) == INDICATOR:
                    candidates.append((r0 - 1, c, "U"))
        if r1 + 1 < g.shape[0]:
            for c in range(c0, c1 + 1):
                if int(g[r1 + 1, c]) == INDICATOR:
                    candidates.append((r1 + 1, c, "D"))
        if c0 - 1 >= 0:
            for r in range(r0, r1 + 1):
                if int(g[r, c0 - 1]) == INDICATOR:
                    candidates.append((r, c0 - 1, "L"))
        if c1 + 1 < g.shape[1]:
            for r in range(r0, r1 + 1):
                if int(g[r, c1 + 1]) == INDICATOR:
                    candidates.append((r, c1 + 1, "R"))

        assert len(candidates) == 1, "each square must have exactly one side indicator"
        ir, ic, direction = candidates[0]
        matched_indicators.add((int(ir), int(ic)))

        if direction in ("U", "D"):
            offset = int(ic - c0)
        else:
            offset = int(ir - r0)

        objects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "size": int(size),
                "area": int(size * size),
                "dir": direction,
                "offset": int(offset),
                "indicator": (int(ir), int(ic)),
            }
        )

    all_indicators = {tuple(map(int, rc)) for rc in np.argwhere(g == INDICATOR)}
    assert matched_indicators == all_indicators, "every indicator must belong to one square"
    return objects


def _simulate(
    grid: np.ndarray,
    objects: list[dict],
    *,
    block_rays: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    initial_non_bg = {tuple(map(int, rc)) for rc in np.argwhere(g != BACKGROUND)}
    ray_cells: set[tuple[int, int]] = set()
    traces: list[dict] = []

    for idx, obj in enumerate(objects):
        color = int(obj["color"])
        ir, ic = obj["indicator"]
        dr, dc = DIRS[obj["dir"]]
        out[ir, ic] = color
        ray_cells.add((int(ir), int(ic)))
        drawn = [(int(ir), int(ic))]

        r = int(ir)
        c = int(ic)
        stop_kind = "edge"
        stop_at = None
        while True:
            nr = r + dr
            nc = c + dc
            if not _in_bounds(out.shape, nr, nc):
                stop_kind = "edge"
                break

            occupied = int(out[nr, nc]) != BACKGROUND
            if occupied:
                cell = (int(nr), int(nc))
                if cell in initial_non_bg:
                    stop_kind = "object"
                    stop_at = cell
                    break
                if block_rays:
                    stop_kind = "ray"
                    stop_at = cell
                    break

            out[nr, nc] = color
            ray_cells.add((int(nr), int(nc)))
            drawn.append((int(nr), int(nc)))
            r, c = nr, nc

        traces.append(
            {
                "index": int(idx),
                "color": color,
                "cells": drawn,
                "stop_kind": stop_kind,
                "stop_at": stop_at,
            }
        )

    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw one outward ray per square, with larger squares claiming space before smaller ones.

    Input structure:
    - The grid has black background, several distinct-color filled squares, and one gray side-adjacent indicator pixel per square.
    - Each indicator sits immediately above, below, left, or right of exactly one square and marks that square's ray direction.
    - Squares are separated from each other, and the raw input contains no pre-drawn rays.

    Rule:
    - Identify each square and its indicator, then sort the squares by decreasing area; break ties by the square bbox's top-left reading order.
    - Recolor each indicator to its square color and extend a 1-pixel ray from that indicator in the indicated direction.
    - Advance one cell at a time and stop before the first out-of-bounds cell or the first non-background cell, where non-background includes original squares and rays drawn earlier in the ordering.
    - Leave every original square pixel unchanged.

    Key insight:
    - The visible ray lengths are controlled by a hidden greedy priority: earlier rays from larger squares become obstacles that truncate later rays from smaller squares.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_scene(g)
    ordered = sorted(objects, key=lambda obj: (-int(obj["area"]), int(obj["r0"]), int(obj["c0"])))
    out, _ = _simulate(g, ordered, block_rays=True)
    return out


def _crossing_case(
    shape: tuple[int, int],
    big_size: int,
    small_size: int,
    placed: list[dict],
    occupied: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> tuple[dict, dict] | None:
    big_specs = _enumerate_specs(shape, big_size)
    for big in big_specs:
        if not _spec_ok(shape, big, placed, occupied, reserved_paths):
            continue

        big_path = _path_cells(shape, big)
        if len(big_path) < 3:
            continue
        candidate_crossings = big_path[1:-1]
        random.shuffle(candidate_crossings)

        big_square = _square_cells(big)
        big_indicator = {_indicator_pos(big)}
        local_occupied = occupied | big_square | big_indicator
        local_reserved = reserved_paths | set(big_path)

        small_specs = _enumerate_specs(shape, small_size, PERPENDICULAR[big["dir"]])
        for crossing in candidate_crossings:
            for small in small_specs:
                if crossing not in set(_path_cells(shape, small)):
                    continue
                if not _spec_ok(shape, small, placed + [big], local_occupied, local_reserved):
                    continue
                return big, small
    return None


def _place_extra(
    shape: tuple[int, int],
    size: int,
    placed: list[dict],
    occupied: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> dict | None:
    specs = _enumerate_specs(shape, size)
    preferred: list[dict] = []
    fallback: list[dict] = []
    for spec in specs:
        if not _spec_ok(shape, spec, placed, occupied, reserved_paths):
            continue
        path = set(_path_cells(shape, spec))
        if path & reserved_paths:
            preferred.append(spec)
        else:
            fallback.append(spec)
    if preferred and random.random() < 0.7:
        return random.choice(preferred)
    if fallback:
        return random.choice(fallback)
    if preferred:
        return random.choice(preferred)
    return None


def _attach_colors(specs: list[dict], colors: list[int]) -> list[dict]:
    out: list[dict] = []
    for spec, color in zip(specs, colors):
        item = dict(spec)
        item["color"] = int(color)
        out.append(item)
    return out


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        objects = _parse_scene(grid)
    except AssertionError:
        return False

    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return False

    areas = [int(obj["area"]) for obj in objects]
    if len(set(areas)) != len(areas):
        return False

    all_path_sets = [set(_path_cells(grid.shape, obj)) for obj in objects]
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(grid != BACKGROUND)}
    for obj, path in zip(objects, all_path_sets):
        if not path:
            return False
        if path & occupied:
            return False
        ir, ic = obj["indicator"]
        if int(grid[ir, ic]) != INDICATOR:
            return False

    ordered = sorted(objects, key=lambda obj: (-int(obj["area"]), int(obj["r0"]), int(obj["c0"])))
    descending_out, traces = _simulate(grid, ordered, block_rays=True)
    ascending_out, _ = _simulate(
        grid,
        sorted(objects, key=lambda obj: (int(obj["area"]), int(obj["r0"]), int(obj["c0"]))),
        block_rays=True,
    )
    no_ray_block_out, _ = _simulate(grid, ordered, block_rays=False)

    if np.array_equal(descending_out, grid):
        return False
    if np.array_equal(descending_out, ascending_out):
        return False
    if np.array_equal(descending_out, no_ray_block_out):
        return False
    if not any(trace["stop_kind"] == "ray" for trace in traces):
        return False

    intersections = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if all_path_sets[i] & all_path_sets[j]:
                intersections += 1
    if intersections == 0:
        return False

    expected = main(grid)
    if not np.array_equal(expected, descending_out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SHAPE_RANGE)
    w = random.randint(*GRID_SHAPE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    sizes = random.sample(SIZE_POOL, count)
    colors = random.sample(list(AVAILABLE_COLORS), count)

    big_size = max(sizes)
    other_sizes = [size for size in sizes if size != big_size]
    small_size = random.choice(other_sizes)

    placed: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()

    pair = _crossing_case((h, w), big_size, small_size, placed, occupied, reserved_paths)
    if pair is None:
        return None

    for spec in pair:
        placed.append(spec)
        occupied |= _square_cells(spec)
        occupied.add(_indicator_pos(spec))
        reserved_paths |= set(_path_cells((h, w), spec))

    remaining_sizes = [size for size in sizes if size not in (big_size, small_size)]
    random.shuffle(remaining_sizes)
    for size in remaining_sizes:
        spec = _place_extra((h, w), size, placed, occupied, reserved_paths)
        if spec is None:
            return None
        placed.append(spec)
        occupied |= _square_cells(spec)
        occupied.add(_indicator_pos(spec))
        reserved_paths |= set(_path_cells((h, w), spec))

    placed = _attach_colors(placed, colors)
    random.shuffle(placed)
    grid = _render((h, w), placed)
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid greedy-ray-blocking ARC input")
