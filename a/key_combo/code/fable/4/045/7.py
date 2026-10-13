from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_family(direction: tuple[int, int]) -> str:
    dr, dc = direction
    if dr == 0:
        return "row"
    if dc == 0:
        return "col"
    if dr == dc:
        return "diag"
    return "anti"


def _ray_positions(
    shape: tuple[int, int], source: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = source
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _analyze_candidate(
    grid: np.ndarray, source: tuple[int, int]
) -> dict[tuple[int, int], dict[str, object]] | None:
    if int(grid[source]) != BG:
        return None

    total_non_bg = int(np.count_nonzero(grid != BG))
    covered = 0
    active_families: set[str] = set()
    active_count = 0
    nonempty_count = 0
    ray_info: dict[tuple[int, int], dict[str, object]] = {}

    for direction in DIRS:
        cells = _ray_positions(grid.shape, source, direction)
        values = [int(grid[r, c]) for r, c in cells]
        nz = [i for i, value in enumerate(values) if value != BG]
        if not nz:
            continue

        nonempty_count += 1
        covered += len(nz)
        first = nz[0]
        if first == len(values) - 1:
            return None

        active = all(value != BG for value in values[first + 1 :])
        if not active and any(b - a == 1 for a, b in zip(nz, nz[1:])):
            return None

        if active:
            active_count += 1
            active_families.add(_line_family(direction))

        ray_info[direction] = {
            "cells": cells,
            "nearest_index": first,
            "nearest_color": values[first],
            "active": active,
        }

    if covered != total_non_bg:
        return None
    if active_count < 2 or len(active_families) < 2:
        return None
    if nonempty_count < 3:
        return None
    return ray_info


def _find_source(grid: np.ndarray) -> tuple[tuple[int, int], dict[tuple[int, int], dict[str, object]]]:
    candidates: list[tuple[int, int, int, tuple[int, int], dict[tuple[int, int], dict[str, object]]]] = []
    for r, c in np.argwhere(grid == BG):
        source = (int(r), int(c))
        ray_info = _analyze_candidate(grid, source)
        if ray_info is None:
            continue
        active_count = sum(1 for info in ray_info.values() if bool(info["active"]))
        nonempty_count = len(ray_info)
        candidates.append((active_count, nonempty_count, len(ray_info), source, ray_info))

    assert candidates, "could not infer a valid hidden light source"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best = candidates[0]
    tied = [item for item in candidates if item[:3] == best[:3]]
    assert len(tied) == 1, "hidden light source must be unique"
    return best[3], best[4]


def _paint_all_shadows(
    grid: np.ndarray,
    ray_info: dict[tuple[int, int], dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for info in ray_info.values():
        cells = info["cells"]
        nearest_index = int(info["nearest_index"])
        nearest_color = int(info["nearest_color"])
        for r, c in cells[nearest_index + 1 :]:
            if out[r, c] == BG:
                out[r, c] = nearest_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the unseen shared light source and complete the missing shadow rays behind colored pixels.

    Input structure:
    - Black background with several colored single-pixel points placed on the eight straight rays from one hidden source cell.
    - Some directions already show a shadow ray: after the nearest point on that ray, every remaining background cell to the edge is filled.
    - Farther points may sit on the same ray, but the source cell itself stays black.

    Rule:
    - Reverse-extend the already filled rays to locate the unique hidden source cell where those lines meet.
    - Along every occupied source-to-edge direction, only the nearest point to the source is a caster.
    - Fill every background cell behind that nearest point to the grid edge with the nearest point's color, while preserving any farther colored points already sitting on the line.

    Key insight:
    - The visible rays are not output targets by themselves; they are geometric evidence for a missing origin, and once that origin is known the nearest point on each aligned ray determines the only shadow that can be completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    source, ray_info = _find_source(g)
    assert int(g[source]) == BG
    return _paint_all_shadows(g, ray_info)


# --------------- generator ---------------

GEN_HEIGHT_RANGE = (12, 24)
GEN_WIDTH_RANGE = (12, 24)
GEN_SOURCE_MARGIN = 3
GEN_DIR_COUNT_RANGE = (3, 5)
GEN_MAX_POINTS = 8
GEN_DUPLICATE_PROB = 0.7
GEN_SECOND_DUPLICATE_PROB = 0.25
GEN_THIRD_POINT_PROB = 0.18
GEN_RETRY_LIMIT = 400


def _edge_distance(
    shape: tuple[int, int], source: tuple[int, int], direction: tuple[int, int]
) -> int:
    return len(_ray_positions(shape, source, direction))


def _sample_distances(max_dist: int, count: int) -> list[int] | None:
    max_count = min(count, (max_dist + 1) // 2)
    if max_count < 1:
        return None
    count = max_count
    nearest_hi = max_dist - 2 * (count - 1) - 1
    if nearest_hi < 1:
        return None

    distances = [random.randint(1, nearest_hi)]
    while len(distances) < count:
        remaining = count - len(distances) - 1
        lo = distances[-1] + 2
        hi = max_dist - 2 * remaining
        if lo > hi:
            return None
        distances.append(random.randint(lo, hi))
    return distances


def _choose_active_dirs(chosen_dirs: list[tuple[int, int]]) -> set[tuple[int, int]] | None:
    if len(chosen_dirs) < 3:
        return None
    size = random.randint(2, len(chosen_dirs) - 1)
    for _ in range(50):
        active = set(random.sample(chosen_dirs, size))
        families = {_line_family(direction) for direction in active}
        if len(families) >= 2:
            return active
    return None


def _build_input(
    shape: tuple[int, int],
    source: tuple[int, int],
    specs: dict[tuple[int, int], dict[str, object]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for direction, spec in specs.items():
        distances = spec["distances"]
        colors = spec["colors"]
        dr, dc = direction
        for dist, color in zip(distances, colors):
            r = source[0] + dr * dist
            c = source[1] + dc * dist
            grid[r, c] = int(color)

    for direction, spec in specs.items():
        if not bool(spec["active"]):
            continue
        dr, dc = direction
        nearest = int(spec["distances"][0])
        color = int(spec["colors"][0])
        max_dist = _edge_distance(shape, source, direction)
        for dist in range(nearest + 1, max_dist + 1):
            r = source[0] + dr * dist
            c = source[1] + dc * dist
            if grid[r, c] == BG:
                grid[r, c] = color

    return grid


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    if height <= 2 * GEN_SOURCE_MARGIN or width <= 2 * GEN_SOURCE_MARGIN:
        return None

    source = (
        random.randint(GEN_SOURCE_MARGIN, height - GEN_SOURCE_MARGIN - 1),
        random.randint(GEN_SOURCE_MARGIN, width - GEN_SOURCE_MARGIN - 1),
    )
    shape = (height, width)
    chosen_count = random.randint(*GEN_DIR_COUNT_RANGE)
    chosen_dirs = random.sample(list(DIRS), chosen_count)
    active_dirs = _choose_active_dirs(chosen_dirs)
    if active_dirs is None:
        return None

    point_counts = {direction: 1 for direction in chosen_dirs}
    duplicate_candidates = [
        direction for direction in chosen_dirs if _edge_distance(shape, source, direction) >= 4
    ]
    random.shuffle(duplicate_candidates)

    if duplicate_candidates and random.random() < GEN_DUPLICATE_PROB:
        direction = duplicate_candidates.pop()
        point_counts[direction] = 3 if random.random() < GEN_THIRD_POINT_PROB else 2
    if duplicate_candidates and random.random() < GEN_SECOND_DUPLICATE_PROB:
        direction = duplicate_candidates.pop()
        point_counts[direction] = 2

    total_points = sum(point_counts.values())
    if total_points > GEN_MAX_POINTS:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    color_cursor = 0
    specs: dict[tuple[int, int], dict[str, object]] = {}
    has_multi_ray = False

    for direction in chosen_dirs:
        max_dist = _edge_distance(shape, source, direction)
        distances = _sample_distances(max_dist, point_counts[direction])
        if distances is None:
            return None
        if len(distances) > 1:
            has_multi_ray = True
        count = len(distances)
        colors = palette[color_cursor : color_cursor + count]
        if len(colors) != count:
            return None
        color_cursor += count
        specs[direction] = {
            "distances": distances,
            "colors": colors,
            "active": direction in active_dirs,
        }

    if not has_multi_ray and random.random() < 0.6:
        return None

    grid = _build_input(shape, source, specs)
    expected = np.array(grid, copy=True)
    for direction, spec in specs.items():
        dr, dc = direction
        nearest = int(spec["distances"][0])
        color = int(spec["colors"][0])
        max_dist = _edge_distance(shape, source, direction)
        for dist in range(nearest + 1, max_dist + 1):
            r = source[0] + dr * dist
            c = source[1] + dc * dist
            if expected[r, c] == BG:
                expected[r, c] = color

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-light-source ARC input")
