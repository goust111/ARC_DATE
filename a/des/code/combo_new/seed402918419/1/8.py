import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
DIRS = {
    "h": (0, 1),
    "v": (1, 0),
    "d1": (1, 1),
    "d2": (1, -1),
}

GRID_MIN = 12
GRID_MAX = 20
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 5
MIN_STEPS = 2
MAX_STEPS = 6
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))
    assert steps > 0, "marker pair must use two distinct cells"
    if dr == 0:
        sr, sc = 0, 1 if dc > 0 else -1
    elif dc == 0:
        sr, sc = 1 if dr > 0 else -1, 0
    else:
        assert abs(dr) == abs(dc), "markers must align horizontally, vertically, or diagonally"
        sr = 1 if dr > 0 else -1
        sc = 1 if dc > 0 else -1
    return [(r0 + step * sr, c0 + step * sc) for step in range(steps + 1)]


def _steps_to_edge(shape: tuple[int, int], cell: tuple[int, int], step: tuple[int, int]) -> int:
    height, width = shape
    r, c = cell
    dr, dc = step
    limits = []
    if dr > 0:
        limits.append(height - 1 - r)
    elif dr < 0:
        limits.append(r)
    if dc > 0:
        limits.append(width - 1 - c)
    elif dc < 0:
        limits.append(c)
    return min(limits) if limits else 0


def _max_half_length(shape: tuple[int, int], center: tuple[int, int], orientation: str) -> int:
    step = DIRS[orientation]
    reverse = (-step[0], -step[1])
    return min(_steps_to_edge(shape, center, step), _steps_to_edge(shape, center, reverse))


def _center_segment(
    center: tuple[int, int], orientation: str, half_length: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = DIRS[orientation]
    r, c = center
    return (r - half_length * dr, c - half_length * dc), (r + half_length * dr, c + half_length * dc)


def _endpoint_spacing_ok(
    new_points: Iterable[tuple[int, int]], existing_points: Iterable[tuple[int, int]]
) -> bool:
    existing = list(existing_points)
    for r0, c0 in new_points:
        for r1, c1 in existing:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _place_crossing_segments(
    shape: tuple[int, int], orientations: list[str], occupied: set[tuple[int, int]], endpoints: set[tuple[int, int]], rng: random.Random
) -> list[dict[str, object]] | None:
    height, width = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        center = (rng.randrange(2, height - 2), rng.randrange(2, width - 2))
        if any(_max_half_length(shape, center, orient) < MIN_STEPS for orient in orientations):
            continue

        segments: list[dict[str, object]] = []
        cluster_cells: set[tuple[int, int]] = set()
        cluster_endpoints: list[tuple[int, int]] = []
        ok = True

        for orient in orientations:
            max_half = min(MAX_STEPS, _max_half_length(shape, center, orient))
            half_length = rng.randint(MIN_STEPS, max_half)
            a, b = _center_segment(center, orient, half_length)
            cells = _line_cells(a, b)

            if any(cell in occupied for cell in cells):
                ok = False
                break
            if any(cell in cluster_cells and cell != center for cell in cells):
                ok = False
                break

            segments.append({"a": a, "b": b, "cells": cells, "orientation": orient})
            cluster_cells.update(cells)
            cluster_endpoints.extend([a, b])

        if not ok:
            continue
        if len(set(cluster_endpoints)) != len(cluster_endpoints):
            continue
        if not _endpoint_spacing_ok(cluster_endpoints, endpoints):
            continue
        return segments
    return None


def _place_isolated_segment(
    shape: tuple[int, int], occupied: set[tuple[int, int]], endpoints: set[tuple[int, int]], rng: random.Random
) -> dict[str, object] | None:
    height, width = shape
    orientations = list(DIRS)
    for _ in range(PLACEMENT_ATTEMPTS):
        orient = rng.choice(orientations)
        dr, dc = DIRS[orient]
        steps = rng.randint(MIN_STEPS, MAX_STEPS)
        a = (rng.randrange(height), rng.randrange(width))
        b = (a[0] + dr * steps, a[1] + dc * steps)
        if not (0 <= b[0] < height and 0 <= b[1] < width):
            continue

        cells = _line_cells(a, b)
        if any(cell in occupied for cell in cells):
            continue
        if not _endpoint_spacing_ok([a, b], endpoints):
            continue
        return {"a": a, "b": b, "cells": cells, "orientation": orient}
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with a straight segment and turn multi-color intersections yellow.

    Input structure:
    - The grid has a black background and several foreground marker pixels.
    - Each foreground color appears exactly twice, forming one matched pair.
    - Every pair lies on one horizontal line, one vertical line, or one 45-degree diagonal.

    Rule:
    - For each same-color pair, draw the full straight segment between its two markers, including the endpoints.
    - If two or more segments of different colors occupy the same cell, recolor that shared cell yellow.
    - Leave all non-segment background cells unchanged.

    Key insight:
    - The task is not just pairing markers and drawing the obvious line; the critical hidden rule is that color conflicts are resolved by a fixed third color, yellow, at every intersection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground pair"

    cell_to_colors: dict[tuple[int, int], set[int]] = {}
    for color in colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        for cell in _line_cells(coords[0], coords[1]):
            cell_to_colors.setdefault(cell, set()).add(color)

    out = np.full_like(g, bg)
    for cell, cell_colors in cell_to_colors.items():
        out[cell] = YELLOW if len(cell_colors) > 1 else next(iter(cell_colors))
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(GRID_MIN, GRID_MAX), rng.randint(GRID_MIN, GRID_MAX))
    total_pairs = rng.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    want_crossing = bool(rng.randrange(2))

    occupied: set[tuple[int, int]] = set()
    endpoints: set[tuple[int, int]] = set()
    segments: list[dict[str, object]] = []

    if want_crossing:
        cross_count = rng.randint(2, min(4, total_pairs))
        orientations = rng.sample(list(DIRS), cross_count)
        crossing = _place_crossing_segments(shape, orientations, occupied, endpoints, rng)
        if crossing is None:
            return None
        for segment in crossing:
            segments.append(segment)
            occupied.update(segment["cells"])
            endpoints.add(segment["a"])
            endpoints.add(segment["b"])

    while len(segments) < total_pairs:
        segment = _place_isolated_segment(shape, occupied, endpoints, rng)
        if segment is None:
            return None
        segments.append(segment)
        occupied.update(segment["cells"])
        endpoints.add(segment["a"])
        endpoints.add(segment["b"])

    colors = rng.sample(PALETTE, len(segments))
    grid = np.full(shape, BACKGROUND, dtype=int)
    for color, segment in zip(colors, segments):
        grid[segment["a"]] = color
        grid[segment["b"]] = color

    out = main(grid)
    has_yellow = bool(np.any(out == YELLOW))
    if has_yellow != want_crossing:
        return None
    if len({tuple(cell) for segment in segments for cell in (segment["a"], segment["b"])}) != 2 * len(segments):
        return None
    return grid
