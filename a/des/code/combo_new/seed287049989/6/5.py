from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
YELLOW = 4
COLOR_POOL = [1, 2, 3, 5, 6, 8, 9]

GRID_SIZE_MIN = 13
GRID_SIZE_MAX = 22
PAIR_COUNT_MIN = 3
PAIR_COUNT_MAX = 5
MIN_ARM = 2
MAX_ARM = 8
CENTER_TRIES = 120
SEGMENT_TRIES = 160
GENERATION_ATTEMPTS = 500

ORIENTATIONS = ("h", "v", "d1", "d2")
TWO_WAY_CROSSES = (
    ("h", "v"),
    ("h", "d1"),
    ("h", "d2"),
    ("v", "d1"),
    ("v", "d2"),
    ("d1", "d2"),
)
THREE_WAY_CROSSES = (
    ("h", "v", "d1"),
    ("h", "v", "d2"),
    ("h", "d1", "d2"),
    ("v", "d1", "d2"),
)

Coord = tuple[int, int]


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _segment_cells(a: Coord, b: Coord) -> list[Coord] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0

    if dr == 0 and dc != 0:
        step_r, step_c = 0, _sign(dc)
        length = abs(dc)
    elif dc == 0 and dr != 0:
        step_r, step_c = _sign(dr), 0
        length = abs(dr)
    elif abs(dr) == abs(dc) and dr != 0:
        step_r, step_c = _sign(dr), _sign(dc)
        length = abs(dr)
    else:
        return None

    return [(r0 + i * step_r, c0 + i * step_c) for i in range(length + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored marker pair with a straight segment and color shared path cells yellow.

    Input structure:
    - The background is black.
    - Each non-black color appears exactly twice as two single-pixel markers.
    - The two markers of one color are aligned horizontally, vertically, or on an exact 45-degree diagonal.

    Rule:
    - For every non-black color, draw the full inclusive straight path between its two markers using that color.
    - Allowed paths are horizontal, vertical, and exact 45-degree diagonal segments.
    - If a pixel is covered by lines of two or more different colors, recolor that pixel yellow.
    - Pixels covered by exactly one line keep that line's original color.

    Key insight:
    - The input only gives endpoint markers; the output is recovered by pairing equal colors, reconstructing their straight segments, and then applying a separate overlap-to-yellow rule at shared cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert colors, "expected at least one marker pair"
    assert YELLOW not in colors, "yellow is reserved for output intersections"

    coverage: defaultdict[Coord, set[int]] = defaultdict(set)
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each marker color must appear exactly twice"
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        cells = _segment_cells(a, b)
        assert cells is not None, "marker pairs must define horizontal, vertical, or 45-degree diagonal segments"
        for cell in cells:
            coverage[cell].add(color)

    out = np.full(g.shape, BLACK, dtype=int)
    for (r, c), cell_colors in coverage.items():
        out[r, c] = YELLOW if len(cell_colors) >= 2 else next(iter(cell_colors))
    return out


def _arm_limits(shape: tuple[int, int], center: Coord, orientation: str) -> tuple[int, int]:
    height, width = shape
    r, c = center

    if orientation == "h":
        return c, width - 1 - c
    if orientation == "v":
        return r, height - 1 - r
    if orientation == "d1":
        return min(r, c), min(height - 1 - r, width - 1 - c)
    assert orientation == "d2"
    return min(r, width - 1 - c), min(height - 1 - r, c)


def _build_segment(
    shape: tuple[int, int],
    orientation: str,
    center: Coord,
    color: int,
    rng: random.Random,
) -> dict[str, object] | None:
    neg_max, pos_max = _arm_limits(shape, center, orientation)
    neg_limit = min(neg_max, MAX_ARM)
    pos_limit = min(pos_max, MAX_ARM)
    if min(neg_limit, pos_limit) < MIN_ARM:
        return None

    neg_len = rng.randint(MIN_ARM, neg_limit)
    pos_len = rng.randint(MIN_ARM, pos_limit)
    r, c = center

    if orientation == "h":
        a = (r, c - neg_len)
        b = (r, c + pos_len)
    elif orientation == "v":
        a = (r - neg_len, c)
        b = (r + pos_len, c)
    elif orientation == "d1":
        a = (r - neg_len, c - neg_len)
        b = (r + pos_len, c + pos_len)
    else:
        a = (r - neg_len, c + neg_len)
        b = (r + pos_len, c - pos_len)

    cells = _segment_cells(a, b)
    assert cells is not None
    return {
        "color": color,
        "a": a,
        "b": b,
        "cells": cells,
        "orientation": orientation,
    }


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _can_place(
    candidate: dict[str, object],
    segments: list[dict[str, object]],
    allowed_overlaps: set[Coord],
) -> bool:
    candidate_cells = set(candidate["cells"])
    candidate_endpoints = {candidate["a"], candidate["b"]}

    for segment in segments:
        other_cells = set(segment["cells"])
        other_endpoints = {segment["a"], segment["b"]}

        overlap = candidate_cells & other_cells
        if overlap - allowed_overlaps:
            return False
        if candidate_endpoints & other_cells:
            return False
        if other_endpoints & candidate_cells:
            return False
        for endpoint in candidate_endpoints:
            for other_endpoint in other_endpoints:
                if _chebyshev(endpoint, other_endpoint) <= 1:
                    return False
    return True


def _sample_center(
    shape: tuple[int, int],
    orientations: tuple[str, ...] | list[str],
    rng: random.Random,
) -> Coord | None:
    height, width = shape
    for _ in range(CENTER_TRIES):
        center = (rng.randrange(height), rng.randrange(width))
        if all(min(_arm_limits(shape, center, orientation)) >= MIN_ARM for orientation in orientations):
            return center
    return None


def _add_cross_cluster(
    shape: tuple[int, int],
    segments: list[dict[str, object]],
    colors: list[int],
    cross_kind: str,
    rng: random.Random,
) -> tuple[list[str], list[dict[str, object]]] | None:
    orientation_group = rng.choice(TWO_WAY_CROSSES if cross_kind == "two" else THREE_WAY_CROSSES)
    center = _sample_center(shape, orientation_group, rng)
    if center is None:
        return None

    new_segments: list[dict[str, object]] = []
    for color, orientation in zip(colors, orientation_group):
        placed = None
        for _ in range(SEGMENT_TRIES):
            candidate = _build_segment(shape, orientation, center, color, rng)
            if candidate is not None and _can_place(candidate, segments + new_segments, {center}):
                placed = candidate
                break
        if placed is None:
            return None
        new_segments.append(placed)

    return list(orientation_group), new_segments


def _extra_orientation_plan(count: int, used: list[str], rng: random.Random) -> list[str]:
    remaining = [orientation for orientation in ORIENTATIONS if orientation not in used]
    rng.shuffle(remaining)
    plan = remaining[:count]
    while len(plan) < count:
        plan.append(rng.choice(ORIENTATIONS))
    return plan


def _add_free_segment(
    shape: tuple[int, int],
    segments: list[dict[str, object]],
    orientation: str,
    color: int,
    rng: random.Random,
) -> dict[str, object] | None:
    for _ in range(SEGMENT_TRIES):
        center = _sample_center(shape, (orientation,), rng)
        if center is None:
            return None
        candidate = _build_segment(shape, orientation, center, color, rng)
        if candidate is not None and _can_place(candidate, segments, set()):
            return candidate
    return None


def _validate_generated_grid(grid: np.ndarray) -> bool:
    colors = [int(color) for color in np.unique(grid) if int(color) != BLACK]
    if len(colors) < PAIR_COUNT_MIN:
        return False
    if any(np.count_nonzero(grid == color) != 2 for color in colors):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return bool(np.any(out == YELLOW))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    shape = (height, width)

    cross_kind = rng.choice(("two", "three"))
    cluster_size = 2 if cross_kind == "two" else 3
    pair_count = rng.randint(max(PAIR_COUNT_MIN, cluster_size), PAIR_COUNT_MAX)
    colors = rng.sample(COLOR_POOL, pair_count)

    cluster_result = _add_cross_cluster(shape, [], colors[:cluster_size], cross_kind, rng)
    if cluster_result is None:
        return None
    used_orientations, segments = cluster_result

    extra_orientations = _extra_orientation_plan(pair_count - cluster_size, used_orientations, rng)
    for index, orientation in enumerate(extra_orientations, start=cluster_size):
        segment = _add_free_segment(shape, segments, orientation, colors[index], rng)
        if segment is None:
            return None
        segments.append(segment)

    grid = np.full(shape, BLACK, dtype=int)
    for segment in segments:
        grid[segment["a"]] = int(segment["color"])
        grid[segment["b"]] = int(segment["color"])

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
