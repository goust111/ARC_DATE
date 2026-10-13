from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
GRID_SIDE_RANGE = (12, 26)
LEGEND_LEN_RANGE = (3, 5)
POINT_COUNT_RANGE = (3, 6)
FIRST_GAP_RANGE = (1, 4)
DIFF_VALUES = (0, 0, 1, 1, 2)
MAX_GENERATION_ATTEMPTS = 600

DIRS = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)
CORNER_SPECS = {
    "tl": ((0, 0), (1, 1)),
    "tr": ((0, -1), (1, -1)),
    "bl": ((-1, 0), (-1, 1)),
    "br": ((-1, -1), (-1, -1)),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _row_major_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _primitive_step(delta: tuple[int, int]) -> tuple[tuple[int, int], int]:
    dr, dc = map(int, delta)
    g = math.gcd(abs(dr), abs(dc))
    assert g > 0, "consecutive marker points must differ"
    return (dr // g, dc // g), g


def _corner_cells(shape: tuple[int, int], corner: str, length: int) -> list[tuple[int, int]]:
    h, w = shape
    (r0, c0), (dr, dc) = CORNER_SPECS[corner]
    r = 0 if r0 == 0 else h - 1
    c = 0 if c0 == 0 else w - 1
    cells = []
    for _ in range(length):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _scan_corner_legend(g: np.ndarray, bg: int, corner: str) -> list[tuple[int, int]]:
    h, w = g.shape
    (r0, c0), (dr, dc) = CORNER_SPECS[corner]
    r = 0 if r0 == 0 else h - 1
    c = 0 if c0 == 0 else w - 1
    cells: list[tuple[int, int]] = []
    while _in_bounds(g.shape, r, c) and int(g[r, c]) != bg:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    legend_candidates = []
    for corner in CORNER_SPECS:
        cells = _scan_corner_legend(g, bg, corner)
        if len(cells) >= 2:
            legend_candidates.append((corner, cells))
    assert len(legend_candidates) == 1, "expected exactly one corner diagonal legend"
    legend_corner, legend_cells = legend_candidates[0]
    legend_colors = [int(g[r, c]) for r, c in legend_cells]

    mask = g != bg
    for r, c in legend_cells:
        mask[r, c] = False
    points = _row_major_coords(list(map(tuple, np.argwhere(mask))))
    assert len(points) >= 3, "expected at least three non-legend marker points"

    deltas = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        deltas.append((r1 - r0, c1 - c0))
    step, first_gap = _primitive_step(deltas[0])
    assert step[0] > 0 or (step[0] == 0 and step[1] > 0), "row-major marker direction is invalid"
    gaps = [first_gap]
    for delta in deltas[1:]:
        delta_step, gap = _primitive_step(delta)
        assert delta_step == step, "marker points must lie on one line"
        gaps.append(gap)
    gap_diff = gaps[1] - gaps[0]
    assert gap_diff >= 0, "gap arithmetic difference must be non-negative"
    assert all(gaps[i + 1] - gaps[i] == gap_diff for i in range(len(gaps) - 1))

    for i, (r, c) in enumerate(points):
        assert int(g[r, c]) == legend_colors[i % len(legend_colors)], "marker colors must follow legend cycle"

    return {
        "bg": bg,
        "legend_corner": legend_corner,
        "legend_cells": legend_cells,
        "legend_colors": legend_colors,
        "points": points,
        "step": step,
        "gaps": gaps,
        "gap_diff": gap_diff,
    }


def _continue_points(
    shape: tuple[int, int],
    start_point: tuple[int, int],
    step: tuple[int, int],
    last_gap: int,
    gap_diff: int,
) -> list[tuple[int, int]]:
    added = []
    r, c = start_point
    gap = last_gap
    while True:
        gap += gap_diff
        nr = r + step[0] * gap
        nc = c + step[1] * gap
        if not _in_bounds(shape, nr, nc):
            return added
        added.append((nr, nc))
        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the accelerating marker line until the next point would leave the grid, recoloring each new point by cycling through the corner legend.

    Input structure:
    - A dominant background fills most of the grid.
    - Exactly one corner contains a tight diagonal legend made of single colored cells touching that corner.
    - Away from the legend, there is one row-major ordered single-pixel marker line with at least three collinear points.

    Rule:
    - Read the legend colors from the corner inward along its diagonal.
    - Sort the marker points in row-major order, recover their common line direction, and read the consecutive gap lengths along that line.
    - Those gap lengths form an arithmetic progression; keep extending the line from its last point using the next gap, then the next one after that, until the next point would be out of bounds.
    - Color each newly added point by continuing the legend sequence cyclically, while leaving the legend and all original points unchanged.

    Key insight:
    - The invariant is not equal spacing between markers but equal change in spacing, and the corner diagonal is only a color cycle legend rather than part of the line to extend.
    """
    parsed = _parse_instance(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = parsed["bg"]
    points = parsed["points"]
    step = parsed["step"]
    gaps = parsed["gaps"]
    gap_diff = parsed["gap_diff"]
    legend_colors = parsed["legend_colors"]

    added = _continue_points(g.shape, points[-1], step, gaps[-1], gap_diff)
    for i, (r, c) in enumerate(added, start=len(points)):
        assert int(out[r, c]) == bg, "new marker must land on background"
        out[r, c] = legend_colors[i % len(legend_colors)]
    return out


def _build_expected(
    shape: tuple[int, int],
    bg: int,
    legend_corner: str,
    legend_colors: list[int],
    points: list[tuple[int, int]],
    step: tuple[int, int],
    gaps: list[int],
    gap_diff: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    inp = np.full(shape, bg, dtype=int)
    legend_cells = _corner_cells(shape, legend_corner, len(legend_colors))
    for (r, c), color in zip(legend_cells, legend_colors):
        inp[r, c] = color
    for i, (r, c) in enumerate(points):
        inp[r, c] = legend_colors[i % len(legend_colors)]

    out = inp.copy()
    added = _continue_points(shape, points[-1], step, gaps[-1], gap_diff)
    for i, (r, c) in enumerate(added, start=len(points)):
        out[r, c] = legend_colors[i % len(legend_colors)]
    return inp, out, added


def _naive_equal_gap_output(parsed: dict, grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    points = parsed["points"]
    step = parsed["step"]
    last_gap = parsed["gaps"][-1]
    legend_colors = parsed["legend_colors"]
    added = _continue_points(out.shape, points[-1], step, last_gap, 0)
    for i, (r, c) in enumerate(added, start=len(points)):
        if out[r, c] == parsed["bg"]:
            out[r, c] = legend_colors[i % len(legend_colors)]
    return out


def _naive_last_color_output(parsed: dict, grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    points = parsed["points"]
    step = parsed["step"]
    gaps = parsed["gaps"]
    gap_diff = parsed["gap_diff"]
    color = int(grid[points[-1]])
    added = _continue_points(out.shape, points[-1], step, gaps[-1], gap_diff)
    for r, c in added:
        if out[r, c] == parsed["bg"]:
            out[r, c] = color
    return out


def _positions_for_gaps(start: tuple[int, int], step: tuple[int, int], gaps: list[int]) -> list[tuple[int, int]]:
    positions = [start]
    r, c = start
    for gap in gaps:
        r += step[0] * gap
        c += step[1] * gap
        positions.append((r, c))
    return positions


def _is_strictly_interior(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)


def _valid_starts(
    shape: tuple[int, int],
    legend_cells: set[tuple[int, int]],
    step: tuple[int, int],
    gaps: list[int],
    gap_diff: int,
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    h, w = shape
    options = []
    for r0 in range(h):
        for c0 in range(w):
            points = _positions_for_gaps((r0, c0), step, gaps)
            if not all(_in_bounds(shape, r, c) for r, c in points):
                continue
            if not _is_strictly_interior(shape, points):
                continue
            if any((r, c) in legend_cells for r, c in points):
                continue
            added = _continue_points(shape, points[-1], step, gaps[-1], gap_diff)
            if not (1 <= len(added) <= 5):
                continue
            if any((r, c) in legend_cells for r, c in added):
                continue
            if len(set(points + added)) != len(points) + len(added):
                continue
            options.append((points, added))
    return options


def _sample_params() -> tuple[tuple[int, int], int, str, list[int], tuple[int, int], list[int], int, list[tuple[int, int]]] | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    shape = (h, w)
    bg = random.choice(PALETTE)

    legend_len = random.randint(*LEGEND_LEN_RANGE)
    legend_colors = random.sample([c for c in PALETTE if c != bg], legend_len)
    legend_corner = random.choice(tuple(CORNER_SPECS))
    legend_cells = set(_corner_cells(shape, legend_corner, legend_len))

    point_count = random.randint(*POINT_COUNT_RANGE)
    step = random.choice(DIRS)
    first_gap = random.randint(*FIRST_GAP_RANGE)
    gap_diff = random.choice(DIFF_VALUES)
    gaps = [first_gap + i * gap_diff for i in range(point_count - 1)]

    starts = _valid_starts(shape, legend_cells, step, gaps, gap_diff)
    if not starts:
        return None

    multi_add = [item for item in starts if len(item[1]) >= 2]
    if multi_add and random.random() < 0.7:
        starts = multi_add
    wraps_cycle = [item for item in starts if point_count + len(item[1]) > legend_len]
    if wraps_cycle and random.random() < 0.6:
        starts = wraps_cycle
    points, added = random.choice(starts)

    # Keep the color cycle visually active and reject degenerate "all one color" shortcuts.
    if np.array_equal(
        np.array([legend_colors[i % legend_len] for i in range(point_count)], dtype=int),
        np.full(point_count, legend_colors[0], dtype=int),
    ):
        return None
    if gap_diff > 0 and len(added) < 1:
        return None
    return shape, bg, legend_corner, legend_colors, step, gaps, gap_diff, points


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    intended: dict,
) -> bool:
    try:
        parsed = _parse_instance(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if parsed["legend_corner"] != intended["legend_corner"]:
        return False
    if parsed["legend_colors"] != intended["legend_colors"]:
        return False
    if parsed["points"] != intended["points"]:
        return False
    if parsed["step"] != intended["step"]:
        return False
    if parsed["gaps"] != intended["gaps"]:
        return False
    if parsed["gap_diff"] != intended["gap_diff"]:
        return False

    if np.array_equal(inp, expected):
        return False
    if np.array_equal(_naive_last_color_output(parsed, inp), expected):
        return False
    if intended["gap_diff"] > 0 and np.array_equal(_naive_equal_gap_output(parsed, inp), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_params()
        if sampled is None:
            continue
        shape, bg, legend_corner, legend_colors, step, gaps, gap_diff, points = sampled
        inp, expected, _ = _build_expected(shape, bg, legend_corner, legend_colors, points, step, gaps, gap_diff)
        intended = {
            "legend_corner": legend_corner,
            "legend_colors": legend_colors,
            "points": points,
            "step": step,
            "gaps": gaps,
            "gap_diff": gap_diff,
        }
        if _semantic_ok(inp, expected, intended):
            return inp
    raise RuntimeError("failed to generate a valid accelerating-markers input")
