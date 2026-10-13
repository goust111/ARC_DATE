from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BG = 5
RED = 2
BLUE = 1
GREEN = 3
OBSTACLE_COLORS = (BLUE, GREEN)
GEN_ATTEMPTS = 300
RNG = np.random.default_rng()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("path segments must be axis-aligned")


def _left_of(forward: tuple[int, int]) -> tuple[int, int]:
    dr, dc = forward
    return (-dc, dr)


def _right_of(forward: tuple[int, int]) -> tuple[int, int]:
    dr, dc = forward
    return (dc, -dr)


def _find_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    starts = np.argwhere(grid == RED)
    assert starts.shape == (1, 2), "expected exactly one red border marker"
    r, c = map(int, starts[0])
    h, w = grid.shape
    if r == 0:
        return (r, c), (1, 0)
    if r == h - 1:
        return (r, c), (-1, 0)
    if c == 0:
        return (r, c), (0, 1)
    if c == w - 1:
        return (r, c), (0, -1)
    raise AssertionError("red marker must lie on the boundary")


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _parse_bars(
    grid: np.ndarray,
    marker: tuple[int, int],
    forward: tuple[int, int],
) -> tuple[int, list[dict[str, int]]]:
    h, w = grid.shape
    dr, dc = forward
    lane = marker[1] if dr else marker[0]
    start_axis = marker[0] if dr else marker[1]
    sign = dr if dr else dc
    bars: list[dict[str, int]] = []

    for color in OBSTACLE_COLORS:
        for coords in _components_of_color(grid, color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            if dr:
                r0 = int(rows.min())
                r1 = int(rows.max())
                c0 = int(cols.min())
                c1 = int(cols.max())
                assert r0 == r1, "obstacles must be one-cell-thick horizontal bars for vertical travel"
                assert np.array_equal(np.sort(cols), np.arange(c0, c1 + 1))
                assert 1 <= c0 <= c1 <= w - 2
                assert c0 <= lane <= c1, "every obstacle must intersect the original lane"
                axis = r0
                side_min = c0
                side_max = c1
            else:
                r0 = int(rows.min())
                r1 = int(rows.max())
                c0 = int(cols.min())
                c1 = int(cols.max())
                assert c0 == c1, "obstacles must be one-cell-thick vertical bars for horizontal travel"
                assert np.array_equal(np.sort(rows), np.arange(r0, r1 + 1))
                assert 1 <= r0 <= r1 <= h - 2
                assert r0 <= lane <= r1, "every obstacle must intersect the original lane"
                axis = c0
                side_min = r0
                side_max = r1

            dist = (axis - start_axis) * sign
            assert dist >= 1, "obstacles must be ahead of the marker"
            bars.append(
                {
                    "color": int(color),
                    "axis": int(axis),
                    "side_min": int(side_min),
                    "side_max": int(side_max),
                    "dist": int(dist),
                }
            )

    assert 2 <= len(bars) <= 4, "expected between two and four obstacles"
    bars.sort(key=lambda bar: bar["dist"])
    return lane, bars


def _far_boundary(
    shape: tuple[int, int],
    marker: tuple[int, int],
    forward: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    dr, dc = forward
    if dr == 1:
        return (h - 1, marker[1])
    if dr == -1:
        return (0, marker[1])
    if dc == 1:
        return (marker[0], w - 1)
    return (marker[0], 0)


def _paint_line(
    out: np.ndarray,
    src: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
    *,
    skip_first: bool,
) -> list[tuple[int, int]]:
    cells = _line_cells(start, end)
    if skip_first:
        cells = cells[1:]
    for r, c in cells:
        assert _in_bounds(src.shape, r, c)
        assert int(src[r, c]) == BG, "the path may only overwrite background cells"
        out[r, c] = color
    return cells


def _render_from_structure(
    grid: np.ndarray,
    marker: tuple[int, int],
    forward: tuple[int, int],
    lane: int,
    bars: list[dict[str, int]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = np.array(grid, copy=True)
    current = marker
    current_color = RED
    dr, dc = forward
    left = _left_of(forward)
    right = _right_of(forward)
    ops: list[dict[str, object]] = []

    for bar in bars:
        if dr:
            hit = (bar["axis"], lane)
        else:
            hit = (lane, bar["axis"])
        turn_start = (hit[0] - dr, hit[1] - dc)
        straight_cells = _paint_line(
            out,
            grid,
            current,
            turn_start,
            current_color,
            skip_first=True,
        )
        ops.append({"kind": "straight", "color": current_color, "cells": straight_cells})

        side = left if bar["color"] == BLUE else right
        if dr:
            if side[1] > 0:
                outside = (bar["axis"], bar["side_max"] + 1)
            else:
                outside = (bar["axis"], bar["side_min"] - 1)
        else:
            if side[0] > 0:
                outside = (bar["side_max"] + 1, bar["axis"])
            else:
                outside = (bar["side_min"] - 1, bar["axis"])

        before_bar = (outside[0] - dr, outside[1] - dc)
        after_bar = (outside[0] + dr, outside[1] + dc)
        return_cell = (hit[0] + dr, hit[1] + dc)

        cells = []
        cells.extend(
            _paint_line(out, grid, turn_start, before_bar, bar["color"], skip_first=True)
        )
        cells.extend(
            _paint_line(out, grid, before_bar, after_bar, bar["color"], skip_first=True)
        )
        cells.extend(
            _paint_line(out, grid, after_bar, return_cell, bar["color"], skip_first=True)
        )
        ops.append({"kind": "detour", "color": int(bar["color"]), "cells": cells})

        current = return_cell
        current_color = int(bar["color"])

    final_cells = _paint_line(
        out,
        grid,
        current,
        _far_boundary(grid.shape, marker, forward),
        current_color,
        skip_first=True,
    )
    ops.append({"kind": "straight", "color": current_color, "cells": final_cells})
    return out, ops


def main(grid: np.ndarray) -> np.ndarray:
    """Trace inward from the border marker, detour around each colored bar, and recolor each resulting segment by the last obstacle hit.

    Input structure:
    - A gray background contains exactly one red marker on the outer border.
    - Blue and green obstacles are one-cell-thick straight bars perpendicular to the marker's inward direction.
    - Each obstacle crosses the marker's original lane, and each bar leaves at least one empty cell beyond both ends.

    Rule:
    - Start from the red border marker and move straight inward along the unique direction perpendicular to that border.
    - When the path is about to enter a blue bar, detour around the bar's left end relative to travel; for a green bar, detour around the right end.
    - A detour runs sideways to the cell just outside the chosen end, moves forward past the obstacle line, then returns sideways to the original lane.
    - The first straight segment keeps the marker's red color; every later straight segment and the whole detour that triggered it take the triggering obstacle's color.
    - Bars and the marker remain unchanged; only background cells on the traced path are painted.

    Key insight:
    - The geometry comes from obstacle-guided detours, but the colors depend on history: each new segment echoes the color of the most recently bypassed obstacle rather than using one uniform path color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30
    palette = set(map(int, np.unique(g)))
    assert palette <= {BG, RED, BLUE, GREEN}, "unexpected colors in grid"

    marker, forward = _find_marker(g)
    lane, bars = _parse_bars(g, marker, forward)
    out, _ops = _render_from_structure(g, marker, forward, lane, bars)
    return out


def _sample_distances(axis_len: int, count: int) -> tuple[int, ...] | None:
    candidates = [
        combo
        for combo in itertools.combinations(range(2, axis_len - 2), count)
        if all(combo[i + 1] - combo[i] >= 3 for i in range(count - 1))
    ]
    if not candidates:
        return None
    return tuple(candidates[int(RNG.integers(len(candidates)))])


def _sample_colors(count: int) -> list[int]:
    colors = [BLUE, GREEN]
    while len(colors) < count:
        colors.append(BLUE if int(RNG.integers(2)) == 0 else GREEN)
    RNG.shuffle(colors)
    return colors


def _sample_extents(lane: int, lateral_len: int, count: int) -> list[tuple[int, int]] | None:
    max_neg = lane - 1
    max_pos = lateral_len - lane - 2
    limit_neg = min(4, max_neg)
    limit_pos = min(4, max_pos)
    if min(limit_neg, limit_pos) < 1:
        return None

    extents: list[tuple[int, int]] = []
    symmetric_index = int(RNG.integers(count))
    for idx in range(count):
        if idx == symmetric_index:
            arm = int(RNG.integers(1, min(limit_neg, limit_pos) + 1))
            extents.append((arm, arm))
        else:
            neg = int(RNG.integers(1, limit_neg + 1))
            pos = int(RNG.integers(1, limit_pos + 1))
            extents.append((neg, pos))
    return extents


def _marker_for_direction(
    shape: tuple[int, int],
    lane: int,
    forward: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    dr, dc = forward
    if dr == 1:
        return (0, lane)
    if dr == -1:
        return (h - 1, lane)
    if dc == 1:
        return (lane, 0)
    return (lane, w - 1)


def _build_instance(
    shape: tuple[int, int],
    lane: int,
    forward: tuple[int, int],
    distances: tuple[int, ...],
    colors: list[int],
    extents: list[tuple[int, int]],
) -> tuple[np.ndarray, list[dict[str, int]], tuple[int, int]]:
    grid = np.full(shape, BG, dtype=int)
    marker = _marker_for_direction(shape, lane, forward)
    grid[marker] = RED
    bars: list[dict[str, int]] = []
    dr, dc = forward

    for dist, color, (neg, pos) in zip(distances, colors, extents):
        if dr:
            axis = marker[0] + dr * dist
            side_min = lane - neg
            side_max = lane + pos
            grid[axis, side_min : side_max + 1] = color
        else:
            axis = marker[1] + dc * dist
            side_min = lane - neg
            side_max = lane + pos
            grid[side_min : side_max + 1, axis] = color
        bars.append(
            {
                "color": int(color),
                "axis": int(axis),
                "side_min": int(side_min),
                "side_max": int(side_max),
                "dist": int(dist),
            }
        )

    return grid, bars, marker


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    marker: tuple[int, int],
    forward: tuple[int, int],
    bars: list[dict[str, int]],
) -> bool:
    try:
        parsed_marker, parsed_forward = _find_marker(grid)
        if parsed_marker != marker or parsed_forward != forward:
            return False
        lane, parsed_bars = _parse_bars(grid, parsed_marker, parsed_forward)
        if lane != (marker[1] if forward[0] else marker[0]):
            return False
        if len(parsed_bars) != len(bars):
            return False
        for want, got in zip(sorted(bars, key=lambda b: b["dist"]), parsed_bars):
            for key in ("color", "axis", "side_min", "side_max", "dist"):
                if int(want[key]) != int(got[key]):
                    return False

        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if not np.array_equal(actual[grid != BG], grid[grid != BG]):
            return False

        changed = actual != grid
        if changed.any() and not np.all(grid[changed] == BG):
            return False

        _, ops = _render_from_structure(grid, marker, forward, lane, bars)
        if not ops or ops[0]["color"] != RED:
            return False
        if not ops[0]["cells"]:
            return False
        if not ops[-1]["cells"]:
            return False

        detour_ops = [op for op in ops if op["kind"] == "detour"]
        if len(detour_ops) != len(bars):
            return False
        for bar, op in zip(bars, detour_ops):
            if int(op["color"]) != int(bar["color"]):
                return False
            if len(op["cells"]) < 3:
                return False
            for r, c in op["cells"]:
                if int(actual[r, c]) != int(bar["color"]):
                    return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        count = int(RNG.integers(2, 5))
        axis_len = int(RNG.integers(max(13, 3 * count + 4), 21))
        lateral_len = int(RNG.integers(10, 19))
        if int(RNG.integers(2)) == 0:
            shape = (axis_len, lateral_len)
            forward = ((1, 0), (-1, 0))[int(RNG.integers(2))]
        else:
            shape = (lateral_len, axis_len)
            forward = ((0, 1), (0, -1))[int(RNG.integers(2))]

        lane_axis = shape[1] if forward[0] else shape[0]
        if lane_axis < 6:
            continue
        lane = int(RNG.integers(2, lane_axis - 2))

        distances = _sample_distances(axis_len, count)
        if distances is None:
            continue
        colors = _sample_colors(count)
        extents = _sample_extents(lane, lane_axis, count)
        if extents is None:
            continue

        grid, bars, marker = _build_instance(shape, lane, forward, distances, colors, extents)
        expected, _ops = _render_from_structure(grid, marker, forward, lane, bars)
        if _validate_instance(grid, expected, marker, forward, bars):
            return grid
    raise RuntimeError("failed to generate a valid segmented-detour puzzle")
