from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
PATH = 5
AXIS = 8

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GRID_MIN = 18
GRID_MAX = 30
CHAIN_CHOICES = (1, 2, 3, 4)
AXIS_LEN_RANGE = (3, 6)
LEG_LEN_RANGE = (2, 4)
DECOY_AXIS_CHOICES = (0, 1, 1, 2)
MAX_GENERATION_ATTEMPTS = 600


Coord = tuple[int, int]
AxisSpec = dict[str, object]


def _add(a: Coord, b: Coord, k: int = 1) -> Coord:
    return a[0] + b[0] * k, a[1] + b[1] * k


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _orthogonal_dirs(direction: Coord) -> list[Coord]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _axis_orientation_for_approach(prev: Coord, end: Coord) -> str:
    if prev[0] == end[0]:
        return "vertical"
    return "horizontal"


def _trace_polyline(dirs: list[Coord], lengths: list[int]) -> list[Coord]:
    coords = [(0, 0)]
    r, c = 0, 0
    for direction, length in zip(dirs, lengths):
        for _ in range(length):
            r, c = _add((r, c), direction)
            coords.append((r, c))
    assert len(coords) == len(set(coords)), "polyline generator must be simple"
    return coords


def _reflect_point(point: Coord, axis: AxisSpec) -> Coord:
    orient = axis["orient"]
    if orient == "vertical":
        col = int(axis["fixed"])
        return point[0], 2 * col - point[1]
    row = int(axis["fixed"])
    return 2 * row - point[0], point[1]


def _reflect_segment(segment: list[Coord], axis: AxisSpec) -> list[Coord]:
    hit = segment[-1]
    assert hit in axis["cells"]
    reflected = [hit]
    reflected.extend(_reflect_point(point, axis) for point in reversed(segment[:-1]))
    assert len(reflected) == len(segment)
    assert len(reflected) == len(set(reflected)), "reflected segment must stay simple"
    return reflected


def _bbox(coords: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _translate_coords(coords: list[Coord], dr: int, dc: int) -> list[Coord]:
    return [(r + dr, c + dc) for r, c in coords]


def _translate_axis(axis: AxisSpec, dr: int, dc: int) -> AxisSpec:
    cells = {(r + dr, c + dc) for r, c in axis["cells"]}
    hit = axis["hit"]
    fixed = int(axis["fixed"]) + (dc if axis["orient"] == "vertical" else dr)
    return {
        "orient": axis["orient"],
        "fixed": fixed,
        "hit": (hit[0] + dr, hit[1] + dc),
        "cells": cells,
    }


def _path_degrees(path_cells: set[Coord]) -> dict[Coord, int]:
    return {cell: sum(nei in path_cells for nei in _neighbors4(*cell)) for cell in path_cells}


def _extract_axis_components(grid: np.ndarray) -> tuple[list[AxisSpec], dict[Coord, AxisSpec], set[Coord]]:
    axis_mask = grid == AXIS
    seen = np.zeros_like(axis_mask, dtype=bool)
    axes: list[AxisSpec] = []
    cell_to_axis: dict[Coord, AxisSpec] = {}

    for r, c in np.argwhere(axis_mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells: set[Coord] = set()
        while q:
            rr, cc = q.popleft()
            cells.add((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(grid.shape, nr, nc) and axis_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

        rows = {rr for rr, _ in cells}
        cols = {cc for _, cc in cells}
        if len(cols) == 1:
            orient = "vertical"
            fixed = next(iter(cols))
            ordered = sorted(rows)
            assert ordered == list(range(ordered[0], ordered[-1] + 1))
        else:
            assert len(rows) == 1, "axis components must be horizontal or vertical segments"
            orient = "horizontal"
            fixed = next(iter(rows))
            ordered = sorted(cols)
            assert ordered == list(range(ordered[0], ordered[-1] + 1))

        axis = {"orient": orient, "fixed": int(fixed), "cells": cells}
        axes.append(axis)
        for cell in cells:
            assert cell not in cell_to_axis, "axis segments must not overlap"
            cell_to_axis[cell] = axis

    return axes, cell_to_axis, set(cell_to_axis)


def _extract_generator_segment(grid: np.ndarray, axis_cells: set[Coord]) -> list[Coord]:
    reds = np.argwhere(grid == RED)
    assert len(reds) == 1, "input must contain exactly one red start pixel"
    start = tuple(map(int, reds[0]))

    path_cells = {
        tuple(map(int, coord))
        for coord in np.argwhere((grid == RED) | (grid == PATH))
    }
    assert start in path_cells
    degrees = _path_degrees(path_cells)
    assert degrees[start] == 1, "the red pixel must be one endpoint of the generator path"
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "the generator path must be a simple non-branching path"
    gray_endpoints = [cell for cell in endpoints if cell != start and grid[cell] == PATH]
    assert len(gray_endpoints) == 1, "the far endpoint must be gray before it touches the axis"
    gray_end = gray_endpoints[0]
    for cell, degree in degrees.items():
        if cell not in endpoints:
            assert degree == 2, "generator path must not branch"

    ordered = [start]
    prev = None
    cur = start
    while cur != gray_end:
        nxts = [nei for nei in _neighbors4(*cur) if nei in path_cells and nei != prev]
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    hit_axes = [nei for nei in _neighbors4(*gray_end) if nei in axis_cells]
    assert len(hit_axes) == 1, "the folded segment must terminate on a unique axis cell"
    ordered.append(hit_axes[0])
    return ordered


def _paint_segment(out: np.ndarray, segment: list[Coord], axis_cells: set[Coord], *, final_red: bool) -> None:
    for idx, (r, c) in enumerate(segment):
        if idx == 0 and (r, c) in axis_cells:
            continue
        if idx == len(segment) - 1:
            if final_red:
                out[r, c] = RED
            elif (r, c) not in axis_cells and out[r, c] == BG:
                out[r, c] = PATH
            continue
        if (r, c) not in axis_cells and out[r, c] == BG:
            out[r, c] = PATH


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold the hidden broken line by repeatedly mirroring its latest segment across the axis cell it just reaches.

    Input structure:
    - The background is black.
    - Cyan pixels form one or more horizontal or vertical mirror-axis segments.
    - A single red start pixel and a connected gray path show only the first folded generator segment.
    - That gray segment ends adjacent to exactly one cyan axis cell, which is the hidden contact point.

    Rule:
    - Recover the ordered first segment from the red endpoint to that axis contact cell.
    - Reflect the latest segment across the contacted axis line to create the next segment in the chain.
    - Keep reflecting whenever the new endpoint lands on another axis cell; stop when the new endpoint is off every axis.
    - Reflected path cells paint only background, axis cells stay cyan, and the final off-axis endpoint becomes red.

    Key insight:
    - The visible gray path is only the first folded generator; every contacted axis tells you to mirror the entire newest segment, not to extend a straight line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    axes, cell_to_axis, axis_cells = _extract_axis_components(g)
    assert axes, "expected at least one mirror axis"
    first_segment = _extract_generator_segment(g, axis_cells)
    hit = first_segment[-1]
    assert hit in cell_to_axis

    out = g.copy()
    current = first_segment
    while True:
        axis = cell_to_axis[current[-1]]
        nxt = _reflect_segment(current, axis)
        end = nxt[-1]
        final_red = end not in axis_cells
        _paint_segment(out, nxt, axis_cells, final_red=final_red)
        if final_red:
            return out
        current = nxt


def _sample_generator_segment() -> list[Coord]:
    legs = random.choice([2, 3, 3])
    lengths: list[int] = [random.randint(*LEG_LEN_RANGE) for _ in range(legs)]
    last_dir = random.choice(CARDINALS)
    if legs == 2:
        dirs = [random.choice(_orthogonal_dirs(last_dir)), last_dir]
    else:
        middle = random.choice(_orthogonal_dirs(last_dir))
        first = random.choice([last_dir, (-last_dir[0], -last_dir[1])])
        dirs = [first, middle, last_dir]

    coords = _trace_polyline(dirs, lengths)
    assert len({coords[i][0] for i in range(len(coords))}) > 1
    assert len({coords[i][1] for i in range(len(coords))}) > 1
    return coords


def _choose_used_axis(
    hit: Coord,
    orient: str,
    occupied_paths: set[Coord],
    occupied_axes: set[Coord],
) -> AxisSpec | None:
    lengths = list(range(AXIS_LEN_RANGE[0], AXIS_LEN_RANGE[1] + 1))
    random.shuffle(lengths)
    signs = [-1, 1]
    random.shuffle(signs)
    step = (1, 0) if orient == "vertical" else (0, 1)

    for length in lengths:
        for sign in signs:
            cells = {hit}
            ok = True
            for i in range(1, length):
                cell = _add(hit, step, sign * i)
                if cell in occupied_paths or cell in occupied_axes:
                    ok = False
                    break
                if any(nei in occupied_axes for nei in _neighbors4(*cell)):
                    ok = False
                    break
                cells.add(cell)
            if ok and any(nei in occupied_axes for nei in _neighbors4(*hit)):
                ok = False
            if ok:
                return {
                    "orient": orient,
                    "fixed": hit[1] if orient == "vertical" else hit[0],
                    "hit": hit,
                    "cells": cells,
                }
    return None


def _segments_overlap_only_at_joint(new_segment: list[Coord], occupied_paths: set[Coord]) -> bool:
    shared = set(new_segment) & occupied_paths
    return shared <= {new_segment[0]}


def _build_core_plan() -> dict[str, object] | None:
    first_segment = _sample_generator_segment()
    chain_len = random.choice(CHAIN_CHOICES)
    segments = [first_segment]
    used_axes: list[AxisSpec] = []
    occupied_paths = set(first_segment)
    occupied_axes: set[Coord] = set()

    current = first_segment
    for step_index in range(chain_len):
        prev, hit = current[-2], current[-1]
        orient = _axis_orientation_for_approach(prev, hit)
        axis = _choose_used_axis(hit, orient, occupied_paths - {hit}, occupied_axes)
        if axis is None:
            return None

        used_axes.append(axis)
        occupied_axes |= set(axis["cells"])

        nxt = _reflect_segment(current, axis)
        if not _segments_overlap_only_at_joint(nxt, occupied_paths):
            return None
        if set(nxt[1:-1]) & occupied_axes:
            return None
        if nxt[-1] in occupied_axes:
            return None
        if len(nxt) < 4:
            return None

        occupied_paths |= set(nxt)
        segments.append(nxt)
        current = nxt

    final_end = segments[-1][-1]
    if final_end in occupied_axes:
        return None

    if chain_len > 1 and final_end == segments[1][-1]:
        return None

    all_cells = occupied_paths | occupied_axes
    r0, c0, r1, c1 = _bbox(all_cells)
    core_h = r1 - r0 + 1
    core_w = c1 - c0 + 1
    if core_h > GRID_MAX - 2 or core_w > GRID_MAX - 2:
        return None

    return {
        "segments": segments,
        "used_axes": used_axes,
        "bbox": (r0, c0, r1, c1),
        "chain_len": chain_len,
    }


def _paint_initial_segment(grid: np.ndarray, segment: list[Coord]) -> None:
    grid[segment[0]] = RED
    for cell in segment[1:-1]:
        grid[cell] = PATH


def _render_expected(shape: tuple[int, int], axes: list[AxisSpec], segments: list[list[Coord]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    axis_cells = set()
    for axis in axes:
        for cell in axis["cells"]:
            out[cell] = AXIS
        axis_cells |= set(axis["cells"])

    out[segments[0][0]] = RED
    for cell in segments[0][1:-1]:
        out[cell] = PATH

    for segment in segments[1:-1]:
        _paint_segment(out, segment, axis_cells, final_red=False)
    _paint_segment(out, segments[-1], axis_cells, final_red=True)
    return out


def _choose_canvas_and_translate(plan: dict[str, object]) -> dict[str, object] | None:
    r0, c0, r1, c1 = plan["bbox"]
    core_h = r1 - r0 + 1
    core_w = c1 - c0 + 1

    min_h = max(GRID_MIN, core_h + 2)
    min_w = max(GRID_MIN, core_w + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    row_margin = height - core_h
    col_margin = width - core_w
    dr = random.randint(1, row_margin - 1) - r0
    dc = random.randint(1, col_margin - 1) - c0

    segments = [_translate_coords(segment, dr, dc) for segment in plan["segments"]]
    axes = [_translate_axis(axis, dr, dc) for axis in plan["used_axes"]]
    return {"shape": (height, width), "segments": segments, "axes": axes, "chain_len": plan["chain_len"]}


def _occupancy_mask(shape: tuple[int, int], axes: list[AxisSpec], segments: list[list[Coord]]) -> np.ndarray:
    occ = np.zeros(shape, dtype=bool)
    for axis in axes:
        for cell in axis["cells"]:
            occ[cell] = True
    for segment in segments:
        for cell in segment:
            occ[cell] = True
    return occ


def _place_decoy_axes(shape: tuple[int, int], axes: list[AxisSpec], segments: list[list[Coord]]) -> list[AxisSpec]:
    decoys: list[AxisSpec] = []
    occupancy = _occupancy_mask(shape, axes, segments)
    n_decoys = random.choice(DECOY_AXIS_CHOICES)

    for _ in range(n_decoys):
        placed = False
        for _ in range(120):
            orient = random.choice(["vertical", "horizontal"])
            length = random.randint(3, 6)
            if orient == "vertical":
                r0 = random.randint(0, shape[0] - length)
                c = random.randint(0, shape[1] - 1)
                cells = {(r0 + i, c) for i in range(length)}
                fixed = c
            else:
                r = random.randint(0, shape[0] - 1)
                c0 = random.randint(0, shape[1] - length)
                cells = {(r, c0 + i) for i in range(length)}
                fixed = r
            if any(occupancy[cell] for cell in cells):
                continue
            if any(
                _in_bounds(shape, nr, nc) and occupancy[nr, nc]
                for cell in cells
                for nr, nc in _neighbors4(*cell)
            ):
                continue

            for cell in cells:
                occupancy[cell] = True
            decoys.append({"orient": orient, "fixed": fixed, "hit": None, "cells": cells})
            placed = True
            break
        if not placed:
            break
    return decoys


def _render_input(shape: tuple[int, int], axes: list[AxisSpec], first_segment: list[Coord]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for axis in axes:
        for cell in axis["cells"]:
            grid[cell] = AXIS
    _paint_initial_segment(grid, first_segment)
    return grid


def _validated_instance() -> tuple[np.ndarray, np.ndarray] | None:
    core = _build_core_plan()
    if core is None:
        return None
    placed = _choose_canvas_and_translate(core)
    if placed is None:
        return None

    decoys = _place_decoy_axes(placed["shape"], placed["axes"], placed["segments"])
    all_axes = placed["axes"] + decoys
    inp = _render_input(placed["shape"], all_axes, placed["segments"][0])
    expected = _render_expected(placed["shape"], all_axes, placed["segments"])

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    if np.count_nonzero(inp == RED) != 1:
        return None
    if np.count_nonzero(expected == RED) != 2:
        return None
    if np.count_nonzero(expected == PATH) <= np.count_nonzero(inp == PATH):
        return None
    if placed["chain_len"] > 1:
        one_step = _render_expected(placed["shape"], all_axes, placed["segments"][:2])
        if np.array_equal(one_step, expected):
            return None

    first = _extract_generator_segment(inp, set().union(*(axis["cells"] for axis in all_axes)))
    if first != placed["segments"][0]:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        instance = _validated_instance()
        if instance is not None:
            return instance[0]
    raise RuntimeError("failed to generate a valid chain-unfold input")
