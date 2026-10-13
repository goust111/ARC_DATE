from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GRAY = 5

Coord = tuple[int, int]
Component = dict[str, object]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def find_components(grid: np.ndarray, fg: set[int]) -> list[np.ndarray]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if seen[r, c] or int(grid[r, c]) not in fg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) not in fg:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bfs_path(passable: set[Coord], start: Coord, goal: Coord) -> list[Coord]:
    q = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in neighbors4(*cur):
            if nxt not in passable or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    assert goal in parent
    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def mirror_step(step: Coord, axis: str) -> Coord:
    dr, dc = step
    if axis == "vertical":
        return int(dr), int(-dc)
    assert axis == "horizontal"
    return int(-dr), int(dc)


def path_is_simple(path: list[Coord], forbidden: set[Coord]) -> bool:
    if len(set(path)) != len(path):
        return False
    for i, cell in enumerate(path):
        if cell in forbidden:
            return False
        prev = path[i - 1] if i > 0 else None
        nxt = path[i + 1] if i + 1 < len(path) else None
        for other in path:
            if other == cell or other == prev or other == nxt:
                continue
            if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1:
                return False
    return True


def gate_info(grid: np.ndarray, comp: np.ndarray) -> tuple[Coord, str, tuple[Coord, Coord], tuple[Coord, Coord]]:
    cells = {tuple(map(int, rc)) for rc in comp}
    gates: list[tuple[Coord, str, tuple[Coord, Coord], tuple[Coord, Coord]]] = []
    for r, c in cells:
        if int(grid[r, c]) != GREEN:
            continue
        horiz = ((r, c - 1), (r, c + 1))
        vert = ((r - 1, c), (r + 1, c))
        if all(p in cells and int(grid[p]) == RED for p in horiz):
            others = tuple(p for p in vert if p in cells and int(grid[p]) == GREEN)
            if len(others) == 2:
                gates.append(((r, c), "vertical", horiz, others))
        if all(p in cells and int(grid[p]) == RED for p in vert):
            others = tuple(p for p in horiz if p in cells and int(grid[p]) == GREEN)
            if len(others) == 2:
                gates.append(((r, c), "horizontal", vert, others))
    assert len(gates) == 1
    return gates[0]


def component_summary(grid: np.ndarray, comp: np.ndarray) -> Component:
    cells = {tuple(map(int, rc)) for rc in comp}
    center, axis, gate_ends, side_neighbors = gate_info(grid, comp)
    marker_cells = [cell for cell in cells if int(grid[cell]) == GRAY]
    assert len(marker_cells) == 1
    marker = marker_cells[0]
    passable = {cell for cell in cells if int(grid[cell]) in (GREEN, GRAY)}
    passable_wo_center = passable - {center}

    side_paths = []
    for side in side_neighbors:
        if side not in passable_wo_center:
            continue
        q = deque([side])
        seen = {side}
        while q:
            cur = q.popleft()
            for nxt in neighbors4(*cur):
                if nxt not in passable_wo_center or nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        side_paths.append((side, seen))
    assert len(side_paths) == 2

    containing = [item for item in side_paths if marker in item[1]]
    assert len(containing) == 1
    parking, near_cells = containing[0]
    far_start = next(side for side, _cells in side_paths if side != parking)
    far_cells = next(cells0 for side, cells0 in side_paths if side == far_start)

    near_path = bfs_path(near_cells, marker, parking)
    far_green = {cell for cell in far_cells if int(grid[cell]) == GREEN}

    return {
        "cells": cells,
        "center": center,
        "axis": axis,
        "gate_ends": gate_ends,
        "marker": marker,
        "parking": parking,
        "far_start": far_start,
        "near_path": near_path,
        "far_green": far_green,
    }


def mirrored_far_path(summary: Component) -> list[Coord]:
    axis = str(summary["axis"])
    near_path = list(summary["near_path"])
    far_start = tuple(summary["far_start"])
    outward = list(reversed(near_path))
    steps = [
        (int(b[0] - a[0]), int(b[1] - a[1]))
        for a, b in zip(outward, outward[1:])
    ]
    out = [far_start]
    cur = far_start
    for step in steps:
        dr, dc = mirror_step(step, axis)
        cur = (int(cur[0] + dr), int(cur[1] + dc))
        out.append(cur)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray marker into its gate, erase the consumed near-side path, and redraw that trajectory as a mirrored far-side path.

    Input structure:
    - The grid has black background and 1 or 2 independent 4-connected foreground components.
    - Each component contains a green one-cell-wide path, one gray marker at a path endpoint, and one gate made of a green center with two red endpoints.
    - The gate endpoints are collinear, while the green path reaches the gate center from the two orthogonal side-neighbors.

    Rule:
    - In each component, move the gray marker along the unique green path until it stops on the gate-adjacent green cell on the marker side.
    - Erase every green cell consumed by that slide, keep the gate center green and the gate endpoints red, and leave the marker on the stopping cell.
    - Erase all green path cells on the far side of the gate.
    - Take the consumed trajectory shape from the stopping cell outward, mirror its step pattern across the axis through the gate center that is perpendicular to the red gate segment, and redraw it from the far-side gate-adjacent cell in green.

    Key insight:
    - The gate is used twice: it determines where the marker stops, and its orientation chooses the mirror axis that transforms the consumed near-side trajectory into the replacement far-side path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    comps = find_components(g, {RED, GREEN, GRAY})
    assert 1 <= len(comps) <= 2

    for comp in comps:
        summary = component_summary(g, comp)
        center = tuple(summary["center"])
        gate_ends = tuple(summary["gate_ends"])
        parking = tuple(summary["parking"])
        near_path = list(summary["near_path"])
        far_green = set(summary["far_green"])
        far_path = mirrored_far_path(summary)

        for cell in near_path:
            out[cell] = BLACK
        for cell in far_green:
            out[cell] = BLACK

        out[center] = GREEN
        for cell in gate_ends:
            out[cell] = RED
        out[parking] = GRAY

        for r, c in far_path:
            if in_bounds(out.shape, r, c):
                out[r, c] = GREEN

    return out


GRID_H_RANGE = (11, 17)
GRID_W_RANGE = (13, 19)
COMPONENT_COUNT_CHOICES = (1, 2)
SIDE_POINTS_RANGE = (4, 6)
MAX_SIDE_SPAN = 5
MAX_COMPONENT_ATTEMPTS = 200
MAX_GRID_ATTEMPTS = 300


def half_plane_ok(pos: Coord, side_axis: int, sign: int) -> bool:
    val = pos[side_axis]
    return sign * val >= 1


def relative_dirs(first_lateral: bool, side_axis: int) -> list[Coord]:
    lateral = [(0, 1), (0, -1)] if side_axis == 0 else [(1, 0), (-1, 0)]
    axial = [(1, 0), (-1, 0)] if side_axis == 0 else [(0, 1), (0, -1)]
    return lateral if first_lateral else lateral + axial


def build_side_walk(
    side_axis: int,
    sign: int,
    length: int,
    forbidden: set[Coord],
) -> list[Coord] | None:
    start = (sign, 0) if side_axis == 0 else (0, sign)
    used = [start]
    used_set = {start}

    def rec() -> bool:
        if len(used) == length:
            return True
        cur = used[-1]
        dirs = relative_dirs(len(used) == 1, side_axis)
        random.shuffle(dirs)
        for dr, dc in dirs:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in forbidden or nxt in used_set:
                continue
            if not half_plane_ok(nxt, side_axis, sign):
                continue
            if max(abs(nxt[0]), abs(nxt[1])) > MAX_SIDE_SPAN:
                continue
            if any(abs(nxt[0] - r) + abs(nxt[1] - c) == 1 for r, c in used_set if (r, c) != cur):
                continue
            used.append(nxt)
            used_set.add(nxt)
            if rec():
                return True
            used.pop()
            used_set.remove(nxt)
        return False

    if rec():
        return used
    return None


def build_component_spec() -> dict[str, object] | None:
    gate_horizontal = random.choice([False, True])
    if gate_horizontal:
        axis = "vertical"
        side_axis = 0
        gate_ends = {(0, -1), (0, 1)}
    else:
        axis = "horizontal"
        side_axis = 1
        gate_ends = {(-1, 0), (1, 0)}

    approach_sign = random.choice([-1, 1])
    near_len = random.randint(*SIDE_POINTS_RANGE)
    far_len = random.randint(max(near_len, 4), 6)
    forbidden = {(0, 0), *gate_ends}

    near_walk = build_side_walk(side_axis, approach_sign, near_len, forbidden)
    if near_walk is None:
        return None

    axis_name = axis
    outward_steps = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(near_walk, near_walk[1:])
    ]
    far_target = [((-approach_sign, 0) if side_axis == 0 else (0, -approach_sign))]
    cur = far_target[0]
    for step in outward_steps:
        dr, dc = mirror_step(step, axis_name)
        cur = (cur[0] + dr, cur[1] + dc)
        far_target.append(cur)

    opposite_sign = -approach_sign
    if not all(half_plane_ok(cell, side_axis, opposite_sign) for cell in far_target):
        return None
    if not path_is_simple(far_target, forbidden):
        return None

    far_input = None
    for _ in range(MAX_COMPONENT_ATTEMPTS):
        candidate = build_side_walk(side_axis, opposite_sign, far_len, forbidden)
        if candidate is None:
            continue
        if candidate != far_target:
            far_input = candidate
            break
    if far_input is None:
        return None

    marker = near_walk[-1]
    green_cells = {(0, 0)} | set(gate_ends) | set(near_walk[:-1]) | set(far_input)
    all_cells = green_cells | {marker}

    if len(all_cells) != len(green_cells) + 1:
        return None

    rel_colors: dict[Coord, int] = {(0, 0): GREEN}
    for cell in gate_ends:
        rel_colors[cell] = RED
    for cell in near_walk[:-1]:
        rel_colors[cell] = GREEN
    for cell in far_input:
        rel_colors[cell] = GREEN
    rel_colors[marker] = GRAY

    return {
        "axis": axis,
        "side_axis": side_axis,
        "near_walk": near_walk,
        "far_target": far_target,
        "far_input": far_input,
        "colors": rel_colors,
    }


def cells_with_margin(cells: set[Coord]) -> set[Coord]:
    padded = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            padded.add((nr, nc))
    return padded


def place_component(
    grid: np.ndarray,
    occupied: set[Coord],
    spec: dict[str, object],
) -> bool:
    rel_colors = dict(spec["colors"])
    rel_cells = list(rel_colors)
    rows = [r for r, _ in rel_cells]
    cols = [c for _, c in rel_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    top_min = 1 - r0
    top_max = grid.shape[0] - 2 - r1
    left_min = 1 - c0
    left_max = grid.shape[1] - 2 - c1
    if top_min > top_max or left_min > left_max:
        return False

    positions = [
        (tr, tc)
        for tr in range(top_min, top_max + 1)
        for tc in range(left_min, left_max + 1)
    ]
    random.shuffle(positions)

    for tr, tc in positions:
        abs_cells = {(r + tr, c + tc) for r, c in rel_cells}
        if any(cell in occupied for cell in cells_with_margin(abs_cells)):
            continue
        for (rr, cc), color in rel_colors.items():
            grid[rr + tr, cc + tc] = int(color)
        occupied |= abs_cells
        return True
    return False


def validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    comps = find_components(inp, {RED, GREEN, GRAY})
    assert 1 <= len(comps) <= 2
    orientations = set()
    for comp in comps:
        summary = component_summary(inp, comp)
        orientations.add(str(summary["axis"]))
        near_path = list(summary["near_path"])
        far_green = set(summary["far_green"])
        far_target = mirrored_far_path(summary)
        assert len(near_path) >= 4
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(reversed(near_path), reversed(near_path[:-1]))]
        if str(summary["axis"]) == "vertical":
            assert any(dc != 0 for _dr, dc in steps)
        else:
            assert any(dr != 0 for dr, _dc in steps)
        assert far_green != set(far_target)
        assert out[tuple(summary["parking"])] == GRAY
        for cell in far_green:
            if cell not in far_target:
                assert out[cell] == BLACK
    assert orientations <= {"vertical", "horizontal"}


def generate_input() -> np.ndarray:
    for _ in range(MAX_GRID_ATTEMPTS):
        grid = np.zeros(
            (
                random.randint(*GRID_H_RANGE),
                random.randint(*GRID_W_RANGE),
            ),
            dtype=int,
        )
        occupied: set[Coord] = set()
        count = random.choice(COMPONENT_COUNT_CHOICES)
        ok = True
        for _idx in range(count):
            spec = None
            for _ in range(MAX_COMPONENT_ATTEMPTS):
                spec = build_component_spec()
                if spec is not None:
                    break
            if spec is None or not place_component(grid, occupied, spec):
                ok = False
                break
        if not ok:
            continue
        try:
            out = main(grid)
            validate_semantics(grid, out)
        except Exception:
            continue
        if not np.array_equal(grid, out):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
