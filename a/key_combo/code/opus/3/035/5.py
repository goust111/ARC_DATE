from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask


BLACK = Color.BLACK
RED = Color.RED
YELLOW = Color.YELLOW
GREY = Color.GREY
ORANGE = Color.ORANGE
PLUS_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 24
LINE_COUNT_CHOICES = (2, 3)
LINE_GAP = 5
MOTIF_CENTER_GAP = 4


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = center
    return [(row + dr, col + dc) for dr, dc in PLUS_DELTAS]


def _find_crosses(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = map(int, grid.shape)
    yellow_components = components_mask(grid == YELLOW, connectivity=8)
    assert yellow_components, "expected at least one yellow marker object"

    crosses: list[dict[str, object]] = []
    for coords in yellow_components:
        component = {tuple(map(int, coord)) for coord in coords}
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        candidates: list[tuple[int, int]] = []
        for row in range(max(1, int(r0) - 2), min(height - 1, int(r1) + 3)):
            for col in range(max(1, int(c0) - 2), min(width - 1, int(c1) + 3)):
                cells = _plus_cells((row, col))
                cell_set = set(cells)
                if not component <= cell_set:
                    continue
                if any(int(grid[r, c]) not in {GREY, YELLOW} for r, c in cells):
                    continue
                if {cell for cell in cell_set if int(grid[cell]) == YELLOW} != component:
                    continue
                candidates.append((row, col))

        assert len(candidates) == 1, "expected each yellow object to imply one plus motif"
        center = candidates[0]
        crosses.append({"center": center, "cells": _plus_cells(center)})

    crosses.sort(key=lambda item: item["center"])
    return crosses


def _open_component_count(grid: np.ndarray) -> int:
    return len(components_mask(grid != GREY, connectivity=4))


def _grey_is_single_component(grid: np.ndarray) -> bool:
    return len(components_mask(grid == GREY, connectivity=4)) == 1


def _connected8(cells: list[tuple[int, int]]) -> bool:
    if not cells:
        return False
    remaining = set(cells)
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        row, col = stack.pop()
        for other in list(remaining - seen):
            if max(abs(row - other[0]), abs(col - other[1])) <= 1:
                seen.add(other)
                stack.append(other)
    return len(seen) == len(cells)


def _connected_subsets(dead_arms: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    subsets: list[list[tuple[int, int]]] = []
    for mask in range(1, 1 << len(dead_arms)):
        cells = [dead_arms[i] for i in range(len(dead_arms)) if (mask >> i) & 1]
        if _connected8(cells):
            subsets.append(cells)
    subsets.sort(key=lambda cells: (len(cells), cells))
    return subsets


def _pick_spaced(rng: random.Random, values: range, count: int, min_gap: int) -> list[int] | None:
    candidates = list(values)
    for _ in range(300):
        rng.shuffle(candidates)
        chosen: list[int] = []
        for value in candidates:
            if all(abs(value - other) >= min_gap for other in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _well_separated(center: tuple[int, int], centers: list[tuple[int, int]]) -> bool:
    row, col = center
    return all(max(abs(row - r0), abs(col - c0)) >= MOTIF_CENTER_GAP for r0, c0 in centers)


def _candidate_regions(rows: list[int], cols: list[int], shape: tuple[int, int]) -> list[tuple[int, int, int, int, list[str]]]:
    height, width = map(int, shape)
    row_bounds = [-1, *rows, height]
    col_bounds = [-1, *cols, width]
    regions: list[tuple[int, int, int, int, list[str]]] = []
    for ridx in range(len(row_bounds) - 1):
        for cidx in range(len(col_bounds) - 1):
            r_top = row_bounds[ridx]
            r_bottom = row_bounds[ridx + 1]
            c_left = col_bounds[cidx]
            c_right = col_bounds[cidx + 1]
            inner_h = r_bottom - r_top - 1
            inner_w = c_right - c_left - 1
            if inner_h < 5 or inner_w < 5:
                continue
            attach_sides: list[str] = []
            if cidx > 0:
                attach_sides.append("left")
            if cidx < len(col_bounds) - 2:
                attach_sides.append("right")
            if ridx > 0:
                attach_sides.append("up")
            if ridx < len(row_bounds) - 2:
                attach_sides.append("down")
            if attach_sides:
                regions.append((r_top, r_bottom, c_left, c_right, attach_sides))
    return regions


def _try_place_topological_plus(
    rng: random.Random,
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    used_centers: list[tuple[int, int]],
) -> dict[str, object] | None:
    height, width = map(int, grid.shape)
    candidates: list[tuple[str, int, int]] = []

    for col in cols:
        for row in range(2, height - 2):
            center = (row, col)
            if not _well_separated(center, used_centers):
                continue
            if any(abs(row - wall_row) <= 1 for wall_row in rows):
                continue
            if int(grid[row, col]) != GREY or int(grid[row - 1, col]) != GREY or int(grid[row + 1, col]) != GREY:
                continue
            if int(grid[row, col - 1]) != BLACK or int(grid[row, col + 1]) != BLACK:
                continue
            candidates.append(("vertical", row, col))

    for row in rows:
        for col in range(2, width - 2):
            center = (row, col)
            if not _well_separated(center, used_centers):
                continue
            if any(abs(col - wall_col) <= 1 for wall_col in cols):
                continue
            if int(grid[row, col]) != GREY or int(grid[row, col - 1]) != GREY or int(grid[row, col + 1]) != GREY:
                continue
            if int(grid[row - 1, col]) != BLACK or int(grid[row + 1, col]) != BLACK:
                continue
            candidates.append(("horizontal", row, col))

    if not candidates:
        return None

    orientation, row, col = rng.choice(candidates)
    if orientation == "vertical":
        grid[row, col - 1] = GREY
        grid[row, col + 1] = GREY
        yellow = [(row, col - 1 if rng.random() < 0.5 else col + 1)]
    else:
        grid[row - 1, col] = GREY
        grid[row + 1, col] = GREY
        yellow = [(row - 1 if rng.random() < 0.5 else row + 1, col)]

    for y_row, y_col in yellow:
        grid[y_row, y_col] = YELLOW
    return {"center": (row, col), "kind": "topo"}


def _try_place_decorative_plus(
    rng: random.Random,
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    used_centers: list[tuple[int, int]],
) -> dict[str, object] | None:
    regions = _candidate_regions(rows, cols, grid.shape)
    rng.shuffle(regions)

    for r_top, r_bottom, c_left, c_right, attach_sides in regions:
        sides = list(attach_sides)
        rng.shuffle(sides)
        for attach in sides:
            for _ in range(25):
                if attach == "left":
                    row = rng.randint(r_top + 2, r_bottom - 3)
                    col = rng.randint(c_left + 3, c_right - 2)
                    corridor = [(row, cc) for cc in range(c_left + 1, col - 1)]
                    dead_arms = [(row - 1, col), (row + 1, col), (row, col + 1)]
                elif attach == "right":
                    row = rng.randint(r_top + 2, r_bottom - 3)
                    col = rng.randint(c_left + 2, c_right - 3)
                    corridor = [(row, cc) for cc in range(col + 2, c_right)]
                    dead_arms = [(row - 1, col), (row + 1, col), (row, col - 1)]
                elif attach == "up":
                    row = rng.randint(r_top + 3, r_bottom - 2)
                    col = rng.randint(c_left + 2, c_right - 3)
                    corridor = [(rr, col) for rr in range(r_top + 1, row - 1)]
                    dead_arms = [(row, col - 1), (row, col + 1), (row + 1, col)]
                else:
                    row = rng.randint(r_top + 2, r_bottom - 3)
                    col = rng.randint(c_left + 2, c_right - 3)
                    corridor = [(rr, col) for rr in range(row + 2, r_bottom)]
                    dead_arms = [(row, col - 1), (row, col + 1), (row - 1, col)]

                center = (row, col)
                if not _well_separated(center, used_centers):
                    continue
                if any(abs(row - wall_row) <= 1 for wall_row in rows):
                    continue
                if any(abs(col - wall_col) <= 1 for wall_col in cols):
                    continue

                plus_cells = _plus_cells(center)
                shape_cells = plus_cells + corridor
                if any(int(grid[r, c]) != BLACK for r, c in shape_cells):
                    continue

                yellow_options = _connected_subsets(dead_arms)
                rng.shuffle(yellow_options)
                for yellow_cells in yellow_options:
                    trial = grid.copy()
                    for r, c in shape_cells:
                        trial[r, c] = GREY
                    for r, c in yellow_cells:
                        trial[r, c] = YELLOW
                    try:
                        crosses = _find_crosses(trial)
                    except AssertionError:
                        continue
                    if (row, col) not in {tuple(item["center"]) for item in crosses}:
                        continue
                    grid[:, :] = trial
                    return {"center": center, "kind": "deco"}
    return None


def _apply_single_cross(grid: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    out = grid.copy()
    for row, col in _plus_cells(center):
        if int(out[row, col]) == GREY:
            out[row, col] = RED
    return out


def _validate_generated(grid: np.ndarray, motifs: list[dict[str, object]]) -> bool:
    try:
        crosses = _find_crosses(grid)
    except AssertionError:
        return False

    if len(crosses) != len(motifs):
        return False
    expected_centers = {tuple(motif["center"]) for motif in motifs}
    found_centers = {tuple(cross["center"]) for cross in crosses}
    if expected_centers != found_centers:
        return False
    if not _grey_is_single_component(grid):
        return False

    base_open_components = _open_component_count(grid)
    saw_topo = False
    saw_deco = False
    for motif in motifs:
        after = _apply_single_cross(grid, tuple(motif["center"]))
        merged_open_components = _open_component_count(after)
        if motif["kind"] == "topo":
            if merged_open_components >= base_open_components:
                return False
            saw_topo = True
        else:
            if merged_open_components != base_open_components:
                return False
            saw_deco = True

    if not (saw_topo and saw_deco):
        return False

    out = main(grid)
    return bool((out == ORANGE).any() and (out == BLACK).any())


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    h_count = rng.choice(LINE_COUNT_CHOICES)
    v_count = rng.choice(LINE_COUNT_CHOICES)
    rows = _pick_spaced(rng, range(3, height - 3), h_count, LINE_GAP)
    cols = _pick_spaced(rng, range(3, width - 3), v_count, LINE_GAP)
    if rows is None or cols is None:
        return None
    if not _candidate_regions(rows, cols, (height, width)):
        return None

    grid = np.zeros((height, width), dtype=int)
    for row in rows:
        grid[row, :] = GREY
    for col in cols:
        grid[:, col] = GREY

    total_motifs = rng.choice((2, 3))
    topo_count = 1 if total_motifs == 2 else rng.choice((1, 2))
    deco_count = total_motifs - topo_count

    motifs: list[dict[str, object]] = []
    used_centers: list[tuple[int, int]] = []

    for _ in range(topo_count):
        motif = _try_place_topological_plus(rng, grid, rows, cols, used_centers)
        if motif is None:
            return None
        motifs.append(motif)
        used_centers.append(tuple(motif["center"]))

    for _ in range(deco_count):
        motif = _try_place_decorative_plus(rng, grid, rows, cols, used_centers)
        if motif is None:
            return None
        motifs.append(motif)
        used_centers.append(tuple(motif["center"]))

    if not _validate_generated(grid, motifs):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each yellow-marked plus, remove its grey cells from the wall network, and fill the newly cross-connected open regions.

    Input structure:
    - A black background contains a single 4-connected grey wall network made from thin horizontal and vertical segments.
    - Two or three 8-connected yellow marker objects sit inside five-cell plus motifs whose other cells are grey wall cells.
    - The rest of the grid uses only black, grey, and yellow.

    Rule:
    - For each yellow object, find the unique in-bounds plus center whose center/up/down/left/right cells contain exactly that yellow object and otherwise only grey cells.
    - Recolor the grey cells of every confirmed plus red while keeping the yellow marker cells unchanged.
    - Treat the remaining grey cells as the only walls; red, yellow, and black cells are all open space for connectivity.
    - Find the 4-connected open components and mark every component that contains at least one completed plus cell.
    - Fill every black cell inside a marked open component with orange, while preserving the grey walls, yellow markers, and red plus cells.

    Key insight:
    - The plus completion is not just recoloring: once its grey cells stop being walls, open-space components can merge through the red/yellow cross, so the fill target depends on the wall topology after the cross is removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, YELLOW, GREY}, "expected only black, grey, and yellow input colors"

    crosses = _find_crosses(g)
    out = g.copy()
    cross_cells: set[tuple[int, int]] = set()
    for cross in crosses:
        cells = [tuple(cell) for cell in cross["cells"]]
        cross_cells.update(cells)
        for row, col in cells:
            if int(out[row, col]) == GREY:
                out[row, col] = RED

    # Once grey is removed from a plus, the red/yellow cells belong to open space.
    open_components = components_mask(out != GREY, connectivity=4)
    for coords in open_components:
        component_cells = {tuple(map(int, coord)) for coord in coords}
        if not component_cells & cross_cells:
            continue
        black_mask = out[coords[:, 0], coords[:, 1]] == BLACK
        if np.any(black_mask):
            black_coords = coords[black_mask]
            out[black_coords[:, 0], black_coords[:, 1]] = ORANGE

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
