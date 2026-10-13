from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, bbox_coords, components_mask


BACKGROUND = Color.BLACK
BAR_COLOR = Color.RED
OBJECT_COLOR = Color.GREEN
OBSTACLE_COLOR = Color.GREY
TRAIL_YELLOW = Color.YELLOW
TRAIL_ORANGE = Color.ORANGE
COPY_COLOR = Color.TEAL

GRID_HEIGHT_RANGE = (13, 19)
GRID_WIDTH_RANGE = (13, 19)
OBJECT_AREA_RANGE = (3, 7)
BAR_LENGTH_RANGE = (4, 9)
BAR_STEPS_RANGE = (3, 7)
MAX_ATTEMPTS = 400


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_polyomino(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
        cells = _normalize_shape(cells)
    return cells


def _extract_unique_component(grid: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = components_mask(grid == int(color), connectivity=4)
    assert len(comps) == 1, f"expected exactly one component of color {color}"
    coords = comps[0]
    return coords, bbox_coords(coords)


def _infer_direction(obj_box: tuple[int, int, int, int], bar_box: tuple[int, int, int, int]) -> tuple[int, int]:
    obj_center_r = (obj_box[0] + obj_box[2]) / 2
    obj_center_c = (obj_box[1] + obj_box[3]) / 2
    bar_center_r = (bar_box[0] + bar_box[2]) / 2
    bar_center_c = (bar_box[1] + bar_box[3]) / 2

    dr = bar_center_r - obj_center_r
    dc = bar_center_c - obj_center_c
    if abs(dr) > abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    if abs(dc) > abs(dr):
        return (0, 1) if dc > 0 else (0, -1)

    if obj_box[2] < bar_box[0]:
        return 1, 0
    if bar_box[2] < obj_box[0]:
        return -1, 0
    if obj_box[3] < bar_box[1]:
        return 0, 1
    if bar_box[3] < obj_box[1]:
        return 0, -1
    raise AssertionError("object and bar must define a unique cardinal sliding direction")


def _reflect_bar(
    bar_cells: set[tuple[int, int]],
    final_box: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> set[tuple[int, int]]:
    dr, dc = direction
    r0, c0, r1, c1 = final_box
    if dr != 0:
        return {(r0 + r1 - r, c) for r, c in bar_cells}
    return {(r, c0 + c1 - c) for r, c in bar_cells}


def _simulate(grid: np.ndarray) -> dict:
    obj_coords, obj_box = _extract_unique_component(grid, OBJECT_COLOR)
    bar_coords, bar_box = _extract_unique_component(grid, BAR_COLOR)
    direction = _infer_direction(obj_box, bar_box)

    object_cells = _coords_to_set(obj_coords)
    bar_cells = _coords_to_set(bar_coords)
    obstacle_cells = _coords_to_set(np.argwhere(grid == OBSTACLE_COLOR))

    positions = [object_cells]
    current = object_cells
    max_steps = grid.shape[0] + grid.shape[1]
    for _ in range(max_steps):
        nxt = _shift_cells(current, *direction)
        if any(r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1] for r, c in nxt):
            raise AssertionError("object should stop on an obstacle or bar before leaving the grid")
        if nxt & obstacle_cells:
            branch = "obstacle"
            break
        if nxt & bar_cells:
            branch = "bar"
            break
        current = nxt
        positions.append(current)
    else:
        raise AssertionError("failed to reach a stopping condition")

    final_cells = positions[-1]

    stop_steps = len(positions) - 1
    assert stop_steps >= 1, "movement must be non-zero"

    trail_layers = []
    for step in range(stop_steps):
        vacated = positions[step] - positions[step + 1]
        assert vacated, "every successful shift must vacate at least one cell"
        trail_layers.append(vacated)

    final_box = bbox_coords(np.array(sorted(final_cells), dtype=int))
    return {
        "branch": branch,
        "direction": direction,
        "object_cells": object_cells,
        "bar_cells": bar_cells,
        "obstacle_cells": obstacle_cells,
        "final_cells": final_cells,
        "final_box": final_box,
        "trail_layers": trail_layers,
        "stop_steps": stop_steps,
    }


def _render_solution(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    _paint_cells(out, info["object_cells"], BACKGROUND)

    if info["branch"] == "obstacle":
        _paint_cells(out, info["obstacle_cells"], BACKGROUND)
        for layer in info["trail_layers"]:
            _paint_cells(out, layer, TRAIL_YELLOW)
        _paint_cells(out, info["final_cells"], COPY_COLOR)
        return out

    for idx, layer in enumerate(info["trail_layers"]):
        color = TRAIL_YELLOW if idx % 2 == 0 else TRAIL_ORANGE
        _paint_cells(out, layer, color)

    _paint_cells(out, info["final_cells"], OBJECT_COLOR)
    copy_cells = _reflect_bar(info["bar_cells"], info["final_box"], info["direction"])
    assert not (copy_cells & info["bar_cells"]), "copied bar must be on the opposite side"
    assert not (copy_cells & info["final_cells"]), "copied bar must not overlap the object"
    assert all(0 <= r < out.shape[0] and 0 <= c < out.shape[1] for r, c in copy_cells)
    _paint_cells(out, copy_cells, COPY_COLOR)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the green object toward the red bar, then branch on whether an obstacle or the bar stops the motion.

    Input structure:
    - A black background contains one green 4-connected object, one red one-cell-thick bar, and zero to two gray single-pixel obstacles.
    - The green object and red bar are aligned along one cardinal direction, so translating the object along that axis can reach the bar.
    - Any gray obstacle lies on that sliding route and can be hit before the bar.

    Rule:
    - Infer the sliding direction from the green object's bbox center toward the red bar's bbox center, restricted to the dominant cardinal axis.
    - Translate the whole green object one cell at a time along that direction.
    - If the next step would hit a gray obstacle, stop at the previous position, erase all gray obstacles, recolor the stopped object cyan, and color every vacated trail slice yellow.
    - Otherwise stop when the next step would hit the red bar, keep the moved object green, color successive vacated trail slices yellow then orange by step parity, and place a cyan mirrored copy of the red bar on the object's far side.

    Key insight:
    - The slide itself is shared, but the stopping cause is the only branch test and simultaneously determines the trail pattern, the moved object's color, and whether an opposite-side bar copy appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    info = _simulate(g)
    return _render_solution(g, info)


def _leading_face(shape: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 1:
        edge = max(r for r, _ in shape)
        return [(r, c) for r, c in shape if r == edge]
    if dr == -1:
        edge = min(r for r, _ in shape)
        return [(r, c) for r, c in shape if r == edge]
    if dc == 1:
        edge = max(c for _, c in shape)
        return [(r, c) for r, c in shape if c == edge]
    edge = min(c for _, c in shape)
    return [(r, c) for r, c in shape if c == edge]


def _paint_shape(grid: np.ndarray, top: int, left: int, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = int(color)


def _build_case(direction: tuple[int, int], obstacle_branch: bool) -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)

    area = random.randint(*OBJECT_AREA_RANGE)
    shape = _random_polyomino(area)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    obj_h = max_r + 1
    obj_w = max_c + 1

    bar_length = random.randint(*BAR_LENGTH_RANGE)
    lead = _leading_face(shape, direction)
    dr, dc = direction

    if obstacle_branch:
        min_steps = 4
    else:
        min_steps = 3

    if dr == 0:
        bar_steps_hi = min(BAR_STEPS_RANGE[1] - 1, W - obj_w - 3)
    else:
        bar_steps_hi = min(BAR_STEPS_RANGE[1] - 1, H - obj_h - 3)
    if bar_steps_hi < min_steps:
        return None
    bar_steps = random.randint(min_steps, bar_steps_hi)

    grid = np.full((H, W), BACKGROUND, dtype=int)

    if direction == (0, 1):
        start_r = random.randint(0, H - obj_h)
        start_c_hi = W - max_c - bar_steps - 2
        if start_c_hi < 0:
            return None
        start_c = random.randint(0, start_c_hi)
        bar_col = start_c + max_c + bar_steps + 1
        anchor_row = start_r + random.choice(lead)[0]
        top_lo = max(0, anchor_row - bar_length + 1)
        top_hi = min(anchor_row, H - bar_length)
        if top_lo > top_hi:
            return None
        bar_top = random.randint(top_lo, top_hi)
        bar_cells = {(bar_top + i, bar_col) for i in range(bar_length)}
    elif direction == (0, -1):
        start_r = random.randint(0, H - obj_h)
        start_c_lo = bar_steps + 1
        start_c_hi = W - obj_w
        if start_c_lo > start_c_hi:
            return None
        start_c = random.randint(start_c_lo, start_c_hi)
        bar_col = start_c - bar_steps - 1
        anchor_row = start_r + random.choice(lead)[0]
        top_lo = max(0, anchor_row - bar_length + 1)
        top_hi = min(anchor_row, H - bar_length)
        if top_lo > top_hi:
            return None
        bar_top = random.randint(top_lo, top_hi)
        bar_cells = {(bar_top + i, bar_col) for i in range(bar_length)}
    elif direction == (1, 0):
        start_c = random.randint(0, W - obj_w)
        start_r_hi = H - max_r - bar_steps - 2
        if start_r_hi < 0:
            return None
        start_r = random.randint(0, start_r_hi)
        bar_row = start_r + max_r + bar_steps + 1
        anchor_col = start_c + random.choice(lead)[1]
        left_lo = max(0, anchor_col - bar_length + 1)
        left_hi = min(anchor_col, W - bar_length)
        if left_lo > left_hi:
            return None
        bar_left = random.randint(left_lo, left_hi)
        bar_cells = {(bar_row, bar_left + i) for i in range(bar_length)}
    else:
        start_c = random.randint(0, W - obj_w)
        start_r_lo = bar_steps + 1
        start_r_hi = H - obj_h
        if start_r_lo > start_r_hi:
            return None
        start_r = random.randint(start_r_lo, start_r_hi)
        bar_row = start_r - bar_steps - 1
        anchor_col = start_c + random.choice(lead)[1]
        left_lo = max(0, anchor_col - bar_length + 1)
        left_hi = min(anchor_col, W - bar_length)
        if left_lo > left_hi:
            return None
        bar_left = random.randint(left_lo, left_hi)
        bar_cells = {(bar_row, bar_left + i) for i in range(bar_length)}

    _paint_shape(grid, start_r, start_c, shape, OBJECT_COLOR)
    _paint_cells(grid, bar_cells, BAR_COLOR)

    if obstacle_branch:
        obstacle_count = random.randint(1, 2)
        obstacle_cells: set[tuple[int, int]] = set()
        available_steps = list(range(3, bar_steps))
        random.shuffle(available_steps)
        lead_choices = lead[:]
        random.shuffle(lead_choices)
        for idx in range(obstacle_count):
            if idx >= len(available_steps):
                break
            lr, lc = lead_choices[idx % len(lead_choices)]
            step = available_steps[idx]
            obstacle_cells.add((start_r + lr + step * dr, start_c + lc + step * dc))
        if not obstacle_cells:
            return None
        for r, c in obstacle_cells:
            if grid[r, c] != BACKGROUND:
                return None
            grid[r, c] = OBSTACLE_COLOR

    try:
        info = _simulate(grid)
        out = _render_solution(grid, info)
        _validate_case(grid, out, obstacle_branch)
    except AssertionError:
        return None
    return grid


def _validate_case(inp: np.ndarray, out: np.ndarray, expected_obstacle_branch: bool) -> None:
    info = _simulate(inp)
    assert (info["branch"] == "obstacle") == bool(expected_obstacle_branch)
    assert np.array_equal(out, main(inp))

    bar_coords, _ = _extract_unique_component(inp, BAR_COLOR)
    bar_cells = _coords_to_set(bar_coords)
    assert np.all(out[np.array(sorted(list(bar_cells)))[:, 0], np.array(sorted(list(bar_cells)))[:, 1]] == BAR_COLOR)

    if expected_obstacle_branch:
        assert int(np.sum(inp == OBSTACLE_COLOR)) >= 1
        assert int(np.sum(out == OBSTACLE_COLOR)) == 0
        assert int(np.sum(out == COPY_COLOR)) == len(info["final_cells"])
        for layer in info["trail_layers"]:
            for r, c in layer:
                assert int(out[r, c]) == TRAIL_YELLOW
    else:
        assert int(np.sum(inp == OBSTACLE_COLOR)) == 0
        assert int(np.sum(out == OBJECT_COLOR)) >= len(info["final_cells"])
        copy_cells = _reflect_bar(info["bar_cells"], info["final_box"], info["direction"])
        for r, c in copy_cells:
            assert int(out[r, c]) == COPY_COLOR
        for idx, layer in enumerate(info["trail_layers"]):
            expected = TRAIL_YELLOW if idx % 2 == 0 else TRAIL_ORANGE
            for r, c in layer:
                if (r, c) in copy_cells:
                    assert int(out[r, c]) == COPY_COLOR
                else:
                    assert int(out[r, c]) == expected
        assert any(int(out[r, c]) == TRAIL_ORANGE for layer in info["trail_layers"][1::2] for r, c in layer)


def generate_input() -> np.ndarray:
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for _ in range(MAX_ATTEMPTS):
        direction = random.choice(directions)
        obstacle_branch = random.choice([False, True])
        grid = _build_case(direction, obstacle_branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-branch puzzle input")
