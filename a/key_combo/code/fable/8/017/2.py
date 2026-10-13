from __future__ import annotations

from collections import Counter, deque

import numpy as np


class Color:
    BLACK = 0
    GREEN = 3


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PATH_BLOB_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
LOCAL_SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (((0, 0),),),
    2: (((0, 0), (1, 0)), ((0, 0), (0, 1))),
    3: (((0, 0), (1, 0), (2, 0)), ((0, 0), (1, 0), (0, 1))),
    4: (
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (0, 1)),
    ),
    5: (
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1), (2, 1)),
    ),
}


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _turn_idx(dir_idx: int, turn: int) -> int:
    return (dir_idx + turn) % 4


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    cell_to_component: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comp_id = len(components)
            for cell in cells:
                cell_to_component[cell] = comp_id
            components.append({"color": color, "cells": tuple(cells), "area": len(cells)})

    return components, cell_to_component


def _initial_direction(grid: np.ndarray, bg: int, start: tuple[int, int]) -> int:
    seen_dirs: list[int] = []
    for dir_idx, (dr, dc) in enumerate(DIRS):
        r, c = start
        while True:
            r += dr
            c += dc
            if not _in_bounds(grid.shape, r, c):
                break
            if int(grid[r, c]) != bg:
                seen_dirs.append(dir_idx)
                break
    assert len(seen_dirs) == 1, "expected exactly one visible blob from the start"
    return seen_dirs[0]


def _trace_ray(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    assert bg == Color.BLACK, "expected a black background"

    starts = np.argwhere(g == Color.GREEN)
    assert len(starts) == 1, "expected exactly one green start pixel"
    start = tuple(map(int, starts[0]))

    components, cell_to_component = _find_components(g, bg)
    start_comp = cell_to_component[start]
    assert components[start_comp]["area"] == 1, "start must be a single pixel"

    for comp_id, comp in enumerate(components):
        if comp_id == start_comp:
            continue
        assert 1 <= int(comp["area"]) <= 5, "control blob areas must stay in 1..5"

    dir_idx = _initial_direction(g, bg, start)
    out = g.copy()
    encountered: list[int] = []
    path_cells: list[tuple[int, int]] = []
    seen_states: set[tuple[int, int, int, int]] = set()
    prev_area = 1
    r, c = start

    while True:
        state = (r, c, dir_idx, prev_area)
        assert state not in seen_states, "ray must terminate instead of cycling"
        seen_states.add(state)

        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            return out, {"encountered": encountered, "path": path_cells}

        if int(g[nr, nc]) == bg:
            out[nr, nc] = Color.GREEN
            path_cells.append((nr, nc))
            r, c = nr, nc
            continue

        comp_id = cell_to_component[(nr, nc)]
        assert comp_id != start_comp, "ray may not turn on the start pixel"
        area = int(components[comp_id]["area"])
        assert area != prev_area, "successive blob areas must differ"
        encountered.append(comp_id)
        dir_idx = _turn_idx(dir_idx, 1 if area > prev_area else -1)
        prev_area = area


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a green ray that turns before each visible blob according to whether that blob grew or shrank in area.

    Input structure:
    - A black background contains exactly one green start pixel and several disjoint monochrome control blobs.
    - Each control blob has area 1 to 5, and colors may repeat without encoding the turn direction.
    - The start has exactly one cardinal sightline whose first foreground object is a control blob.

    Rule:
    - Launch the ray from the green start toward that unique visible blob, painting every traversed background cell green.
    - Whenever the next step would enter a control blob, stay on the current cell and turn 90 degrees.
    - Turn right if the encountered blob area is larger than the previous encountered area, otherwise turn left; the start counts as the initial area-1 reference.

    Key insight:
    - The blobs are only turn triggers, and the actual instruction is carried by the sign of the area change rather than by the blob colors.
    """
    out, _meta = _trace_ray(grid)
    return out


def _sample_areas(rng: np.random.Generator, count: int) -> list[int]:
    for _ in range(200):
        areas: list[int] = []
        prev = 1
        for i in range(count):
            choices = [area for area in range(1, 6) if area != prev]
            if i == 0:
                choices = [area for area in choices if area > prev]
            area = int(rng.choice(choices))
            areas.append(area)
            prev = area
        if any(curr < prev for prev, curr in zip(areas, areas[1:])):
            return areas
    raise AssertionError("failed to sample non-degenerate area sequence")


def _shape_cells(
    anchor: tuple[int, int],
    dir_idx: int,
    turn: int,
    area: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    forward = DIRS[dir_idx]
    side = DIRS[_turn_idx(dir_idx, -turn)]
    local_shape = LOCAL_SHAPES[area][int(rng.integers(len(LOCAL_SHAPES[area])))]
    cells = {
        (
            anchor[0] + a * forward[0] + b * side[0],
            anchor[1] + a * forward[1] + b * side[1],
        )
        for a, b in local_shape
    }
    assert len(cells) == area
    return cells


def _shift(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * steps, cell[1] + direction[1] * steps


def _sample_colors(
    rng: np.random.Generator,
    count: int,
    turns: list[int],
) -> list[int]:
    pool_size = int(rng.integers(2, min(5, count + 1)))
    pool = list(map(int, rng.choice(PATH_BLOB_COLORS, size=pool_size, replace=False)))
    colors = [int(rng.choice(pool)) for _ in range(count)]
    left_indices = [idx for idx, turn in enumerate(turns) if turn == -1]
    if left_indices:
        colors[int(rng.choice(left_indices))] = colors[0]
    return colors


def _plan_scene(rng: np.random.Generator) -> dict | None:
    count = int(rng.integers(3, 6))
    areas = _sample_areas(rng, count)

    turns: list[int] = []
    prev_area = 1
    for area in areas:
        turns.append(1 if area > prev_area else -1)
        prev_area = area

    directions = [int(rng.integers(4))]
    for turn in turns:
        directions.append(_turn_idx(directions[-1], turn))

    path_cells: set[tuple[int, int]] = {(0, 0)}
    fg_cells: set[tuple[int, int]] = {(0, 0)}
    current = (0, 0)
    blobs: list[dict] = []

    for i, (area, dir_idx, turn) in enumerate(zip(areas, directions[:-1], turns)):
        length = int(rng.integers(1, 5))
        segment = [_shift(current, DIRS[dir_idx], step) for step in range(1, length + 1)]
        if any(cell in path_cells or cell in fg_cells for cell in segment):
            return None

        anchor = _shift(current, DIRS[dir_idx], length + 1)
        next_dir_idx = _turn_idx(dir_idx, turn)
        next_after_turn = _shift(segment[-1], DIRS[next_dir_idx])
        blob_cells = _shape_cells(anchor, dir_idx, turn, area, rng)

        if next_after_turn in blob_cells:
            return None
        if any(cell in path_cells or cell in fg_cells for cell in blob_cells):
            return None
        if any(nei in fg_cells for cell in blob_cells for nei in _neighbors4(cell) if nei not in blob_cells):
            return None

        path_cells.update(segment)
        fg_cells.update(blob_cells)
        blobs.append(
            {
                "area": area,
                "dir_idx": dir_idx,
                "turn": turn,
                "cells": blob_cells,
                "anchor": anchor,
                "turnpoint": segment[-1],
                "index": i,
            }
        )
        current = segment[-1]

    all_scene = path_cells | fg_cells
    rows = [cell[0] for cell in all_scene]
    cols = [cell[1] for cell in all_scene]
    final_dir_idx = directions[-1]
    final_dir = DIRS[final_dir_idx]
    extra = int(rng.integers(2, 5))
    if final_dir_idx == 0:
        exit_length = current[0] - min(rows) + extra
    elif final_dir_idx == 1:
        exit_length = max(cols) - current[1] + extra
    elif final_dir_idx == 2:
        exit_length = max(rows) - current[0] + extra
    else:
        exit_length = current[1] - min(cols) + extra

    final_segment = [_shift(current, final_dir, step) for step in range(1, exit_length + 1)]
    if any(cell in path_cells or cell in fg_cells for cell in final_segment):
        return None
    path_cells.update(final_segment)

    all_scene = path_cells | fg_cells
    min_r = min(cell[0] for cell in all_scene)
    max_r = max(cell[0] for cell in all_scene)
    min_c = min(cell[1] for cell in all_scene)
    max_c = max(cell[1] for cell in all_scene)

    top = 0 if final_dir_idx == 0 else int(rng.integers(1, 4))
    right = 0 if final_dir_idx == 1 else int(rng.integers(1, 4))
    bottom = 0 if final_dir_idx == 2 else int(rng.integers(1, 4))
    left = 0 if final_dir_idx == 3 else int(rng.integers(1, 4))

    height = max_r - min_r + 1 + top + bottom
    width = max_c - min_c + 1 + left + right
    if height > 30 or width > 30:
        return None

    offset_r = top - min_r
    offset_c = left - min_c
    colors = _sample_colors(rng, count, turns)

    return {
        "height": height,
        "width": width,
        "offset": (offset_r, offset_c),
        "start": (offset_r, offset_c),
        "blobs": blobs,
        "colors": colors,
        "turns": turns,
        "areas": areas,
        "path_cells": path_cells,
    }


def _render_plan(plan: dict) -> np.ndarray:
    grid = np.full((plan["height"], plan["width"]), Color.BLACK, dtype=int)
    start_r, start_c = plan["start"]
    grid[start_r, start_c] = Color.GREEN
    for blob, color in zip(plan["blobs"], plan["colors"]):
        for r, c in blob["cells"]:
            grid[r + plan["offset"][0], c + plan["offset"][1]] = int(color)
    return grid


def _validate_scene(grid: np.ndarray, expected_blob_count: int) -> bool:
    try:
        out, meta = _trace_ray(grid)
    except AssertionError:
        return False

    if len(meta["encountered"]) != expected_blob_count:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        plan = _plan_scene(rng)
        if plan is None:
            continue
        grid = _render_plan(plan)
        if _validate_scene(grid, expected_blob_count=len(plan["blobs"])):
            return grid
    raise AssertionError("failed to generate a valid input")
