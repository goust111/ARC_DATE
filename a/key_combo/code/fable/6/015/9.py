from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


RED = Color.RED
GREY = Color.GREY
PALETTE = tuple(color for color in Color.ALL_COLORS if color not in {RED, GREY})
L_CORNERS = {(0, 0), (0, 1), (1, 0), (1, 1)}
MAX_GENERATION_ATTEMPTS = 400


def _object_color(grid: np.ndarray, comp: dict) -> int | None:
    colors = {int(grid[int(r), int(c)]) for r, c in comp["coords"]}
    if len(colors) != 1:
        return None
    return next(iter(colors))


def _l_info(comp: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    assert int(coords.shape[0]) == 3, "L triomino must have area 3"
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "L triomino must occupy a 2x2 bbox"

    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = L_CORNERS - rel
    assert len(missing) == 1, "expected exactly one missing 2x2 corner"
    mr, mc = next(iter(missing))
    missing_cell = (int(r0 + mr), int(c0 + mc))
    direction = (-1 if mr == 0 else 1, -1 if mc == 0 else 1)
    return missing_cell, direction


def _is_l_triomino(comp: dict) -> bool:
    try:
        _l_info(comp)
    except AssertionError:
        return False
    return True


def _follow_chain(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert objects, "expected foreground objects"

    cell_to_object: dict[tuple[int, int], int] = {}
    red_l_ids: list[int] = []
    grey_l_ids: set[int] = set()

    for obj_id, comp in enumerate(objects):
        for r, c in comp["coords"]:
            cell_to_object[(int(r), int(c))] = obj_id
        color = _object_color(g, comp)
        if color is None or not _is_l_triomino(comp):
            continue
        if color == RED:
            red_l_ids.append(obj_id)
        elif color == GREY:
            grey_l_ids.add(obj_id)

    assert len(red_l_ids) == 1, "expected exactly one red L triomino"
    current_id = red_l_ids[0]
    visited = {current_id}

    while True:
        missing_cell, direction = _l_info(objects[current_id])
        r, c = missing_cell
        dr, dc = direction
        r += dr
        c += dc

        next_id: int | None = None
        while in_bounds(g.shape, r, c):
            if int(g[r, c]) != int(bg):
                next_id = cell_to_object[(int(r), int(c))]
                break
            r += dr
            c += dc

        assert next_id is not None, "pointer ray must hit another object"
        if next_id in grey_l_ids:
            assert next_id not in visited, "pointer chain must not loop"
            visited.add(next_id)
            current_id = next_id
            continue
        return objects, next_id


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Start at the unique red L triomino, follow the missing-corner diagonal relay
    through grey L triominoes, and recolor the terminal colored blob red.
    """
    g = np.asarray(input_grid, dtype=int)
    objects, target_id = _follow_chain(g)
    out = g.copy()
    for r, c in objects[target_id]["coords"]:
        out[int(r), int(c)] = RED
    return out


def _l_cells(top: int, left: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (top + rr, left + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _l_record(top: int, left: int, missing: tuple[int, int], color: int) -> dict:
    mr, mc = missing
    missing_cell = (top + mr, left + mc)
    direction = (-1 if mr == 0 else 1, -1 if mc == 0 else 1)
    return {
        "kind": "l",
        "color": int(color),
        "cells": _l_cells(top, left, missing),
        "missing_cell": missing_cell,
        "direction": direction,
    }


def _neighbor4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _touches_occupied(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for cell in cells:
        if cell in occupied:
            return True
        for nbr in _neighbor4(cell):
            if nbr in occupied:
                return True
    return False


def _available_steps(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = start
    dr, dc = direction
    steps = 0
    r += dr
    c += dc
    while in_bounds(shape, r, c):
        steps += 1
        r += dr
        c += dc
    return steps


def _segment_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    *,
    include_hit: bool,
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(1, steps + 1):
        cells.append((r + dr * step, c + dc * step))
    if not include_hit:
        cells = cells[:-1]
    return cells


def _ray_parameter(
    start: tuple[int, int],
    direction: tuple[int, int],
    cell: tuple[int, int],
) -> int | None:
    sr, sc = start
    dr, dc = direction
    rr = cell[0] - sr
    cc = cell[1] - sc
    if dr == 0 or dc == 0:
        return None
    if rr == 0 or cc == 0:
        return None
    if rr % dr != 0 or cc % dc != 0:
        return None
    step_r = rr // dr
    step_c = cc // dc
    if step_r != step_c or step_r <= 0:
        return None
    return int(step_r)


def _sample_red_start(shape: tuple[int, int]) -> dict | None:
    H, W = shape
    for _ in range(120):
        missing = random.choice(tuple(L_CORNERS))
        top = random.randint(0, H - 2)
        left = random.randint(0, W - 2)
        record = _l_record(top, left, missing, RED)
        if _available_steps(record["missing_cell"], record["direction"], shape) < 4:
            continue
        return record
    return None


def _place_next_l(
    source: dict,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    remaining_links: int,
) -> tuple[dict, set[tuple[int, int]]] | None:
    max_steps = _available_steps(source["missing_cell"], source["direction"], shape)
    if max_steps < 2:
        return None

    for _ in range(200):
        missing = random.choice(tuple(L_CORNERS))
        steps = random.randint(2, max_steps)
        hit_r = source["missing_cell"][0] + source["direction"][0] * steps
        hit_c = source["missing_cell"][1] + source["direction"][1] * steps

        out_dr = -1 if missing[0] == 0 else 1
        out_dc = -1 if missing[1] == 0 else 1
        missing_cell = (hit_r + out_dr, hit_c + out_dc)
        top = missing_cell[0] - missing[0]
        left = missing_cell[1] - missing[1]
        if not (0 <= top <= shape[0] - 2 and 0 <= left <= shape[1] - 2):
            continue

        record = _l_record(top, left, missing, GREY)
        cells = set(record["cells"])
        if _touches_occupied(cells, occupied):
            continue

        segment = _segment_cells(source["missing_cell"], source["direction"], steps, include_hit=False)
        if any(cell in occupied or cell in cells or cell in reserved_bg for cell in segment):
            continue
        if _available_steps(record["missing_cell"], record["direction"], shape) < max(2, remaining_links * 2):
            continue
        return record, set(segment)
    return None


def _grow_blob(
    anchor: tuple[int, int],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    source_start: tuple[int, int],
    source_direction: tuple[int, int],
    hit_step: int,
) -> set[tuple[int, int]] | None:
    target_size = random.randint(2, 7)
    blob = {anchor}

    for _ in range(80):
        frontier: list[tuple[int, int]] = []
        for cell in tuple(blob):
            for nbr in _neighbor4(cell):
                if nbr in blob or nbr in blocked or not in_bounds(shape, nbr[0], nbr[1]):
                    continue
                param = _ray_parameter(source_start, source_direction, nbr)
                if param is not None and param != hit_step:
                    continue
                frontier.append(nbr)
        if not frontier:
            break
        blob.add(random.choice(frontier))
        if len(blob) >= target_size:
            break

    if len(blob) < 2:
        return None
    return blob


def _place_endpoint_blob(
    source: dict,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[dict, set[tuple[int, int]]] | None:
    max_steps = _available_steps(source["missing_cell"], source["direction"], shape)
    if max_steps < 2:
        return None

    endpoint_colors = [color for color in PALETTE if color != source["color"]]
    for _ in range(240):
        steps = random.randint(2, max_steps)
        anchor = (
            source["missing_cell"][0] + source["direction"][0] * steps,
            source["missing_cell"][1] + source["direction"][1] * steps,
        )
        segment = _segment_cells(source["missing_cell"], source["direction"], steps, include_hit=False)
        if any(cell in occupied or cell in reserved_bg for cell in segment):
            continue

        blocked = set(occupied) | set(reserved_bg)
        blocked.update(segment)
        blob = _grow_blob(anchor, shape, blocked, source["missing_cell"], source["direction"], steps)
        if blob is None or _touches_occupied(blob, occupied):
            continue

        color = random.choice(endpoint_colors)
        return {"kind": "blob", "color": int(color), "cells": tuple(sorted(blob))}, set(segment)
    return None


def _place_decoy_blob(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    color: int,
) -> dict | None:
    H, W = shape
    blocked = set(occupied) | set(reserved_bg)
    for _ in range(160):
        seed = (random.randrange(H), random.randrange(W))
        if seed in blocked:
            continue
        blob = {seed}
        target_size = random.randint(1, 6)
        for _ in range(60):
            frontier: list[tuple[int, int]] = []
            for cell in tuple(blob):
                for nbr in _neighbor4(cell):
                    if nbr in blob or nbr in blocked or not in_bounds(shape, nbr[0], nbr[1]):
                        continue
                    frontier.append(nbr)
            if not frontier:
                break
            blob.add(random.choice(frontier))
            if len(blob) >= target_size:
                break
        if _touches_occupied(blob, occupied):
            continue
        return {"kind": "blob", "color": int(color), "cells": tuple(sorted(blob))}
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(14, 24)
    W = random.randint(14, 24)
    shape = (H, W)

    bg = random.choice(PALETTE)
    colors_for_blobs = [color for color in PALETTE if color != bg]
    if len(colors_for_blobs) < 2:
        return None

    start = _sample_red_start(shape)
    if start is None:
        return None

    objects = [start]
    occupied = set(start["cells"])
    reserved_bg: set[tuple[int, int]] = set()
    source = start

    grey_count = random.randint(1, 4)
    for remaining in range(grey_count, 0, -1):
        placed = _place_next_l(source, shape, occupied, reserved_bg, remaining_links=remaining - 1)
        if placed is None:
            return None
        record, segment = placed
        objects.append(record)
        occupied.update(record["cells"])
        reserved_bg.update(segment)
        source = record

    endpoint = _place_endpoint_blob(source, shape, occupied, reserved_bg)
    if endpoint is None:
        return None
    endpoint_record, segment = endpoint
    objects.append(endpoint_record)
    occupied.update(endpoint_record["cells"])
    reserved_bg.update(segment)

    decoy_count = random.randint(1, 4)
    for _ in range(decoy_count):
        color = random.choice(colors_for_blobs)
        decoy = _place_decoy_blob(shape, occupied, reserved_bg, color)
        if decoy is None:
            continue
        objects.append(decoy)
        occupied.update(decoy["cells"])

    inp = np.full(shape, bg, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            inp[int(r), int(c)] = int(obj["color"])

    out = inp.copy()
    for r, c in endpoint_record["cells"]:
        out[int(r), int(c)] = RED

    try:
        got = main(inp)
    except AssertionError:
        return None
    if np.array_equal(got, out):
        return inp, out
    return None


def _generate_valid_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid pointer-chain ARC input")


def generate_input() -> np.ndarray:
    return _generate_valid_case()[0]
