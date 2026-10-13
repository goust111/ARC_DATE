from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PAINT_COLORS = (1, 2, 3, 4, 6, 7, 8)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
CARDINALS = (UP, DOWN, LEFT, RIGHT)
CLOCKWISE = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}
HALO8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
)

GEN_RETRIES = 500
RAY_SOURCE_TRIES = 24
RAY_SHAPE_TRIES = 20


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _step(pos: tuple[int, int], direction: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return pos[0] + direction[0] * k, pos[1] + direction[1] * k


def _source_direction(shape: tuple[int, int], pos: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = pos
    if r == 0 and 0 < c < w - 1:
        return DOWN
    if r == h - 1 and 0 < c < w - 1:
        return UP
    if c == 0 and 0 < r < h - 1:
        return RIGHT
    if c == w - 1 and 0 < r < h - 1:
        return LEFT
    raise ValueError("sources must be non-corner edge pixels")


def _orth_step(direction: tuple[int, int]) -> tuple[int, int]:
    if direction[0] != 0:
        return (0, 1)
    return (1, 0)


def _lens_cells(center: tuple[int, int], orient: str) -> tuple[tuple[int, int], ...]:
    r, c = center
    if orient == "h":
        return ((r, c - 1), (r, c), (r, c + 1))
    return ((r - 1, c), (r, c), (r + 1, c))


def _reserve_group(
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    cells: Iterable[tuple[int, int]],
) -> None:
    cells = tuple(cells)
    for cell in cells:
        occupied_fixed.add(cell)
    for r, c in cells:
        for dr, dc in HALO8:
            forbidden_fixed.add((r + dr, c + dc))


def _can_place_fixed_group(
    shape: tuple[int, int],
    cells: Iterable[tuple[int, int]],
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied_fixed or (r, c) in forbidden_fixed or (r, c) in occupied_path:
            return False
    return True


def _free_run_cells(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    occupied_fixed: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    for k in range(1, steps + 1):
        cell = _step(pos, direction, k)
        if cell in occupied_fixed or cell in occupied_path:
            return None
        if not _in_bounds(shape, *cell):
            return None
        cells.append(cell)
    return cells


def _detect_lenses(
    grid: np.ndarray,
) -> tuple[list[dict], dict[tuple[int, int], dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    lenses: list[dict] = []
    by_cell: dict[tuple[int, int], dict] = {}
    centers: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color in (BLACK, GREY, MAROON):
                continue
            horiz = int(g[r, c - 1]) == GREY and int(g[r, c + 1]) == GREY
            vert = int(g[r - 1, c]) == GREY and int(g[r + 1, c]) == GREY
            if horiz == vert:
                continue
            orient = "h" if horiz else "v"
            cells = _lens_cells((r, c), orient)
            if any(cell in by_cell for cell in cells):
                continue
            lens = {
                "center": (r, c),
                "orient": orient,
                "color": color,
                "cells": cells,
            }
            lenses.append(lens)
            centers.add((r, c))
            for cell in cells:
                by_cell[cell] = lens
    return lenses, by_cell, centers


def _edge_sources(grid: np.ndarray, lens_centers: set[tuple[int, int]]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources: list[dict] = []
    edge_cells: list[tuple[int, int]] = []
    for c in range(w):
        edge_cells.append((0, c))
        edge_cells.append((h - 1, c))
    for r in range(1, h - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, w - 1))
    for r, c in sorted(set(edge_cells)):
        if (r, c) in lens_centers:
            continue
        color = int(g[r, c])
        if color in (BLACK, GREY, MAROON):
            continue
        direction = _source_direction(g.shape, (r, c))
        sources.append({"pos": (r, c), "color": color, "dir": direction})
    return sources


def _trace_one_ray(
    grid: np.ndarray,
    out: np.ndarray,
    lens_by_cell: dict[tuple[int, int], dict],
    blockers: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
    source: dict,
) -> None:
    pos = tuple(source["pos"])
    direction = tuple(source["dir"])
    color = int(source["color"])
    seen: set[tuple[int, int, int, int, int, int]] = set()
    while True:
        state = (pos[0], pos[1], direction[0], direction[1], color, len(occupied_path))
        if state in seen:
            break
        seen.add(state)

        nr, nc = _step(pos, direction)
        if not _in_bounds(grid.shape, nr, nc):
            break
        if (nr, nc) in blockers:
            break

        lens = lens_by_cell.get((nr, nc))
        if lens is not None:
            center = tuple(lens["center"])
            orient = str(lens["orient"])
            if orient == "h":
                if direction[0] != 0 and (nr, nc) == center:
                    pos = center
                    color = int(lens["color"])
                    continue
                if direction[1] != 0 and (nr, nc) != center:
                    pos = center
                    direction = CLOCKWISE[direction]
                    color = int(lens["color"])
                    continue
            else:
                if direction[1] != 0 and (nr, nc) == center:
                    pos = center
                    color = int(lens["color"])
                    continue
                if direction[0] != 0 and (nr, nc) != center:
                    pos = center
                    direction = CLOCKWISE[direction]
                    color = int(lens["color"])
                    continue
            break

        if (nr, nc) in occupied_path:
            break
        if int(grid[nr, nc]) != BLACK:
            break

        out[nr, nc] = color
        occupied_path.add((nr, nc))
        pos = (nr, nc)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _lenses, lens_by_cell, lens_centers = _detect_lenses(g)
    blockers = {tuple(map(int, cell)) for cell in np.argwhere(g == MAROON)}
    sources = _edge_sources(g, lens_centers)

    occupied_path: set[tuple[int, int]] = set()
    for source in sources:
        _trace_one_ray(g, out, lens_by_cell, blockers, occupied_path, source)
    return out


def _edge_candidates(
    shape: tuple[int, int],
    min_key: tuple[int, int],
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        candidates.append((0, c))
    for r in range(1, h - 1):
        candidates.append((r, 0))
        candidates.append((r, w - 1))
    for c in range(1, w - 1):
        candidates.append((h - 1, c))
    out: list[tuple[int, int]] = []
    for cell in sorted(candidates):
        if cell <= min_key:
            continue
        if not _can_place_fixed_group(shape, [cell], occupied_fixed, forbidden_fixed, occupied_path):
            continue
        out.append(cell)
    return out


def _sample_sequence(style: str) -> list[str]:
    if style == "front":
        seq = ["front"]
        if random.random() < 0.45:
            seq.append("front")
        return seq
    if style == "turn":
        seq = ["turn"]
        if random.random() < 0.45:
            seq.append("front")
        return seq
    if style == "mixed":
        return random.choice((["front", "turn"], ["turn", "front"]))
    return []


def _pick_new_color(current_color: int) -> int:
    choices = [color for color in PAINT_COLORS if color != current_color]
    return random.choice(choices)


def _choose_front_event(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    current_color: int,
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
) -> dict | None:
    run_lengths = list(range(1, 6))
    random.shuffle(run_lengths)
    occupied_path = global_path | local_path
    ortho = _orth_step(direction)
    orient = "h" if direction[0] != 0 else "v"
    for run_len in run_lengths:
        painted = _free_run_cells(shape, pos, direction, run_len, occupied_fixed, occupied_path)
        if painted is None:
            continue
        center = _step(pos, direction, run_len + 1)
        cells = (
            (center[0] - ortho[0], center[1] - ortho[1]),
            center,
            (center[0] + ortho[0], center[1] + ortho[1]),
        )
        if not _can_place_fixed_group(shape, cells, occupied_fixed, forbidden_fixed, occupied_path):
            continue
        return {
            "painted": painted,
            "center": center,
            "orient": orient,
            "lens_cells": cells,
            "new_pos": center,
            "new_dir": direction,
            "new_color": _pick_new_color(current_color),
            "kind": "front",
        }
    return None


def _choose_turn_event(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    current_color: int,
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
) -> dict | None:
    run_lengths = list(range(1, 5))
    random.shuffle(run_lengths)
    occupied_path = global_path | local_path
    orient = "v" if direction[0] != 0 else "h"
    new_dir = CLOCKWISE[direction]
    for run_len in run_lengths:
        painted = _free_run_cells(shape, pos, direction, run_len, occupied_fixed, occupied_path)
        if painted is None:
            continue
        center = _step(pos, direction, run_len + 2)
        cells = (
            _step(center, direction, -1),
            center,
            _step(center, direction, 1),
        )
        if not _can_place_fixed_group(shape, cells, occupied_fixed, forbidden_fixed, occupied_path):
            continue
        exit_cell = _step(center, new_dir)
        if not _in_bounds(shape, *exit_cell):
            continue
        if exit_cell in local_path:
            continue
        return {
            "painted": painted,
            "center": center,
            "orient": orient,
            "lens_cells": cells,
            "new_pos": center,
            "new_dir": new_dir,
            "new_color": _pick_new_color(current_color),
            "kind": "turn",
        }
    return None


def _choose_boundary_termination(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    occupied_fixed: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> dict | None:
    painted: list[tuple[int, int]] = []
    step_count = 1
    while True:
        cell = _step(pos, direction, step_count)
        if not _in_bounds(shape, *cell):
            break
        if cell in occupied_fixed or cell in occupied_path:
            return None
        painted.append(cell)
        step_count += 1
    if not painted:
        return None
    return {"painted": painted, "kind": "boundary"}


def _choose_blocker_termination(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
) -> dict | None:
    run_lengths = list(range(1, 6))
    random.shuffle(run_lengths)
    occupied_path = global_path | local_path
    for run_len in run_lengths:
        painted = _free_run_cells(shape, pos, direction, run_len, occupied_fixed, occupied_path)
        if painted is None:
            continue
        blocker = _step(pos, direction, run_len + 1)
        if not (0 < blocker[0] < shape[0] - 1 and 0 < blocker[1] < shape[1] - 1):
            continue
        if not _can_place_fixed_group(shape, [blocker], occupied_fixed, forbidden_fixed, occupied_path):
            continue
        return {"painted": painted, "blocker": blocker, "kind": "blocker"}
    return None


def _choose_path_termination(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    occupied_fixed: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
) -> dict | None:
    painted: list[tuple[int, int]] = []
    step_count = 1
    while True:
        cell = _step(pos, direction, step_count)
        if not _in_bounds(shape, *cell):
            return None
        if cell in local_path or cell in occupied_fixed:
            return None
        if cell in global_path:
            if painted:
                return {"painted": painted, "kind": "path"}
            return None
        painted.append(cell)
        step_count += 1


def _try_plan_from_source(
    shape: tuple[int, int],
    source_pos: tuple[int, int],
    source_color: int,
    style: str,
    termination: str,
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
) -> dict | None:
    for _ in range(RAY_SHAPE_TRIES):
        local_occ = set(occupied_fixed)
        local_forbidden = set(forbidden_fixed)
        local_path: set[tuple[int, int]] = set()
        ray_paints: list[tuple[tuple[int, int], int]] = []
        new_lenses: list[dict] = []
        new_blockers: list[tuple[int, int]] = []

        _reserve_group(local_occ, local_forbidden, [source_pos])
        pos = source_pos
        direction = _source_direction(shape, source_pos)
        current_color = source_color
        sequence = _sample_sequence(style)
        ok = True

        for kind in sequence:
            chooser = _choose_front_event if kind == "front" else _choose_turn_event
            event = chooser(
                shape,
                pos,
                direction,
                current_color,
                local_occ,
                local_forbidden,
                global_path,
                local_path,
            )
            if event is None:
                ok = False
                break
            for cell in event["painted"]:
                local_path.add(cell)
                ray_paints.append((cell, current_color))
            _reserve_group(local_occ, local_forbidden, event["lens_cells"])
            new_lenses.append(
                {
                    "center": tuple(event["center"]),
                    "orient": str(event["orient"]),
                    "color": int(event["new_color"]),
                }
            )
            pos = tuple(event["new_pos"])
            direction = tuple(event["new_dir"])
            current_color = int(event["new_color"])

        if not ok:
            continue

        occupied_path = global_path | local_path
        if termination == "path":
            term = _choose_path_termination(shape, pos, direction, local_occ, global_path, local_path)
        elif termination == "blocker":
            term = _choose_blocker_termination(
                shape,
                pos,
                direction,
                local_occ,
                local_forbidden,
                global_path,
                local_path,
            )
        else:
            term = _choose_boundary_termination(shape, pos, direction, local_occ, occupied_path)
        if term is None:
            continue

        for cell in term["painted"]:
            local_path.add(cell)
            ray_paints.append((cell, current_color))
        if term["kind"] == "blocker":
            blocker = tuple(term["blocker"])
            _reserve_group(local_occ, local_forbidden, [blocker])
            new_blockers.append(blocker)

        return {
            "source": {"pos": source_pos, "color": source_color},
            "lenses": new_lenses,
            "blockers": new_blockers,
            "paints": ray_paints,
            "used_front": "front" in sequence,
            "used_turn": "turn" in sequence,
        }
    return None


def _try_build_ray(
    shape: tuple[int, int],
    min_key: tuple[int, int],
    remaining_after: int,
    source_color: int,
    style: str,
    termination: str,
    occupied_fixed: set[tuple[int, int]],
    forbidden_fixed: set[tuple[int, int]],
    global_path: set[tuple[int, int]],
) -> dict | None:
    candidates = _edge_candidates(shape, min_key, occupied_fixed, forbidden_fixed, global_path)
    if len(candidates) <= remaining_after:
        return None
    usable = candidates[: len(candidates) - remaining_after]
    random.shuffle(usable)
    for source_pos in usable[:RAY_SOURCE_TRIES]:
        ray = _try_plan_from_source(
            shape,
            source_pos,
            source_color,
            style,
            termination,
            occupied_fixed,
            forbidden_fixed,
            global_path,
        )
        if ray is not None:
            return ray
    return None


def _sample_roles(source_count: int) -> list[tuple[str, str]]:
    roles: list[tuple[str, str]] = [
        ("front", random.choice(("boundary", "blocker"))),
        ("turn", random.choice(("boundary", "blocker"))),
    ]
    if source_count >= 3:
        if random.random() < 0.65:
            roles.append(("straight", "path"))
        else:
            roles.append((random.choice(("straight", "mixed", "front")), random.choice(("boundary", "blocker"))))
    if source_count >= 4:
        if not any(term == "path" for _style, term in roles) and random.random() < 0.5:
            roles.append(("straight", "path"))
        else:
            roles.append((random.choice(("straight", "mixed", "front", "turn")), random.choice(("boundary", "blocker"))))
    if not any(term == "blocker" for _style, term in roles) and random.random() < 0.7:
        idx = random.randrange(len(roles))
        style, term = roles[idx]
        if term != "path":
            roles[idx] = (style, "blocker")
    if all(term != "boundary" for _style, term in roles):
        idx = random.randrange(len(roles))
        style, term = roles[idx]
        if term != "path":
            roles[idx] = (style, "boundary")
    return roles


def _render_input(
    shape: tuple[int, int],
    sources: list[dict],
    lenses: list[dict],
    blockers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for source in sources:
        r, c = source["pos"]
        grid[r, c] = int(source["color"])
    for lens in lenses:
        cells = _lens_cells(tuple(lens["center"]), str(lens["orient"]))
        for r, c in cells:
            grid[r, c] = GREY
        cr, cc = lens["center"]
        grid[cr, cc] = int(lens["color"])
    for r, c in blockers:
        grid[r, c] = MAROON
    return grid


def _render_output(grid: np.ndarray, paints: list[tuple[tuple[int, int], int]]) -> np.ndarray:
    out = grid.copy()
    occupied: set[tuple[int, int]] = set()
    for cell, color in paints:
        if cell in occupied:
            raise ValueError("paint overlap")
        occupied.add(cell)
        out[cell] = int(color)
    return out


def _valid_generated_pair(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    edge_colors = {
        int(inp[r, c])
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1)
    }
    if len(edge_colors - {BLACK}) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 17)
    w = random.randint(11, 17)
    shape = (h, w)
    source_count = random.choice((3, 4))
    source_colors = random.sample(PAINT_COLORS, source_count)
    roles = _sample_roles(source_count)

    occupied_fixed: set[tuple[int, int]] = set()
    forbidden_fixed: set[tuple[int, int]] = set()
    global_path: set[tuple[int, int]] = set()
    sources: list[dict] = []
    lenses: list[dict] = []
    blockers: list[tuple[int, int]] = []
    paints: list[tuple[tuple[int, int], int]] = []
    min_key = (-1, -1)
    used_front = False
    used_turn = False

    for idx, ((style, termination), source_color) in enumerate(zip(roles, source_colors)):
        ray = _try_build_ray(
            shape,
            min_key,
            source_count - idx - 1,
            source_color,
            style,
            termination,
            occupied_fixed,
            forbidden_fixed,
            global_path,
        )
        if ray is None:
            return None

        source = ray["source"]
        source_pos = tuple(source["pos"])
        _reserve_group(occupied_fixed, forbidden_fixed, [source_pos])
        min_key = source_pos
        sources.append(source)

        for lens in ray["lenses"]:
            cells = _lens_cells(tuple(lens["center"]), str(lens["orient"]))
            if not _can_place_fixed_group(shape, cells, occupied_fixed, set(), global_path):
                return None
            _reserve_group(occupied_fixed, forbidden_fixed, cells)
            lenses.append(lens)

        for blocker in ray["blockers"]:
            if blocker in occupied_fixed or blocker in global_path:
                return None
            _reserve_group(occupied_fixed, forbidden_fixed, [blocker])
            blockers.append(blocker)

        for cell, color in ray["paints"]:
            if cell in global_path or cell in occupied_fixed:
                return None
            global_path.add(cell)
            paints.append((cell, color))

        used_front = used_front or bool(ray["used_front"])
        used_turn = used_turn or bool(ray["used_turn"])

    if not used_front or not used_turn:
        return None
    if len(lenses) < 2:
        return None

    inp = _render_input(shape, sources, lenses, blockers)
    expected = _render_output(inp, paints)
    if not _valid_generated_pair(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-lens task")
