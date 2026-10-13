import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

MARKER_COLORS = (BLUE, GREEN, YELLOW)
COLOR_SEQUENCE = (RED, PINK)
BASE_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
STEP_SIZES = (1, 2, 3)
GEN_RETRIES = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _ray_cells(shape: tuple[int, int], markers: list[tuple[int, int]], step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    r, c = markers[-1][0] + dr, markers[-1][1] + dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _state_color(base_color: int, collisions: int) -> int:
    if collisions <= 0:
        return int(base_color)
    return RED if collisions == 1 else PINK


def _parse_groups(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, GREY})
    assert 2 <= len(colors) <= 3, "expected 2-3 marker colors"
    assert set(colors).issubset(set(MARKER_COLORS)), "unexpected input colors"

    groups: list[dict] = []
    occupied_markers: set[tuple[int, int]] = set()
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 3, "each marker color must appear exactly three times"
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        deltas = np.diff(coords, axis=0)
        assert np.array_equal(deltas[0], deltas[1]), "markers must be equally spaced"
        dr, dc = map(int, deltas[0])
        assert dr > 0 or (dr == 0 and dc > 0), "markers must be sorted in row-major direction"
        assert (
            (dr == 0 and dc > 0)
            or (dc == 0 and dr > 0)
            or (abs(dr) == abs(dc) and dr > 0)
        ), "marker line must be horizontal, vertical, or 45-degree diagonal"
        markers = [tuple(map(int, coord)) for coord in coords]
        for cell in markers:
            assert cell not in occupied_markers
            occupied_markers.add(cell)
        groups.append(
            {
                "color": int(color),
                "markers": markers,
                "step": (dr, dc),
            }
        )
    return groups


def _parse_obstacles(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    comps = _components(grid == GREY)
    assert 1 <= len(comps) <= 3, "expected 1-3 obstacle blocks"
    lookup: dict[tuple[int, int], int] = {}
    obstacles: list[dict] = []
    for idx, coords in enumerate(comps):
        r0, c0, r1, c1 = _bbox(coords)
        assert coords.shape[0] == 4, "each obstacle must have area 4"
        assert r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2, "each obstacle must be a 2x2 block"
        cells = {tuple(map(int, rc)) for rc in coords}
        assert cells == {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
        for cell in cells:
            lookup[cell] = idx
        obstacles.append({"bbox": (r0, c0, r1, c1), "cells": cells})
    return obstacles, lookup


def _trace_group(grid: np.ndarray, group: dict, obstacle_lookup: dict[tuple[int, int], int]) -> dict:
    ray = _ray_cells(grid.shape, group["markers"], group["step"])
    collisions = 0
    prev_obstacle = None
    paints: list[tuple[int, int, int]] = []
    painted_by_state: dict[int, list[tuple[int, int]]] = defaultdict(list)
    hit_order: list[int] = []
    hit_steps: dict[int, list[int]] = defaultdict(list)

    for step_idx, (r, c) in enumerate(ray):
        obstacle_id = obstacle_lookup.get((r, c))
        if obstacle_id is not None:
            hit_steps[obstacle_id].append(step_idx)
            if obstacle_id != prev_obstacle:
                hit_order.append(obstacle_id)
                collisions += 1
            prev_obstacle = obstacle_id
            continue

        prev_obstacle = None
        assert int(grid[r, c]) == BG, "ray may only paint background outside obstacles"
        color = _state_color(group["color"], collisions)
        paints.append((r, c, color))
        painted_by_state[collisions].append((r, c))

    assert 1 <= len(hit_order) <= 2, "each ray must cross 1-2 obstacles"
    assert len(hit_order) == len(set(hit_order)), "a straight ray must not count the same obstacle twice"
    return {
        "ray": ray,
        "paints": paints,
        "painted_by_state": painted_by_state,
        "hit_order": hit_order,
        "hit_steps": hit_steps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate each equally spaced marker line through grey gates, recoloring later segments after each obstacle crossing.

    Input structure:
    - A black background contains 2-3 marker groups, each made of exactly three same-color pixels.
    - Each group is strictly equally spaced on one horizontal, vertical, or 45-degree diagonal line.
    - The grid also contains 1-3 grey 2x2 obstacle blocks that stay fixed.

    Rule:
    - For each marker group, sort its three pixels in row-major direction and use the repeated delta as the continuation step.
    - Start one step beyond the third marker and follow that same step until leaving the grid.
    - Background cells on the ray are painted with the current state color.
    - Crossing a grey obstacle does not stop the ray and does not recolor the obstacle; instead it advances the ray color state once per obstacle block.
    - The state colors are: before any obstacle = the marker color, after one obstacle = red, after two obstacles = pink.

    Key insight:
    - Grey blocks act as pass-through color gates: the marker line fixes the ray geometry, while obstacle crossings partition that ray into visibly recolored segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert np.all(np.isin(g, [BG, BLUE, RED, GREEN, YELLOW, GREY, PINK]))

    groups = _parse_groups(g)
    _obstacles, obstacle_lookup = _parse_obstacles(g)

    out = g.copy()
    painted_cells: set[tuple[int, int]] = set()
    for group in groups:
        info = _trace_group(g, group, obstacle_lookup)
        for r, c, color in info["paints"]:
            assert (r, c) not in painted_cells, "ray paths must not intersect"
            painted_cells.add((r, c))
            out[r, c] = color
    return out


def _enumerate_candidates(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    for mag in STEP_SIZES:
        for base_dr, base_dc in BASE_DIRECTIONS:
            dr = base_dr * mag
            dc = base_dc * mag
            for r0 in range(h):
                for c0 in range(w):
                    markers = [(r0 + k * dr, c0 + k * dc) for k in range(3)]
                    if not all(_in_bounds(shape, r, c) for r, c in markers):
                        continue
                    ray = _ray_cells(shape, markers, (dr, dc))
                    if len(ray) < 4:
                        continue
                    occupied = set(markers) | set(ray)
                    candidates.append(
                        {
                            "markers": markers,
                            "step": (dr, dc),
                            "ray": ray,
                            "occupied": occupied,
                            "kind": (base_dr, base_dc),
                            "mag": mag,
                        }
                    )
    return candidates


def _choose_groups(shape: tuple[int, int], group_count: int) -> list[dict] | None:
    candidates = _enumerate_candidates(shape)
    for _ in range(200):
        chosen: list[dict] = []
        pool = candidates[:]
        random.shuffle(pool)
        target_diagonal = random.random() < 0.45
        while pool and len(chosen) < group_count:
            if target_diagonal and not any(abs(c["step"][0]) == abs(c["step"][1]) and c["step"][0] != 0 for c in chosen):
                diag_pool = [
                    cand for cand in pool
                    if abs(cand["step"][0]) == abs(cand["step"][1]) and cand["step"][0] != 0
                ]
                cand = random.choice(diag_pool) if diag_pool else random.choice(pool)
            else:
                cand = random.choice(pool)
            if all(not (cand["occupied"] & other["occupied"]) for other in chosen):
                chosen.append(cand)
                pool = [other for other in pool if all(not (other["occupied"] & pick["occupied"]) for pick in chosen)]
            else:
                pool.remove(cand)
        if len(chosen) == group_count:
            return chosen
    return None


def _block_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}


def _obstacle_candidates(
    shape: tuple[int, int],
    group: dict,
    other_occupied: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    ray = group["ray"]
    candidates: dict[tuple[int, int], dict] = {}
    for idx, (r, c) in enumerate(ray):
        for r0 in (r - 1, r):
            for c0 in (c - 1, c):
                if not (0 <= r0 < h - 1 and 0 <= c0 < w - 1):
                    continue
                top_left = (r0, c0)
                cells = _block_cells(top_left)
                if cells & marker_cells:
                    continue
                if cells & other_occupied:
                    continue
                hit_indices = [j for j, cell in enumerate(ray) if cell in cells]
                if not hit_indices:
                    continue
                if top_left not in candidates:
                    candidates[top_left] = {
                        "top_left": top_left,
                        "cells": cells,
                        "first_hit": min(hit_indices),
                        "last_hit": max(hit_indices),
                        "hit_indices": tuple(hit_indices),
                        "thick": len(hit_indices) >= 2,
                    }
    return list(candidates.values())


def _select_group_obstacles(
    group: dict,
    placements: list[dict],
    count: int,
    used_cells: set[tuple[int, int]],
    require_thick: bool,
) -> list[dict] | None:
    valid = [p for p in placements if not (p["cells"] & used_cells)]
    if count == 1:
        random.shuffle(valid)
        thick_first = sorted(valid, key=lambda p: (not p["thick"], random.random()))
        for placement in thick_first:
            if placement["first_hit"] < 1:
                continue
            if placement["last_hit"] >= len(group["ray"]) - 1:
                continue
            if require_thick and not placement["thick"]:
                continue
            return [placement]
        if require_thick:
            return None
        return None

    pairs: list[tuple[dict, dict]] = []
    for first in valid:
        for second in valid:
            if first["top_left"] == second["top_left"]:
                continue
            if first["cells"] & second["cells"]:
                continue
            ordered = sorted([first, second], key=lambda p: p["first_hit"])
            a, b = ordered
            if a["first_hit"] < 1:
                continue
            if b["last_hit"] >= len(group["ray"]) - 1:
                continue
            if b["first_hit"] <= a["last_hit"] + 1:
                continue
            if require_thick and not (a["thick"] or b["thick"]):
                continue
            pairs.append((a, b))
    if not pairs:
        return None
    return list(random.choice(pairs))


def _render_input(shape: tuple[int, int], groups: list[dict], obstacle_top_lefts: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group in groups:
        for r, c in group["markers"]:
            grid[r, c] = group["color"]
    for r0, c0 in obstacle_top_lefts:
        grid[r0 : r0 + 2, c0 : c0 + 2] = GREY
    return grid


def _validate_semantics(grid: np.ndarray, expected_group_count: int, expected_obstacle_count: int, require_two_collision: bool, require_thick: bool) -> bool:
    try:
        groups = _parse_groups(grid)
        obstacles, lookup = _parse_obstacles(grid)
    except AssertionError:
        return False

    if len(groups) != expected_group_count or len(obstacles) != expected_obstacle_count:
        return False

    painted_cells: set[tuple[int, int]] = set()
    thick_seen = False
    two_collision_seen = False
    for group in groups:
        try:
            info = _trace_group(grid, group, lookup)
        except AssertionError:
            return False
        if not info["painted_by_state"][0]:
            return False
        if len(info["hit_order"]) == 2:
            two_collision_seen = True
            if not info["painted_by_state"][1] or not info["painted_by_state"][2]:
                return False
        else:
            if info["painted_by_state"][2]:
                return False
        if not info["painted_by_state"][len(info["hit_order"])]:
            return False
        thick_seen = thick_seen or any(len(hit_steps) >= 2 for hit_steps in info["hit_steps"].values())
        for r, c, _color in info["paints"]:
            if (r, c) in painted_cells:
                return False
            painted_cells.add((r, c))

    if require_two_collision and not two_collision_seen:
        return False
    if require_thick and not thick_seen:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape:
        return False
    if np.any(grid == GREY) and not np.array_equal(out[grid == GREY], grid[grid == GREY]):
        return False
    if np.any(np.isin(grid, MARKER_COLORS)) and not np.array_equal(out[np.isin(grid, MARKER_COLORS)], grid[np.isin(grid, MARKER_COLORS)]):
        return False
    if require_two_collision and not np.any(out == PINK):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(15, 19), random.randint(15, 19))
    group_count = 2 if random.random() < 0.65 else 3
    chosen = _choose_groups(shape, group_count)
    if chosen is None:
        return None

    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    groups = []
    for color, base in zip(colors, chosen):
        groups.append(
            {
                "color": color,
                "markers": list(base["markers"]),
                "step": tuple(base["step"]),
                "ray": list(base["ray"]),
            }
        )

    obstacle_counts = [1] * group_count
    require_two_collision = False
    if group_count == 2 and random.random() < 0.6:
        long_groups = [idx for idx, group in enumerate(groups) if len(group["ray"]) >= 7]
        if long_groups:
            obstacle_counts[random.choice(long_groups)] = 2
            require_two_collision = True

    require_thick = random.random() < 0.35
    marker_cells = {cell for group in groups for cell in group["markers"]}
    global_obstacles: list[dict] = []
    used_obstacle_cells: set[tuple[int, int]] = set()
    thick_budget = require_thick

    order = sorted(range(group_count), key=lambda idx: (-obstacle_counts[idx], -len(groups[idx]["ray"])))
    for idx in order:
        group = groups[idx]
        other_occupied = set().union(*(groups[j]["occupied"] if "occupied" in groups[j] else set(groups[j]["markers"]) | set(groups[j]["ray"]) for j in range(group_count) if j != idx))
        placements = _obstacle_candidates(shape, group, other_occupied, marker_cells)
        picked = _select_group_obstacles(group, placements, obstacle_counts[idx], used_obstacle_cells, thick_budget)
        if picked is None:
            return None
        for placement in picked:
            global_obstacles.append(placement)
            used_obstacle_cells |= placement["cells"]
        if any(placement["thick"] for placement in picked):
            thick_budget = False

    if require_thick and thick_budget:
        return None

    inp = _render_input(shape, groups, [placement["top_left"] for placement in global_obstacles])
    expected_obstacle_count = sum(obstacle_counts)
    if not _validate_semantics(inp, group_count, expected_obstacle_count, require_two_collision, require_thick):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-state extrapolation input")
