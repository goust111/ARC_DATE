from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
WALL_COLORS = (4, 5, 6, 7, 8, 9)
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
EDGE_NAMES = ("top", "bottom", "left", "right")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _arrow_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0, c1 - c0) != (1, 1):
        return None

    pts = set(coords)
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    missing = [p for p in corners if p not in pts]
    if len(missing) != 1:
        return None

    mr, mc = missing[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    degrees = {}
    for r, c in coords:
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts:
                deg += 1
        degrees[(r, c)] = deg
    start = max(degrees, key=degrees.get)
    return start, (dr, dc)


def _parse_grid(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict] = []
    walls: dict[tuple[int, int], int] = {}
    arrow_cells: set[tuple[int, int]] = set()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BLACK):
        for comp in _connected_components(g == color):
            arrow_info = _arrow_signature(comp)
            if arrow_info is not None:
                start, direction = arrow_info
                arrows.append(
                    {
                        "color": int(color),
                        "cells": tuple(sorted(comp)),
                        "start": start,
                        "dir": direction,
                    }
                )
                arrow_cells.update(comp)
            elif len(comp) == 1:
                walls[comp[0]] = int(color)
            else:
                raise AssertionError("unexpected foreground object")

    assert len(arrows) == 2, "expected exactly two arrows"
    assert len(walls) <= 3, "expected at most three wall pixels"
    assert len({arrow["color"] for arrow in arrows}) == 2, "arrow colors must differ"
    assert len({arrow["dir"] for arrow in arrows}) == 2, "arrow directions must differ"
    return arrows, walls, arrow_cells


def _ray_state(arrow: dict) -> dict:
    return {
        "pos": tuple(arrow["start"]),
        "dir": tuple(arrow["dir"]),
        "color": int(arrow["color"]),
        "trail": set(),
        "moves": 0,
        "wall_bounces": 0,
        "boundary_bounces": 0,
        "touched_walls": set(),
        "seen": set(),
    }


def _advance_one(
    state: dict,
    wall_map: dict[tuple[int, int], int],
    arrow_cells: set[tuple[int, int]],
    visible_grid: np.ndarray,
) -> dict:
    h, w = visible_grid.shape
    r, c = state["pos"]
    dr, dc = state["dir"]
    color = int(state["color"])
    signature = (r, c, dr, dc, color)
    if signature in state["seen"]:
        return {"invalid": True, "reason": "loop"}

    next_state = {
        "pos": state["pos"],
        "dir": state["dir"],
        "color": state["color"],
        "trail": set(state["trail"]),
        "moves": int(state["moves"]),
        "wall_bounces": int(state["wall_bounces"]),
        "boundary_bounces": int(state["boundary_bounces"]),
        "touched_walls": set(state["touched_walls"]),
        "seen": set(state["seen"]),
    }
    next_state["seen"].add(signature)

    nr = r + dr
    nc = c + dc
    row_out = not (0 <= nr < h)
    col_out = not (0 <= nc < w)
    if row_out or col_out:
        if row_out:
            dr = -dr
            next_state["boundary_bounces"] += 1
        if col_out:
            dc = -dc
            next_state["boundary_bounces"] += 1
        next_state["dir"] = (dr, dc)
        return {
            "invalid": False,
            "state": next_state,
            "arrival": None,
            "paint": None,
            "wall_updates": [],
        }

    if (nr, nc) in arrow_cells:
        return {"invalid": True, "reason": "ray hit arrow"}

    if (nr, nc) in wall_map:
        wall_color = int(wall_map[(nr, nc)])
        next_state["dir"] = (-dr, -dc)
        next_state["color"] = wall_color
        next_state["wall_bounces"] += 1
        next_state["touched_walls"].add((nr, nc))
        return {
            "invalid": False,
            "state": next_state,
            "arrival": None,
            "paint": None,
            "wall_updates": [((nr, nc), wall_color)],
        }

    next_state["pos"] = (nr, nc)
    next_state["moves"] += 1
    paint = None
    if int(visible_grid[nr, nc]) == BLACK:
        next_state["trail"].add((nr, nc))
        paint = ((nr, nc), color)

    row_block = None
    col_block = None
    if not (0 <= nr + dr < h):
        row_block = ("boundary", None, None)
    elif (nr + dr, nc) in wall_map:
        wall_color = int(wall_map[(nr + dr, nc)])
        row_block = ("wall", (nr + dr, nc), wall_color)

    if not (0 <= nc + dc < w):
        col_block = ("boundary", None, None)
    elif (nr, nc + dc) in wall_map:
        wall_color = int(wall_map[(nr, nc + dc)])
        col_block = ("wall", (nr, nc + dc), wall_color)

    touched_wall_cells = [block[1] for block in (row_block, col_block) if block is not None and block[0] == "wall"]
    touched_wall_colors = {int(block[2]) for block in (row_block, col_block) if block is not None and block[0] == "wall"}
    if len(touched_wall_colors) > 1:
        return {"invalid": True, "reason": "ambiguous multi-wall color bounce"}

    wall_updates: list[tuple[tuple[int, int], int]] = []
    if touched_wall_colors:
        new_color = next(iter(touched_wall_colors))
        next_state["color"] = new_color
        next_state["wall_bounces"] += len(touched_wall_cells)
        for wall_cell in touched_wall_cells:
            next_state["touched_walls"].add(wall_cell)
            wall_updates.append((wall_cell, new_color))

    if row_block is not None:
        dr = -dr
        if row_block[0] == "boundary":
            next_state["boundary_bounces"] += 1
    if col_block is not None:
        dc = -dc
        if col_block[0] == "boundary":
            next_state["boundary_bounces"] += 1
    next_state["dir"] = (dr, dc)

    return {
        "invalid": False,
        "state": next_state,
        "arrival": (nr, nc),
        "paint": paint,
        "wall_updates": wall_updates,
    }


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    arrows, wall_map, arrow_cells = _parse_grid(g)
    out = g.copy()
    states = [_ray_state(arrow) for arrow in arrows]
    used_wall_cells: set[tuple[int, int]] = set()
    max_ticks = g.shape[0] * g.shape[1] * 12

    for _tick in range(max_ticks):
        proposals = []
        for state in states:
            prop = _advance_one(state, wall_map, arrow_cells, out)
            if prop["invalid"]:
                return {"ok": False, "reason": prop["reason"]}
            proposals.append(prop)

        prior_trails = [set(state["trail"]) for state in states]
        arrivals = [prop["arrival"] for prop in proposals]
        collision_points: list[tuple[int, int]] = []
        collision_kind = None

        if arrivals[0] is not None and arrivals[1] is not None and arrivals[0] == arrivals[1]:
            collision_points.append(arrivals[0])
            collision_kind = "same_step_same_cell"
        if arrivals[0] is not None and arrivals[0] in prior_trails[1]:
            collision_points.append(arrivals[0])
        if arrivals[1] is not None and arrivals[1] in prior_trails[0]:
            collision_points.append(arrivals[1])
            if arrivals[0] is not None and arrivals[0] in prior_trails[1] and arrivals[0] != arrivals[1]:
                collision_kind = "simultaneous_distinct"

        for prop in proposals:
            for wall_cell, wall_color in prop["wall_updates"]:
                wall_map[wall_cell] = int(wall_color)
                out[wall_cell] = int(wall_color)
                used_wall_cells.add(wall_cell)

        for prop in proposals:
            if prop["paint"] is not None:
                (pr, pc), paint_color = prop["paint"]
                out[pr, pc] = int(paint_color)

        states = [prop["state"] for prop in proposals]

        if collision_points:
            collision = min(collision_points)
            cr, cc = collision
            if not (1 <= cr < g.shape[0] - 1 and 1 <= cc < g.shape[1] - 1):
                return {"ok": False, "reason": "cross would leave grid"}
            for dr, dc in PLUS_OFFSETS:
                out[cr + dr, cc + dc] = GREEN
            return {
                "ok": True,
                "output": out,
                "collision": collision,
                "collision_kind": collision_kind or "trail_hit",
                "states": states,
                "used_wall_cells": used_wall_cells,
            }

    return {"ok": False, "reason": "simulation timeout"}


def _simulate_no_reflection(grid: np.ndarray) -> tuple[int, int] | None:
    g = np.asarray(grid, dtype=int)
    arrows, _wall_map, _arrow_cells = _parse_grid(g)
    states = [
        {
            "pos": tuple(arrow["start"]),
            "dir": tuple(arrow["dir"]),
            "trail": set(),
        }
        for arrow in arrows
    ]
    max_ticks = g.shape[0] * g.shape[1]
    for _tick in range(max_ticks):
        arrivals = []
        for state in states:
            r, c = state["pos"]
            dr, dc = state["dir"]
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                arrivals.append(None)
            else:
                arrivals.append((nr, nc))

        if arrivals[0] is not None and arrivals[1] is not None and arrivals[0] == arrivals[1]:
            return arrivals[0]
        if arrivals[0] is not None and arrivals[0] in states[1]["trail"]:
            return arrivals[0]
        if arrivals[1] is not None and arrivals[1] in states[0]["trail"]:
            return arrivals[1]

        any_move = False
        for idx in range(2):
            if arrivals[idx] is not None:
                states[idx]["pos"] = arrivals[idx]
                states[idx]["trail"].add(arrivals[idx])
                any_move = True
        if not any_move:
            return None
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace two reflecting diagonal rays and place a green cross at their first collision.

    Input structure:
    - The grid has a black background, exactly two monochromatic 3-pixel L arrows, and zero to three isolated wall pixels in other colors.
    - Each arrow occupies three corners of a 2x2 box, so its missing corner uniquely determines one diagonal launch direction.
    - The arrows use different colors and directions, and the first ray collision lies far enough from the border to hold a centered 3x3 cross.

    Rule:
    - For each arrow, launch one diagonal ray from the arrow's right-angle corner in the direction away from the missing corner.
    - Whenever a ray enters a black cell, paint that cell with the ray's current color.
    - A diagonal wall hit flips both axes; a side touch flips only the touched axis; a bounce inherits the touched wall color, while boundary bounces flip the blocked axes without changing color.
    - Advance the two rays independently but step them in lockstep; the first cell reached on the other ray's painted trail is the collision point, and a simultaneous two-point collision uses lexicographic tie-breaking.
    - Stop both rays at that collision and overwrite the centered 3x3 plus pattern there with green.

    Key insight:
    - The green cross is not anchored to any static marker; its center emerges only after both reflected ray paths are traced and their first shared reachable point is resolved.
    """
    result = _simulate(np.asarray(grid, dtype=int))
    assert result["ok"], result.get("reason", "invalid puzzle instance")
    return result["output"]


def _arrow_patch(color: int, missing_corner: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), BLACK, dtype=int)
    for rr in range(2):
        for cc in range(2):
            if (rr, cc) != missing_corner:
                patch[rr, cc] = int(color)
    return patch


def _edge_positions(height: int, width: int, edge: str) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if edge == "top":
        rows = [0, 1]
        cols = list(range(1, width - 2))
    elif edge == "bottom":
        rows = [height - 2, height - 3]
        cols = list(range(1, width - 2))
    elif edge == "left":
        rows = list(range(1, height - 2))
        cols = [0, 1]
    else:
        rows = list(range(1, height - 2))
        cols = [width - 2, width - 3]
    for r0 in rows:
        for c0 in cols:
            if 0 <= r0 <= height - 2 and 0 <= c0 <= width - 2:
                cells.append((r0, c0))
    random.shuffle(cells)
    return cells


def _place_arrows(height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    edges = list(EDGE_NAMES)
    random.shuffle(edges)
    chosen_edges = edges[:2]
    missing_corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
    placements = []
    occupied: set[tuple[int, int]] = set()

    for color, edge in zip((BLUE, RED), chosen_edges):
        candidates = _edge_positions(height, width, edge)
        random.shuffle(missing_corners)
        placed = False
        for missing in missing_corners:
            patch = _arrow_patch(color, missing)
            arrow_cells = [(rr, cc) for rr, cc in np.argwhere(patch == color)]
            for r0, c0 in candidates:
                coords = {(r0 + rr, c0 + cc) for rr, cc in arrow_cells}
                if any(cell in occupied for cell in coords):
                    continue
                if any(abs(r0 - pr0) <= 2 and abs(c0 - pc0) <= 2 for pr0, pc0, _pm in placements):
                    continue
                for rr, cc in coords:
                    grid[rr, cc] = color
                placements.append((r0, c0, missing))
                occupied.update(coords)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    try:
        arrows, _walls, _arrow_cells = _parse_grid(grid)
    except AssertionError:
        return None
    if len({arrow["dir"] for arrow in arrows}) != 2:
        return None
    return grid


def _wall_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    arrows, _walls, arrow_cells = _parse_grid(grid)
    candidates: set[tuple[int, int]] = set()
    h, w = grid.shape

    for arrow in arrows:
        r, c = arrow["start"]
        dr, dc = arrow["dir"]
        for _ in range(max(h, w)):
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                break
            if (nr, nc) not in arrow_cells:
                candidates.add((nr, nc))
            row_side = (nr + dr, nc)
            col_side = (nr, nc + dc)
            if _in_bounds(grid.shape, *row_side) and row_side not in arrow_cells:
                candidates.add(row_side)
            if _in_bounds(grid.shape, *col_side) and col_side not in arrow_cells:
                candidates.add(col_side)
            r, c = nr, nc

    filtered = [
        cell
        for cell in candidates
        if 1 <= cell[0] < h - 1 and 1 <= cell[1] < w - 1 and int(grid[cell]) == BLACK
    ]
    random.shuffle(filtered)
    return filtered


def _with_walls(base: np.ndarray, n_walls: int) -> np.ndarray | None:
    candidates = _wall_candidates(base)
    if len(candidates) < n_walls:
        return None

    grid = base.copy()
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if any(abs(cell[0] - other[0]) <= 1 and abs(cell[1] - other[1]) <= 1 for other in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == n_walls:
            break
    if len(chosen) != n_walls:
        return None

    colors = random.sample(WALL_COLORS, n_walls)
    for cell, color in zip(chosen, colors):
        grid[cell] = int(color)
    return grid


def _scenario_ok(grid: np.ndarray, scenario: str) -> bool:
    result = _simulate(grid)
    if not result["ok"]:
        return False

    arrows, walls, _arrow_cells = _parse_grid(grid)
    states = result["states"]
    collision = result["collision"]
    if collision is None:
        return False
    if not all(state["moves"] >= 2 for state in states):
        return False
    if any(len(state["trail"]) == 0 for state in states):
        return False
    if set(walls) != set(result["used_wall_cells"]):
        return False

    total_wall_bounces = sum(int(state["wall_bounces"]) for state in states)
    total_boundary_bounces = sum(int(state["boundary_bounces"]) for state in states)
    if scenario == "wall" and not (total_wall_bounces >= 1 and total_boundary_bounces == 0):
        return False
    if scenario == "boundary" and not (total_boundary_bounces >= 1 and total_wall_bounces == 0 and len(walls) == 0):
        return False
    if scenario == "mixed" and not (total_wall_bounces >= 1 and total_boundary_bounces >= 1):
        return False

    direct_collision = _simulate_no_reflection(grid)
    if direct_collision == collision:
        return False

    for wall_cell in walls:
        test_grid = grid.copy()
        test_grid[wall_cell] = BLACK
        changed = _simulate(test_grid)
        if changed["ok"] and changed["collision"] == collision and np.array_equal(changed["output"], result["output"]):
            return False

    wallless = grid.copy()
    for wall_cell in walls:
        wallless[wall_cell] = BLACK
    wallless_result = _simulate(wallless)
    if walls and wallless_result["ok"] and wallless_result["collision"] == collision:
        return False

    for arrow in arrows:
        sr, sc = arrow["start"]
        if abs(sr - collision[0]) + abs(sc - collision[1]) <= 1:
            return False
    return True


def _gen_attempt(scenario: str) -> np.ndarray | None:
    height = random.randint(13, 18)
    width = random.randint(13, 18)
    base = _place_arrows(height, width)
    if base is None:
        return None

    if scenario == "boundary":
        grid = base
    elif scenario == "wall":
        grid = _with_walls(base, 1)
    else:
        grid = _with_walls(base, random.choice((1, 2)))
    if grid is None:
        return None

    if _scenario_ok(grid, scenario):
        return grid
    return None


def generate_input() -> np.ndarray:
    scenarios = ("boundary", "mixed", "wall")
    weights = (0.50, 0.40, 0.10)
    budgets = {"boundary": 80, "mixed": 320, "wall": 520}

    primary = random.choices(scenarios, weights=weights, k=1)[0]
    ordered = [primary] + [name for name in scenarios if name != primary]
    for scenario in ordered:
        for _ in range(budgets[scenario]):
            grid = _gen_attempt(scenario)
            if grid is not None:
                return grid
    raise RuntimeError("failed to construct a valid puzzle instance")
