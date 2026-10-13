import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MISSING_TO_REL = {
    (-1, -1): (0, 0),
    (-1, 1): (0, 1),
    (1, -1): (1, 0),
    (1, 1): (1, 1),
}
REL_TO_MISSING = {rel: vec for vec, rel in MISSING_TO_REL.items()}

GRID_MIN = 15
GRID_MAX = 18
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _piece_cells(anchor: tuple[int, int], missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = map(int, anchor)
    miss = MISSING_TO_REL[missing]
    cells = []
    for rr in (0, 1):
        for cc in (0, 1):
            if (rr, cc) != miss:
                cells.append((r0 + rr, c0 + cc))
    return tuple(cells)


def _shift_cells(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r + dr), int(c + dc)) for r, c in cells)


def _all_in_bounds(shape: tuple[int, int], cells: tuple[tuple[int, int], ...]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _move_from_color_missing(color: int, missing: tuple[int, int]) -> tuple[int, int]:
    if int(color) == GREY:
        return missing
    return (-missing[0], -missing[1])


def _parse_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    for color in (RED, GREY):
        for coords in _components_mask(grid == color):
            assert int(coords.shape[0]) == 3, "expected triomino components"
            r0, c0, r1, c1 = _bbox_coords(coords)
            assert (r1 - r0, c1 - c0) == (1, 1), "expected 2x2 L triomino bbox"
            occ = {(int(r - r0), int(c - c0)) for r, c in coords}
            missing_rel = next(rel for rel in MISSING_TO_REL.values() if rel not in occ)
            missing = REL_TO_MISSING[missing_rel]
            move = _move_from_color_missing(color, missing)
            cells = tuple(sorted((int(r), int(c)) for r, c in coords))
            pieces.append(
                {
                    "color": int(color),
                    "anchor": (int(r0), int(c0)),
                    "missing": missing,
                    "move": move,
                    "cells": cells,
                }
            )
    return pieces


def _next_stop_reason(
    next_cells: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    obstacles: set[tuple[int, int]],
    stopped: set[tuple[int, int]],
) -> str | None:
    h, w = shape
    for r, c in next_cells:
        if not (0 <= r < h and 0 <= c < w):
            return "wall"
    if any((r, c) in obstacles for r, c in next_cells):
        return "obstacle"
    if any((r, c) in stopped for r, c in next_cells):
        return "piece"
    return None


def _free_steps(
    cells: tuple[tuple[int, int], ...],
    move: tuple[int, int],
    shape: tuple[int, int],
    obstacles: set[tuple[int, int]],
) -> int:
    cur = cells
    steps = 0
    while True:
        nxt = _shift_cells(cur, move[0], move[1])
        if _next_stop_reason(nxt, shape, obstacles, set()) is not None:
            return steps
        cur = nxt
        steps += 1


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pieces = _parse_pieces(g)
    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(g == TEAL)}

    for piece in pieces:
        piece["static_steps"] = _free_steps(piece["cells"], piece["move"], (h, w), obstacles)

    pieces.sort(key=lambda p: (int(p["static_steps"]), int(p["anchor"][0]), int(p["anchor"][1])))

    out = g.copy()
    out[np.isin(out, (RED, GREY))] = BG
    stopped: set[tuple[int, int]] = set()
    trace = []

    for piece in pieces:
        cells = piece["cells"]
        travel = 0
        while True:
            nxt = _shift_cells(cells, piece["move"][0], piece["move"][1])
            reason = _next_stop_reason(nxt, (h, w), obstacles, stopped)
            if reason is not None:
                break
            cells = nxt
            travel += 1

        out_color = YELLOW if int(piece["color"]) == GREY else GREEN
        for r, c in cells:
            out[r, c] = out_color
        stopped.update(cells)
        trace.append(
            {
                "color": int(piece["color"]),
                "output_color": int(out_color),
                "anchor": tuple(map(int, piece["anchor"])),
                "missing": tuple(map(int, piece["missing"])),
                "move": tuple(map(int, piece["move"])),
                "input_cells": set(piece["cells"]),
                "final_cells": set(cells),
                "travel": int(travel),
                "static_steps": int(piece["static_steps"]),
                "stop_reason": str(reason),
            }
        )

    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each L triomino diagonally in the direction implied by its missing corner and recolor it after stopping.

    Input structure:
    - The background is black, the fixed blockers are teal, and every moving object is a 4-connected 3-cell L triomino.
    - Each triomino occupies three cells of a 2x2 box, so one corner of that box is visibly missing.
    - Moving triomino colors are grey and red; teal blocker pixels or tiny teal clusters remain fixed.

    Rule:
    - For each triomino, read the missing 2x2 corner as a diagonal direction.
    - Grey triominoes slide one step at a time along that missing-corner diagonal, while red triominoes slide one step at a time along the opposite diagonal.
    - First sort the triominoes by how many free steps they have before the next step would hit a teal blocker or the grid boundary, breaking ties by row-major source position.
    - Then process them in that order: a triomino keeps translating diagonally until its next step would leave the grid, hit teal, or overlap a previously stopped triomino.
    - Remove all source triominoes, draw each stopped grey triomino in yellow, draw each stopped red triomino in green, and leave teal blockers unchanged.

    Key insight:
    - The geometry of the L shape determines a diagonal, but the color decides whether to move along that diagonal or against it; the final layout only emerges after respecting the stop-order and piece-on-piece collisions.
    """
    return _solve_with_trace(grid)[0]


def _chebyshev_too_close(a: set[tuple[int, int]], b: set[tuple[int, int]], gap: int = 3) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= int(gap):
                return True
    return False


def _boundary_anchor(shape: tuple[int, int], move: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if random.random() < 0.5:
        r = h - 2 if move[0] == 1 else 0
        c = random.randint(0, w - 2)
    else:
        r = random.randint(0, h - 2)
        c = w - 2 if move[1] == 1 else 0
    return int(r), int(c)


def _build_plan_from_final(
    shape: tuple[int, int],
    color: int,
    missing: tuple[int, int],
    final_anchor: tuple[int, int],
    travel: int,
) -> dict | None:
    move = _move_from_color_missing(color, missing)
    input_anchor = (
        int(final_anchor[0] - travel * move[0]),
        int(final_anchor[1] - travel * move[1]),
    )
    path_steps = []
    cells = _piece_cells(input_anchor, missing)
    if not _all_in_bounds(shape, cells):
        return None
    path_steps.append(set(cells))
    for _ in range(int(travel)):
        cells = _shift_cells(cells, move[0], move[1])
        if not _all_in_bounds(shape, cells):
            return None
        path_steps.append(set(cells))
    if tuple(sorted(cells)) != tuple(sorted(_piece_cells(final_anchor, missing))):
        return None
    return {
        "color": int(color),
        "missing": tuple(map(int, missing)),
        "move": tuple(map(int, move)),
        "input_anchor": tuple(map(int, input_anchor)),
        "final_anchor": tuple(map(int, final_anchor)),
        "travel": int(travel),
        "input_cells": set(path_steps[0]),
        "final_cells": set(path_steps[-1]),
        "path_steps": path_steps,
    }


def _plan_is_clear(plan: dict, plans: list[dict], obstacles: set[tuple[int, int]]) -> bool:
    if plan is None:
        return False
    for step in plan["path_steps"]:
        if step & obstacles:
            return False
    for other in plans:
        if _chebyshev_too_close(plan["input_cells"], other["input_cells"], gap=3):
            return False
        if plan["final_cells"] & other["final_cells"]:
            return False
        for step in plan["path_steps"]:
            if step & other["final_cells"]:
                return False
    return True


def _search_chain_pair(shape: tuple[int, int], missing: tuple[int, int]) -> list[dict] | None:
    move = missing
    for _ in range(160):
        front_final = _boundary_anchor(shape, move)
        front_travel = random.randint(1, 3)
        back_final = (int(front_final[0] - move[0]), int(front_final[1] - move[1]))
        back_travel = front_travel + random.randint(3, 5)

        front = _build_plan_from_final(shape, GREY, missing, front_final, front_travel)
        back = _build_plan_from_final(shape, GREY, missing, back_final, back_travel)
        if not _plan_is_clear(front, [], set()):
            continue
        if not _plan_is_clear(back, [front], set()):
            continue

        next_from_front = _shift_cells(tuple(sorted(front["final_cells"])), move[0], move[1])
        if _all_in_bounds(shape, next_from_front):
            continue

        next_from_back = set(_shift_cells(tuple(sorted(back["final_cells"])), move[0], move[1]))
        if not (next_from_back & front["final_cells"]):
            continue
        return [front, back]
    return None


def _search_boundary_piece(
    shape: tuple[int, int],
    plans: list[dict],
    obstacles: set[tuple[int, int]],
    color: int,
    missing: tuple[int, int],
    min_travel: int = 1,
    max_travel: int = 5,
) -> dict | None:
    move = _move_from_color_missing(color, missing)
    for _ in range(140):
        final_anchor = _boundary_anchor(shape, move)
        travel = random.randint(min_travel, max_travel)
        plan = _build_plan_from_final(shape, color, missing, final_anchor, travel)
        if not _plan_is_clear(plan, plans, obstacles):
            continue
        next_cells = _shift_cells(tuple(sorted(plan["final_cells"])), move[0], move[1])
        if _all_in_bounds(shape, next_cells):
            continue
        return plan
    return None


def _search_obstacle_piece(
    shape: tuple[int, int],
    plans: list[dict],
    obstacles: set[tuple[int, int]],
    move: tuple[int, int],
    color: int,
) -> tuple[dict, set[tuple[int, int]]] | None:
    h, w = shape
    missing = move if int(color) == GREY else (-move[0], -move[1])
    final_rows = list(range(max(1, h // 2 - 1), h - 1))
    final_cols = list(range(0, w - 1))
    for _ in range(220):
        final_anchor = (random.choice(final_rows), random.choice(final_cols))
        travel = random.randint(1, 4)
        plan = _build_plan_from_final(shape, color, missing, final_anchor, travel)
        if plan is None:
            continue
        next_cells = set(_shift_cells(tuple(sorted(plan["final_cells"])), move[0], move[1]))
        if not _all_in_bounds(shape, tuple(next_cells)):
            continue
        if not any(r >= h // 2 for r, _ in next_cells):
            continue

        blocker = random.choice(sorted(next_cells))
        local_obstacles = set(obstacles)
        local_obstacles.add(blocker)
        if not _plan_is_clear(plan, plans, local_obstacles):
            continue

        cluster = {blocker}
        random_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(random_dirs)
        for dr, dc in random_dirs:
            if len(cluster) >= random.randint(1, 3):
                break
            nr, nc = int(blocker[0] + dr), int(blocker[1] + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nr < h // 2:
                continue
            if (nr, nc) in cluster:
                continue
            if any((nr, nc) in step for step in plan["path_steps"]):
                continue
            if any((nr, nc) in other["final_cells"] for other in plans):
                continue
            cluster.add((nr, nc))

        if not _plan_is_clear(plan, plans, obstacles | cluster):
            continue
        return plan, cluster
    return None


def _add_extra_cluster(
    shape: tuple[int, int],
    plans: list[dict],
    obstacles: set[tuple[int, int]],
) -> None:
    h, w = shape
    reserved = set(obstacles)
    for plan in plans:
        for step in plan["path_steps"]:
            reserved.update(step)
    for _ in range(80):
        root = (random.randint(h // 2, h - 1), random.randint(0, w - 1))
        if root in reserved:
            continue
        cluster = {root}
        candidates = [root]
        while candidates and len(cluster) < random.randint(1, 3):
            r, c = candidates.pop(0)
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if nr < h // 2:
                    continue
                if (nr, nc) in reserved or (nr, nc) in cluster:
                    continue
                cluster.add((nr, nc))
                candidates.append((nr, nc))
                if len(cluster) >= 3:
                    break
        if cluster.isdisjoint(reserved):
            obstacles.update(cluster)
            return


def _build_grid(shape: tuple[int, int], plans: list[dict], obstacles: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in obstacles:
        grid[r, c] = TEAL
    for plan in plans:
        for r, c in plan["input_cells"]:
            grid[r, c] = int(plan["color"])
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        pieces = _parse_pieces(grid)
        out, trace = _solve_with_trace(grid)
    except Exception:
        return False

    if not (4 <= len(pieces) <= 7):
        return False
    if {int(p["color"]) for p in pieces} != {RED, GREY}:
        return False
    if len(np.argwhere(grid == TEAL)) < 2:
        return False
    if not bool((np.argwhere(grid == TEAL)[:, 0] >= grid.shape[0] // 2).all()):
        return False
    if not any(t["travel"] > 0 for t in trace):
        return False
    if not any(t["stop_reason"] == "piece" for t in trace):
        return False
    if not any(t["stop_reason"] == "obstacle" for t in trace):
        return False
    if not any(t["stop_reason"] == "wall" for t in trace):
        return False
    if not any(t["stop_reason"] == "piece" and t["travel"] < t["static_steps"] for t in trace):
        return False
    if len({t["move"] for t in trace}) < 3:
        return False
    if not np.any(out == YELLOW) or not np.any(out == GREEN):
        return False
    if np.array_equal(grid, out):
        return False

    grey_missing = {t["missing"] for t in trace if int(t["color"]) == GREY}
    red_missing = {t["missing"] for t in trace if int(t["color"]) == RED}
    if not (grey_missing & red_missing):
        return False

    for i in range(len(trace)):
        for j in range(i + 1, len(trace)):
            if _chebyshev_too_close(trace[i]["input_cells"], trace[j]["input_cells"], gap=3):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    plans: list[dict] = []
    obstacles: set[tuple[int, int]] = set()

    core_missing = random.choice(DIAGS)
    chain = _search_chain_pair(shape, core_missing)
    if chain is None:
        return None
    plans.extend(chain)

    opposite_color_piece = _search_boundary_piece(
        shape,
        plans,
        obstacles,
        RED,
        core_missing,
        min_travel=1,
        max_travel=5,
    )
    if opposite_color_piece is None:
        return None
    plans.append(opposite_color_piece)

    obstacle_move = random.choice([(1, 1), (1, -1)])
    obstacle_color = random.choice([GREY, RED])
    obstacle_result = _search_obstacle_piece(shape, plans, obstacles, obstacle_move, obstacle_color)
    if obstacle_result is None:
        return None
    obstacle_piece, cluster = obstacle_result
    plans.append(obstacle_piece)
    obstacles.update(cluster)

    remaining_dirs = [d for d in DIAGS if d not in {core_missing, (-core_missing[0], -core_missing[1]), obstacle_move}]
    if remaining_dirs and random.random() < 0.7:
        extra_dir = random.choice(remaining_dirs)
        extra_color = random.choice([GREY, RED])
        extra_missing = extra_dir if int(extra_color) == GREY else (-extra_dir[0], -extra_dir[1])
        extra_piece = _search_boundary_piece(
            shape,
            plans,
            obstacles,
            extra_color,
            extra_missing,
            min_travel=1,
            max_travel=4,
        )
        if extra_piece is not None:
            plans.append(extra_piece)

    if len(plans) < 7 and random.random() < 0.3:
        bonus_move = random.choice(DIAGS)
        bonus_color = random.choice([GREY, RED])
        if bonus_move[0] == 1 and random.random() < 0.6:
            bonus_result = _search_obstacle_piece(shape, plans, obstacles, bonus_move, bonus_color)
            if bonus_result is not None:
                bonus_piece, bonus_cluster = bonus_result
                plans.append(bonus_piece)
                obstacles.update(bonus_cluster)
        else:
            bonus_missing = bonus_move if int(bonus_color) == GREY else (-bonus_move[0], -bonus_move[1])
            bonus_piece = _search_boundary_piece(
                shape,
                plans,
                obstacles,
                bonus_color,
                bonus_missing,
                min_travel=1,
                max_travel=4,
            )
            if bonus_piece is not None:
                plans.append(bonus_piece)

    extra_clusters = 1 + random.randint(0, 1)
    for _ in range(extra_clusters):
        _add_extra_cluster(shape, plans, obstacles)

    grid = _build_grid(shape, plans, obstacles)
    if _validate_generated(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
