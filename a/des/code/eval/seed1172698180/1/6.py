from __future__ import annotations

import random

import numpy as np


BG = 0
PORTAL_COLORS = [1, 2, 4, 6, 7, 8]
GREEN = 3
WALL = 5
MAROON = 9

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_SEG_STEPS = 4
MAX_SEG_STEPS = 8
GRID_MIN = 16
GRID_MAX = 23
GEN_ATTEMPTS = 300


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _step(cell: tuple[int, int], direction: tuple[int, int], amount: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * amount, cell[1] + direction[1] * amount


def _opp(direction: tuple[int, int]) -> tuple[int, int]:
    return -direction[0], -direction[1]


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _distance_to_edge(cell: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = cell
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _wall_bar(
    wall_seed: tuple[int, int],
    wall_dir: tuple[int, int],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells = {wall_seed}
    if wall_dir[0] == 0:
        perps = [(-1, 0), (1, 0)]
    else:
        perps = [(0, -1), (0, 1)]

    local_forbidden = set(forbidden)
    local_forbidden.add(wall_seed)
    for pr, pc in perps:
        rr, cc = wall_seed[0] + pr, wall_seed[1] + pc
        if _in_bounds(h, w, rr, cc) and (rr, cc) not in local_forbidden:
            cells.add((rr, cc))
            local_forbidden.add((rr, cc))
    return cells


def _arrow_cells(head: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    back = _opp(direction)
    return [head] + [_step(head, back, k) for k in range(1, 4)]


def _portal_maps(grid: np.ndarray) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    exit_dirs: dict[tuple[int, int], tuple[int, int]] = {}
    for color in sorted(set(map(int, np.unique(g))) - {BG, GREEN, WALL, MAROON}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each portal color must appear exactly twice"

        wall_dirs: list[tuple[int, int]] = []
        for cell in coords:
            adjacent = [
                (dr, dc)
                for dr, dc in DIRS
                if _in_bounds(h, w, cell[0] + dr, cell[1] + dc)
                and int(g[cell[0] + dr, cell[1] + dc]) == WALL
            ]
            assert len(adjacent) == 1, "each portal must touch exactly one wall"
            wall_dirs.append(adjacent[0])

        a, b = coords
        wa, wb = wall_dirs
        pairs[a] = b
        pairs[b] = a
        exit_dirs[a] = _opp(wa)
        exit_dirs[b] = _opp(wb)

    return pairs, exit_dirs


def _detect_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    arrows: list[dict[str, object]] = []
    used_green: set[tuple[int, int]] = set()
    heads = [tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)]
    assert heads, "expected at least one arrow"

    for head in heads:
        hr, hc = head
        candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for br, bc in DIRS:
            nr, nc = hr + br, hc + bc
            if not _in_bounds(h, w, nr, nc) or int(g[nr, nc]) != GREEN:
                continue

            direction = (-br, -bc)
            body = [_step(head, (br, bc), k) for k in range(1, 4)]
            if not all(_in_bounds(h, w, r, c) and int(g[r, c]) == GREEN for r, c in body):
                continue

            tail = _step(head, (br, bc), 4)
            if _in_bounds(h, w, tail[0], tail[1]) and int(g[tail]) == GREEN:
                continue
            candidates.append((direction, body))

        assert len(candidates) == 1, "each maroon head must define one arrow"
        direction, body = candidates[0]
        for cell in body:
            assert cell not in used_green, "green arrow bodies may not overlap"
            used_green.add(cell)
        arrows.append({"head": head, "dir": direction, "body": body})

    green_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)}
    assert green_cells == used_green, "all green pixels must belong to arrow bodies"
    return arrows


def _trace_arrow(
    grid: np.ndarray,
    head: tuple[int, int],
    direction: tuple[int, int],
    pairs: dict[tuple[int, int], tuple[int, int]],
    exit_dirs: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    r, c = head
    dr, dc = direction
    trail = [(r, c)]
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "teleport cycle detected"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(h, w, nr, nc):
            break
        if int(g[nr, nc]) == WALL:
            break
        if (nr, nc) in pairs:
            r, c = pairs[(nr, nc)]
            dr, dc = exit_dirs[(r, c)]
            continue

        r, c = nr, nc
        trail.append((r, c))

    return trail


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recognize same-color portal pairs and move each maroon-headed arrow as a point
    that starts at the head, painting a green trail, teleporting through paired
    portals, and stopping at walls or edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs, exit_dirs = _portal_maps(g)
    arrows = _detect_arrows(g)

    out = g.copy()
    out[g == GREEN] = BG
    out[g == MAROON] = BG

    for arrow in arrows:
        head = tuple(arrow["head"])
        direction = tuple(arrow["dir"])
        for r, c in _trace_arrow(g, head, direction, pairs, exit_dirs):
            if (r, c) not in pairs:
                out[r, c] = GREEN

    return out


def _state_ok_for_portals(
    portal_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for cell in portal_cells:
        count = 0
        for nr, nc in _neighbors(cell):
            if _in_bounds(h, w, nr, nc) and (nr, nc) in wall_cells:
                count += 1
        if count != 1:
            return False
    return True


def _make_segment_candidate(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    portal_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    start_kind: str,
    end_kind: str,
) -> dict[str, object] | None:
    end = _step(start, direction, steps)
    if not _in_bounds(h, w, end[0], end[1]):
        return None

    if end_kind == "edge":
        if _distance_to_edge(start, direction, h, w) != steps:
            return None
    elif not _in_bounds(h, w, end[0] + direction[0], end[1] + direction[1]):
        return None

    new_arrow_cells: set[tuple[int, int]] = set()
    head: tuple[int, int] | None = None
    if start_kind == "arrow":
        head = start
        new_arrow_cells = set(_arrow_cells(start, direction))
        if any(not _in_bounds(h, w, r, c) for r, c in new_arrow_cells):
            return None
    else:
        wall_seed = _step(start, _opp(direction))
        if not _in_bounds(h, w, wall_seed[0], wall_seed[1]):
            return None

    path_cells = {_step(start, direction, k) for k in range(steps + 1)}

    new_portal_cells: set[tuple[int, int]] = set()
    start_portal: tuple[int, int] | None = None
    end_portal: tuple[int, int] | None = None
    start_wall_dir: tuple[int, int] | None = None
    end_wall_dir: tuple[int, int] | None = None

    if start_kind == "portal":
        start_portal = start
        start_wall_dir = _opp(direction)
        new_portal_cells.add(start)
    if end_kind == "portal":
        end_portal = end
        end_wall_dir = direction
        new_portal_cells.add(end)

    new_walls: set[tuple[int, int]] = set()
    forbidden = blocked | path_cells | new_arrow_cells | new_portal_cells
    if start_kind == "portal":
        start_seed = _step(start, _opp(direction))
        new_walls |= _wall_bar(start_seed, _opp(direction), h, w, forbidden | new_walls)
    if end_kind in {"portal", "wall"}:
        end_seed = _step(end, direction)
        new_walls |= _wall_bar(end_seed, direction, h, w, forbidden | new_walls)

    candidate_nonblack = new_arrow_cells | new_portal_cells | new_walls
    candidate_all = path_cells | candidate_nonblack
    if any(cell in blocked for cell in candidate_all):
        return None

    for cell in new_arrow_cells:
        for nbr in _neighbors(cell):
            if nbr in arrow_cells or nbr in portal_cells:
                return None
    for cell in new_portal_cells:
        for nbr in _neighbors(cell):
            if nbr in arrow_cells or nbr in portal_cells:
                return None

    wall_union = wall_cells | new_walls
    portal_union = portal_cells | new_portal_cells
    if not _state_ok_for_portals(portal_union, wall_union, h, w):
        return None

    return {
        "blocked": candidate_all,
        "arrow_cells": new_arrow_cells,
        "portal_cells": new_portal_cells,
        "wall_cells": new_walls,
        "head": head,
        "direction": direction,
        "start_portal": start_portal,
        "start_wall_dir": start_wall_dir,
        "end_portal": end_portal,
        "end_wall_dir": end_wall_dir,
    }


def _find_segment_candidate(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    portal_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    direction: tuple[int, int],
    start_kind: str,
    end_kind: str,
) -> dict[str, object] | None:
    starts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(starts)

    for start in starts:
        if end_kind == "edge":
            steps = _distance_to_edge(start, direction, h, w)
            step_choices = [steps]
        else:
            step_choices = list(range(MIN_SEG_STEPS, MAX_SEG_STEPS + 1))
            random.shuffle(step_choices)

        for steps in step_choices:
            if not (MIN_SEG_STEPS <= steps <= MAX_SEG_STEPS):
                continue
            candidate = _make_segment_candidate(
                h,
                w,
                blocked,
                arrow_cells,
                portal_cells,
                wall_cells,
                start,
                direction,
                steps,
                start_kind,
                end_kind,
            )
            if candidate is not None:
                return candidate
    return None


def _commit_segment(
    state: dict[str, object],
    candidate: dict[str, object],
    arrows: list[dict[str, object]],
) -> None:
    blocked: set[tuple[int, int]] = state["blocked"]  # type: ignore[assignment]
    arrow_cells: set[tuple[int, int]] = state["arrow_cells"]  # type: ignore[assignment]
    portal_cells: set[tuple[int, int]] = state["portal_cells"]  # type: ignore[assignment]
    wall_cells: set[tuple[int, int]] = state["wall_cells"]  # type: ignore[assignment]

    blocked.update(candidate["blocked"])  # type: ignore[arg-type]
    arrow_cells.update(candidate["arrow_cells"])  # type: ignore[arg-type]
    portal_cells.update(candidate["portal_cells"])  # type: ignore[arg-type]
    wall_cells.update(candidate["wall_cells"])  # type: ignore[arg-type]

    if candidate["head"] is not None:
        arrows.append({"head": candidate["head"], "dir": candidate["direction"]})


def _sample_directions(count: int) -> list[tuple[int, int]]:
    dirs = [random.choice(DIRS)]
    for _ in range(1, count):
        choices = [d for d in DIRS if d != dirs[-1]]
        dirs.append(random.choice(choices))
    return dirs


def _place_isolated_portal_endpoint(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    portal_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    starts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(starts)
    dirs = list(DIRS)

    for cell in starts:
        random.shuffle(dirs)
        for wall_dir in dirs:
            wall_seed = _step(cell, wall_dir)
            if not _in_bounds(h, w, wall_seed[0], wall_seed[1]):
                continue

            new_portal = {cell}
            new_walls = _wall_bar(wall_seed, wall_dir, h, w, blocked | new_portal)
            candidate_all = new_portal | new_walls
            if any(pos in blocked for pos in candidate_all):
                continue

            for nbr in _neighbors(cell):
                if nbr in arrow_cells or nbr in portal_cells:
                    break
            else:
                if not _state_ok_for_portals(portal_cells | new_portal, wall_cells | new_walls, h, w):
                    continue
                return cell, wall_dir, candidate_all, new_walls
    return None


def _paint_input(
    h: int,
    w: int,
    wall_cells: set[tuple[int, int]],
    portal_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    arrows: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in wall_cells:
        grid[r, c] = WALL

    colors = random.sample(PORTAL_COLORS, len(portal_pairs))
    for color, (a, b) in zip(colors, portal_pairs):
        grid[a] = int(color)
        grid[b] = int(color)

    for arrow in arrows:
        head = tuple(arrow["head"])
        direction = tuple(arrow["dir"])
        grid[head] = MAROON
        for r, c in _arrow_cells(head, direction)[1:]:
            grid[r, c] = GREEN

    return grid


def _scenario_plan() -> tuple[list[int], int]:
    scenario = random.choice(["direct", "one_tp", "two_tp", "dual"])
    if scenario == "direct":
        return [0], random.randint(1, 2)
    if scenario == "one_tp":
        return [1], random.randint(0, 1)
    if scenario == "two_tp":
        return [2], random.randint(0, 1)
    return [0, random.choice([1, 2])], random.randint(0, 1)


def _attempt_generate() -> np.ndarray | None:
    teleports_per_arrow, extra_pairs = _scenario_plan()
    h = random.randint(GRID_MIN + (2 if len(teleports_per_arrow) > 1 else 0), GRID_MAX)
    w = random.randint(GRID_MIN + (2 if max(teleports_per_arrow, default=0) > 1 else 0), GRID_MAX)

    total_pairs = sum(teleports_per_arrow) + extra_pairs
    if total_pairs > len(PORTAL_COLORS):
        return None

    state: dict[str, object] = {
        "blocked": set(),
        "arrow_cells": set(),
        "portal_cells": set(),
        "wall_cells": set(),
    }
    blocked: set[tuple[int, int]] = state["blocked"]  # type: ignore[assignment]
    arrow_cells: set[tuple[int, int]] = state["arrow_cells"]  # type: ignore[assignment]
    portal_cells: set[tuple[int, int]] = state["portal_cells"]  # type: ignore[assignment]
    wall_cells: set[tuple[int, int]] = state["wall_cells"]  # type: ignore[assignment]

    arrows: list[dict[str, object]] = []
    portal_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for teleport_count in teleports_per_arrow:
        directions = _sample_directions(teleport_count + 1)
        pending_source: tuple[tuple[int, int], tuple[int, int]] | None = None

        for i, direction in enumerate(directions):
            start_kind = "arrow" if i == 0 else "portal"
            end_kind = random.choice(["wall", "edge"]) if i == len(directions) - 1 else "portal"
            candidate = _find_segment_candidate(
                h,
                w,
                blocked,
                arrow_cells,
                portal_cells,
                wall_cells,
                direction,
                start_kind,
                end_kind,
            )
            if candidate is None:
                return None

            _commit_segment(state, candidate, arrows)

            if start_kind == "portal":
                assert pending_source is not None
                start_portal = candidate["start_portal"]
                assert start_portal is not None
                portal_pairs.append((pending_source[0], start_portal))
                pending_source = None

            if end_kind == "portal":
                end_portal = candidate["end_portal"]
                end_wall_dir = candidate["end_wall_dir"]
                assert end_portal is not None and end_wall_dir is not None
                pending_source = (end_portal, end_wall_dir)

        assert pending_source is None

    for _ in range(extra_pairs):
        endpoint_a = _place_isolated_portal_endpoint(h, w, blocked, arrow_cells, portal_cells, wall_cells)
        if endpoint_a is None:
            return None
        cell_a, _wall_dir_a, block_a, walls_a = endpoint_a
        blocked.update(block_a)
        portal_cells.add(cell_a)
        wall_cells.update(walls_a)

        endpoint_b = _place_isolated_portal_endpoint(h, w, blocked, arrow_cells, portal_cells, wall_cells)
        if endpoint_b is None:
            return None
        cell_b, _wall_dir_b, block_b, walls_b = endpoint_b
        blocked.update(block_b)
        portal_cells.add(cell_b)
        wall_cells.update(walls_b)

        portal_pairs.append((cell_a, cell_b))

    grid = _paint_input(h, w, wall_cells, portal_pairs, arrows)
    out = main(grid)

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.any(out == MAROON):
        return None
    if len(set(np.unique(grid)) - {BG}) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-arrow scene")
