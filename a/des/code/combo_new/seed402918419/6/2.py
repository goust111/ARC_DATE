from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PATH_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

OUTER_MARGIN = 1
REGION_GAP = 2
MIN_REGION = 8
GRID_MIN = 18
GRID_MAX = 30
ROUTE_COUNT_MIN = 2
ROUTE_COUNT_MAX = 4
ROUTE_ATTEMPTS = 120
GEN_ATTEMPTS = 300

Coord = tuple[int, int]
Region = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        coords: list[Coord] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return comps


def _is_straight_segment(coords: list[Coord]) -> bool:
    if not 2 <= len(coords) <= 4:
        return False
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        row = next(iter(rows))
        ordered = sorted(c for _, c in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1)) and len(ordered) == len(coords) and row >= 0
    if len(cols) == 1:
        col = next(iter(cols))
        ordered = sorted(r for r, _ in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1)) and len(ordered) == len(coords) and col >= 0
    return False


def _segment_endpoints(coords: list[Coord]) -> tuple[Coord, Coord, str]:
    assert _is_straight_segment(coords), "segment must be a short straight component"
    rows = {r for r, _ in coords}
    if len(rows) == 1:
        row = next(iter(rows))
        cols = sorted(c for _, c in coords)
        return (row, cols[0]), (row, cols[-1]), "h"
    col = next(iter(c for _, c in coords))
    rows_sorted = sorted(r for r, _ in coords)
    return (rows_sorted[0], col), (rows_sorted[-1], col), "v"


def _axial_entry_candidates(coords: list[Coord], grid: np.ndarray) -> list[Coord]:
    h, w = grid.shape
    end_a, end_b, axis = _segment_endpoints(coords)
    candidates: list[Coord] = []
    if axis == "h":
        probes = [(end_a[0], end_a[1] - 1), (end_b[0], end_b[1] + 1)]
    else:
        probes = [(end_a[0] - 1, end_a[1]), (end_b[0] + 1, end_b[1])]
    for r, c in probes:
        if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG:
            candidates.append((r, c))
    return candidates


def _bfs_path(walkable: np.ndarray, start: Coord, goal: Coord) -> list[Coord] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev: dict[Coord, Coord | None] = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in prev or not bool(walkable[nr, nc]):
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path: list[Coord] = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _path_color_components(grid: np.ndarray) -> tuple[dict[int, list[list[Coord]]], int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    route_colors: dict[int, list[list[Coord]]] = {}
    wall_colors: list[int] = []
    for color in colors:
        comps = _components(grid == color)
        if len(comps) == 2 and all(_is_straight_segment(comp) for comp in comps):
            route_colors[color] = comps
        else:
            wall_colors.append(color)
    assert route_colors, "expected at least one routed color"
    assert len(wall_colors) == 1, "expected a single wall color"
    return route_colors, wall_colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored segment pair through the unique wall-bounded corridor between them.

    Input structure:
    - Black cells are the walkable corridor background.
    - One wall color forms thin grey barriers around and between the corridors.
    - Every other non-background color appears as exactly two short horizontal or vertical
      line segments of length 2-4.
    - The same-colored segments sit at corridor ends, and each pair is linked by a unique
      4-connected path through black cells.

    Rule:
    - Identify every non-wall color that appears as two short straight segments.
    - For each segment, inspect the two axial cells just beyond its endpoints and keep the
      black endpoint-adjacent corridor cells.
    - For each color pair, find the unique 4-connected path through black cells that joins
      one valid endpoint-adjacent cell of the first segment to one valid endpoint-adjacent
      cell of the second segment.
    - Recolor every black cell on that corridor path to the segment color, keeping the
      original segments and all wall pixels unchanged.

    Key insight:
    The short colored bars are not complete paths; they are terminal markers attached to a
    maze-like corridor system, so solving requires pairing colors first and then tracing the
    only corridor route allowed by the walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color_comps, _wall_color = _path_color_components(g)
    out = g.copy()

    for color, comps in color_comps.items():
        starts = _axial_entry_candidates(comps[0], g)
        goals = _axial_entry_candidates(comps[1], g)
        assert starts and goals, "each segment must expose a corridor-facing endpoint"

        walkable = out == BG
        routes: dict[tuple[Coord, ...], list[Coord]] = {}
        for start in starts:
            for goal in goals:
                if not (bool(walkable[start]) and bool(walkable[goal])):
                    continue
                path = _bfs_path(walkable, start, goal)
                if path is not None:
                    routes[tuple(path)] = path
        assert len(routes) == 1, f"color {color} must have one unique corridor route"

        for r, c in next(iter(routes.values())):
            assert int(out[r, c]) == BG, "routes of different colors must stay disjoint"
            out[r, c] = int(color)

    return out


def _side_axis_len(side: str, h: int, w: int) -> int:
    return w if side in ("left", "right") else h


def _start_segment(side: str, length: int, line: int, h: int, w: int) -> list[Coord]:
    if side == "left":
        return [(line, c) for c in range(length)]
    if side == "right":
        return [(line, c) for c in range(w - 1, w - length - 1, -1)]
    if side == "top":
        return [(r, line) for r in range(length)]
    assert side == "bottom"
    return [(r, line) for r in range(h - 1, h - length - 1, -1)]


def _end_segment(side: str, length: int, line: int, h: int, w: int) -> list[Coord]:
    if side == "left":
        return [(line, c) for c in range(length - 1, -1, -1)]
    if side == "right":
        return [(line, c) for c in range(w - length, w)]
    if side == "top":
        return [(r, line) for r in range(length - 1, -1, -1)]
    assert side == "bottom"
    return [(r, line) for r in range(h - length, h)]


def _segment_line_choices(side: str, h: int, w: int) -> list[int]:
    if side in ("left", "right"):
        if h <= 2:
            return []
        return list(range(1, h - 1))
    if w <= 2:
        return []
    return list(range(1, w - 1))


def _touches_forbidden(cell: Coord, forbidden: set[Coord], allow: set[Coord]) -> bool:
    for adj in _neighbors4(*cell):
        if adj in forbidden and adj not in allow:
            return True
    return False


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _search_core_path(
    h: int,
    w: int,
    start: Coord,
    goal: Coord,
    blocked: set[Coord],
    forbidden_touch: set[Coord],
    goal_allow: set[Coord],
    rng: random.Random,
    min_turns: int,
) -> list[Coord] | None:
    visited = {start}
    path = [start]
    max_steps = h * w

    def dfs(prev_dir: Coord | None, turns: int) -> list[Coord] | None:
        cur = path[-1]
        if cur == goal:
            return list(path) if turns >= min_turns else None
        if len(path) >= max_steps:
            return None

        scored: list[tuple[float, Coord, Coord, int]] = []
        for nr, nc in _neighbors4(*cur):
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in visited or (nxt in blocked and nxt != goal):
                continue
            if _touches_forbidden(nxt, forbidden_touch, goal_allow if nxt == goal else {cur}):
                continue
            collision = False
            for adj in _neighbors4(nr, nc):
                if adj == cur:
                    continue
                if adj in visited:
                    collision = True
                    break
            if collision:
                continue

            step_dir = (nr - cur[0], nc - cur[1])
            turn_inc = 1 if prev_dir is not None and step_dir != prev_dir else 0
            need_turn_bonus = -0.45 if turns < min_turns and prev_dir is not None and step_dir != prev_dir else 0.0
            straight_penalty = 0.25 if turns < min_turns and prev_dir is not None and step_dir == prev_dir else 0.0
            score = float(_manhattan(nxt, goal)) + straight_penalty + need_turn_bonus + rng.random() * 0.35
            scored.append((score, nxt, step_dir, turns + turn_inc))

        scored.sort(key=lambda item: item[0])
        for _score, nxt, step_dir, next_turns in scored:
            if nxt != goal and len(path) + _manhattan(nxt, goal) > max_steps:
                continue
            visited.add(nxt)
            path.append(nxt)
            found = dfs(step_dir, next_turns)
            if found is not None:
                return found
            path.pop()
            visited.remove(nxt)
        return None

    return dfs(None, 0)


def _add_dead_end_branches(
    h: int,
    w: int,
    main_route: list[Coord],
    occupied: set[Coord],
    segment_cells: set[Coord],
    rng: random.Random,
) -> set[Coord]:
    branches: set[Coord] = set()
    candidates = main_route[2:-2]
    if not candidates:
        return branches

    target_count = rng.randint(0, 2)
    for _ in range(target_count):
        rng.shuffle(candidates)
        placed = False
        for base in candidates:
            used_neighbors = {adj for adj in _neighbors4(*base) if adj in occupied}
            free_dirs = []
            for nr, nc in _neighbors4(*base):
                nxt = (nr, nc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if nxt in occupied:
                    continue
                if _touches_forbidden(nxt, occupied | segment_cells, {base}):
                    continue
                free_dirs.append((nr - base[0], nc - base[1]))
            rng.shuffle(free_dirs)
            for dr, dc in free_dirs:
                length = rng.randint(1, 3)
                new_cells: list[Coord] = []
                prev = base
                cur = (base[0] + dr, base[1] + dc)
                ok = True
                for _step in range(length):
                    r, c = cur
                    if not (0 <= r < h and 0 <= c < w) or cur in occupied or cur in new_cells:
                        ok = False
                        break
                    if _touches_forbidden(cur, occupied | segment_cells | set(new_cells), {prev}):
                        ok = False
                        break
                    new_cells.append(cur)
                    prev = cur
                    cur = (cur[0] + dr, cur[1] + dc)
                if ok and new_cells:
                    for cell in new_cells:
                        occupied.add(cell)
                        branches.add(cell)
                    placed = True
                    break
            if placed:
                break
    return branches


def _line_path(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    assert a[1] == b[1], "path segment must be axis-aligned"
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _is_simple_ordered_path(route: list[Coord]) -> bool:
    if len(route) != len(set(route)):
        return False
    route_set = set(route)
    for idx in range(len(route) - 1):
        if _manhattan(route[idx], route[idx + 1]) != 1:
            return False
    endpoint_count = 0
    for cell in route:
        degree = sum(adj in route_set for adj in _neighbors4(*cell))
        if degree == 1:
            endpoint_count += 1
        elif degree != 2:
            return False
    return endpoint_count == 2


def _build_route_patch(h: int, w: int, color: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(ROUTE_ATTEMPTS):
        rotation = rng.randrange(4)
        bh, bw = (h, w) if rotation % 2 == 0 else (w, h)
        if bh < 4 or bw < 7:
            continue

        start_len = rng.randint(2, min(4, bw - 3))
        end_len = rng.randint(2, min(4, bw - 3))
        inner_start = start_len - 1
        inner_end = bw - end_len
        if inner_end - inner_start < 2:
            continue

        rows_available = list(range(1, bh - 1))
        if not rows_available:
            continue
        row_count = rng.choice([count for count in (1, 2, 3, 4, 5) if count <= len(rows_available)])
        rows = sorted(rng.sample(rows_available, row_count))
        if rng.random() < 0.5:
            rows.reverse()

        left_lo = start_len + 1
        left_hi = max(left_lo, min(bw - end_len - 3, start_len + max(1, (bw - start_len - end_len) // 3)))
        right_lo = max(left_hi + 2, bw - end_len - 3)
        right_hi = inner_end - 1
        if left_lo > left_hi or right_lo > right_hi:
            continue

        left_sweep = rng.randint(left_lo, left_hi)
        right_sweep = rng.randint(right_lo, right_hi)

        seg_a = [(rows[0], c) for c in range(start_len)]
        seg_b = [(rows[-1], c) for c in range(inner_end, bw)]

        main_route: list[Coord] = list(seg_a)
        cur = seg_a[-1]
        for idx, row in enumerate(rows):
            if idx == len(rows) - 1:
                target_col = inner_end
            else:
                target_col = right_sweep if idx % 2 == 0 else left_sweep
            horiz = _line_path(cur, (row, target_col))
            main_route.extend(horiz[1:])
            cur = horiz[-1]
            if idx < len(rows) - 1:
                next_row = rows[idx + 1]
                vert = _line_path(cur, (next_row, target_col))
                main_route.extend(vert[1:])
                cur = vert[-1]

        main_route.extend(seg_b[1:])
        if not _is_simple_ordered_path(main_route):
            continue

        set_a = set(seg_a)
        set_b = set(seg_b)
        occupied = set(main_route)
        branches = _add_dead_end_branches(bh, bw, main_route, occupied, set_a | set_b, rng)

        patch = np.full((bh, bw), GREY, dtype=int)
        for r, c in occupied | branches:
            patch[r, c] = BG
        for r, c in seg_a:
            patch[r, c] = int(color)
        for r, c in seg_b:
            patch[r, c] = int(color)

        patch = np.rot90(patch, rotation).copy()
        try:
            main(patch)
        except AssertionError:
            continue
        return patch
    return None


def _split_regions(h: int, w: int, count: int, rng: random.Random) -> list[Region] | None:
    regions: list[Region] = [(OUTER_MARGIN, h - OUTER_MARGIN, OUTER_MARGIN, w - OUTER_MARGIN)]
    while len(regions) < count:
        splittable = []
        for idx, (r0, r1, c0, c1) in enumerate(regions):
            rh = r1 - r0
            rw = c1 - c0
            can_h = rh >= 2 * MIN_REGION + REGION_GAP
            can_v = rw >= 2 * MIN_REGION + REGION_GAP
            if can_h or can_v:
                splittable.append((rh * rw, idx, can_h, can_v))
        if not splittable:
            return None

        _area, idx, can_h, can_v = max(splittable, key=lambda item: item[0])
        r0, r1, c0, c1 = regions.pop(idx)
        rh = r1 - r0
        rw = c1 - c0
        if can_h and can_v:
            split_h = rh >= rw if rng.random() < 0.7 else rh < rw
        else:
            split_h = can_h

        if split_h:
            cut = rng.randint(r0 + MIN_REGION, r1 - MIN_REGION - REGION_GAP)
            reg_a = (r0, cut, c0, c1)
            reg_b = (cut + REGION_GAP, r1, c0, c1)
        else:
            cut = rng.randint(c0 + MIN_REGION, c1 - MIN_REGION - REGION_GAP)
            reg_a = (r0, r1, c0, cut)
            reg_b = (r0, r1, cut + REGION_GAP, c1)
        regions.extend([reg_a, reg_b])
    rng.shuffle(regions)
    return regions


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        route_count = rng.randint(ROUTE_COUNT_MIN, ROUTE_COUNT_MAX)
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)

        regions = _split_regions(h, w, route_count, rng)
        if regions is None:
            continue

        grid = np.full((h, w), GREY, dtype=int)
        colors = rng.sample(PATH_PALETTE, route_count)
        ok = True
        for (r0, r1, c0, c1), color in zip(regions, colors):
            patch = _build_route_patch(r1 - r0, c1 - c0, color, rng)
            if patch is None:
                ok = False
                break
            grid[r0:r1, c0:c1] = patch
        if not ok:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        assert out.shape == grid.shape
        return grid

    raise RuntimeError("failed to generate a valid routed-corridor sample")
