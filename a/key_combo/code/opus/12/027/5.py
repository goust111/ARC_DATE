from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
ACTIVE_COLORS = (BLUE, YELLOW, ORANGE)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), size]
    bands = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    assert all(start < stop for start, stop in bands), "separator lines must leave non-empty partitions"
    return bands


def _parse_partitions(grid: np.ndarray) -> dict:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert len(row_bands) * len(col_bands) == 6, "expected six separator-delimited partitions"
    assert len(row_bands) in (2, 3) and len(col_bands) in (2, 3)

    zone_rows = np.full(h, -1, dtype=int)
    zone_cols = np.full(w, -1, dtype=int)
    for i, (start, stop) in enumerate(row_bands):
        zone_rows[start:stop] = i
    for i, (start, stop) in enumerate(col_bands):
        zone_cols[start:stop] = i
    for r in sep_rows:
        zone_rows[r] = -1
    for c in sep_cols:
        zone_cols[c] = -1

    indicators: dict[tuple[int, int], int] = {}
    indicator_cells: set[tuple[int, int]] = set()
    for zr, (r0, _r1) in enumerate(row_bands):
        for zc, (c0, _c1) in enumerate(col_bands):
            color = int(grid[r0, c0])
            assert color in ACTIVE_COLORS, "each partition must start with a blue/orange/yellow indicator"
            indicators[(zr, zc)] = color
            indicator_cells.add((r0, c0))

    return {
        "row_bands": row_bands,
        "col_bands": col_bands,
        "zone_rows": zone_rows,
        "zone_cols": zone_cols,
        "indicators": indicators,
        "indicator_cells": indicator_cells,
    }


def _zone_at(parsed: dict, r: int, c: int) -> tuple[int, int] | None:
    zr = int(parsed["zone_rows"][r])
    zc = int(parsed["zone_cols"][c])
    if zr < 0 or zc < 0:
        return None
    return zr, zc


def _is_active_obstacle(grid: np.ndarray, parsed: dict, r: int, c: int) -> bool:
    value = int(grid[r, c])
    if value not in ACTIVE_COLORS or (r, c) in parsed["indicator_cells"]:
        return False
    zone = _zone_at(parsed, r, c)
    if zone is None:
        return False
    return value == int(parsed["indicators"][zone])


def _parse_edge_line(grid: np.ndarray, color: int) -> dict:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
    assert coords, f"missing color {color} line"
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    h, w = grid.shape

    if len(rows) == 1:
        row = rows[0]
        lo, hi = min(cols), max(cols)
        assert cols == list(range(lo, hi + 1)), "edge line must be contiguous"
        if lo == 0:
            direction = (0, 1)
            anchor = (row, hi)
        else:
            assert hi == w - 1, "horizontal edge line must touch the left or right border"
            direction = (0, -1)
            anchor = (row, lo)
    else:
        assert len(cols) == 1, "edge line must be horizontal or vertical"
        col = cols[0]
        lo, hi = min(rows), max(rows)
        assert rows == list(range(lo, hi + 1)), "edge line must be contiguous"
        if lo == 0:
            direction = (1, 0)
            anchor = (hi, col)
        else:
            assert hi == h - 1, "vertical edge line must touch the top or bottom border"
            direction = (-1, 0)
            anchor = (lo, col)

    return {"cells": set(coords), "direction": direction, "anchor": anchor}


def _dir_index(direction: tuple[int, int]) -> int:
    return DIRS.index(direction)


def _state_successors(grid: np.ndarray, parsed: dict, state: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    r, c, d = state
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < h and 0 <= nc < w):
        return []
    if _is_active_obstacle(grid, parsed, nr, nc):
        return [(r, c, (d - 1) % 4), (r, c, (d + 1) % 4)]
    if int(grid[nr, nc]) == RED:
        return []
    return [(nr, nc, d)]


def _is_goal_state(grid: np.ndarray, state: tuple[int, int, int]) -> bool:
    h, w = grid.shape
    r, c, d = state
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    return 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == RED


def _solve(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size
    parsed = _parse_partitions(g)
    start = _parse_edge_line(g, GREEN)
    _parse_edge_line(g, RED)

    start_state = (*start["anchor"], _dir_index(start["direction"]))
    queue = deque([start_state])
    dist = {start_state: 0}
    count = {start_state: 1}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    goal_states: list[tuple[int, int, int]] = []

    while queue:
        state = queue.popleft()
        if _is_goal_state(g, state):
            goal_states.append(state)
            continue
        for nxt in _state_successors(g, parsed, state):
            nd = dist[state] + 1
            if nxt not in dist:
                dist[nxt] = nd
                count[nxt] = count[state]
                parent[nxt] = state
                queue.append(nxt)
            elif dist[nxt] == nd:
                count[nxt] += count[state]

    assert goal_states, "no path from the green start line to the red end line"
    best = min(dist[state] for state in goal_states)
    best_goals = [state for state in goal_states if dist[state] == best]
    shortest_count = sum(count[state] for state in best_goals)
    goal = min(best_goals)

    path_states = [goal]
    while path_states[-1] in parent:
        path_states.append(parent[path_states[-1]])
    path_states.reverse()

    path_positions = {(r, c) for r, c, _d in path_states}
    zones_crossed = set()
    for r, c in path_positions:
        zone = _zone_at(parsed, r, c)
        if zone is not None:
            zones_crossed.add(zone)

    return {
        "parsed": parsed,
        "start": start,
        "goal": goal,
        "best_dist": best,
        "shortest_count": shortest_count,
        "path_states": path_states,
        "path_positions": path_positions,
        "zones_crossed": zones_crossed,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color only the black cells on the unique shortest route through partitions whose obstacle color changes by zone.

    Input structure:
    - A black grid is split into six rectangular partitions by full grey separator rows and columns.
    - Each partition's top-left cell is a blue/orange/yellow indicator, and the partitions also contain scattered blue/orange/yellow pixels.
    - A short green edge line marks the start and initial direction; a short red edge line marks the destination.

    Rule:
    - Read each partition indicator color C_zone from that partition's top-left cell.
    - Inside a partition, non-indicator pixels of color C_zone are turn-triggering obstacles, while other colored pixels and the grey separator lines are passable.
    - Run BFS over (row, col, direction) states, moving straight when the next cell is passable and turning left or right in place only when the next cell is that partition's active obstacle; stop when the next cell is red.
    - Paint the recovered shortest-route cells green only where the input cell was black, leaving inert colored pixels, indicators, separators, and the start/end markers unchanged.

    Key insight:
    - The same visible color does not have one global meaning: whether a blue/orange/yellow pixel blocks motion depends on the local partition indicator, so pathfinding must continuously consult partition context.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    info = _solve(g)
    for r, c in info["path_positions"]:
        if int(out[r, c]) == BLACK:
            out[r, c] = GREEN
    return out


def _draw_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segments must be axis-aligned")


def _polyline_cells(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(vertices) - 1):
        part = _draw_segment(vertices[i], vertices[i + 1])
        if i:
            part = part[1:]
        cells.extend(part)
    return cells


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _line_direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    if a[0] == b[0]:
        return (0, 1 if b[1] > a[1] else -1)
    return (1 if b[0] > a[0] else -1, 0)


def _paint_route_only_black(grid: np.ndarray, cells: set[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c in cells:
        if int(out[r, c]) == BLACK:
            out[r, c] = GREEN
    return out


def _route_length(vertices: list[tuple[int, int]]) -> int:
    moves = 0
    for a, b in zip(vertices, vertices[1:]):
        moves += abs(a[0] - b[0]) + abs(a[1] - b[1])
    return moves + len(vertices) - 2


def _route_is_valid(grid: np.ndarray, vertices: list[tuple[int, int]], initial_dir: tuple[int, int]) -> bool:
    parsed = _parse_partitions(grid)
    direction = initial_dir
    current = vertices[0]
    if len(vertices) < 2:
        return False

    for i in range(len(vertices) - 1):
        nxt_vertex = vertices[i + 1]
        expected = _line_direction(current, nxt_vertex)
        if expected != direction:
            return False
        segment = _draw_segment(current, nxt_vertex)
        for cell in segment[1:]:
            r, c = cell
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                return False
            if int(grid[r, c]) == RED or _is_active_obstacle(grid, parsed, r, c):
                return False
        current = nxt_vertex
        if i == len(vertices) - 2:
            dr, dc = direction
            rr, cc = current[0] + dr, current[1] + dc
            return 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) == RED
        next_dir = _line_direction(vertices[i + 1], vertices[i + 2])
        if next_dir == direction:
            return False
        dr, dc = direction
        rr, cc = current[0] + dr, current[1] + dc
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return False
        if not _is_active_obstacle(grid, parsed, rr, cc):
            return False
        turn_options = {DIRS[(_dir_index(direction) - 1) % 4], DIRS[(_dir_index(direction) + 1) % 4]}
        if next_dir not in turn_options:
            return False
        direction = next_dir
    return False


def _indicator_assignment(zone_count: int) -> list[int]:
    while True:
        colors = [random.choice(ACTIVE_COLORS) for _ in range(zone_count)]
        if len(set(colors)) >= 2:
            return colors


def _build_layout() -> dict | None:
    orientation = random.choice(["2x3", "3x2"])
    if orientation == "2x3":
        row_sizes = [random.randint(5, 6) for _ in range(2)]
        col_sizes = [random.randint(4, 6) for _ in range(3)]
    else:
        row_sizes = [random.randint(4, 6) for _ in range(3)]
        col_sizes = [random.randint(5, 6) for _ in range(2)]

    h = sum(row_sizes) + len(row_sizes) - 1
    w = sum(col_sizes) + len(col_sizes) - 1
    grid = np.full((h, w), BLACK, dtype=int)

    row_bands: list[tuple[int, int]] = []
    start = 0
    for size in row_sizes:
        row_bands.append((start, start + size))
        start += size + 1

    col_bands: list[tuple[int, int]] = []
    start = 0
    for size in col_sizes:
        col_bands.append((start, start + size))
        start += size + 1

    sep_rows = [band[1] for band in row_bands[:-1]]
    sep_cols = [band[1] for band in col_bands[:-1]]
    for r in sep_rows:
        grid[r, :] = GREY
    for c in sep_cols:
        grid[:, c] = GREY

    zone_bounds: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    colors = _indicator_assignment(len(row_bands) * len(col_bands))
    color_i = 0
    for zr, (r0, r1) in enumerate(row_bands):
        for zc, (c0, c1) in enumerate(col_bands):
            zone_bounds[(zr, zc)] = (r0, r1, c0, c1)
            grid[r0, c0] = colors[color_i]
            color_i += 1

    return {
        "grid": grid,
        "orientation": orientation,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "zone_bounds": zone_bounds,
    }


def _rand_between(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _build_template(layout: dict) -> dict | None:
    grid = layout["grid"]
    h, w = grid.shape
    row_bands = layout["row_bands"]
    col_bands = layout["col_bands"]
    start_len = random.randint(2, 3)
    end_len = random.randint(2, 3)

    if layout["orientation"] == "2x3":
        (rt0, rt1), (rb0, rb1) = row_bands
        (cl0, cl1), (cm0, cm1), (cr0, cr1) = col_bands

        x_mid = _rand_between(cm0 + 1, cm1 - 2)
        x_left = _rand_between(cl0 + 2, cl1 - 2)
        x_exit = _rand_between(cr0 + 1, cr1 - 2)
        y_top = _rand_between(rt0 + 2, rt1 - 2)
        y_goal = h - end_len - 1
        y_j = _rand_between(rb0 + 1, min(rb1 - 3, y_goal))
        if None in (x_mid, x_left, x_exit, y_top, y_j):
            return None

        start_cells = [(r, x_mid) for r in range(start_len)]
        end_cells = [(r, x_exit) for r in range(h - end_len, h)]

        s = (start_len - 1, x_mid)
        j = (y_j, x_mid)
        p = (y_j, x_exit)
        l = (y_j, x_left)
        u = (y_top, x_left)
        rturn = (y_top, x_exit)
        a = (y_goal, x_exit)

        primary_vertices = [s, j, p, a]
        alternate_vertices = [s, j, l, u, rturn, a]
        obstacles = {
            (j[0] + 1, j[1]),
            (p[0], p[1] + 1),
            (l[0], l[1] - 1),
            (u[0] - 1, u[1]),
            (rturn[0], rturn[1] + 1),
        }
        initial_dir = (1, 0)
    else:
        (rt0, rt1), (rm0, rm1), (rb0, rb1) = row_bands
        (cl0, cl1), (cr0, cr1) = col_bands

        y_mid = _rand_between(rm0 + 1, rm1 - 2)
        y_top = _rand_between(rt0 + 2, rt1 - 2)
        y_exit = _rand_between(rb0 + 1, rb1 - 2)
        x_goal = w - end_len - 1
        x_j = _rand_between(cr0 + 1, min(cr1 - 3, x_goal))
        x_left = _rand_between(cl0 + 2, cl1 - 2)
        if None in (y_mid, y_top, y_exit, x_j, x_left):
            return None

        start_cells = [(y_mid, c) for c in range(start_len)]
        end_cells = [(y_exit, c) for c in range(w - end_len, w)]

        s = (y_mid, start_len - 1)
        j = (y_mid, x_j)
        p = (y_exit, x_j)
        u = (y_top, x_j)
        l = (y_top, x_left)
        dturn = (y_exit, x_left)
        a = (y_exit, x_goal)

        primary_vertices = [s, j, p, a]
        alternate_vertices = [s, j, u, l, dturn, a]
        obstacles = {
            (j[0], j[1] + 1),
            (p[0] + 1, p[1]),
            (u[0] - 1, u[1]),
            (l[0], l[1] - 1),
            (dturn[0] + 1, dturn[1]),
        }
        initial_dir = (0, 1)

    if any(not (0 <= r < h and 0 <= c < w) for r, c in obstacles):
        return None

    primary_cells = _polyline_cells(primary_vertices)
    alternate_cells = _polyline_cells(alternate_vertices)
    return {
        "start_cells": start_cells,
        "end_cells": end_cells,
        "primary_vertices": primary_vertices,
        "alternate_vertices": alternate_vertices,
        "primary_cells": primary_cells,
        "alternate_cells": alternate_cells,
        "obstacles": obstacles,
        "initial_dir": initial_dir,
    }


def _coord_zone_from_bands(row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], r: int, c: int):
    zr = next((i for i, (a, b) in enumerate(row_bands) if a <= r < b), None)
    zc = next((i for i, (a, b) in enumerate(col_bands) if a <= c < b), None)
    if zr is None or zc is None:
        return None
    return zr, zc


def _fill_zone_colors(layout: dict, template: dict) -> bool:
    grid = layout["grid"]
    row_bands = layout["row_bands"]
    col_bands = layout["col_bands"]
    reserved = set(template["start_cells"]) | set(template["end_cells"]) | set(template["obstacles"])
    path_union = set(template["primary_cells"]) | set(template["alternate_cells"])
    turn_points = set(template["primary_vertices"][1:-1]) | set(template["alternate_vertices"][1:-1])

    buffer = set(reserved) | path_union
    for r, c in list(path_union):
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                buffer.add((nr, nc))

    fixed: dict[tuple[int, int], dict[int, set[tuple[int, int]]]] = {}
    for zone in layout["zone_bounds"]:
        fixed[zone] = {color: set() for color in ACTIVE_COLORS}

    for r, c in template["obstacles"]:
        zone = _coord_zone_from_bands(row_bands, col_bands, r, c)
        if zone is None:
            return False
        fixed[zone][int(grid[layout["zone_bounds"][zone][0], layout["zone_bounds"][zone][2]])].add((r, c))

    passive_candidates_by_zone: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in path_union:
        if cell in reserved or cell in turn_points:
            continue
        zone = _coord_zone_from_bands(row_bands, col_bands, cell[0], cell[1])
        if zone is None:
            continue
        passive_candidates_by_zone.setdefault(zone, []).append(cell)

    passive_count = 0
    visited_zones = sorted({zone for zone in passive_candidates_by_zone})
    random.shuffle(visited_zones)
    for zone in visited_zones:
        indicator = int(grid[layout["zone_bounds"][zone][0], layout["zone_bounds"][zone][2]])
        choices = [color for color in ACTIVE_COLORS if color != indicator]
        cells = passive_candidates_by_zone[zone][:]
        random.shuffle(cells)
        take = 1 if cells and (random.random() < 0.75 or passive_count == 0) else 0
        for cell in cells[:take]:
            color = random.choice(choices)
            fixed[zone][color].add(cell)
            passive_count += 1

    if passive_count == 0:
        zones_with_cells = [zone for zone, cells in passive_candidates_by_zone.items() if cells]
        if not zones_with_cells:
            return False
        zone = random.choice(zones_with_cells)
        indicator = int(grid[layout["zone_bounds"][zone][0], layout["zone_bounds"][zone][2]])
        color = random.choice([c for c in ACTIVE_COLORS if c != indicator])
        fixed[zone][color].add(passive_candidates_by_zone[zone][0])

    for zone, (r0, r1, c0, c1) in layout["zone_bounds"].items():
        indicator = int(grid[r0, c0])
        candidates = [
            (r, c)
            for r in range(r0, r1)
            for c in range(c0, c1)
            if (r, c) != (r0, c0) and (r, c) not in buffer
        ]
        random.shuffle(candidates)

        need = {}
        for color in ACTIVE_COLORS:
            base = len(fixed[zone][color])
            lo = max(2, base)
            hi = max(lo, 4 if color != indicator else 5)
            need[color] = random.randint(lo, hi)

        missing = sum(need[color] - len(fixed[zone][color]) for color in ACTIVE_COLORS)
        if missing > len(candidates):
            return False

        idx = 0
        for color in ACTIVE_COLORS:
            while len(fixed[zone][color]) < need[color]:
                fixed[zone][color].add(candidates[idx])
                idx += 1

        for color, cells in fixed[zone].items():
            for r, c in cells:
                grid[r, c] = color
    return True


def _semantic_checks(grid: np.ndarray, template: dict) -> bool:
    try:
        info = _solve(grid)
    except AssertionError:
        return False

    primary_cells = set(template["primary_cells"])
    alternate_cells = set(template["alternate_cells"])
    if info["shortest_count"] != 1:
        return False
    if info["best_dist"] != _route_length(template["primary_vertices"]):
        return False
    if _route_length(template["alternate_vertices"]) <= info["best_dist"]:
        return False
    if info["path_positions"] != primary_cells:
        return False
    if len(info["zones_crossed"]) < 3:
        return False
    if not _route_is_valid(grid, template["primary_vertices"], template["initial_dir"]):
        return False
    if not _route_is_valid(grid, template["alternate_vertices"], template["initial_dir"]):
        return False

    parsed = info["parsed"]
    color_roles_ok = False
    for color in ACTIVE_COLORS:
        has_obstacle = False
        has_inert = False
        for r, c in np.argwhere(grid == color):
            rr, cc = int(r), int(c)
            if (rr, cc) in parsed["indicator_cells"]:
                continue
            if _is_active_obstacle(grid, parsed, rr, cc):
                has_obstacle = True
            else:
                has_inert = True
        if has_obstacle and has_inert:
            color_roles_ok = True
            break
    if not color_roles_ok:
        return False

    out = main(grid)
    expected = _paint_route_only_black(grid, primary_cells)
    if not np.array_equal(out, expected):
        return False

    changed = np.argwhere((grid == BLACK) & (out == GREEN))
    return len(changed) >= 6 and not primary_cells == alternate_cells


def _gen_attempt() -> np.ndarray | None:
    layout = _build_layout()
    if layout is None:
        return None
    template = _build_template(layout)
    if template is None:
        return None

    grid = layout["grid"]
    for r, c in template["start_cells"]:
        grid[r, c] = GREEN
    for r, c in template["end_cells"]:
        grid[r, c] = RED
    for r, c in template["obstacles"]:
        zone = _coord_zone_from_bands(layout["row_bands"], layout["col_bands"], r, c)
        if zone is None:
            return None
        zr, zc = zone
        r0, _r1, c0, _c1 = layout["zone_bounds"][zone]
        grid[r, c] = int(grid[r0, c0])

    if not _fill_zone_colors(layout, template):
        return None
    if not _semantic_checks(grid, template):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partition-aware pathfinding instance")
