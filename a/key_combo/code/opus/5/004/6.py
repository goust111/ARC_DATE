import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r, c):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _reflect(coord, axis_kind, axis_idx):
    r, c = coord
    if axis_kind == "row":
        return 2 * axis_idx - r, c
    return r, 2 * axis_idx - c


def _line(a, b):
    r0, c0 = a
    r1, c1 = b
    if r0 != r1 and c0 != c1:
        raise ValueError("polyline segments must be axis-aligned")
    out = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            out.append((r0, c))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            out.append((r, c0))
    return out


def _polyline(points):
    cells = []
    for idx in range(len(points) - 1):
        seg = _line(points[idx], points[idx + 1])
        if idx:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _build_graph_nodes(grid, forbidden):
    nodes = set()
    for r, c in np.argwhere((grid == TEAL) | (grid == MAROON)):
        rr = int(r)
        cc = int(c)
        if not forbidden[rr, cc]:
            nodes.add((rr, cc))
    return nodes


def _endpoints(grid, forbidden):
    pts = []
    for r, c in np.argwhere(grid == MAROON):
        rr = int(r)
        cc = int(c)
        if not forbidden[rr, cc]:
            pts.append((rr, cc))
    return pts


def _adjacent_nodes(nodes, cell, blocked_edge=None):
    out = []
    for nxt in _neighbors4(*cell):
        if nxt not in nodes:
            continue
        if blocked_edge is not None and ((cell, nxt) == blocked_edge or (nxt, cell) == blocked_edge):
            continue
        out.append(nxt)
    return out


def _bfs_path(nodes, start, goal):
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _adjacent_nodes(nodes, cur):
            if nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _connected(nodes, start, goal, blocked_edge=None):
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        for nxt in _adjacent_nodes(nodes, cur, blocked_edge=blocked_edge):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return False


def _is_unique_path(nodes, start, goal):
    path = _bfs_path(nodes, start, goal)
    if path is None:
        return False, None
    for a, b in zip(path, path[1:]):
        if _connected(nodes, start, goal, blocked_edge=(a, b)):
            return False, path
    return True, path


def _forbidden_mask(grid):
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(grid == ORANGE):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True
    return mask


def _find_axis(grid):
    h, w = grid.shape
    candidates = []
    for r in range(h):
        grey_count = int(np.count_nonzero(grid[r, :] == GREY))
        non_grey = w - grey_count
        if grey_count >= w - 2 and non_grey <= 2:
            candidates.append(("row", r, grey_count, -non_grey))
    for c in range(w):
        grey_count = int(np.count_nonzero(grid[:, c] == GREY))
        non_grey = h - grey_count
        if grey_count >= h - 2 and non_grey <= 2:
            candidates.append(("col", c, grey_count, -non_grey))
    if not candidates:
        raise AssertionError("expected a near-complete grey axis")
    candidates.sort(key=lambda item: (item[2], item[3]), reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][2:] == best[2:]:
        raise AssertionError("axis detection is ambiguous")
    return best[0], best[1]


def _path_from_grid(grid):
    forbidden = _forbidden_mask(grid)
    endpoints = _endpoints(grid, forbidden)
    assert len(endpoints) == 2, "expected two usable endpoints"
    nodes = _build_graph_nodes(grid, forbidden)
    unique, path = _is_unique_path(nodes, endpoints[0], endpoints[1])
    assert unique and path is not None, "safe path must be unique"
    return path


def _graph_degrees(nodes):
    return {cell: len(_adjacent_nodes(nodes, cell)) for cell in nodes}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the obstacle-safe path, mirror it across the grey guide axis, and classify overlap versus extension.

    Input structure:
    - The background is black.
    - Teal track pixels form one 4-connected rail network with branches and dead ends, and exactly two maroon endpoints lie on that network.
    - One nearly complete grey row or column marks the reflection axis, while 1-3 orange obstacle pixels sit near misleading track segments.

    Rule:
    - Mark every obstacle pixel and its full 8-neighborhood as forbidden.
    - In the remaining teal-and-maroon 4-connected graph, find the unique path between the two maroon endpoints.
    - Recolor the non-endpoint teal cells on that path to maroon.
    - Reflect every path coordinate across the grey axis.
    - If a reflected coordinate already belongs to the maroon path, recolor it yellow; if it lands on black background, recolor it blue; if it lands on teal track, an orange obstacle, or the grey axis, leave it unchanged.

    Key insight:
    The visible yellow cells are not a pre-existing symmetric object: they appear only after obstacle exclusion isolates the intended path and that derived path is mirrored back onto the scene.
    """
    g = np.asarray(grid, dtype=int)
    axis_kind, axis_idx = _find_axis(g)
    path = _path_from_grid(g)

    out = g.copy()
    endpoint_set = {path[0], path[-1]}
    for cell in path:
        if cell not in endpoint_set and out[cell] == TEAL:
            out[cell] = MAROON

    base = out.copy()
    reflected = {_reflect(cell, axis_kind, axis_idx) for cell in path}
    for rr, cc in reflected:
        if not _in_bounds(out.shape, rr, cc):
            continue
        if base[rr, cc] == MAROON:
            out[rr, cc] = YELLOW
        elif base[rr, cc] == BG:
            out[rr, cc] = BLUE
    return out


def _vertical_attempt(rng):
    h = rng.randint(14, 19)
    w = rng.randint(17, 22)
    axis = rng.randint(8, w - 8)

    hole_top = rng.randint(3, max(4, h // 2 - 1))
    hole_bottom = rng.randint(hole_top + 4, h - 3)
    if hole_top - 2 < 1 or hole_bottom + 2 > h - 2:
        return None
    r_start = rng.randint(1, hole_top - 2)
    r_end = rng.randint(hole_bottom + 2, h - 2)

    c_entry = axis - rng.randint(2, 3)
    c_exit = axis - rng.randint(3, 4)
    c_start = axis - rng.randint(4, 5)
    c_end = axis - rng.randint(4, 5)
    c_chord = axis + 2
    c_right = axis + rng.randint(4, 5)
    if c_right > w - 2:
        return None

    main_points = [
        (r_start, c_start),
        (r_start, c_entry),
        (hole_top, c_entry),
        (hole_top, c_right),
        (hole_bottom, c_right),
        (hole_bottom, c_exit),
        (r_end, c_exit),
        (r_end, c_end),
    ]
    path = _polyline(main_points)
    if len(path) != len(set(path)):
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[:, axis] = GREY

    chord = {(r, c_chord) for r in range(hole_top, hole_bottom + 1)}
    chord_body = {cell for cell in chord if cell not in {path[0], path[-1]}}
    if any(cell in path for cell in chord_body if cell[0] not in {hole_top, hole_bottom}):
        return None

    track_cells = set(path) | chord
    branch_attempts = 0
    while branch_attempts < 8:
        branch_attempts += 1
        anchor_row = rng.randint(hole_top + 1, hole_bottom - 1)
        anchor = (anchor_row, c_right)
        if anchor not in track_cells:
            continue
        length = rng.randint(1, 4)
        branch = [(anchor_row, c_right + dc) for dc in range(1, length + 1)]
        if any(not _in_bounds(grid.shape, r, c) or c == axis or (r, c) in track_cells for r, c in branch):
            continue
        track_cells.update(branch)
        break

    for r, c in track_cells:
        grid[r, c] = TEAL

    grid[path[0]] = MAROON
    grid[path[-1]] = MAROON

    obstacle_candidates = list(range(hole_top + 2, hole_bottom - 1))
    if not obstacle_candidates:
        return None
    obstacle_count = rng.randint(1, min(3, len(obstacle_candidates)))
    obstacle_rows = sorted(rng.sample(obstacle_candidates, obstacle_count))
    for row in obstacle_rows:
        grid[row, c_chord - 1] = ORANGE

    return grid


def _augment_and_orient(grid, rng):
    out = grid
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = out.T.copy()
    else:
        out = out.copy()
    return out


def _validate_generated(grid):
    axis_kind, axis_idx = _find_axis(grid)
    forbidden = _forbidden_mask(grid)
    safe_endpoints = _endpoints(grid, forbidden)
    if len(safe_endpoints) != 2:
        return False

    safe_nodes = _build_graph_nodes(grid, forbidden)
    full_nodes = _build_graph_nodes(grid, np.zeros_like(forbidden, dtype=bool))
    safe_unique, safe_path = _is_unique_path(safe_nodes, safe_endpoints[0], safe_endpoints[1])
    if not safe_unique or safe_path is None:
        return False

    full_endpoints = _endpoints(grid, np.zeros_like(forbidden, dtype=bool))
    if len(full_endpoints) != 2:
        return False
    full_unique, _ = _is_unique_path(full_nodes, full_endpoints[0], full_endpoints[1])
    if full_unique:
        return False

    degrees = _graph_degrees(full_nodes)
    non_endpoint_dead_ends = [cell for cell, deg in degrees.items() if deg == 1 and cell not in full_endpoints]
    junctions = [cell for cell, deg in degrees.items() if deg >= 3]
    if not non_endpoint_dead_ends or not junctions:
        return False

    base = grid.copy()
    endpoint_set = {safe_path[0], safe_path[-1]}
    for cell in safe_path:
        if cell not in endpoint_set and base[cell] == TEAL:
            base[cell] = MAROON
    reflected = {_reflect(cell, axis_kind, axis_idx) for cell in safe_path if _in_bounds(grid.shape, *_reflect(cell, axis_kind, axis_idx))}
    overlap = [cell for cell in reflected if base[cell] == MAROON]
    extensions = [cell for cell in reflected if base[cell] == BG]
    off_axis_overlap = [
        cell for cell in overlap
        if not ((axis_kind == "row" and cell[0] == axis_idx) or (axis_kind == "col" and cell[1] == axis_idx))
    ]
    if not off_axis_overlap or not extensions:
        return False

    out = main(grid)
    if np.count_nonzero(out == YELLOW) < 1 or np.count_nonzero(out == BLUE) < 1:
        return False
    if np.count_nonzero(grid == ORANGE) < 1 or np.count_nonzero(grid == ORANGE) > 3:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _vertical_attempt(rng)
        if grid is None:
            continue
        grid = _augment_and_orient(grid, rng)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirrored-safe-path puzzle")
