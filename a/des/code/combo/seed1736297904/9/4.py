from __future__ import annotations

import random

import numpy as np


BLACK = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
BASE_ROOM_RANGE = (4, 5)
ROOT_ROOM_RANGE = (7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _gap_run(segment: np.ndarray) -> tuple[int, int] | None:
    values = [int(v) for v in segment]
    if any(v not in (BLACK, WALL) for v in values):
        raise AssertionError("wall segments may only contain wall or door cells")
    idxs = [i for i, v in enumerate(values) if v == BLACK]
    if not idxs:
        return None
    if idxs != list(range(idxs[0], idxs[-1] + 1)):
        raise AssertionError("door gap must be contiguous")
    if len(idxs) > 2:
        raise AssertionError("door gap must be width 1 or 2")
    return idxs[0], len(idxs)


def _detect_wall_rows(grid: np.ndarray) -> list[int]:
    width = int(grid.shape[1])
    counts = np.sum(grid == WALL, axis=1)
    rows = [int(r) for r, cnt in enumerate(counts) if int(cnt) >= max(3, width // 2)]
    if not rows or rows[0] != 0 or rows[-1] != int(grid.shape[0]) - 1:
        raise AssertionError("failed to identify horizontal wall lines")
    return rows


def _detect_wall_cols(grid: np.ndarray) -> list[int]:
    height = int(grid.shape[0])
    counts = np.sum(grid == WALL, axis=0)
    cols = [int(c) for c, cnt in enumerate(counts) if int(cnt) >= max(3, height // 2)]
    if not cols or cols[0] != 0 or cols[-1] != int(grid.shape[1]) - 1:
        raise AssertionError("failed to identify vertical wall lines")
    return cols


def _extract_object(room_view: np.ndarray) -> np.ndarray | None:
    mask = (room_view != BLACK) & (room_view != WALL)
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    patch = np.array(room_view[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)
    values = np.unique(patch)
    if len(values) != 1 or int(values[0]) in (BLACK, WALL):
        raise AssertionError("objects must be solid monochromatic rectangles")
    return patch


def _build_room_model(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    wall_rows = _detect_wall_rows(g)
    wall_cols = _detect_wall_cols(g)

    row_count = len(wall_rows) - 1
    col_count = len(wall_cols) - 1
    rooms: dict[object, dict[str, object]] = {}
    room_of: dict[tuple[int, int], int] = {}

    rid = 0
    for rr in range(row_count):
        for cc in range(col_count):
            top = wall_rows[rr] + 1
            bottom = wall_rows[rr + 1] - 1
            left = wall_cols[cc] + 1
            right = wall_cols[cc + 1] - 1
            if top > bottom or left > right:
                raise AssertionError("degenerate room interior")
            room_of[(rr, cc)] = rid
            rooms[rid] = {
                "cell": (rr, cc),
                "bounds": (top, bottom, left, right),
                "anchor": (top, left),
                "object": _extract_object(g[top : bottom + 1, left : right + 1]),
            }
            rid += 1

    neighbors: dict[object, list[object]] = {rid: [] for rid in rooms}
    for rr in range(row_count):
        for cc in range(col_count):
            room_id = room_of[(rr, cc)]
            if rr + 1 < row_count:
                wall_r = wall_rows[rr + 1]
                left = wall_cols[cc] + 1
                right = wall_cols[cc + 1]
                if _gap_run(g[wall_r, left:right]) is not None:
                    other = room_of[(rr + 1, cc)]
                    neighbors[room_id].append(other)
                    neighbors[other].append(room_id)
            if cc + 1 < col_count:
                wall_c = wall_cols[cc + 1]
                top = wall_rows[rr] + 1
                bottom = wall_rows[rr + 1]
                if _gap_run(g[top:bottom, wall_c]) is not None:
                    other = room_of[(rr, cc + 1)]
                    neighbors[room_id].append(other)
                    neighbors[other].append(room_id)

    edge_count = sum(len(v) for v in neighbors.values()) // 2
    if edge_count != len(rooms) - 1:
        raise AssertionError("room graph must be a tree")
    return rooms, neighbors


def _simulate_flow(rooms, neighbors, root):
    order_keys = sorted(rooms, key=lambda rid: rooms[rid]["anchor"])
    payloads = {
        rid: ([] if rooms[rid]["object"] is None else [rooms[rid]["object"]]) for rid in rooms
    }
    closed = {rid: set() for rid in rooms}
    moved: set[object] = set()
    total_objects = sum(len(items) for items in payloads.values())

    while True:
        ready = []
        for rid in order_keys:
            if rid == root or rid in moved or not payloads[rid]:
                continue
            open_neighbors = [nbr for nbr in neighbors[rid] if nbr not in closed[rid]]
            if len(open_neighbors) == 1:
                ready.append(rid)
        if not ready:
            break
        for rid in ready:
            if rid in moved or not payloads[rid]:
                continue
            open_neighbors = [nbr for nbr in neighbors[rid] if nbr not in closed[rid]]
            if len(open_neighbors) != 1:
                continue
            dst = open_neighbors[0]
            payloads[dst].extend(payloads[rid])
            payloads[rid] = []
            moved.add(rid)
            closed[dst].add(rid)

    if len(moved) != len(rooms) - 1:
        return None
    if any(payloads[rid] for rid in rooms if rid != root):
        return None
    if len(payloads[root]) != total_objects:
        return None
    return payloads[root]


def _choose_root_and_order(rooms, neighbors):
    ordered = sorted(rooms, key=lambda rid: rooms[rid]["anchor"])
    empties = [rid for rid in ordered if rooms[rid]["object"] is None]
    empties.sort(key=lambda rid: (-len(neighbors[rid]), rooms[rid]["anchor"]))

    for pool in (empties, ordered):
        for rid in pool:
            final_order = _simulate_flow(rooms, neighbors, rid)
            if final_order is not None:
                return rid, final_order
    raise AssertionError("no valid terminal room found")


def _shelf_pack(patches: list[np.ndarray], height: int, width: int):
    placements: list[tuple[int, int]] = []
    row = 0
    col = 0
    shelf_h = 0
    for patch in patches:
        h, w = map(int, patch.shape)
        if h > height or w > width:
            return None
        if col + w > width:
            row += shelf_h
            col = 0
            shelf_h = 0
        if row + h > height:
            return None
        placements.append((row, col))
        col += w
        shelf_h = max(shelf_h, h)
    return placements


def _render_solution(grid: np.ndarray, rooms, neighbors, root, final_order=None) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    if final_order is None:
        final_order = _simulate_flow(rooms, neighbors, root)
    if final_order is None:
        raise AssertionError("room flow does not resolve to the chosen root")

    mask = (g != BLACK) & (g != WALL)
    g[mask] = BLACK

    top, bottom, left, right = rooms[root]["bounds"]
    height = bottom - top + 1
    width = right - left + 1
    placements = _shelf_pack(final_order, height, width)
    if placements is None:
        raise AssertionError("terminal room is too small for the packed objects")

    for patch, (rr, cc) in zip(final_order, placements):
        h, w = map(int, patch.shape)
        g[top + rr : top + rr + h, left + cc : left + cc + w] = patch
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each room's object toward the unique empty terminal room through tree-like door closures.
    """
    g = np.asarray(grid, dtype=int)
    rooms, neighbors = _build_room_model(g)
    root, final_order = _choose_root_and_order(rooms, neighbors)
    return _render_solution(g, rooms, neighbors, root, final_order)


def _cell_grid_neighbors(rows: int, cols: int):
    neighbors = {}
    for rr in range(rows):
        for cc in range(cols):
            cell = (rr, cc)
            links = []
            if rr > 0:
                links.append((rr - 1, cc))
            if rr + 1 < rows:
                links.append((rr + 1, cc))
            if cc > 0:
                links.append((rr, cc - 1))
            if cc + 1 < cols:
                links.append((rr, cc + 1))
            neighbors[cell] = links
    return neighbors


def _edges_to_neighbors(cells, edges):
    neighbors = {cell: [] for cell in cells}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    return neighbors


def _snake_path_edges(rows: int, cols: int, rng: random.Random):
    cells = []
    if rng.random() < 0.5:
        for rr in range(rows):
            seq = range(cols) if rr % 2 == 0 else range(cols - 1, -1, -1)
            for cc in seq:
                cells.append((rr, cc))
    else:
        for cc in range(cols):
            seq = range(rows) if cc % 2 == 0 else range(rows - 1, -1, -1)
            for rr in seq:
                cells.append((rr, cc))
    if rng.random() < 0.5:
        cells.reverse()
    return [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]


def _branchy_tree_edges(rows: int, cols: int, rng: random.Random):
    grid_neighbors = _cell_grid_neighbors(rows, cols)
    cells = list(grid_neighbors)
    start = rng.choice(cells)
    visited = {start}
    edges = []
    frontier = [(start, nbr) for nbr in grid_neighbors[start]]

    while len(visited) < len(cells):
        idx = rng.randrange(len(frontier))
        a, b = frontier.pop(idx)
        if b in visited:
            continue
        visited.add(b)
        edges.append((a, b))
        for nbr in grid_neighbors[b]:
            if nbr not in visited:
                frontier.append((b, nbr))
    return edges


def _choose_root(mode: str, neighbors, rng: random.Random):
    degrees = {cell: len(nbrs) for cell, nbrs in neighbors.items()}
    if mode == "path":
        candidates = [cell for cell, deg in degrees.items() if deg == 2]
        if not candidates:
            candidates = list(neighbors)
    else:
        best = max(degrees.values())
        candidates = [cell for cell, deg in degrees.items() if deg == best]
    return rng.choice(candidates)


def _root_parent_map(neighbors, root):
    parent = {root: None}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nbr in neighbors[cur]:
            if nbr in parent:
                continue
            parent[nbr] = cur
            stack.append(nbr)
    return parent


def _wall_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    for size in sizes:
        positions.append(positions[-1] + size + 1)
    return positions


def _room_bounds(row_heights: list[int], col_widths: list[int]):
    wall_rows = _wall_positions(row_heights)
    wall_cols = _wall_positions(col_widths)
    bounds = {}
    for rr in range(len(row_heights)):
        for cc in range(len(col_widths)):
            bounds[(rr, cc)] = (
                wall_rows[rr] + 1,
                wall_rows[rr + 1] - 1,
                wall_cols[cc] + 1,
                wall_cols[cc + 1] - 1,
            )
    return wall_rows, wall_cols, bounds


def _carve_doors(grid: np.ndarray, wall_rows, wall_cols, edges, rng: random.Random):
    door_meta = {}
    for a, b in edges:
        ra, ca = a
        rb, cb = b
        if abs(ra - rb) + abs(ca - cb) != 1:
            raise AssertionError("tree edges must connect adjacent rooms")
        if ra == rb:
            left_cell, right_cell = (a, b) if ca < cb else (b, a)
            rr, cc = left_cell
            wall_c = wall_cols[cc + 1]
            top = wall_rows[rr] + 1
            bottom = wall_rows[rr + 1] - 1
            span = bottom - top + 1
            width = 1 if span == 1 else rng.choice((1, 2))
            width = min(width, span)
            start = rng.randint(top, bottom - width + 1)
            grid[start : start + width, wall_c] = BLACK
            door_meta[(left_cell, right_cell)] = {"side": "R", "wall": wall_c, "start": start, "width": width}
            door_meta[(right_cell, left_cell)] = {"side": "L", "wall": wall_c, "start": start, "width": width}
        else:
            upper_cell, lower_cell = (a, b) if ra < rb else (b, a)
            rr, cc = upper_cell
            wall_r = wall_rows[rr + 1]
            left = wall_cols[cc] + 1
            right = wall_cols[cc + 1] - 1
            span = right - left + 1
            width = 1 if span == 1 else rng.choice((1, 2))
            width = min(width, span)
            start = rng.randint(left, right - width + 1)
            grid[wall_r, start : start + width] = BLACK
            door_meta[(upper_cell, lower_cell)] = {"side": "D", "wall": wall_r, "start": start, "width": width}
            door_meta[(lower_cell, upper_cell)] = {"side": "U", "wall": wall_r, "start": start, "width": width}
    return door_meta


def _anchor_to_parent(bounds, patch: np.ndarray, door):
    top, bottom, left, right = bounds
    h, w = map(int, patch.shape)
    side = door["side"]
    center = door["start"] + (door["width"] - 1) / 2.0

    if side == "U":
        rr = top
        cc = int(round(center - (w - 1) / 2.0))
        cc = min(max(cc, left), right - w + 1)
    elif side == "D":
        rr = bottom - h + 1
        cc = int(round(center - (w - 1) / 2.0))
        cc = min(max(cc, left), right - w + 1)
    elif side == "L":
        cc = left
        rr = int(round(center - (h - 1) / 2.0))
        rr = min(max(rr, top), bottom - h + 1)
    else:
        cc = right - w + 1
        rr = int(round(center - (h - 1) / 2.0))
        rr = min(max(rr, top), bottom - h + 1)
    return rr, cc


def _sample_layout(mode: str, rng: random.Random):
    for _ in range(100):
        rows = rng.randint(*ROOM_ROWS_RANGE)
        cols = rng.randint(*ROOM_COLS_RANGE)
        if rows * cols > 9:
            continue
        if mode == "branch" and rows * cols < 5:
            continue
        return rows, cols
    raise RuntimeError("failed to sample a room grid size")


def _gen_attempt(rng: random.Random):
    mode = "branch" if rng.random() < 0.5 else "path"
    rows, cols = _sample_layout(mode, rng)
    cells = [(rr, cc) for rr in range(rows) for cc in range(cols)]

    if mode == "path":
        edges = _snake_path_edges(rows, cols, rng)
    else:
        edges = _branchy_tree_edges(rows, cols, rng)

    neighbors = _edges_to_neighbors(cells, edges)
    degrees = [len(neighbors[cell]) for cell in cells]
    if mode == "branch" and max(degrees) < 3:
        return None

    root = _choose_root(mode, neighbors, rng)
    parent = _root_parent_map(neighbors, root)

    row_heights = [rng.randint(*BASE_ROOM_RANGE) for _ in range(rows)]
    col_widths = [rng.randint(*BASE_ROOM_RANGE) for _ in range(cols)]
    row_heights[root[0]] = rng.randint(*ROOT_ROOM_RANGE)
    col_widths[root[1]] = rng.randint(*ROOT_ROOM_RANGE)
    wall_rows, wall_cols, bounds = _room_bounds(row_heights, col_widths)

    height = wall_rows[-1] + 1
    width = wall_cols[-1] + 1
    if height > 30 or width > 30:
        return None

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    rooms = {}
    for cell in cells:
        rooms[cell] = {
            "cell": cell,
            "bounds": bounds[cell],
            "anchor": cell,
            "object": None,
        }
    for idx, cell in enumerate(cell for cell in cells if cell != root):
        patch = np.full(
            (rng.randint(1, 3), rng.randint(1, 3)),
            colors[idx],
            dtype=int,
        )
        rooms[cell]["object"] = patch

    final_order = _simulate_flow(rooms, neighbors, root)
    if final_order is None:
        return None
    root_top, root_bottom, root_left, root_right = bounds[root]
    if _shelf_pack(final_order, root_bottom - root_top + 1, root_right - root_left + 1) is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL
    door_meta = _carve_doors(grid, wall_rows, wall_cols, edges, rng)

    for cell in cells:
        if cell == root:
            continue
        patch = rooms[cell]["object"]
        door = door_meta[(cell, parent[cell])]
        rr, cc = _anchor_to_parent(bounds[cell], patch, door)
        h, w = map(int, patch.shape)
        grid[rr : rr + h, cc : cc + w] = patch

    expected = _render_solution(grid, rooms, neighbors, root, final_order)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-flow input")
