from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
SEED = 2
GEN_RETRIES = 400
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (2, 5)
SIDE_MARGIN_RANGE = (1, 4)
EXTRA_EDGE_PROB_RANGE = (0.05, 0.35)
CYCLE_REST_COLORS = (1, 3, 4, 6, 7, 8, 9)


def _legend_segments(values: np.ndarray) -> list[tuple[int, int, list[int]]]:
    segments: list[tuple[int, int, list[int]]] = []
    start: int | None = None
    for idx, value in enumerate(values.tolist()):
        if int(value) not in (BG, WALL):
            if start is None:
                start = idx
        elif start is not None:
            colors = [int(v) for v in values[start:idx].tolist()]
            if 3 <= len(colors) <= 4 and len(colors) == len(set(colors)):
                segments.append((start, idx, colors))
            start = None
    if start is not None:
        colors = [int(v) for v in values[start:].tolist()]
        if 3 <= len(colors) <= 4 and len(colors) == len(set(colors)):
            segments.append((start, len(values), colors))
    return segments


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int, list[int]]] = []
    for side, values in (
        ("top", grid[0, :]),
        ("bottom", grid[h - 1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, w - 1]),
    ):
        for start, end, colors in _legend_segments(values):
            candidates.append((side, start, end, colors))

    assert candidates, "expected a unique edge legend strip"
    best_len = max(len(colors) for _, _, _, colors in candidates)
    best = [cand for cand in candidates if len(cand[3]) == best_len]
    assert len(best) == 1, "legend strip must be unique"
    side, start, end, colors = best[0]

    mask = np.zeros_like(grid, dtype=bool)
    if side == "top":
        mask[0, start:end] = True
    elif side == "bottom":
        mask[h - 1, start:end] = True
    elif side == "left":
        mask[start:end, 0] = True
    else:
        mask[start:end, w - 1] = True
    return colors, mask


def _split_wall_lines(counts: np.ndarray) -> list[int]:
    distinct = sorted({int(v) for v in counts.tolist()})
    assert len(distinct) >= 2, "wall lines must be separable from room rows/cols"
    gap_idx = max(range(len(distinct) - 1), key=lambda i: distinct[i + 1] - distinct[i])
    threshold = (distinct[gap_idx] + distinct[gap_idx + 1]) / 2.0
    return [idx for idx, value in enumerate(counts.tolist()) if float(value) > threshold]


def _recover_room_lines(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[list[int], list[int]]:
    wall_mask = (grid == WALL) & (~legend_mask)
    coords = np.argwhere(wall_mask)
    assert len(coords) > 0, "expected grey wall pixels"
    r0, c0 = coords.min(axis=0).tolist()
    r1, c1 = coords.max(axis=0).tolist()

    row_counts = wall_mask[r0 : r1 + 1, c0 : c1 + 1].sum(axis=1)
    col_counts = wall_mask[r0 : r1 + 1, c0 : c1 + 1].sum(axis=0)
    row_offsets = _split_wall_lines(row_counts)
    col_offsets = _split_wall_lines(col_counts)

    wall_rows = [r0 + off for off in row_offsets]
    wall_cols = [c0 + off for off in col_offsets]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "need at least one rectangular room"
    assert all(b - a >= 2 for a, b in zip(wall_rows, wall_rows[1:], strict=False))
    assert all(b - a >= 2 for a, b in zip(wall_cols, wall_cols[1:], strict=False))
    return wall_rows, wall_cols


def _rooms_from_lines(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr in range(len(wall_rows) - 1):
        for cc in range(len(wall_cols) - 1):
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1] - 1
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            assert r0 <= r1 and c0 <= c1, "each room must have a non-empty floor rectangle"
            rooms[(rr, cc)] = (r0, c0, r1, c1)
    return rooms


def _room_graph(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {
        (r, c): set() for r in range(room_rows) for c in range(room_cols)
    }

    for wall_r_idx in range(1, len(wall_rows) - 1):
        wall_r = wall_rows[wall_r_idx]
        upper = wall_r_idx - 1
        lower = wall_r_idx
        for cc in range(room_cols):
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            if np.any(grid[wall_r, c0 : c1 + 1] == BG):
                a = (upper, cc)
                b = (lower, cc)
                graph[a].add(b)
                graph[b].add(a)

    for wall_c_idx in range(1, len(wall_cols) - 1):
        wall_c = wall_cols[wall_c_idx]
        left = wall_c_idx - 1
        right = wall_c_idx
        for rr in range(room_rows):
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1] - 1
            if np.any(grid[r0 : r1 + 1, wall_c] == BG):
                a = (rr, left)
                b = (rr, right)
                graph[a].add(b)
                graph[b].add(a)

    return graph


def _find_seed_room(grid: np.ndarray, rooms: dict[tuple[int, int], tuple[int, int, int, int]], legend_mask: np.ndarray) -> tuple[int, int]:
    seed_cells = np.argwhere((grid == SEED) & (~legend_mask))
    assert len(seed_cells) > 0, "expected a red seed square"
    seed_room: tuple[int, int] | None = None
    for room, (r0, c0, r1, c1) in rooms.items():
        inside = (
            (seed_cells[:, 0] >= r0)
            & (seed_cells[:, 0] <= r1)
            & (seed_cells[:, 1] >= c0)
            & (seed_cells[:, 1] <= c1)
        )
        if np.any(inside):
            if seed_room is None:
                seed_room = room
            else:
                assert seed_room == room, "seed square must lie in a single room"
    assert seed_room is not None, "seed square must belong to one room"
    return seed_room


def _bfs_distances(graph: dict[tuple[int, int], set[tuple[int, int]]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-color the room network from the seed room, advancing one step in the edge legend cycle per doorway crossed.

    Input structure:
    - Black background pixels form rectangular room floors inside a grey wall network with 1-cell-thick walls.
    - Interior walls may contain single-cell door gaps that connect neighboring rooms without changing the room rectangles.
    - Exactly one room contains a small red seed square, and a separate edge legend strip lists 3-4 distinct colors in cycle order.

    Rule:
    - Read the edge legend strip to obtain the repeating room-color cycle.
    - Recover the room grid from the grey wall rows and columns, then detect which room pairs are linked by a door gap on their shared wall segment.
    - Starting from the room that contains the red seed square, assign each reachable room the legend color at index equal to its doorway distance from the seed room modulo the cycle length.
    - Recolor only room-floor pixels; keep walls, door-gap pixels, the legend strip, and all untouched background outside the room network unchanged.

    Key insight:
    - Door gaps merge floor pixels geometrically, so the task is not ordinary flood fill on black cells; you must first reconstruct the hidden room graph from the grey wall lattice and then color rooms by graph distance.
    """
    g = np.asarray(grid, dtype=int)
    cycle, legend_mask = _find_legend(g)
    wall_rows, wall_cols = _recover_room_lines(g, legend_mask)
    rooms = _rooms_from_lines(wall_rows, wall_cols)
    graph = _room_graph(g, wall_rows, wall_cols)
    seed_room = _find_seed_room(g, rooms, legend_mask)
    distances = _bfs_distances(graph, seed_room)

    out = g.copy()
    for room, distance in distances.items():
        r0, c0, r1, c1 = rooms[room]
        out[r0 : r1 + 1, c0 : c1 + 1] = cycle[distance % len(cycle)]
    return out


def _all_room_edges(room_rows: int, room_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            if rr + 1 < room_rows:
                edges.append(((rr, cc), (rr + 1, cc)))
            if cc + 1 < room_cols:
                edges.append(((rr, cc), (rr, cc + 1)))
    return edges


def _choose_connected_edges(room_rows: int, room_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    nodes = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    parent = {node: node for node in nodes}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    edges = _all_room_edges(room_rows, room_cols)
    random.shuffle(edges)

    chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for edge in edges:
        if union(*edge):
            chosen.append(edge)

    extra_prob = random.uniform(*EXTRA_EDGE_PROB_RANGE)
    for edge in edges:
        if edge in chosen:
            continue
        if random.random() < extra_prob:
            chosen.append(edge)
    return chosen


def _graph_from_edges(
    room_rows: int,
    room_cols: int,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {
        (r, c): set() for r in range(room_rows) for c in range(room_cols)
    }
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _choose_seed_room(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    cycle_len: int,
) -> tuple[tuple[int, int], dict[tuple[int, int], int]] | None:
    nodes = list(graph)
    want_wrap = len(nodes) >= cycle_len + 1 and random.random() < 0.5
    min_distance = cycle_len if want_wrap else min(2, len(nodes) - 1)

    candidates: list[tuple[tuple[int, int], dict[tuple[int, int], int]]] = []
    for node in nodes:
        distances = _bfs_distances(graph, node)
        if len(distances) != len(nodes):
            return None
        if max(distances.values()) >= min_distance:
            candidates.append((node, distances))
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_margins(network_size: int, leading_min: int, trailing_min: int) -> tuple[int, int]:
    slack = 30 - network_size - leading_min - trailing_min
    if slack < 0:
        raise ValueError("network is too large for ARC bounds")
    leading = leading_min + random.randint(0, min(4, slack))
    slack -= leading - leading_min
    trailing = trailing_min + random.randint(0, min(4, slack))
    return leading, trailing


def _build_layout():
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    network_h = sum(room_heights) + room_rows + 1
    network_w = sum(room_widths) + room_cols + 1

    legend_side = random.choice(["top", "bottom"])
    top_min, bottom_min = (2, 1) if legend_side == "top" else (1, 2)
    top_margin, bottom_margin = _pick_margins(network_h, top_min, bottom_min)
    left_margin, right_margin = _pick_margins(network_w, 1, 1)
    height = network_h + top_margin + bottom_margin
    width = network_w + left_margin + right_margin

    cycle_len = random.choice([3, 4])
    cycle = [SEED] + random.sample(list(CYCLE_REST_COLORS), cycle_len - 1)

    return {
        "room_rows": room_rows,
        "room_cols": room_cols,
        "room_heights": room_heights,
        "room_widths": room_widths,
        "top_margin": top_margin,
        "left_margin": left_margin,
        "height": height,
        "width": width,
        "legend_side": legend_side,
        "cycle": cycle,
    }


def _draw_walls(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> None:
    for rr in wall_rows:
        grid[rr, wall_cols[0] : wall_cols[-1] + 1] = WALL
    for cc in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, cc] = WALL


def _room_bounds(
    top_margin: int,
    left_margin: int,
    room_heights: list[int],
    room_widths: list[int],
) -> tuple[list[int], list[int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    wall_rows = [top_margin]
    for size in room_heights:
        wall_rows.append(wall_rows[-1] + size + 1)
    wall_cols = [left_margin]
    for size in room_widths:
        wall_cols.append(wall_cols[-1] + size + 1)
    rooms = _rooms_from_lines(wall_rows, wall_cols)
    return wall_rows, wall_cols, rooms


def _carve_doors(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    for a, b in edges:
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            rr = r1
            wall_c = wall_cols[max(c1, c2)]
            door_r = random.randint(wall_rows[rr] + 1, wall_rows[rr + 1] - 1)
            grid[door_r, wall_c] = BG
        else:
            cc = c1
            wall_r = wall_rows[max(r1, r2)]
            door_c = random.randint(wall_cols[cc] + 1, wall_cols[cc + 1] - 1)
            grid[wall_r, door_c] = BG


def _place_seed(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    seed_room: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = rooms[seed_room]
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    square = random.randint(1, min(2, room_h, room_w))
    sr = random.randint(r0, r1 - square + 1)
    sc = random.randint(c0, c1 - square + 1)
    grid[sr : sr + square, sc : sc + square] = SEED


def _place_legend(grid: np.ndarray, cycle: list[int], side: str) -> None:
    length = len(cycle)
    if side == "top":
        row = 0
        start = random.randint(0, grid.shape[1] - length)
        grid[row, start : start + length] = cycle
    else:
        row = grid.shape[0] - 1
        start = random.randint(0, grid.shape[1] - length)
        grid[row, start : start + length] = cycle


def _expected_output(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    distances: dict[tuple[int, int], int],
    cycle: list[int],
) -> np.ndarray:
    out = grid.copy()
    for room, distance in distances.items():
        r0, c0, r1, c1 = rooms[room]
        out[r0 : r1 + 1, c0 : c1 + 1] = cycle[distance % len(cycle)]
    return out


def _gen_attempt() -> np.ndarray | None:
    layout = _build_layout()
    grid = np.full((layout["height"], layout["width"]), BG, dtype=int)

    wall_rows, wall_cols, rooms = _room_bounds(
        layout["top_margin"],
        layout["left_margin"],
        layout["room_heights"],
        layout["room_widths"],
    )
    _draw_walls(grid, wall_rows, wall_cols)

    edges = _choose_connected_edges(layout["room_rows"], layout["room_cols"])
    graph = _graph_from_edges(layout["room_rows"], layout["room_cols"], edges)
    seed_choice = _choose_seed_room(graph, len(layout["cycle"]))
    if seed_choice is None:
        return None
    seed_room, distances = seed_choice

    _carve_doors(grid, wall_rows, wall_cols, edges)
    _place_seed(grid, rooms, seed_room)
    _place_legend(grid, layout["cycle"], layout["legend_side"])

    expected = _expected_output(grid, rooms, distances, layout["cycle"])
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-flood sample")
