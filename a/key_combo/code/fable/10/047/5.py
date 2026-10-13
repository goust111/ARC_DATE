from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_ATTEMPTS = 400

ROUTE_TEMPLATES = [
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    [(0, 1), (0, 2), (1, 2), (1, 1), (2, 1)],
]

DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _edge_background(grid: np.ndarray) -> int:
    edge = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    return int(Counter(map(int, edge.tolist())).most_common(1)[0][0])


def _distances(passable: np.ndarray, sources: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in sources:
        rr = int(r)
        cc = int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _count_shortest_paths(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int]:
    dist = _distances(passable, np.array([start], dtype=int))
    gr, gc = goal
    shortest = int(dist[gr, gc])
    if shortest < 0:
        return -1, 0
    counts = np.zeros_like(dist, dtype=np.int64)
    sr, sc = start
    counts[sr, sc] = 1
    layers: list[list[tuple[int, int]]] = [[] for _ in range(shortest + 1)]
    for r, c in np.argwhere(dist >= 0):
        d = int(dist[r, c])
        if d <= shortest:
            layers[d].append((int(r), int(c)))
    for d in range(1, shortest + 1):
        for r, c in layers[d]:
            total = 0
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < dist.shape[0] and 0 <= nc < dist.shape[1] and dist[nr, nc] == d - 1:
                    total += int(counts[nr, nc])
            counts[r, c] = total
    return shortest, int(counts[gr, gc])


def _reconstruct_path(passable: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list[tuple[int, int]]:
    dist = _distances(passable, start)
    goal_cells = [tuple(map(int, cell)) for cell in goal.tolist()]
    reachable = [cell for cell in goal_cells if dist[cell] >= 0]
    assert reachable
    end = min(reachable, key=lambda cell: (int(dist[cell]), cell[0], cell[1]))
    path = [end]
    current = end
    while dist[current] > 0:
        prevs = []
        for nr, nc in _neighbors4(*current):
            if 0 <= nr < dist.shape[0] and 0 <= nc < dist.shape[1] and dist[nr, nc] == dist[current] - 1:
                prevs.append((nr, nc))
        assert prevs
        current = min(prevs)
        path.append(current)
    path.reverse()
    return path


def _parse_linear_legend(grid: np.ndarray, coords: np.ndarray) -> tuple[list[int], np.ndarray]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == 1 or w == 1
    mask = np.zeros_like(grid, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    if h == 1:
        cols = list(range(c0, c1 + 1))
        assert all(mask[r0, c] for c in cols)
        order = [(r0, c) for c in cols]
    else:
        rows = list(range(r0, r1 + 1))
        assert all(mask[r, c0] for r in rows)
        order = [(r, c0) for r in rows]
    return [int(grid[r, c]) for r, c in order], np.array(order, dtype=int)


def _find_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    h, w = grid.shape
    rooms = []
    seen_boxes = set()
    max_n = min(6, h - 2, w - 2)
    for n in range(2, max_n + 1):
        perim = 4 * (n + 2) - 4
        for r in range(1, h - n - 1):
            for c in range(1, w - n - 1):
                interior = grid[r : r + n, c : c + n]
                if np.any(interior == wall):
                    continue
                top = grid[r - 1, c - 1 : c + n + 1]
                bottom = grid[r + n, c - 1 : c + n + 1]
                left = grid[r - 1 : r + n + 1, c - 1]
                right = grid[r - 1 : r + n + 1, c + n]
                corners = (
                    int(grid[r - 1, c - 1]) == wall
                    and int(grid[r - 1, c + n]) == wall
                    and int(grid[r + n, c - 1]) == wall
                    and int(grid[r + n, c + n]) == wall
                )
                if not corners:
                    continue
                side_counts = [
                    int(np.count_nonzero(top == wall)),
                    int(np.count_nonzero(bottom == wall)),
                    int(np.count_nonzero(left == wall)),
                    int(np.count_nonzero(right == wall)),
                ]
                if min(side_counts) < n + 1:
                    continue
                ring = np.concatenate([top, bottom, left[1:-1], right[1:-1]])
                if int(np.count_nonzero(ring == wall)) < perim - 2:
                    continue
                box = (r, c, n)
                if box in seen_boxes:
                    continue
                seen_boxes.add(box)
                mask = np.zeros_like(grid, dtype=bool)
                mask[r : r + n, c : c + n] = True
                rooms.append({"r": r, "c": c, "n": n, "mask": mask})
    return rooms


def _room_sequence_along_path(path: list[tuple[int, int]], rooms: list[dict]) -> list[int]:
    room_hits = []
    last_room = None
    for r, c in path:
        room_id = None
        for idx, room in enumerate(rooms):
            if room["mask"][r, c]:
                room_id = idx
                break
        if room_id is not None and room_id != last_room:
            room_hits.append(room_id)
        last_room = room_id
    return room_hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the framed rooms visited by the unique shortest endpoint path according to the ordered legend strip.

    Input structure:
    - The grid contains a background, one wall color forming corridors and square framed rooms, and a single legend strip attached to one outer edge.
    - Two different-colored solid endpoint blocks lie at the two ends of the corridor system.
    - The legend strip is a straight 1-cell-thick bar; its first cell matches exactly one endpoint color, and its remaining cells correspond to the rooms traversed by the shortest path.

    Rule:
    - Treat wall cells as blocked and all other cells as passable.
    - Find the unique shortest 4-connected path between the two endpoint components.
    - Start from the endpoint whose color matches the first legend cell, read the legend from its visible first cell toward the other end, and recolor each traversed room interior in first-visit order with legend cells 2, 3, ...
    - Leave every non-traversed room, all corridors, the walls, both endpoints, and the legend unchanged.

    Key insight:
    - The legend is not a local color key: it is a path-order sequence that must be consumed along the shortest-path traversal.
    - The first legend color determines which endpoint is the start, so reversing the reading direction reverses the room colors and gives the wrong answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _edge_background(g)
    non_bg = [int(c) for c in np.unique(g) if int(c) != bg]
    assert non_bg
    wall = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    fg_mask = (g != bg) & (g != wall)
    comps = _mask_components(fg_mask)
    assert len(comps) == 3

    legend_coords = None
    endpoint_coords = []
    endpoint_colors = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        colors = {int(g[r, c]) for r, c in coords}
        touches_edge = r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1
        if touches_edge and (h == 1 or w == 1) and coords.shape[0] == max(h, w):
            assert legend_coords is None
            legend_coords = coords
            continue
        assert len(colors) == 1
        endpoint_coords.append(coords)
        endpoint_colors.append(next(iter(colors)))

    assert legend_coords is not None
    assert len(endpoint_coords) == 2

    legend, legend_order = _parse_linear_legend(g, legend_coords)
    assert len(legend) >= 2
    assert endpoint_colors[0] != endpoint_colors[1]
    assert legend[0] in endpoint_colors

    if endpoint_colors[0] == legend[0]:
        start_idx, goal_idx = 0, 1
    else:
        start_idx, goal_idx = 1, 0

    passable = g != wall
    path = _reconstruct_path(passable, endpoint_coords[start_idx], endpoint_coords[goal_idx])
    rooms = _find_rooms(g, bg, wall)
    route_room_ids = _room_sequence_along_path(path, rooms)
    assert len(route_room_ids) == len(legend) - 1

    endpoint_mask = np.zeros_like(g, dtype=bool)
    for coords in endpoint_coords:
        endpoint_mask[coords[:, 0], coords[:, 1]] = True

    out = g.copy()
    for color, room_id in zip(legend[1:], route_room_ids):
        room_mask = rooms[room_id]["mask"] & ~endpoint_mask
        out[room_mask] = int(color)
    return out


def _apply_transform(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(points, dtype=int)
    if random.random() < 0.5:
        arr[:, 1] *= -1
    for _ in range(random.randrange(4)):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    arr -= arr.min(axis=0)
    return [tuple(map(int, row)) for row in arr.tolist()]


def _room_top_left(lr: int, lc: int, stride: int, margin: int) -> tuple[int, int]:
    return margin + lr * stride, margin + lc * stride


def _room_outer_box(room: dict) -> tuple[int, int, int, int]:
    r = int(room["r"])
    c = int(room["c"])
    s = int(room["n"])
    return r - 1, c - 1, r + s, c + s


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _door(room: dict, side: str) -> tuple[int, int]:
    r = int(room["r"])
    c = int(room["c"])
    s = int(room["n"])
    mid_r = r + s // 2
    mid_c = c + s // 2
    if side == "N":
        return r - 1, mid_c
    if side == "S":
        return r + s, mid_c
    if side == "W":
        return mid_r, c - 1
    return mid_r, c + s


def _connect_side(a: tuple[int, int], b: tuple[int, int]) -> str:
    if a[0] == b[0]:
        return "E" if a[1] < b[1] else "W"
    return "S" if a[0] < b[0] else "N"


def _carve_line(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    pts = _line_points(start, end)
    for r, c in pts:
        passable[r, c] = True
    return pts


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    pts = []
    if sr == er:
        step = 1 if sc <= ec else -1
        for c in range(sc, ec + step, step):
            pts.append((sr, c))
    else:
        step = 1 if sr <= er else -1
        for r in range(sr, er + step, step):
            pts.append((r, sc))
    return pts


def _free_sides(logical: dict[tuple[int, int], int], pos: tuple[int, int]) -> list[str]:
    lr, lc = pos
    free = []
    for side, (dr, dc) in DIRS.items():
        if (lr + dr, lc + dc) not in logical:
            free.append(side)
    return free


def _pick_endpoint_side(logical: dict[tuple[int, int], int], pos: tuple[int, int], blocked: set[str]) -> str:
    choices = [side for side in _free_sides(logical, pos) if side not in blocked]
    assert choices
    return random.choice(choices)


def _attempt_case() -> np.ndarray | None:
    room_size = random.choice([2, 3])
    outer = room_size + 2
    stride = outer + 3
    margin = 4

    route_logical = _apply_transform(random.choice(ROUTE_TEMPLATES))
    logical_index = {pos: i for i, pos in enumerate(route_logical)}
    route_rooms = []
    for lr, lc in route_logical:
        r, c = _room_top_left(lr, lc, stride, margin)
        route_rooms.append({"r": r, "c": c, "n": room_size})

    used_sides = [set() for _ in route_rooms]
    corridor_segments = []
    passable_pts: set[tuple[int, int]] = set()
    all_rooms = [dict(room) for room in route_rooms]

    def add_room(room: dict):
        for rr in range(room["r"], room["r"] + room["n"]):
            for cc in range(room["c"], room["c"] + room["n"]):
                passable_pts.add((rr, cc))

    for room in all_rooms:
        add_room(room)

    for i in range(len(route_rooms) - 1):
        a_log = route_logical[i]
        b_log = route_logical[i + 1]
        side_a = _connect_side(a_log, b_log)
        side_b = OPPOSITE[side_a]
        used_sides[i].add(side_a)
        used_sides[i + 1].add(side_b)
        da = _door(route_rooms[i], side_a)
        db = _door(route_rooms[i + 1], side_b)
        seg = _line_points(da, db)
        corridor_segments.append({"index": i, "a": da, "b": db, "side": side_a, "cells": seg})
        passable_pts.update(seg)

    start_side = _pick_endpoint_side(logical_index, route_logical[0], used_sides[0])
    end_side = _pick_endpoint_side(logical_index, route_logical[-1], used_sides[-1])
    endpoint_len_a = random.randint(2, 3)
    endpoint_len_b = random.randint(2, 3)

    endpoint_corridors = []
    endpoint_cells = []
    for room, side, length in (
        (route_rooms[0], start_side, endpoint_len_a),
        (route_rooms[-1], end_side, endpoint_len_b),
    ):
        dr, dc = DIRS[side]
        door = _door(room, side)
        end = (door[0] + dr * length, door[1] + dc * length)
        pts = []
        if door[0] == end[0]:
            step = 1 if door[1] <= end[1] else -1
            for c in range(door[1], end[1] + step, step):
                pts.append((door[0], c))
        else:
            step = 1 if door[0] <= end[0] else -1
            for r in range(door[0], end[0] + step, step):
                pts.append((r, door[1]))
        endpoint_corridors.append(pts)
        endpoint_cells.append(end)
        passable_pts.update(pts)

    candidate_dead = []
    for seg in corridor_segments:
        if len(seg["cells"]) < 3:
            continue
        junction = seg["cells"][len(seg["cells"]) // 2]
        if seg["side"] in {"E", "W"}:
            jr, jc = junction
            for sign in (-1, 1):
                if sign < 0:
                    room = {"r": jr - room_size - 1, "c": jc - room_size // 2, "n": room_size}
                    door_side = "S"
                else:
                    room = {"r": jr + 3, "c": jc - room_size // 2, "n": room_size}
                    door_side = "N"
                box = _room_outer_box(room)
                if any(_boxes_overlap(box, _room_outer_box(other), pad=0) for other in all_rooms):
                    continue
                shaft = _line_points(junction, _door(room, door_side))
                inner = shaft[1:-1]
                if any(any(rb0 <= r <= rb1 and cb0 <= c <= cb1 for rb0, cb0, rb1, cb1 in [_room_outer_box(other) for other in all_rooms]) for r, c in inner):
                    continue
                candidate_dead.append((room, shaft))
        else:
            jr, jc = junction
            for sign in (-1, 1):
                if sign < 0:
                    room = {"r": jr - room_size // 2, "c": jc - room_size - 1, "n": room_size}
                    door_side = "E"
                else:
                    room = {"r": jr - room_size // 2, "c": jc + 3, "n": room_size}
                    door_side = "W"
                box = _room_outer_box(room)
                if any(_boxes_overlap(box, _room_outer_box(other), pad=0) for other in all_rooms):
                    continue
                shaft = _line_points(junction, _door(room, door_side))
                inner = shaft[1:-1]
                if any(any(rb0 <= r <= rb1 and cb0 <= c <= cb1 for rb0, cb0, rb1, cb1 in [_room_outer_box(other) for other in all_rooms]) for r, c in inner):
                    continue
                candidate_dead.append((room, shaft))
    if not candidate_dead:
        return None
    dead_end_room, dead_shaft = random.choice(candidate_dead)
    all_rooms.append(dead_end_room)
    add_room(dead_end_room)
    passable_pts.update(dead_shaft)

    room_boxes = [_room_outer_box(room) for room in all_rooms]
    bypass_candidates = []
    bypass_offset = room_size + 2
    for seg in corridor_segments:
        a = seg["a"]
        b = seg["b"]
        side = seg["side"]
        if len(seg["cells"]) < 4:
            continue
        if side in {"E", "W"}:
            row = a[0]
            c0 = min(seg["cells"][1][1], seg["cells"][-2][1])
            c1 = max(seg["cells"][1][1], seg["cells"][-2][1])
            for sign in (-1, 1):
                bypass_row = row + sign * bypass_offset
                leg_a = _line_points((row, c0), (bypass_row, c0))
                span = _line_points((bypass_row, c0), (bypass_row, c1))
                leg_b = _line_points((bypass_row, c1), (row, c1))
                bypass = leg_a + span[1:] + leg_b[1:]
                inner = bypass[1:-1]
                if any(any(rb0 <= r <= rb1 and cb0 <= c <= cb1 for rb0, cb0, rb1, cb1 in room_boxes) for r, c in inner):
                    continue
                bypass_candidates.append((seg, bypass))
        else:
            col = a[1]
            r0 = min(seg["cells"][1][0], seg["cells"][-2][0])
            r1 = max(seg["cells"][1][0], seg["cells"][-2][0])
            for sign in (-1, 1):
                bypass_col = col + sign * bypass_offset
                leg_a = _line_points((r0, col), (r0, bypass_col))
                span = _line_points((r0, bypass_col), (r1, bypass_col))
                leg_b = _line_points((r1, bypass_col), (r1, col))
                bypass = leg_a + span[1:] + leg_b[1:]
                inner = bypass[1:-1]
                if any(any(rb0 <= r <= rb1 and cb0 <= c <= cb1 for rb0, cb0, rb1, cb1 in room_boxes) for r, c in inner):
                    continue
                bypass_candidates.append((seg, bypass))
    if not bypass_candidates:
        return None
    chosen_seg, bypass_cells = random.choice(bypass_candidates)
    passable_pts.update(bypass_cells)
    detour_direct_segment = list(chosen_seg["cells"])

    all_pts = list(passable_pts) + endpoint_cells
    rs = [p[0] for p in all_pts]
    cs = [p[1] for p in all_pts]
    local_h = max(rs) + 3
    local_w = max(cs) + 3
    local = np.full((local_h, local_w), WALL, dtype=int)
    for r, c in passable_pts:
        local[r, c] = BG

    start_color, goal_color = random.sample(PALETTE, 2)
    local[endpoint_cells[0]] = start_color
    local[endpoint_cells[1]] = goal_color

    used_room_boxes = [_room_outer_box(room) for room in all_rooms]
    used_mask = np.zeros_like(local, dtype=bool)
    for r0, c0, r1, c1 in used_room_boxes:
        used_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    used_mask[np.array(rs), np.array(cs)] = True
    r0, c0, r1, c1 = _bbox_coords(np.argwhere(used_mask))
    maze = local[max(0, r0 - 1) : r1 + 2, max(0, c0 - 1) : c1 + 2]

    route_count = len(route_rooms)
    remaining = [c for c in PALETTE if c not in {start_color, goal_color}]
    random.shuffle(remaining)
    room_colors = remaining[:route_count]
    assert len(room_colors) == route_count
    if room_colors[0] == room_colors[-1]:
        return None

    start_from_first = random.random() < 0.5
    legend_body = room_colors if start_from_first else list(reversed(room_colors))
    legend = [start_color if start_from_first else goal_color] + legend_body

    edge = random.choice(["top", "bottom", "left", "right"])
    pads = {"top": 1, "bottom": 1, "left": 1, "right": 1}
    pads[edge] = 2
    final_h = maze.shape[0] + pads["top"] + pads["bottom"]
    final_w = maze.shape[1] + pads["left"] + pads["right"]
    if edge in {"top", "bottom"} and final_w < len(legend) + 4:
        final_w = len(legend) + 4
    if edge in {"left", "right"} and final_h < len(legend) + 4:
        final_h = len(legend) + 4
    if max(final_h, final_w) > 30:
        return None

    grid = np.full((final_h, final_w), BG, dtype=int)
    maze_r = pads["top"]
    maze_c = pads["left"]
    grid[maze_r : maze_r + maze.shape[0], maze_c : maze_c + maze.shape[1]] = maze

    if edge == "top":
        start = random.randint(1, final_w - len(legend) - 1)
        for i, color in enumerate(legend):
            grid[0, start + i] = int(color)
    elif edge == "bottom":
        start = random.randint(1, final_w - len(legend) - 1)
        for i, color in enumerate(legend):
            grid[-1, start + i] = int(color)
    elif edge == "left":
        start = random.randint(1, final_h - len(legend) - 1)
        for i, color in enumerate(legend):
            grid[start + i, 0] = int(color)
    else:
        start = random.randint(1, final_h - len(legend) - 1)
        for i, color in enumerate(legend):
            grid[start + i, -1] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    start_cell = tuple(map(int, endpoint_cells[0]))
    goal_cell = tuple(map(int, endpoint_cells[1]))
    start_cell = (start_cell[0] - max(0, r0 - 1) + maze_r, start_cell[1] - max(0, c0 - 1) + maze_c)
    goal_cell = (goal_cell[0] - max(0, r0 - 1) + maze_r, goal_cell[1] - max(0, c0 - 1) + maze_c)

    scene_bg = _edge_background(grid)
    scene_rooms = _find_rooms(grid, scene_bg, WALL)
    route_ids = _room_sequence_along_path(
        _reconstruct_path(
            grid != WALL,
            np.array([start_cell], dtype=int),
            np.array([goal_cell], dtype=int),
        ),
        scene_rooms,
    )
    if len(route_ids) != route_count:
        return None

    changed = out != grid
    if int(changed.sum()) != route_count * (room_size * room_size):
        return None

    shortest, count = _count_shortest_paths(grid != WALL, start_cell, goal_cell)
    if shortest < 0 or count != 1:
        return None

    direct_block = grid.copy()
    middle = detour_direct_segment[len(detour_direct_segment) // 2]
    middle = (middle[0] - max(0, r0 - 1) + maze_r, middle[1] - max(0, c0 - 1) + maze_c)
    if direct_block[middle] == WALL:
        return None
    direct_block[middle] = WALL
    alt_shortest, _ = _count_shortest_paths(direct_block != WALL, start_cell, goal_cell)
    if alt_shortest <= shortest:
        return None

    reversed_out = grid.copy()
    for color, room_id in zip(list(reversed(legend_body)), route_ids):
        reversed_out[scene_rooms[room_id]["mask"]] = int(color)
    if np.array_equal(reversed_out, out):
        return None

    traversed = set(route_ids)
    if len(scene_rooms) - len(traversed) < 1:
        return None
    for idx, room in enumerate(scene_rooms):
        room_changes = changed & room["mask"]
        if idx in traversed:
            if int(room_changes.sum()) != room_size * room_size:
                return None
        else:
            if np.any(room_changes):
                return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
