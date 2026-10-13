import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
WALL = 5
BAR_COLORS = (3, 6, 7, 8, 9)

GEN_RETRIES = 400


def in_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mask_components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _find_color_bar(grid):
    mask = ~np.isin(grid, [BLACK, BLUE, RED, YELLOW, WALL])
    candidates = []
    for comp in _mask_components(mask):
        if len(comp) != 3:
            continue
        rows = {r for r, _ in comp}
        cols = sorted(c for _, c in comp)
        if len(rows) != 1 or cols != list(range(cols[0], cols[0] + 3)):
            continue
        r = next(iter(rows))
        c0 = cols[0]
        if r > 3 or c0 < grid.shape[1] - 7:
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != 3:
            continue
        candidates.append(colors)
    assert len(candidates) == 1, "expected exactly one 1x3 color bar"
    return candidates[0]


def _find_singleton(grid, color):
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _bfs_distances(grid, start):
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and grid[nr, nc] != WALL and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _shortest_path(grid, start, goal):
    h, w = grid.shape
    parent_r = np.full((h, w), -1, dtype=int)
    parent_c = np.full((h, w), -1, dtype=int)
    q = deque([start])
    parent_r[start] = start[0]
    parent_c[start] = start[1]
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if grid[nr, nc] == WALL or parent_r[nr, nc] != -1:
                continue
            parent_r[nr, nc] = r
            parent_c[nr, nc] = c
            q.append((nr, nc))
    assert parent_r[goal] != -1, "endpoints must be connected"
    path = [goal]
    cur = goal
    while cur != start:
        pr = int(parent_r[cur])
        pc = int(parent_c[cur])
        cur = (pr, pc)
        path.append(cur)
    path.reverse()
    return path


def _prefix_sum(mask):
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix, r0, c0, r1, c1):
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _line_full_open(prefix, r0, c0, r1, c1):
    if r0 > r1 or c0 > c1:
        return False
    return _rect_sum(prefix, r0, c0, r1, c1) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_rooms(grid):
    open_mask = grid != WALL
    prefix = _prefix_sum(open_mask)
    h, w = grid.shape
    candidates = []
    for r0 in range(h):
        for r1 in range(r0 + 2, h):
            room_h = r1 - r0 + 1
            for c0 in range(w):
                for c1 in range(c0 + 2, w):
                    room_w = c1 - c0 + 1
                    area = room_h * room_w
                    if area < 9:
                        continue
                    if _rect_sum(prefix, r0, c0, r1, c1) != area:
                        continue
                    side_openings = {}
                    if r0 > 0:
                        side_openings["top"] = [c for c in range(c0, c1 + 1) if open_mask[r0 - 1, c]]
                    else:
                        side_openings["top"] = []
                    if r1 + 1 < h:
                        side_openings["bottom"] = [c for c in range(c0, c1 + 1) if open_mask[r1 + 1, c]]
                    else:
                        side_openings["bottom"] = []
                    if c0 > 0:
                        side_openings["left"] = [r for r in range(r0, r1 + 1) if open_mask[r, c0 - 1]]
                    else:
                        side_openings["left"] = []
                    if c1 + 1 < w:
                        side_openings["right"] = [r for r in range(r0, r1 + 1) if open_mask[r, c1 + 1]]
                    else:
                        side_openings["right"] = []
                    if any(len(v) > 1 for v in side_openings.values()):
                        continue
                    open_sides = [side for side, vals in side_openings.items() if vals]
                    if len(open_sides) not in (1, 2):
                        continue
                    if len(open_sides) == 2:
                        if set(open_sides) == {"left", "right"}:
                            if side_openings["left"][0] != side_openings["right"][0]:
                                continue
                        elif set(open_sides) == {"top", "bottom"}:
                            if side_openings["top"][0] != side_openings["bottom"][0]:
                                continue
                        else:
                            continue
                    if r0 > 0 and _line_full_open(prefix, r0 - 1, c0, r0 - 1, c1):
                        continue
                    if r1 + 1 < h and _line_full_open(prefix, r1 + 1, c0, r1 + 1, c1):
                        continue
                    if c0 > 0 and _line_full_open(prefix, r0, c0 - 1, r1, c0 - 1):
                        continue
                    if c1 + 1 < w and _line_full_open(prefix, r0, c1 + 1, r1, c1 + 1):
                        continue
                    candidates.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "area": area,
                            "open_sides": tuple(sorted(open_sides)),
                        }
                    )
    rooms = []
    used = np.zeros_like(grid, dtype=bool)
    for cand in sorted(candidates, key=lambda x: (-x["area"], x["bbox"])):
        r0, c0, r1, c1 = cand["bbox"]
        if np.any(used[r0 : r1 + 1, c0 : c1 + 1]):
            continue
        used[r0 : r1 + 1, c0 : c1 + 1] = True
        rooms.append(cand)
    return sorted(rooms, key=lambda x: x["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique non-wall shortest path by a cyclic bar pattern and clear yellow contents from rooms the path never enters.

    Input structure:
    - Grey pixels form 1-cell-thick walls around rectangular room interiors and width-1 corridors on a black floor.
    - A single red pixel and a single blue pixel lie in different rooms, yellow mini-shapes may appear inside some rooms, and an isolated 1x3 color bar sits near the upper-right corner.

    Rule:
    - Read the bar colors from left to right.
    - Treat grey as blocked, find the unique 4-connected shortest path from red to blue through every non-grey cell, and recolor each non-endpoint path cell by cycling through the bar colors.
    - Detect every rectangular room interior of size at least 3x3; keep yellow contents only in rooms touched by the path and clear yellow contents in untouched rooms to black.

    Key insight:
    - The shortest path is reused twice: its traversal order defines the cyclic recoloring, and its room intersections decide which room contents survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bar = _find_color_bar(g)
    start = _find_singleton(g, RED)
    goal = _find_singleton(g, BLUE)
    path = _shortest_path(g, start, goal)
    rooms = _find_rooms(g)
    assert rooms, "expected at least one room"

    out = g.copy()
    path_set = set(path)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        touched = any((r, c) in path_set for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
        if not touched:
            sub = out[r0 : r1 + 1, c0 : c1 + 1]
            sub[sub == YELLOW] = BLACK
    for i, (r, c) in enumerate(path[1:-1], start=0):
        out[r, c] = bar[i % len(bar)]
    out[start] = RED
    out[goal] = BLUE
    return out


def _door_cell(room, door):
    r0, c0, h, w = room["r0"], room["c0"], room["h"], room["w"]
    side = door["side"]
    offset = door["offset"]
    if side == "left":
        return r0 + offset, c0 - 1
    if side == "right":
        return r0 + offset, c0 + w
    if side == "top":
        return r0 - 1, c0 + offset
    if side == "bottom":
        return r0 + h, c0 + offset
    raise ValueError(side)


def _room_cells(room):
    cells = []
    for r in range(room["r0"], room["r0"] + room["h"]):
        for c in range(room["c0"], room["c0"] + room["w"]):
            cells.append((r, c))
    return cells


def _segment_cells(a, b):
    (r0, c0), (r1, c1) = a, b
    assert r0 == r1 or c0 == c1
    cells = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.append((r, c0))
    return cells


def _polyline_cells(points):
    cells = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _non_path_cells(room):
    blocked = set(room["path_cells"]) | set(room.get("endpoint_cells", []))
    return [cell for cell in _room_cells(room) if cell not in blocked]


def _sample_connected_shape(rng, allowed, size):
    allowed = set(allowed)
    start = rng.choice(tuple(allowed))
    shape = {start}
    frontier = {start}
    while len(shape) < size:
        grow = []
        for cell in frontier:
            for nxt in neighbors4(*cell):
                if nxt in allowed and nxt not in shape:
                    grow.append(nxt)
        if not grow:
            return None
        nxt = rng.choice(grow)
        shape.add(nxt)
        frontier = {nxt} | frontier
    return sorted(shape)


def _choose_on_path_cells(room, rng):
    if room["route"] == "through_h":
        rel = room["doors"][0]["offset"]
        return [(room["r0"] + rel, room["c0"] + c) for c in range(room["w"])]
    if room["route"] == "through_v":
        rel = room["doors"][0]["offset"]
        return [(room["r0"] + r, room["c0"] + rel) for r in range(room["h"])]
    if room["route"] == "leaf_h":
        rel = room["doors"][0]["offset"]
        side = room["doors"][0]["side"]
        if side == "right":
            end_c = rng.randrange(0, room["w"] - 1)
            endpoint = (room["r0"] + rel, room["c0"] + end_c)
            path = [(room["r0"] + rel, room["c0"] + c) for c in range(end_c, room["w"])]
        else:
            end_c = rng.randrange(1, room["w"])
            endpoint = (room["r0"] + rel, room["c0"] + end_c)
            path = [(room["r0"] + rel, room["c0"] + c) for c in range(0, end_c + 1)]
        return endpoint, path
    if room["route"] == "leaf_v":
        rel = room["doors"][0]["offset"]
        side = room["doors"][0]["side"]
        if side == "bottom":
            end_r = rng.randrange(0, room["h"] - 1)
            endpoint = (room["r0"] + end_r, room["c0"] + rel)
            path = [(room["r0"] + r, room["c0"] + rel) for r in range(end_r, room["h"])]
        else:
            end_r = rng.randrange(1, room["h"])
            endpoint = (room["r0"] + end_r, room["c0"] + rel)
            path = [(room["r0"] + r, room["c0"] + rel) for r in range(0, end_r + 1)]
        return endpoint, path
    raise ValueError(room["route"])


def _new_room(room_id, r0, c0, h, w, route, doors, path_tag):
    return {
        "id": room_id,
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "route": route,
        "doors": doors,
        "path_tag": path_tag,
        "path_cells": [],
        "endpoint_cells": [],
        "content": [],
    }


def _rand_inner_index(rng, size):
    return 1 if size == 3 else rng.randrange(1, size - 1)


def _build_horizontal(rng):
    rooms = []
    h0, w0 = rng.randint(3, 5), rng.randint(3, 5)
    h1, w1 = rng.randint(3, 5), rng.randint(3, 5)
    h2, w2 = rng.randint(3, 5), rng.randint(3, 5)
    s0, s1, s2 = _rand_inner_index(rng, h0), _rand_inner_index(rng, h1), _rand_inner_index(rng, h2)
    g1, g2 = rng.randint(4, 6), rng.randint(4, 6)
    c0 = 0
    c1 = c0 + w0 + g1 + 1
    c2 = c1 + w1 + g2 + 1
    rooms.append(_new_room("main0", -s0, c0, h0, w0, "leaf_h", [{"side": "right", "offset": s0}], True))
    rooms.append(
        _new_room(
            "main1",
            -s1,
            c1,
            h1,
            w1,
            "through_h",
            [{"side": "left", "offset": s1}, {"side": "right", "offset": s1}],
            True,
        )
    )
    rooms.append(_new_room("main2", -s2, c2, h2, w2, "leaf_h", [{"side": "left", "offset": s2}], True))
    segments = [
        (_door_cell(rooms[0], rooms[0]["doors"][0]), _door_cell(rooms[1], rooms[1]["doors"][0])),
        (_door_cell(rooms[1], rooms[1]["doors"][1]), _door_cell(rooms[2], rooms[2]["doors"][0])),
    ]
    branch_specs = []
    anchors = [
        ("up", rng.randint(_door_cell(rooms[0], rooms[0]["doors"][0])[1] + 1, _door_cell(rooms[1], rooms[1]["doors"][0])[1] - 1)),
        ("down", rng.randint(_door_cell(rooms[0], rooms[0]["doors"][0])[1] + 1, _door_cell(rooms[1], rooms[1]["doors"][0])[1] - 1)),
        ("up", rng.randint(_door_cell(rooms[1], rooms[1]["doors"][1])[1] + 1, _door_cell(rooms[2], rooms[2]["doors"][0])[1] - 1)),
        ("down", rng.randint(_door_cell(rooms[1], rooms[1]["doors"][1])[1] + 1, _door_cell(rooms[2], rooms[2]["doors"][0])[1] - 1)),
    ]
    rng.shuffle(anchors)
    for i, (side, anchor_x) in enumerate(anchors[: rng.randint(1, 3)]):
        bh, bw = rng.randint(3, 5), rng.randint(3, 5)
        off = _rand_inner_index(rng, bw)
        sep = rng.randint(3, 5)
        if side == "up":
            room = _new_room(
                f"branch{i}",
                -sep - bh,
                anchor_x - off,
                bh,
                bw,
                None,
                [{"side": "bottom", "offset": off}],
                False,
            )
        else:
            room = _new_room(
                f"branch{i}",
                sep + 1,
                anchor_x - off,
                bh,
                bw,
                None,
                [{"side": "top", "offset": off}],
                False,
            )
        rooms.append(room)
        segments.append(((0, anchor_x), _door_cell(room, room["doors"][0])))
    return rooms, segments


def _build_vertical(rng):
    rooms = []
    h0, w0 = rng.randint(3, 5), rng.randint(3, 5)
    h1, w1 = rng.randint(3, 5), rng.randint(3, 5)
    h2, w2 = rng.randint(3, 5), rng.randint(3, 5)
    s0, s1, s2 = _rand_inner_index(rng, w0), _rand_inner_index(rng, w1), _rand_inner_index(rng, w2)
    g1, g2 = rng.randint(4, 6), rng.randint(4, 6)
    r0 = 0
    r1 = r0 + h0 + g1 + 1
    r2 = r1 + h1 + g2 + 1
    rooms.append(_new_room("main0", r0, -s0, h0, w0, "leaf_v", [{"side": "bottom", "offset": s0}], True))
    rooms.append(
        _new_room(
            "main1",
            r1,
            -s1,
            h1,
            w1,
            "through_v",
            [{"side": "top", "offset": s1}, {"side": "bottom", "offset": s1}],
            True,
        )
    )
    rooms.append(_new_room("main2", r2, -s2, h2, w2, "leaf_v", [{"side": "top", "offset": s2}], True))
    segments = [
        (_door_cell(rooms[0], rooms[0]["doors"][0]), _door_cell(rooms[1], rooms[1]["doors"][0])),
        (_door_cell(rooms[1], rooms[1]["doors"][1]), _door_cell(rooms[2], rooms[2]["doors"][0])),
    ]
    anchors = [
        ("left", rng.randint(_door_cell(rooms[0], rooms[0]["doors"][0])[0] + 1, _door_cell(rooms[1], rooms[1]["doors"][0])[0] - 1)),
        ("right", rng.randint(_door_cell(rooms[0], rooms[0]["doors"][0])[0] + 1, _door_cell(rooms[1], rooms[1]["doors"][0])[0] - 1)),
        ("left", rng.randint(_door_cell(rooms[1], rooms[1]["doors"][1])[0] + 1, _door_cell(rooms[2], rooms[2]["doors"][0])[0] - 1)),
        ("right", rng.randint(_door_cell(rooms[1], rooms[1]["doors"][1])[0] + 1, _door_cell(rooms[2], rooms[2]["doors"][0])[0] - 1)),
    ]
    rng.shuffle(anchors)
    for i, (side, anchor_y) in enumerate(anchors[: rng.randint(1, 3)]):
        bh, bw = rng.randint(3, 5), rng.randint(3, 5)
        off = _rand_inner_index(rng, bh)
        sep = rng.randint(3, 5)
        if side == "left":
            room = _new_room(
                f"branch{i}",
                anchor_y - off,
                -sep - bw,
                bh,
                bw,
                None,
                [{"side": "right", "offset": off}],
                False,
            )
        else:
            room = _new_room(
                f"branch{i}",
                anchor_y - off,
                sep + 1,
                bh,
                bw,
                None,
                [{"side": "left", "offset": off}],
                False,
            )
        rooms.append(room)
        segments.append(((anchor_y, 0), _door_cell(room, room["doors"][0])))
    return rooms, segments


def _build_lshape(rng):
    rooms = []
    h0, w0 = rng.randint(3, 5), rng.randint(3, 5)
    h1, w1 = rng.randint(3, 5), rng.randint(3, 5)
    h2, w2 = rng.randint(3, 5), rng.randint(3, 5)
    h3, w3 = rng.randint(3, 5), rng.randint(3, 5)
    s0, s1 = _rand_inner_index(rng, h0), _rand_inner_index(rng, h1)
    t2, t3 = _rand_inner_index(rng, w2), _rand_inner_index(rng, w3)
    g1, g2, g3, gv = rng.randint(4, 6), rng.randint(3, 5), rng.randint(4, 6), rng.randint(3, 5)
    c0 = 0
    c1 = c0 + w0 + g1 + 1
    joint_x = c1 + w1 + g2
    r2 = gv + 1
    r3 = r2 + h2 + g3 + 1
    rooms.append(_new_room("main0", -s0, c0, h0, w0, "leaf_h", [{"side": "right", "offset": s0}], True))
    rooms.append(
        _new_room(
            "main1",
            -s1,
            c1,
            h1,
            w1,
            "through_h",
            [{"side": "left", "offset": s1}, {"side": "right", "offset": s1}],
            True,
        )
    )
    rooms.append(
        _new_room(
            "main2",
            r2,
            joint_x - t2,
            h2,
            w2,
            "through_v",
            [{"side": "top", "offset": t2}, {"side": "bottom", "offset": t2}],
            True,
        )
    )
    rooms.append(_new_room("main3", r3, joint_x - t3, h3, w3, "leaf_v", [{"side": "top", "offset": t3}], True))
    segments = [
        (_door_cell(rooms[0], rooms[0]["doors"][0]), _door_cell(rooms[1], rooms[1]["doors"][0])),
        (_door_cell(rooms[1], rooms[1]["doors"][1]), (0, joint_x)),
        ((0, joint_x), _door_cell(rooms[2], rooms[2]["doors"][0])),
        (_door_cell(rooms[2], rooms[2]["doors"][1]), _door_cell(rooms[3], rooms[3]["doors"][0])),
    ]
    branch_targets = [
        ("down", rng.randint(_door_cell(rooms[0], rooms[0]["doors"][0])[1] + 1, _door_cell(rooms[1], rooms[1]["doors"][0])[1] - 1)),
        ("up", rng.randint(_door_cell(rooms[1], rooms[1]["doors"][1])[1] + 1, joint_x - 1)),
        ("left", rng.randint(1, _door_cell(rooms[2], rooms[2]["doors"][0])[0] - 1)),
        ("right", rng.randint(_door_cell(rooms[2], rooms[2]["doors"][1])[0] + 1, _door_cell(rooms[3], rooms[3]["doors"][0])[0] - 1)),
    ]
    rng.shuffle(branch_targets)
    for i, (side, anchor) in enumerate(branch_targets[: rng.randint(1, 2)]):
        bh, bw = rng.randint(3, 5), rng.randint(3, 5)
        sep = rng.randint(3, 5)
        if side in ("up", "down"):
            off = _rand_inner_index(rng, bw)
            if side == "up":
                room = _new_room(
                    f"branch{i}",
                    -sep - bh,
                    anchor - off,
                    bh,
                    bw,
                    None,
                    [{"side": "bottom", "offset": off}],
                    False,
                )
            else:
                room = _new_room(
                    f"branch{i}",
                    sep + 1,
                    anchor - off,
                    bh,
                    bw,
                    None,
                    [{"side": "top", "offset": off}],
                    False,
                )
            segments.append(((0, anchor), _door_cell(room, room["doors"][0])))
        else:
            off = _rand_inner_index(rng, bh)
            if side == "left":
                room = _new_room(
                    f"branch{i}",
                    anchor - off,
                    -sep - bw,
                    bh,
                    bw,
                    None,
                    [{"side": "right", "offset": off}],
                    False,
                )
            else:
                room = _new_room(
                    f"branch{i}",
                    anchor - off,
                    joint_x + sep + 1,
                    bh,
                    bw,
                    None,
                    [{"side": "left", "offset": off}],
                    False,
                )
            segments.append(((anchor, joint_x), _door_cell(room, room["doors"][0])))
        rooms.append(room)
    return rooms, segments


def _apply_shift(rooms, segments, dr, dc):
    shifted_rooms = []
    for room in rooms:
        nr = dict(room)
        nr["r0"] += dr
        nr["c0"] += dc
        nr["doors"] = [dict(door) for door in room["doors"]]
        nr["path_cells"] = [(r + dr, c + dc) for r, c in room["path_cells"]]
        nr["endpoint_cells"] = [(r + dr, c + dc) for r, c in room["endpoint_cells"]]
        nr["content"] = [(r + dr, c + dc) for r, c in room["content"]]
        shifted_rooms.append(nr)
    shifted_segments = []
    for a, b in segments:
        shifted_segments.append(((a[0] + dr, a[1] + dc), (b[0] + dr, b[1] + dc)))
    return shifted_rooms, shifted_segments


def _layout_bounds(rooms, segments):
    cells = set()
    for room in rooms:
        cells.update(_room_cells(room))
        for door in room["doors"]:
            cells.add(_door_cell(room, door))
        cells.update(room["path_cells"])
        cells.update(room["endpoint_cells"])
        cells.update(room["content"])
    for a, b in segments:
        cells.update(_segment_cells(a, b))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _paint_layout(grid, rooms, segments, bar):
    for room in rooms:
        grid[room["r0"] : room["r0"] + room["h"], room["c0"] : room["c0"] + room["w"]] = BLACK
        for door in room["doors"]:
            dr, dc = _door_cell(room, door)
            grid[dr, dc] = BLACK
    for a, b in segments:
        for r, c in _segment_cells(a, b):
            grid[r, c] = BLACK
    for room in rooms:
        for r, c in room["content"]:
            grid[r, c] = YELLOW
    start_room = next(room for room in rooms if room.get("endpoint_color") == RED)
    end_room = next(room for room in rooms if room.get("endpoint_color") == BLUE)
    grid[start_room["endpoint_cells"][0]] = RED
    grid[end_room["endpoint_cells"][0]] = BLUE
    bar_row = 1
    bar_col = grid.shape[1] - 4
    grid[bar_row, bar_col : bar_col + 3] = np.array(bar, dtype=int)


def _rooms_overlap(rooms):
    seen = set()
    for room in rooms:
        cells = set(_room_cells(room))
        if seen & cells:
            return True
        seen |= cells
    return False


def _choose_contents(rng, rooms):
    path_rooms = [room for room in rooms if room["path_tag"]]
    branch_rooms = [room for room in rooms if not room["path_tag"]]
    keep_room = rng.choice(path_rooms)
    clear_room = rng.choice(branch_rooms)
    assigned = set()
    for room in (keep_room, clear_room):
        allowed = _non_path_cells(room)
        shape = _sample_connected_shape(rng, allowed, rng.randint(1, min(4, len(allowed))))
        if shape is None:
            return False
        room["content"] = shape
        assigned.add(room["id"])
    for room in rooms:
        if room["id"] in assigned or rng.random() > 0.35:
            continue
        allowed = _non_path_cells(room)
        if not allowed:
            continue
        shape = _sample_connected_shape(rng, allowed, rng.randint(1, min(4, len(allowed))))
        if shape is not None:
            room["content"] = shape
    return True


def _prepare_rooms(rng, rooms):
    path_rooms = [room for room in rooms if room["path_tag"]]
    assert len(path_rooms) >= 3
    path_rooms[0]["endpoint_color"] = RED
    path_rooms[-1]["endpoint_color"] = BLUE
    for room in path_rooms[1:-1]:
        room["endpoint_color"] = None
        room["path_cells"] = _choose_on_path_cells(room, rng)
    start_endpoint, start_path = _choose_on_path_cells(path_rooms[0], rng)
    end_endpoint, end_path = _choose_on_path_cells(path_rooms[-1], rng)
    path_rooms[0]["endpoint_cells"] = [start_endpoint]
    path_rooms[0]["path_cells"] = start_path
    path_rooms[-1]["endpoint_cells"] = [end_endpoint]
    path_rooms[-1]["path_cells"] = end_path
    for room in rooms:
        if not room["path_tag"]:
            room["endpoint_color"] = None
            room["path_cells"] = []
            room["endpoint_cells"] = []
    return _choose_contents(rng, rooms)


def _count_shortest_paths(grid, start, goal):
    dist = _bfs_distances(grid, start)
    assert dist[goal] >= 0
    coords = [(int(dist[r, c]), r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if dist[r, c] >= 0]
    coords.sort()
    ways = np.zeros_like(dist, dtype=object)
    ways[start] = 1
    for d, r, c in coords:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and dist[nr, nc] == d + 1:
                ways[nr, nc] += ways[r, c]
    return int(ways[goal])


def _validate_instance(grid, rooms, bar):
    red = _find_singleton(grid, RED)
    blue = _find_singleton(grid, BLUE)
    assert _count_shortest_paths(grid, red, blue) == 1
    path = _shortest_path(grid, red, blue)
    path_set = set(path)
    detected = _find_rooms(grid)
    expected_boxes = sorted((room["r0"], room["c0"], room["r0"] + room["h"] - 1, room["c0"] + room["w"] - 1) for room in rooms)
    found_boxes = sorted(room["bbox"] for room in detected)
    assert expected_boxes == found_boxes
    on_path_ids = {
        room["id"]
        for room in rooms
        if any((r, c) in path_set for r in range(room["r0"], room["r0"] + room["h"]) for c in range(room["c0"], room["c0"] + room["w"]))
    }
    assert any(room["path_tag"] and room["content"] for room in rooms)
    assert any((not room["path_tag"]) and room["content"] for room in rooms)
    assert on_path_ids == {room["id"] for room in rooms if room["path_tag"]}
    out = main(grid)
    expected = grid.copy()
    for room in rooms:
        if room["id"] not in on_path_ids:
            for r, c in room["content"]:
                expected[r, c] = BLACK
    for i, (r, c) in enumerate(path[1:-1]):
        expected[r, c] = bar[i % 3]
    expected[red] = RED
    expected[blue] = BLUE
    assert np.array_equal(out, expected)


def _gen_attempt(rng):
    builder = rng.choice([_build_horizontal, _build_vertical, _build_lshape])
    rooms, segments = builder(rng)
    if _rooms_overlap(rooms):
        return None
    if not _prepare_rooms(rng, rooms):
        return None
    min_r, min_c, max_r, max_c = _layout_bounds(rooms, segments)
    dr = 4 - min_r + rng.randint(0, 2)
    dc = 2 - min_c + rng.randint(0, 2)
    rooms, segments = _apply_shift(rooms, segments, dr, dc)
    min_r, min_c, max_r, max_c = _layout_bounds(rooms, segments)
    h = max_r + 3
    w = max_c + 7
    if h > 30 or w > 30:
        return None
    grid = np.full((h, w), WALL, dtype=int)
    bar = rng.sample(BAR_COLORS, 3)
    _paint_layout(grid, rooms, segments, bar)
    if np.any(grid[0:3, 0 : max(0, w - 6)] != WALL):
        return None
    try:
        _validate_instance(grid, rooms, bar)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
