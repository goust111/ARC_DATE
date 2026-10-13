from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
DOOR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PATH_COLORS = [1, 2, 3, 4, 6, 7, 9]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _left_of(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + separators + [size]
    spans = []
    for i in range(len(bounds) - 1):
        lo = bounds[i] + 1
        hi = bounds[i + 1] - 1
        assert lo <= hi, "every room band must have positive thickness"
        spans.append((lo, hi))
    return spans


def _signed_area2(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i, (r0, c0) in enumerate(points):
        r1, c1 = points[(i + 1) % len(points)]
        total += c0 * r1 - c1 * r0
    return total


def _segment_cells(a: tuple[int, int], b: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    if row_first:
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            cells.append((nr, c))
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            cells.append((rr, nc))
    else:
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            cells.append((r, nc))
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            cells.append((nr, cc))
    return cells


def _concat_segments(waypoints: list[tuple[int, int]], room_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = room_bbox
    used = {waypoints[0]}
    path = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        orders = [True, False]
        random.shuffle(orders)
        seg = None
        for row_first in orders:
            trial = _segment_cells(a, b, row_first)
            body = trial[1:]
            if any(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in body):
                continue
            if any(cell in used for cell in body):
                continue
            seg = trial
            break
        if seg is None:
            return None
        path.extend(seg[1:])
        used.update(seg[1:])
    return path


def _room_path(
    room_bbox: tuple[int, int, int, int],
    endpoint_a: tuple[int, int],
    side_a: str,
    endpoint_b: tuple[int, int],
    side_b: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_bbox

    def inward(cell: tuple[int, int], side: str) -> tuple[int, int]:
        r, c = cell
        if side == "left":
            return r, c + 1
        if side == "right":
            return r, c - 1
        if side == "top":
            return r + 1, c
        assert side == "bottom"
        return r - 1, c

    inner_a = inward(endpoint_a, side_a)
    inner_b = inward(endpoint_b, side_b)
    strict_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    assert strict_bbox[0] <= strict_bbox[2] and strict_bbox[1] <= strict_bbox[3]

    interior = [
        (r, c)
        for r in range(strict_bbox[0], strict_bbox[2] + 1)
        for c in range(strict_bbox[1], strict_bbox[3] + 1)
        if (r, c) not in {inner_a, inner_b}
    ]
    for _ in range(80):
        pivot_count = random.choice([0, 1, 1, 2])
        if pivot_count > len(interior):
            continue
        pivots = random.sample(interior, pivot_count)
        waypoints = [inner_a] + pivots + [inner_b]
        route = _concat_segments(waypoints, strict_bbox)
        if route is None:
            continue
        return [endpoint_a] + route + [endpoint_b]

    fallback = _concat_segments([inner_a, inner_b], strict_bbox)
    assert fallback is not None
    return [endpoint_a] + fallback + [endpoint_b]


def _path_order(cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    neighbors = {
        cell: [nbr for nbr in _neighbors4(*cell) if nbr in cells]
        for cell in cells
    }
    ordered = [start]
    prev = None
    curr = start
    while curr != end:
        nxts = [nbr for nbr in neighbors[curr] if nbr != prev]
        assert len(nxts) == 1, "room fragment must be a simple path"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, curr = curr, nxt
    assert len(ordered) == len(cells), "room fragment must be connected"
    return ordered


def _shade_left(out: np.ndarray, grid: np.ndarray, ordered: list[tuple[int, int]]) -> None:
    for i, (r, c) in enumerate(ordered):
        dirs = []
        if len(ordered) == 1:
            continue
        if i == 0:
            nr, nc = ordered[1]
            dirs.append((nr - r, nc - c))
        elif i == len(ordered) - 1:
            pr, pc = ordered[i - 1]
            dirs.append((r - pr, c - pc))
        else:
            pr, pc = ordered[i - 1]
            nr, nc = ordered[i + 1]
            dirs.append((r - pr, c - pc))
            dirs.append((nr - r, nc - c))

        for dr, dc in dirs:
            lr, lc = _left_of(dr, dc)
            rr, cc = r + lr, c + lc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and int(grid[rr, cc]) == BLACK:
                out[rr, cc] = AZURE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect room-local path fragments through same-colored separator doors, orient every room cycle clockwise, and color the black cells immediately to the left of each oriented fragment azure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    separator_rows = [
        r for r in range(h) if np.all(g[r] != BLACK) and int(np.count_nonzero(g[r] == GREY)) >= w // 2
    ]
    separator_cols = [
        c for c in range(w) if np.all(g[:, c] != BLACK) and int(np.count_nonzero(g[:, c] == GREY)) >= h // 2
    ]
    assert separator_rows and separator_cols, "expected internal grey separator lines"
    assert all(b - a > 1 for a, b in zip(separator_rows, separator_rows[1:])), "separator rows must be single-pixel wide"
    assert all(b - a > 1 for a, b in zip(separator_cols, separator_cols[1:])), "separator cols must be single-pixel wide"

    row_spans = _spans(h, separator_rows)
    col_spans = _spans(w, separator_cols)

    separator_mask = np.zeros_like(g, dtype=bool)
    separator_mask[separator_rows, :] = True
    separator_mask[:, separator_cols] = True

    room_of = -np.ones((h, w), dtype=int)
    rooms: dict[int, dict] = {}
    room_id = 0
    for ir, (r0, r1) in enumerate(row_spans):
        for ic, (c0, c1) in enumerate(col_spans):
            for r in range(r0, r1 + 1):
                room_of[r, c0 : c1 + 1] = room_id
            rooms[room_id] = {
                "index": (ir, ic),
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
            }
            room_id += 1

    endpoint_owner: dict[tuple[int, int], int] = {}
    for rid, room in rooms.items():
        r0, c0, r1, c1 = room["bbox"]
        cells = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(g[r, c]) != BLACK
        }
        assert cells, "every room must contain a path fragment"
        colors = {int(g[r, c]) for r, c in cells}
        assert GREY not in colors, "grey must stay on separators"
        assert len(colors) == 1, "each room fragment must be monochrome"

        adj = {cell: [nbr for nbr in _neighbors4(*cell) if nbr in cells] for cell in cells}
        degrees = {cell: len(nbrs) for cell, nbrs in adj.items()}
        assert all(deg in (1, 2) for deg in degrees.values()), "fragment must be a simple 4-connected path"
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        assert len(endpoints) == 2, "fragment must have exactly two endpoints"

        room["cells"] = cells
        room["endpoints"] = endpoints
        room["endpoint_by_color"] = {}
        for cell in endpoints:
            endpoint_owner[cell] = rid

    door_records: dict[int, list[dict]] = {}
    for r, c in np.argwhere(separator_mask & (g != GREY)):
        color = int(g[r, c])
        touching = [cell for cell in _neighbors4(int(r), int(c)) if cell in endpoint_owner]
        assert len(touching) == 1, "each door must touch exactly one fragment endpoint"
        endpoint = touching[0]
        rid = endpoint_owner[endpoint]
        room = rooms[rid]
        room["endpoint_by_color"][color] = endpoint
        door_records.setdefault(color, []).append({"room": rid, "endpoint": endpoint, "door": (int(r), int(c))})

    edge_color_by_pair: dict[frozenset[int], int] = {}
    neighbors: dict[int, list[tuple[int, int]]] = {rid: [] for rid in rooms}
    for color, records in door_records.items():
        assert len(records) == 2, "each door color must appear exactly twice"
        a, b = records
        assert a["room"] != b["room"], "paired doors must belong to different rooms"
        pair = frozenset({a["room"], b["room"]})
        assert pair not in edge_color_by_pair, "room pairs must not be connected twice"
        edge_color_by_pair[pair] = color
        neighbors[a["room"]].append((b["room"], color))
        neighbors[b["room"]].append((a["room"], color))

    assert rooms, "expected at least one room"
    for rid, room in rooms.items():
        assert len(neighbors[rid]) == 2, "every room must have exactly two door connections"
        assert len(room["endpoint_by_color"]) == 2, "each endpoint must map to one door color"

    out = g.copy()
    seen: set[int] = set()
    for start in rooms:
        if start in seen:
            continue

        first_room, _first_color = neighbors[start][0]
        order = [start]
        prev = start
        curr = first_room
        while curr != start:
            order.append(curr)
            options = neighbors[curr]
            nxt_room, _nxt_color = options[0] if options[0][0] != prev else options[1]
            prev, curr = curr, nxt_room

        for rid in order:
            seen.add(rid)

        points = [rooms[rid]["center"] for rid in order]
        area2 = _signed_area2(points)
        assert area2 != 0, "room cycle must have a non-degenerate clockwise direction"
        if area2 < 0:
            order = [order[0]] + list(reversed(order[1:]))

        cycle_colors = [
            edge_color_by_pair[frozenset({order[i], order[(i + 1) % len(order)]})]
            for i in range(len(order))
        ]
        for i, rid in enumerate(order):
            room = rooms[rid]
            incoming = cycle_colors[i - 1]
            outgoing = cycle_colors[i]
            start_cell = room["endpoint_by_color"][incoming]
            end_cell = room["endpoint_by_color"][outgoing]
            ordered = _path_order(room["cells"], start_cell, end_cell)
            _shade_left(out, g, ordered)

    return out


def _build_room_bboxes(row_heights: list[int], col_widths: list[int]) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], list[int], list[int], int, int]:
    separator_rows = [row_heights[0]]
    separator_cols = []
    c = 0
    for width in col_widths[:-1]:
        c += width
        separator_cols.append(c)
        c += 1

    h = sum(row_heights) + len(separator_rows)
    w = sum(col_widths) + len(separator_cols)
    row_spans = _spans(h, separator_rows)
    col_spans = _spans(w, separator_cols)

    room_bboxes = {}
    for ir, (r0, r1) in enumerate(row_spans):
        for ic, (c0, c1) in enumerate(col_spans):
            room_bboxes[(ir, ic)] = (r0, c0, r1, c1)
    return room_bboxes, separator_rows, separator_cols, h, w


def _cycles_for_cols(n_cols: int) -> list[list[tuple[int, int]]]:
    if n_cols == 2:
        return [[(0, 0), (0, 1), (1, 1), (1, 0)]]
    if n_cols == 3:
        return [[(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]]
    assert n_cols == 4
    if random.random() < 0.5:
        return [[(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0)]]
    return [
        [(0, 0), (0, 1), (1, 1), (1, 0)],
        [(0, 2), (0, 3), (1, 3), (1, 2)],
    ]


def _shared_sides(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, str]:
    ar, ac = a
    br, bc = b
    if ar == br and abs(ac - bc) == 1:
        return ("right", "left") if bc > ac else ("left", "right")
    if ac == bc and abs(ar - br) == 1:
        return ("bottom", "top") if br > ar else ("top", "bottom")
    raise AssertionError("cycle edges must join adjacent rooms")


def _door_cell_for_side(room_bbox: tuple[int, int, int, int], side: str, pos: int) -> tuple[int, int]:
    r0, c0, r1, c1 = room_bbox
    if side == "left":
        return pos, c0
    if side == "right":
        return pos, c1
    if side == "top":
        return r0, pos
    assert side == "bottom"
    return r1, pos


def _place_edge_doors(
    room_bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
    grid: np.ndarray,
    color: int,
    a: tuple[int, int],
    b: tuple[int, int],
    per_room: dict[tuple[int, int], list[tuple[int, str, tuple[int, int]]]],
) -> None:
    side_a, side_b = _shared_sides(a, b)
    bbox_a = room_bboxes[a]
    bbox_b = room_bboxes[b]

    if side_a in {"left", "right"}:
        valid = list(range(bbox_a[0] + 1, bbox_a[2]))
        pos_a, pos_b = random.sample(valid, 2)
    else:
        valid = list(range(bbox_a[1] + 1, bbox_a[3]))
        pos_a, pos_b = random.sample(valid, 2)

    endpoint_a = _door_cell_for_side(bbox_a, side_a, pos_a)
    endpoint_b = _door_cell_for_side(bbox_b, side_b, pos_b)

    if side_a == "right":
        door_a = (endpoint_a[0], endpoint_a[1] + 1)
    elif side_a == "left":
        door_a = (endpoint_a[0], endpoint_a[1] - 1)
    elif side_a == "bottom":
        door_a = (endpoint_a[0] + 1, endpoint_a[1])
    else:
        door_a = (endpoint_a[0] - 1, endpoint_a[1])

    if side_b == "right":
        door_b = (endpoint_b[0], endpoint_b[1] + 1)
    elif side_b == "left":
        door_b = (endpoint_b[0], endpoint_b[1] - 1)
    elif side_b == "bottom":
        door_b = (endpoint_b[0] + 1, endpoint_b[1])
    else:
        door_b = (endpoint_b[0] - 1, endpoint_b[1])

    grid[door_a] = color
    grid[door_b] = color
    per_room[a].append((color, side_a, endpoint_a))
    per_room[b].append((color, side_b, endpoint_b))


def _paint_segment(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _generate_attempt() -> np.ndarray | None:
    n_cols = random.choice([2, 3, 4])
    row_heights = [random.randint(5, 7), random.randint(5, 7)]
    col_widths = [random.randint(5, 6) for _ in range(n_cols)]
    room_bboxes, separator_rows, separator_cols, h, w = _build_room_bboxes(row_heights, col_widths)

    grid = np.full((h, w), BLACK, dtype=int)
    for r in separator_rows:
        grid[r, :] = GREY
    for c in separator_cols:
        grid[:, c] = GREY

    cycles = _cycles_for_cols(n_cols)
    edge_count = sum(len(cycle) for cycle in cycles)
    door_colors = random.sample(DOOR_COLORS, edge_count)
    per_room: dict[tuple[int, int], list[tuple[int, str, tuple[int, int]]]] = {
        room: [] for room in room_bboxes
    }

    color_idx = 0
    for cycle in cycles:
        for i, room in enumerate(cycle):
            nxt = cycle[(i + 1) % len(cycle)]
            _place_edge_doors(room_bboxes, grid, door_colors[color_idx], room, nxt, per_room)
            color_idx += 1

    for room, door_specs in per_room.items():
        assert len(door_specs) == 2, "each room must receive two doors"
        (color_a, side_a, endpoint_a), (color_b, side_b, endpoint_b) = door_specs
        if endpoint_a == endpoint_b:
            return None
        path_color_choices = [color for color in PATH_COLORS if color not in {color_a, color_b}]
        if not path_color_choices:
            return None
        path_color = random.choice(path_color_choices)
        cells = _room_path(room_bboxes[room], endpoint_a, side_a, endpoint_b, side_b)
        if len(set(cells)) != len(cells):
            return None
        _paint_segment(grid, cells, path_color)

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not isinstance(produced, np.ndarray):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-linked room puzzle")
