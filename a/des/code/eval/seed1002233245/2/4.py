from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 8
PATTERN_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


RoomId = tuple[int, int]
RoomBox = tuple[int, int, int, int]  # r0, c0, h, w


def _detect_wall_color(grid: np.ndarray) -> int:
    best_color = None
    best_score = -1
    for color in map(int, np.unique(grid)):
        score = int(np.sum(np.all(grid == color, axis=1))) + int(np.sum(np.all(grid == color, axis=0)))
        if score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None and best_score >= 4, "expected a rectilinear wall color"
    return int(best_color)


def _wall_lines(grid: np.ndarray, wall_color: int) -> tuple[list[int], list[int]]:
    wall_rows = [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == wall_color))]
    wall_cols = [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == wall_color))]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected multiple wall lines"
    assert wall_rows[0] == 0 and wall_rows[-1] == grid.shape[0] - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == grid.shape[1] - 1
    for a, b in zip(wall_rows, wall_rows[1:]):
        assert b - a >= 2, "room height must be positive"
    for a, b in zip(wall_cols, wall_cols[1:]):
        assert b - a >= 2, "room width must be positive"
    return wall_rows, wall_cols


def _build_rooms(wall_rows: list[int], wall_cols: list[int]) -> dict[RoomId, RoomBox]:
    rooms: dict[RoomId, RoomBox] = {}
    for ri in range(len(wall_rows) - 1):
        r0 = int(wall_rows[ri] + 1)
        r1 = int(wall_rows[ri + 1] - 1)
        for ci in range(len(wall_cols) - 1):
            c0 = int(wall_cols[ci] + 1)
            c1 = int(wall_cols[ci + 1] - 1)
            rooms[(ri, ci)] = (r0, c0, int(r1 - r0 + 1), int(c1 - c0 + 1))
    return rooms


def _draw_mask(
    grid: np.ndarray,
    room_box: RoomBox,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    r0, c0, h, w = room_box
    ph, pw = mask.shape
    assert 0 <= top <= h - ph and 0 <= left <= w - pw
    patch = grid[r0 + top : r0 + top + ph, c0 + left : c0 + left + pw]
    patch[mask] = int(color)


def _portal_graph(
    grid: np.ndarray,
    rooms: dict[RoomId, RoomBox],
    wall_rows: list[int],
    wall_cols: list[int],
    pattern_color: int,
) -> tuple[dict[RoomId, list[dict[str, object]]], set[tuple[int, int]]]:
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    graph: dict[RoomId, list[dict[str, object]]] = {(ri, ci): [] for ri in range(room_rows) for ci in range(room_cols)}
    portal_cells: set[tuple[int, int]] = set()

    for ri in range(room_rows):
        top, _left, height, _width = rooms[(ri, 0)]
        bottom = int(top + height - 1)
        for ci in range(room_cols - 1):
            wall_c = int(wall_cols[ci + 1])
            matches = [
                int(r)
                for r in range(top, bottom + 1)
                if int(grid[r, wall_c - 1]) == pattern_color and int(grid[r, wall_c + 1]) == pattern_color
            ]
            assert len(matches) <= 1, "expected at most one portal per shared vertical wall"
            if matches:
                row = matches[0]
                offset = int(row - top)
                left_room = (ri, ci)
                right_room = (ri, ci + 1)
                graph[left_room].append({"dst": right_room, "axis": "vertical", "offset": offset})
                graph[right_room].append({"dst": left_room, "axis": "vertical", "offset": offset})
                portal_cells.add((int(row), int(wall_c - 1)))
                portal_cells.add((int(row), int(wall_c + 1)))

    for ri in range(room_rows - 1):
        wall_r = int(wall_rows[ri + 1])
        for ci in range(room_cols):
            _top, left, _height, width = rooms[(ri, ci)]
            right = int(left + width - 1)
            matches = [
                int(c)
                for c in range(left, right + 1)
                if int(grid[wall_r - 1, c]) == pattern_color and int(grid[wall_r + 1, c]) == pattern_color
            ]
            assert len(matches) <= 1, "expected at most one portal per shared horizontal wall"
            if matches:
                col = matches[0]
                offset = int(col - left)
                top_room = (ri, ci)
                bottom_room = (ri + 1, ci)
                graph[top_room].append({"dst": bottom_room, "axis": "horizontal", "offset": offset})
                graph[bottom_room].append({"dst": top_room, "axis": "horizontal", "offset": offset})
                portal_cells.add((int(wall_r - 1), int(col)))
                portal_cells.add((int(wall_r + 1), int(col)))

    return graph, portal_cells


def _source_pattern(
    grid: np.ndarray,
    rooms: dict[RoomId, RoomBox],
    pattern_color: int,
    portal_cells: set[tuple[int, int]],
) -> tuple[RoomId, np.ndarray]:
    source_room = None
    source_cells: list[tuple[int, int]] = []

    for room_id, (r0, c0, h, w) in rooms.items():
        cells = [
            (int(r), int(c))
            for r in range(r0, r0 + h)
            for c in range(c0, c0 + w)
            if int(grid[r, c]) == pattern_color and (int(r), int(c)) not in portal_cells
        ]
        if cells:
            assert source_room is None, "expected a unique source room"
            source_room = room_id
            source_cells = cells

    assert source_room is not None, "expected one room containing the source pattern"
    rows = [r for r, _c in source_cells]
    cols = [c for _r, c in source_cells]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=bool)
    for r, c in source_cells:
        mask[int(r - rmin), int(c - cmin)] = True
    return source_room, mask


def _transform_across_wall(
    mask: np.ndarray,
    src_room: RoomId,
    dst_room: RoomId,
    dst_box: RoomBox,
    axis: str,
    offset: int,
) -> tuple[np.ndarray, int, int]:
    dst_h = int(dst_box[2])
    dst_w = int(dst_box[3])

    if axis == "vertical":
        transformed = np.fliplr(mask)
        ph, pw = transformed.shape
        top = int(offset)
        left = 0 if dst_room[1] > src_room[1] else int(dst_w - pw)
    else:
        assert axis == "horizontal"
        transformed = np.flipud(mask)
        ph, pw = transformed.shape
        top = 0 if dst_room[0] > src_room[0] else int(dst_h - ph)
        left = int(offset)

    assert 0 <= top <= dst_h - ph
    assert 0 <= left <= dst_w - pw
    return transformed, int(top), int(left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a source pattern through portal-marked room walls, flipping on each crossing.

    Rooms are the rectangular cells between complete horizontal and vertical wall lines.
    Portals are same-colored pixel pairs on opposite sides of one shared wall. Crossing a
    vertical wall mirrors left-right; crossing a horizontal wall mirrors up-down. The copied
    pattern is placed flush against the entered wall, and its along-wall offset equals the
    portal offset on that wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_color = _detect_wall_color(g)
    wall_rows, wall_cols = _wall_lines(g, wall_color)
    rooms = _build_rooms(wall_rows, wall_cols)

    colors = [int(c) for c in np.unique(g) if int(c) != wall_color]
    assert len(colors) == 2, "expected background plus one pattern/portal color"
    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    bg_color = max(colors, key=counts.get)
    pattern_colors = [color for color in colors if color != bg_color]
    assert len(pattern_colors) == 1, "expected a unique pattern color"
    pattern_color = int(pattern_colors[0])

    graph, portal_cells = _portal_graph(g, rooms, wall_rows, wall_cols, pattern_color)
    source_room, source_mask = _source_pattern(g, rooms, pattern_color, portal_cells)

    out = g.copy()
    seen = {source_room}
    masks: dict[RoomId, np.ndarray] = {source_room: source_mask}
    queue: deque[RoomId] = deque([source_room])

    while queue:
        room_id = queue.popleft()
        mask = masks[room_id]
        for edge in graph[room_id]:
            dst = edge["dst"]
            assert isinstance(dst, tuple) and len(dst) == 2
            dst_room = (int(dst[0]), int(dst[1]))
            if dst_room in seen:
                continue
            transformed, top, left = _transform_across_wall(
                mask=mask,
                src_room=room_id,
                dst_room=dst_room,
                dst_box=rooms[dst_room],
                axis=str(edge["axis"]),
                offset=int(edge["offset"]),
            )
            _draw_mask(out, rooms[dst_room], transformed, top, left, pattern_color)
            seen.add(dst_room)
            masks[dst_room] = transformed
            queue.append(dst_room)

    return out


def _wall_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cur = 0
    for size in sizes:
        cur += int(size) + 1
        positions.append(int(cur))
    return positions


def _room_grid(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[int], list[int], dict[RoomId, RoomBox]]:
    wall_rows = _wall_positions(row_heights)
    wall_cols = _wall_positions(col_widths)
    height = int(wall_rows[-1] + 1)
    width = int(wall_cols[-1] + 1)
    grid = np.full((height, width), BG, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL
    rooms = _build_rooms(wall_rows, wall_cols)
    return grid, wall_rows, wall_cols, rooms


def _random_pattern(rng: random.Random, ph: int, pw: int) -> np.ndarray | None:
    for _ in range(120):
        cells = {(0, 0)}
        steps = ["D"] * (ph - 1) + ["R"] * (pw - 1)
        rng.shuffle(steps)
        r = 0
        c = 0
        for step in steps:
            if step == "D":
                r += 1
            else:
                c += 1
            cells.add((int(r), int(c)))

        frontier = {
            (int(nr), int(nc))
            for cr, cc in cells
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1))
            if 0 <= nr < ph and 0 <= nc < pw and (int(nr), int(nc)) not in cells
        }
        max_extra = max(1, min(ph * pw - len(cells), (ph * pw) // 2))
        extra = rng.randint(1, max_extra)
        for _ in range(extra):
            if not frontier:
                break
            nr, nc = rng.choice(sorted(frontier))
            cells.add((int(nr), int(nc)))
            frontier = {
                (int(x), int(y))
                for cr, cc in cells
                for x, y in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1))
                if 0 <= x < ph and 0 <= y < pw and (int(x), int(y)) not in cells
            }

        mask = np.zeros((ph, pw), dtype=bool)
        for rr, cc in cells:
            mask[int(rr), int(cc)] = True

        if bool(mask[0, 0]) and bool(mask[-1, -1]) and not np.array_equal(mask, np.fliplr(mask)) and not np.array_equal(mask, np.flipud(mask)):
            return mask
    return None


def _candidate_edges(room_rows: int, room_cols: int) -> list[tuple[RoomId, RoomId, str]]:
    edges: list[tuple[RoomId, RoomId, str]] = []
    for ri in range(room_rows):
        for ci in range(room_cols - 1):
            edges.append(((ri, ci), (ri, ci + 1), "vertical"))
    for ri in range(room_rows - 1):
        for ci in range(room_cols):
            edges.append(((ri, ci), (ri + 1, ci), "horizontal"))
    return edges


def _random_spanning_tree(
    rng: random.Random,
    room_rows: int,
    room_cols: int,
) -> list[tuple[RoomId, RoomId, str]]:
    nodes = [(ri, ci) for ri in range(room_rows) for ci in range(room_cols)]
    visited = {rng.choice(nodes)}
    tree: list[tuple[RoomId, RoomId, str]] = []
    edges = _candidate_edges(room_rows, room_cols)

    while len(visited) < len(nodes):
        frontier = [
            (a, b, axis)
            for a, b, axis in edges
            if (a in visited) ^ (b in visited)
        ]
        assert frontier, "grid graph must stay connected"
        a, b, axis = rng.choice(frontier)
        tree.append((a, b, axis))
        visited.add(a)
        visited.add(b)

    return tree


def _tree_adjacency(
    room_rows: int,
    room_cols: int,
    edges: list[tuple[RoomId, RoomId, str]],
) -> dict[RoomId, list[tuple[RoomId, str]]]:
    adj = {(ri, ci): [] for ri in range(room_rows) for ci in range(room_cols)}
    for a, b, axis in edges:
        adj[a].append((b, axis))
        adj[b].append((a, axis))
    return adj


def _root_metrics(
    source: RoomId,
    adj: dict[RoomId, list[tuple[RoomId, str]]],
) -> tuple[int, bool]:
    seen = {source}
    queue: deque[tuple[RoomId, str | None, int, bool]] = deque([(source, None, 0, False)])
    max_depth = 0
    has_repeat = False

    while queue:
        room, prev_axis, depth, repeat_on_path = queue.popleft()
        max_depth = max(max_depth, int(depth))
        has_repeat = bool(has_repeat or repeat_on_path)
        for nxt, axis in adj[room]:
            if nxt in seen:
                continue
            seen.add(nxt)
            repeat_now = bool(repeat_on_path or (prev_axis == axis and prev_axis is not None))
            queue.append((nxt, axis, int(depth + 1), repeat_now))

    return int(max_depth), bool(has_repeat)


def _place_portals(
    grid: np.ndarray,
    rooms: dict[RoomId, RoomBox],
    wall_rows: list[int],
    wall_cols: list[int],
    edges: list[tuple[RoomId, RoomId, str]],
    pattern_color: int,
    ph: int,
    pw: int,
    rng: random.Random,
) -> None:
    for a, b, axis in edges:
        if axis == "vertical":
            left_room, right_room = (a, b) if a[1] < b[1] else (b, a)
            r0, _c0, h, _w = rooms[left_room]
            max_offset = int(h - ph - 1)
            assert max_offset >= 1
            offset = int(rng.randint(1, max_offset))
            wall_c = int(wall_cols[left_room[1] + 1])
            row = int(r0 + offset)
            grid[row, wall_c - 1] = pattern_color
            grid[row, wall_c + 1] = pattern_color
        else:
            top_room, bottom_room = (a, b) if a[0] < b[0] else (b, a)
            _r0, c0, _h, w = rooms[top_room]
            max_offset = int(w - pw - 1)
            assert max_offset >= 1
            offset = int(rng.randint(1, max_offset))
            wall_r = int(wall_rows[top_room[0] + 1])
            col = int(c0 + offset)
            grid[wall_r - 1, col] = pattern_color
            grid[wall_r + 1, col] = pattern_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = int(rng.choice((2, 3)))
    room_cols = int(rng.choice((2, 3)))
    row_heights = [int(rng.randint(6, 9)) for _ in range(room_rows)]
    col_widths = [int(rng.randint(6, 9)) for _ in range(room_cols)]

    max_ph = min(4, min(row_heights) - 2)
    max_pw = min(4, min(col_widths) - 2)
    if max_ph < 2 or max_pw < 2:
        return None
    ph = int(rng.randint(2, max_ph))
    pw = int(rng.randint(2, max_pw))

    pattern = _random_pattern(rng, ph, pw)
    if pattern is None:
        return None

    edges = _random_spanning_tree(rng, room_rows, room_cols)
    edge_axes = {axis for _a, _b, axis in edges}
    if edge_axes != {"vertical", "horizontal"}:
        return None

    adj = _tree_adjacency(room_rows, room_cols, edges)
    source_candidates = []
    for room_id in adj:
        depth, has_repeat = _root_metrics(room_id, adj)
        if depth >= 2 and has_repeat:
            source_candidates.append(room_id)
    if not source_candidates:
        return None
    source_room = rng.choice(source_candidates)

    grid, wall_rows, wall_cols, rooms = _room_grid(row_heights, col_widths)
    pattern_color = int(rng.choice(PATTERN_COLORS))

    _place_portals(grid, rooms, wall_rows, wall_cols, edges, pattern_color, ph, pw, rng)

    src_box = rooms[source_room]
    src_top = int(rng.randint(1, src_box[2] - ph - 1))
    src_left = int(rng.randint(1, src_box[3] - pw - 1))
    _draw_mask(grid, src_box, pattern, src_top, src_left, pattern_color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-room task")
