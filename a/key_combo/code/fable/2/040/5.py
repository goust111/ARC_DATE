from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
ROOM_COLORS = (2, 3, 4, 6, 7, 8, 9)

GENERATION_ATTEMPTS = 500
ROOM_LAYOUTS = ((2, 2), (2, 3), (3, 2))
ROOM_SIZE_RANGE = (3, 6)
COUNT_VALUES = tuple(range(1, 8))


Room = tuple[int, int]
Span = tuple[int, int]


def _is_divider_line(line: np.ndarray) -> bool:
    values = set(map(int, np.unique(line)))
    grey_count = int(np.count_nonzero(line == GREY))
    return bool(grey_count > len(line) // 2 and values <= {BLACK, GREY})


def _divider_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if _is_divider_line(grid[r, :])]


def _divider_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if _is_divider_line(grid[:, c])]


def _segments(length: int, dividers: list[int]) -> list[Span]:
    spans: list[Span] = []
    start = 0
    previous = -2
    for div in dividers:
        assert div > previous + 1, "divider lines must be separated by room interiors"
        assert start <= div - 1, "divider cannot touch the previous divider or border"
        spans.append((start, div - 1))
        start = div + 1
        previous = div
    assert start <= length - 1, "last room band must be non-empty"
    spans.append((start, length - 1))
    return spans


def _span_index(spans: list[Span], x: int) -> int | None:
    for i, (lo, hi) in enumerate(spans):
        if lo <= int(x) <= hi:
            return int(i)
    return None


def _room_for_cell(row_spans: list[Span], col_spans: list[Span], r: int, c: int) -> Room:
    rr = _span_index(row_spans, r)
    cc = _span_index(col_spans, c)
    assert rr is not None and cc is not None, "cell must lie inside a room"
    return int(rr), int(cc)


def _collect_door_edges(
    grid: np.ndarray,
    div_rows: list[int],
    div_cols: list[int],
    row_spans: list[Span],
    col_spans: list[Span],
) -> set[tuple[Room, Room]]:
    edges: set[tuple[Room, Room]] = set()
    height, width = map(int, grid.shape)

    for band, r in enumerate(div_rows):
        assert 0 <= band < len(row_spans) - 1
        for c in range(width):
            if int(grid[r, c]) != BLACK:
                continue
            room_col = _span_index(col_spans, c)
            assert room_col is not None, "horizontal door must sit inside one room column"
            a = (int(band), int(room_col))
            b = (int(band + 1), int(room_col))
            edge = tuple(sorted((a, b)))
            assert edge not in edges, "each adjacent room pair may have only one door"
            edges.add(edge)

    for band, c in enumerate(div_cols):
        assert 0 <= band < len(col_spans) - 1
        for r in range(height):
            if int(grid[r, c]) != BLACK:
                continue
            room_row = _span_index(row_spans, r)
            assert room_row is not None, "vertical door must sit inside one room row"
            a = (int(room_row), int(band))
            b = (int(room_row), int(band + 1))
            edge = tuple(sorted((a, b)))
            assert edge not in edges, "each adjacent room pair may have only one door"
            edges.add(edge)

    return edges


def _room_infos(
    grid: np.ndarray,
    row_spans: list[Span],
    col_spans: list[Span],
    start_room: Room,
) -> dict[Room, tuple[int, int]]:
    infos: dict[Room, tuple[int, int]] = {}
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert not bool(np.any(patch == GREY)), "grey belongs only to divider lines"

            blue_count = int(np.count_nonzero(patch == BLUE))
            if (i, j) == start_room:
                assert blue_count == 1, "start room must contain exactly one blue marker"
            else:
                assert blue_count == 0, "only the start room may contain a blue marker"

            colors = set(map(int, np.unique(patch))) - {BLACK, BLUE}
            assert len(colors) == 1, "each room must contain exactly one counted color"
            color = int(colors.pop())
            assert color not in (BLACK, BLUE, GREY)
            count = int(np.count_nonzero(patch == color))
            assert count > 0, "each room must contribute a positive segment length"
            infos[(int(i), int(j))] = (color, count)
    return infos


def _visit_order(rooms: list[Room], edges: set[tuple[Room, Room]], start: Room) -> list[Room]:
    adj: dict[Room, list[Room]] = {room: [] for room in rooms}
    for a, b in edges:
        assert a in adj and b in adj, "door must connect two known rooms"
        adj[a].append(b)
        adj[b].append(a)

    assert len(edges) == len(rooms) - 1, "doors must form one chain through all rooms"
    degrees = sorted(len(adj[room]) for room in rooms)
    assert degrees.count(1) == 2 and degrees.count(2) == len(rooms) - 2, "room graph must be an unbranched chain"
    assert len(adj[start]) == 1, "the blue start room must be an endpoint of the chain"

    order: list[Room] = []
    prev: Room | None = None
    cur = start
    for _ in range(len(rooms)):
        assert cur not in order, "door chain must not revisit a room"
        order.append(cur)
        next_rooms = [room for room in adj[cur] if room != prev]
        if not next_rooms:
            break
        assert len(next_rooms) == 1, "current room must have one unvisited exit"
        prev, cur = cur, next_rooms[0]

    assert len(order) == len(rooms), "door chain must visit every room"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the door-ordered room chain into one row of counted color segments.

    Input structure:
    - A black grid is partitioned into rectangular rooms by grey divider rows and columns.
    - Each room contains scattered pixels of exactly one non-black, non-grey, non-blue color.
    - Exactly one endpoint room also contains one blue start marker, and black one-cell gaps in divider lines are doors.

    Rule:
    - Build the room graph whose edges are the black doors between adjacent rooms.
    - Starting from the room with the blue marker, walk through the unvisited door at each step until the opposite endpoint.
    - Output a 1 x N row whose segments follow that visit order; each segment uses the room color and has length equal to that room's counted pixels.

    Key insight:
    - The segment order is topological, from the door chain, while the segment lengths are local pixel counts inside rooms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    div_rows = _divider_rows(g)
    div_cols = _divider_cols(g)
    assert div_rows and div_cols, "expected both grey divider rows and grey divider columns"

    row_spans = _segments(g.shape[0], div_rows)
    col_spans = _segments(g.shape[1], div_cols)
    rooms = [(i, j) for i in range(len(row_spans)) for j in range(len(col_spans))]
    assert len(rooms) >= 4, "expected a rectangular arrangement of several rooms"

    blue_cells = np.argwhere(g == BLUE)
    assert blue_cells.shape[0] == 1, "expected one blue start marker"
    start_r, start_c = map(int, blue_cells[0])
    start_room = _room_for_cell(row_spans, col_spans, start_r, start_c)

    edges = _collect_door_edges(g, div_rows, div_cols, row_spans, col_spans)
    order = _visit_order(rooms, edges, start_room)
    infos = _room_infos(g, row_spans, col_spans, start_room)

    compressed: list[int] = []
    for room in order:
        color, count = infos[room]
        compressed.extend([int(color)] * int(count))

    assert compressed, "output must contain at least one counted pixel"
    assert len(compressed) <= 30, "compressed row must fit ARC dimensions"
    return np.array([compressed], dtype=int)


def _spans_from_sizes(sizes: list[int]) -> tuple[list[Span], list[int], int]:
    spans: list[Span] = []
    dividers: list[int] = []
    pos = 0
    for i, size in enumerate(sizes):
        spans.append((pos, pos + int(size) - 1))
        pos += int(size)
        if i != len(sizes) - 1:
            dividers.append(pos)
            pos += 1
    return spans, dividers, pos


def _snake_path(rows: int, cols: int) -> list[Room]:
    if random.random() < 0.5:
        path: list[Room] = []
        for i in range(rows):
            ordered_cols = range(cols) if i % 2 == 0 else range(cols - 1, -1, -1)
            path.extend((i, j) for j in ordered_cols)
    else:
        path = []
        for j in range(cols):
            ordered_rows = range(rows) if j % 2 == 0 else range(rows - 1, -1, -1)
            path.extend((i, j) for i in ordered_rows)

    if random.random() < 0.5:
        path.reverse()
    return [(int(i), int(j)) for i, j in path]


def _paint_door(
    grid: np.ndarray,
    a: Room,
    b: Room,
    row_spans: list[Span],
    col_spans: list[Span],
    div_rows: list[int],
    div_cols: list[int],
) -> tuple[int, int]:
    ar, ac = a
    br, bc = b
    if ar == br:
        assert abs(ac - bc) == 1
        divider_col = div_cols[min(ac, bc)]
        r0, r1 = row_spans[ar]
        door_r = random.randint(r0, r1)
        grid[door_r, divider_col] = BLACK
        return int(door_r), int(divider_col)

    assert ac == bc and abs(ar - br) == 1
    divider_row = div_rows[min(ar, br)]
    c0, c1 = col_spans[ac]
    door_c = random.randint(c0, c1)
    grid[divider_row, door_c] = BLACK
    return int(divider_row), int(door_c)


def _render_metadata_output(path: list[Room], colors: dict[Room, int], counts: dict[Room, int]) -> np.ndarray:
    row: list[int] = []
    for room in path:
        row.extend([int(colors[room])] * int(counts[room]))
    return np.array([row], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_LAYOUTS)
    room_count = int(room_rows * room_cols)

    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    row_spans, div_rows, height = _spans_from_sizes(heights)
    col_spans, div_cols, width = _spans_from_sizes(widths)
    if max(height, width) > 30:
        return None

    path = _snake_path(room_rows, room_cols)
    reading_order = [(i, j) for i in range(room_rows) for j in range(room_cols)]
    if path == reading_order:
        return None

    move_dirs = {(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])}
    if not any(dr != 0 for dr, _dc in move_dirs):
        return None
    if not any(dc != 0 for _dr, dc in move_dirs):
        return None

    color_values = list(random.sample(ROOM_COLORS, room_count))
    random.shuffle(color_values)
    count_values = list(random.sample(COUNT_VALUES, room_count))
    random.shuffle(count_values)
    if sum(count_values) > 30 or len(set(count_values)) != room_count:
        return None

    rooms = reading_order
    colors = {room: int(color) for room, color in zip(rooms, color_values)}
    counts = {room: int(count) for room, count in zip(rooms, count_values)}
    expected = _render_metadata_output(path, colors, counts)
    reading_expected = _render_metadata_output(reading_order, colors, counts)
    if np.array_equal(expected, reading_expected):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for r in div_rows:
        grid[r, :] = GREY
    for c in div_cols:
        grid[:, c] = GREY

    for a, b in zip(path, path[1:]):
        _paint_door(grid, a, b, row_spans, col_spans, div_rows, div_cols)

    start_room = path[0]
    for room in rooms:
        rr, cc = room
        r0, r1 = row_spans[rr]
        c0, c1 = col_spans[cc]
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        needed = counts[room] + (1 if room == start_room else 0)
        if needed > len(cells):
            return None
        chosen = random.sample(cells, needed)
        for r, c in chosen[: counts[room]]:
            grid[r, c] = colors[room]
        if room == start_room:
            marker_r, marker_c = chosen[-1]
            grid[marker_r, marker_c] = BLUE

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, reading_expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid count-corridor compression input")
