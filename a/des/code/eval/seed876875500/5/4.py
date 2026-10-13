from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

MAX_GENERATION_ATTEMPTS = 400
ROOM_COUNT_RANGE = (2, 4)
ROOM_SPAN_RANGE = (3, 5)
DOOR_PROB_RANGE = (0.25, 0.55)


Room = tuple[int, int]
Box = tuple[int, int, int, int]


def _non_wall_run_lengths(line: np.ndarray, wall_color: int) -> list[int]:
    lengths: list[int] = []
    run = 0
    for value in line.tolist():
        if int(value) != int(wall_color):
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def _find_wall_lines(grid: np.ndarray, axis: int, wall_color: int) -> list[int]:
    lines: list[int] = []
    limit = int(grid.shape[axis])
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if not bool(np.any(line == int(wall_color))):
            continue
        runs = _non_wall_run_lengths(line, wall_color)
        if all(length == 1 for length in runs):
            lines.append(idx)
    assert len(lines) >= 2, "expected an outer wall frame"
    assert lines[0] == 0 and lines[-1] == limit - 1, "outer border must be walls"
    assert all(b - a >= 2 for a, b in zip(lines, lines[1:], strict=False)), "rooms need positive area"
    return lines


def _room_boxes(h_lines: list[int], v_lines: list[int]) -> dict[Room, Box]:
    boxes: dict[Room, Box] = {}
    for rr in range(len(h_lines) - 1):
        for cc in range(len(v_lines) - 1):
            r0 = int(h_lines[rr] + 1)
            r1 = int(h_lines[rr + 1] - 1)
            c0 = int(v_lines[cc] + 1)
            c1 = int(v_lines[cc + 1] - 1)
            assert r0 <= r1 and c0 <= c1
            boxes[(rr, cc)] = (r0, r1, c0, c1)
    return boxes


def _vertical_door(grid: np.ndarray, h_lines: list[int], room_row: int, wall_col: int) -> bool:
    r0 = int(h_lines[room_row] + 1)
    r1 = int(h_lines[room_row + 1])
    segment = grid[r0:r1, int(wall_col)]
    holes = np.flatnonzero(segment != int(BLUE))
    assert all(length == 1 for length in _non_wall_run_lengths(segment, BLUE))
    if holes.size:
        assert bool(np.all(segment[holes] == BLACK))
        return True
    return False


def _horizontal_door(grid: np.ndarray, v_lines: list[int], wall_row: int, room_col: int) -> bool:
    c0 = int(v_lines[room_col] + 1)
    c1 = int(v_lines[room_col + 1])
    segment = grid[int(wall_row), c0:c1]
    holes = np.flatnonzero(segment != int(BLUE))
    assert all(length == 1 for length in _non_wall_run_lengths(segment, BLUE))
    if holes.size:
        assert bool(np.all(segment[holes] == BLACK))
        return True
    return False


def _room_graph(
    grid: np.ndarray,
    h_lines: list[int],
    v_lines: list[int],
) -> dict[Room, set[Room]]:
    room_rows = len(h_lines) - 1
    room_cols = len(v_lines) - 1
    graph: dict[Room, set[Room]] = {
        (rr, cc): set() for rr in range(room_rows) for cc in range(room_cols)
    }

    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            divider_col = int(v_lines[cc + 1])
            if _vertical_door(grid, h_lines, rr, divider_col):
                left = (rr, cc)
                right = (rr, cc + 1)
                graph[left].add(right)
                graph[right].add(left)

    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            divider_row = int(h_lines[rr + 1])
            if _horizontal_door(grid, v_lines, divider_row, cc):
                top = (rr, cc)
                bottom = (rr + 1, cc)
                graph[top].add(bottom)
                graph[bottom].add(top)

    return graph


def _graph_components(graph: dict[Room, set[Room]]) -> list[list[Room]]:
    seen: set[Room] = set()
    components: list[list[Room]] = []
    for start in graph:
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        comp: list[Room] = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def _paint_room(out: np.ndarray, box: Box, count: int) -> None:
    r0, r1, c0, c1 = box
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    if count == 0:
        patch[:] = BLACK
        return
    if count == 1:
        patch[:] = GREEN
        return
    if count == 2:
        patch[:] = BLACK
        patch[::2, :] = YELLOW
        return
    if count == 3:
        patch[:] = BLACK
        patch[:, ::2] = ORANGE
        return

    patch[:] = BLACK
    rows, cols = np.indices(patch.shape)
    patch[(rows + cols) % 2 == 0] = MAGENTA


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group rectangular rooms by single-pixel wall doors, count group markers,
    and fill each group's rooms with the pattern selected by that count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h_lines = _find_wall_lines(g, axis=0, wall_color=BLUE)
    v_lines = _find_wall_lines(g, axis=1, wall_color=BLUE)
    boxes = _room_boxes(h_lines, v_lines)

    for box in boxes.values():
        r0, r1, c0, c1 = box
        interior = g[r0 : r1 + 1, c0 : c1 + 1]
        assert not bool(np.any(interior == BLUE)), "room interiors cannot contain walls"

    graph = _room_graph(g, h_lines, v_lines)
    components = _graph_components(graph)

    room_counts = {
        room: int(np.count_nonzero(g[box[0] : box[1] + 1, box[2] : box[3] + 1] == RED))
        for room, box in boxes.items()
    }

    out = g.copy()
    for comp in components:
        total = sum(room_counts[room] for room in comp)
        for room in comp:
            _paint_room(out, boxes[room], total)
    return out


def _wall_positions(spans: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for span in spans:
        cursor += int(span) + 1
        positions.append(cursor)
    return positions


def _adjacency_edges(room_rows: int, room_cols: int) -> list[tuple[Room, Room, str]]:
    edges: list[tuple[Room, Room, str]] = []
    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            edges.append(((rr, cc), (rr, cc + 1), "vertical"))
    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            edges.append(((rr, cc), (rr + 1, cc), "horizontal"))
    return edges


def _components_from_edges(
    room_rows: int,
    room_cols: int,
    open_edge_ids: set[int],
    edges: list[tuple[Room, Room, str]],
) -> list[list[Room]]:
    graph: dict[Room, set[Room]] = {
        (rr, cc): set() for rr in range(room_rows) for cc in range(room_cols)
    }
    for idx in open_edge_ids:
        a, b, _kind = edges[idx]
        graph[a].add(b)
        graph[b].add(a)
    return _graph_components(graph)


def _sample_open_edges(
    room_rows: int,
    room_cols: int,
) -> tuple[list[tuple[Room, Room, str]], set[int], list[list[Room]]]:
    edges = _adjacency_edges(room_rows, room_cols)
    for _ in range(200):
        prob = random.uniform(*DOOR_PROB_RANGE)
        open_edge_ids = {idx for idx in range(len(edges)) if random.random() < prob}
        if not (0 < len(open_edge_ids) < len(edges)):
            continue
        components = _components_from_edges(room_rows, room_cols, open_edge_ids, edges)
        sizes = sorted(len(comp) for comp in components)
        if len(components) < 2:
            continue
        if not any(size > 1 for size in sizes):
            continue
        return edges, open_edge_ids, components
    raise RuntimeError("failed to sample a useful door graph")


def _box_area(box: Box) -> int:
    r0, r1, c0, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _sample_group_counts(
    components: list[list[Room]],
    boxes: dict[Room, Box],
) -> dict[frozenset[Room], int]:
    shuffled = [list(comp) for comp in components]
    random.shuffle(shuffled)
    category_order = list(range(5))
    random.shuffle(category_order)

    counts: dict[frozenset[Room], int] = {}
    for idx, comp in enumerate(shuffled):
        category = category_order[idx] if idx < min(len(shuffled), 5) else random.randrange(5)
        area = sum(_box_area(boxes[room]) for room in comp)
        if category < 4:
            total = category
        else:
            total = random.randint(4, min(6, area))
        counts[frozenset(comp)] = int(total)
    return counts


def _room_cells(box: Box) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _place_markers(
    grid: np.ndarray,
    components: list[list[Room]],
    boxes: dict[Room, Box],
    counts: dict[frozenset[Room], int],
) -> None:
    for comp in components:
        total = counts[frozenset(comp)]
        if total == 0:
            continue
        all_cells = []
        for room in comp:
            all_cells.extend(_room_cells(boxes[room]))
        picks = random.sample(all_cells, total)
        for r, c in picks:
            grid[r, c] = RED


def _expected_output(
    grid: np.ndarray,
    components: list[list[Room]],
    boxes: dict[Room, Box],
) -> np.ndarray:
    out = grid.copy()
    for comp in components:
        total = 0
        for room in comp:
            r0, r1, c0, c1 = boxes[room]
            total += int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == RED))
        for room in comp:
            _paint_room(out, boxes[room], total)
    return out


def _build_grid(
    room_rows: int,
    room_cols: int,
    row_spans: list[int],
    col_spans: list[int],
    edges: list[tuple[Room, Room, str]],
    open_edge_ids: set[int],
) -> tuple[np.ndarray, list[int], list[int]]:
    h_lines = _wall_positions(row_spans)
    v_lines = _wall_positions(col_spans)
    height = int(h_lines[-1] + 1)
    width = int(v_lines[-1] + 1)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[h_lines, :] = BLUE
    grid[:, v_lines] = BLUE

    for idx in open_edge_ids:
        a, b, kind = edges[idx]
        if kind == "vertical":
            rr = a[0]
            divider_col = int(v_lines[min(a[1], b[1]) + 1])
            door_row = random.randint(int(h_lines[rr] + 1), int(h_lines[rr + 1] - 1))
            grid[door_row, divider_col] = BLACK
        else:
            cc = a[1]
            divider_row = int(h_lines[min(a[0], b[0]) + 1])
            door_col = random.randint(int(v_lines[cc] + 1), int(v_lines[cc + 1] - 1))
            grid[divider_row, door_col] = BLACK

    return grid, h_lines, v_lines


def _generate_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    row_spans = [random.randint(*ROOM_SPAN_RANGE) for _ in range(room_rows)]
    col_spans = [random.randint(*ROOM_SPAN_RANGE) for _ in range(room_cols)]

    edges, open_edge_ids, components = _sample_open_edges(room_rows, room_cols)
    grid, h_lines, v_lines = _build_grid(
        room_rows, room_cols, row_spans, col_spans, edges, open_edge_ids
    )
    boxes = _room_boxes(h_lines, v_lines)

    counts = _sample_group_counts(components, boxes)
    _place_markers(grid, components, boxes, counts)

    expected = _expected_output(grid, components, boxes)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle")
