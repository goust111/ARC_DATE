from __future__ import annotations

from itertools import combinations
import random

import numpy as np


WALL_COLOR = 5
BG_COLOR = 0
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

ROOM_SHAPES = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
ROOM_SIZE_RANGE = (2, 5)
WINDOW_MULTI_MIN_LENGTH = 3
GENERATION_ATTEMPTS = 400
EDGE_LAYOUT_ATTEMPTS = 200


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[counts.argmax()])


def _board_colors(board: np.ndarray) -> tuple[int, int]:
    colors = [int(color) for color in np.unique(board)]
    assert len(colors) == 2, "board must contain only background and wall colors"
    corners = {int(board[0, 0]), int(board[0, -1]), int(board[-1, 0]), int(board[-1, -1])}
    assert len(corners) == 1, "outer border corners must all be wall pixels"
    wall = corners.pop()
    bg = next(color for color in colors if color != wall)
    return bg, wall


def _legend_colors(grid: np.ndarray, bg: int, wall: int) -> list[int]:
    legend = [int(grid[0, c]) for c in range(grid.shape[1]) if int(grid[0, c]) not in {bg, wall}]
    assert legend, "legend row must contain at least one color"
    assert len(legend) == len(set(legend)), "legend colors must be unique"
    return legend


def _wall_axes(board: np.ndarray, wall: int) -> tuple[list[int], list[int]]:
    row_runs = (board[:, :-1] == wall) & (board[:, 1:] == wall)
    col_runs = (board[:-1, :] == wall) & (board[1:, :] == wall)

    wall_rows = [int(r) for r in np.flatnonzero(row_runs.any(axis=1))]
    wall_cols = [int(c) for c in np.flatnonzero(col_runs.any(axis=0))]

    assert wall_rows and wall_cols, "failed to detect wall lines"
    assert wall_rows[0] == 0 and wall_rows[-1] == board.shape[0] - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == board.shape[1] - 1
    assert all(b > a for a, b in zip(wall_rows, wall_rows[1:])), "wall rows must be distinct"
    assert all(b > a for a, b in zip(wall_cols, wall_cols[1:])), "wall cols must be distinct"
    return wall_rows, wall_cols


def _room_rects(wall_rows: list[int], wall_cols: list[int]) -> tuple[list[tuple[int, int, int, int]], int, int]:
    n_room_rows = len(wall_rows) - 1
    n_room_cols = len(wall_cols) - 1
    assert n_room_rows >= 1 and n_room_cols >= 1

    rooms: list[tuple[int, int, int, int]] = []
    for rr in range(n_room_rows):
        r0 = wall_rows[rr] + 1
        r1 = wall_rows[rr + 1] - 1
        assert r0 <= r1, "room height must be positive"
        for cc in range(n_room_cols):
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            assert c0 <= c1, "room width must be positive"
            rooms.append((r0, c0, r1, c1))
    return rooms, n_room_rows, n_room_cols


def _view_graph(
    board: np.ndarray,
    wall: int,
    wall_rows: list[int],
    wall_cols: list[int],
    n_room_rows: int,
    n_room_cols: int,
) -> list[set[int]]:
    room_count = n_room_rows * n_room_cols
    graph = [set() for _ in range(room_count)]

    def rid(rr: int, cc: int) -> int:
        return rr * n_room_cols + cc

    for rr in range(n_room_rows):
        for cc in range(n_room_cols - 1):
            wall_c = wall_cols[cc + 1]
            seg = board[wall_rows[rr] + 1 : wall_rows[rr + 1], wall_c]
            if int(np.count_nonzero(seg != wall)) == 1:
                a = rid(rr, cc)
                b = rid(rr, cc + 1)
                graph[a].add(b)
                graph[b].add(a)

    for rr in range(n_room_rows - 1):
        for cc in range(n_room_cols):
            wall_r = wall_rows[rr + 1]
            seg = board[wall_r, wall_cols[cc] + 1 : wall_cols[cc + 1]]
            if int(np.count_nonzero(seg != wall)) == 1:
                a = rid(rr, cc)
                b = rid(rr + 1, cc)
                graph[a].add(b)
                graph[b].add(a)

    return graph


def _room_fill_colors(
    rooms: list[tuple[int, int, int, int]],
    graph: list[set[int]],
    legend: list[int],
) -> dict[int, int]:
    order = sorted(range(len(rooms)), key=lambda idx: (-len(graph[idx]), rooms[idx][1], rooms[idx][0]))
    colors: dict[int, int] = {}
    legend_idx = 0

    for room_id in order:
        if room_id in colors:
            continue
        assert legend_idx < len(legend), "legend is too short for the number of color groups"
        color = int(legend[legend_idx])
        legend_idx += 1

        stack = [room_id]
        colors[room_id] = color
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if neighbor not in colors:
                    colors[neighbor] = color
                    stack.append(neighbor)

    assert len(colors) == len(rooms)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color wall-separated rooms by propagating legend colors across single-window view links.

    Input structure:
    - The top row is a legend row containing distinct non-black, non-wall colors in left-to-right order.
    - Below the legend, grey one-cell-thick orthogonal walls partition the board into rectangular rooms on a black background.
    - A shared wall segment between two neighboring rooms may contain zero or more one-cell holes that act as windows.

    Rule:
    - Detect the room grid from the wall rows and wall columns.
    - Two adjacent rooms belong to the same view group iff their shared wall segment contains exactly one window.
    - Rank all rooms by descending count of such view links; break ties by left-to-right, then top-to-bottom room position.
    - Traverse that ranking: whenever an uncolored room is reached, assign the next legend color and flood that color through its entire view-connected component.
    - Fill each room interior with its assigned color while keeping the legend and the grey walls unchanged.

    Key insight:
    The windows are not openings to flood through geometrically; they define a graph on rectangular rooms, and the legend colors are assigned to graph components according to a degree-based room ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3

    board = g[1:, :]
    bg, wall = _board_colors(board)
    legend = _legend_colors(g, bg, wall)
    wall_rows, wall_cols = _wall_axes(board, wall)
    rooms, n_room_rows, n_room_cols = _room_rects(wall_rows, wall_cols)
    graph = _view_graph(board, wall, wall_rows, wall_cols, n_room_rows, n_room_cols)
    fill_colors = _room_fill_colors(rooms, graph, legend)

    out = g.copy()
    out_board = out[1:, :].copy()
    for room_id, color in fill_colors.items():
        r0, c0, r1, c1 = rooms[room_id]
        out_board[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    out[1:, :] = out_board
    return out


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True


def _line_positions(lengths: list[int]) -> list[int]:
    pos = [0]
    for size in lengths:
        pos.append(pos[-1] + int(size) + 1)
    return pos


def _room_id(room_row: int, room_col: int, n_room_cols: int) -> int:
    return room_row * n_room_cols + room_col


def _segment_patterns(length: int, count: int) -> list[tuple[int, ...]]:
    if count < 0 or count >= length:
        return []
    patterns: list[tuple[int, ...]] = []
    for holes in combinations(range(length), count):
        cells = [True] + [idx not in holes for idx in range(length)] + [True]
        if any(cells[i] and cells[i + 1] for i in range(len(cells) - 1)):
            patterns.append(tuple(int(x) for x in holes))
    return patterns


def _sample_window_positions(length: int, count: int) -> tuple[int, ...] | None:
    patterns = _segment_patterns(length, count)
    if not patterns:
        return None
    return random.choice(patterns)


def _sample_layout() -> tuple[int, int, list[int], list[int]] | None:
    n_room_rows, n_room_cols = random.choice(ROOM_SHAPES)
    room_count = n_room_rows * n_room_cols

    for _ in range(EDGE_LAYOUT_ATTEMPTS):
        heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_rows)]
        widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_room_cols)]
        board_h = sum(heights) + n_room_rows + 1
        board_w = sum(widths) + n_room_cols + 1
        if board_h <= 29 and board_w <= 30 and board_w >= room_count:
            return n_room_rows, n_room_cols, heights, widths
    return None


def _all_walls(
    n_room_rows: int,
    n_room_cols: int,
    heights: list[int],
    widths: list[int],
) -> list[dict[str, int]]:
    walls: list[dict[str, int]] = []
    for rr in range(n_room_rows):
        for cc in range(n_room_cols - 1):
            walls.append(
                {
                    "kind": 0,
                    "row": rr,
                    "col": cc,
                    "length": int(heights[rr]),
                    "a": _room_id(rr, cc, n_room_cols),
                    "b": _room_id(rr, cc + 1, n_room_cols),
                }
            )
    for rr in range(n_room_rows - 1):
        for cc in range(n_room_cols):
            walls.append(
                {
                    "kind": 1,
                    "row": rr,
                    "col": cc,
                    "length": int(widths[cc]),
                    "a": _room_id(rr, cc, n_room_cols),
                    "b": _room_id(rr + 1, cc, n_room_cols),
                }
            )
    return walls


def _selected_view_walls(
    room_count: int,
    walls: list[dict[str, int]],
    need_hub: bool,
    need_both_orientations: bool,
) -> set[int] | None:
    min_components = 2
    max_components = room_count - 2 if need_hub or need_both_orientations else room_count - 1
    if max_components < min_components:
        return None

    for _ in range(EDGE_LAYOUT_ATTEMPTS):
        target_components = random.randint(min_components, max_components)
        dsu = _DSU(room_count)
        wall_ids = list(range(len(walls)))
        random.shuffle(wall_ids)
        selected: set[int] = set()

        for wall_id in wall_ids:
            if dsu.count == target_components:
                break
            wall = walls[wall_id]
            if dsu.union(wall["a"], wall["b"]):
                selected.add(wall_id)

        if dsu.count != target_components:
            continue

        extra_candidates = [
            wall_id
            for wall_id, wall in enumerate(walls)
            if wall_id not in selected and dsu.find(wall["a"]) == dsu.find(wall["b"])
        ]
        random.shuffle(extra_candidates)
        extra_count = random.randint(0, min(2, len(extra_candidates)))
        for wall_id in extra_candidates[:extra_count]:
            selected.add(wall_id)

        degrees = [0] * room_count
        orientations: set[int] = set()
        for wall_id in selected:
            wall = walls[wall_id]
            degrees[wall["a"]] += 1
            degrees[wall["b"]] += 1
            orientations.add(wall["kind"])

        if need_hub and max(degrees) < 2:
            continue
        if need_both_orientations and len(orientations) < 2:
            continue
        return selected

    return None


def _window_counts(
    walls: list[dict[str, int]],
    selected: set[int],
    need_multiwindow: bool,
) -> list[int] | None:
    counts = [0] * len(walls)
    multi_candidates = [
        wall_id
        for wall_id, wall in enumerate(walls)
        if wall_id not in selected and wall["length"] >= WINDOW_MULTI_MIN_LENGTH and _segment_patterns(wall["length"], 2)
    ]

    forced_multi = None
    if need_multiwindow:
        if not multi_candidates:
            return None
        forced_multi = random.choice(multi_candidates)

    for wall_id, wall in enumerate(walls):
        if wall_id in selected:
            counts[wall_id] = 1
            continue
        if wall_id == forced_multi:
            counts[wall_id] = 2
            continue
        if wall["length"] >= WINDOW_MULTI_MIN_LENGTH and random.random() < 0.35 and _segment_patterns(wall["length"], 2):
            counts[wall_id] = 2
        else:
            counts[wall_id] = 0

    return counts


def _wall_windows(walls: list[dict[str, int]], counts: list[int]) -> list[tuple[int, ...]] | None:
    windows: list[tuple[int, ...]] = []
    for wall, count in zip(walls, counts, strict=False):
        holes = _sample_window_positions(wall["length"], count)
        if holes is None:
            return None
        windows.append(holes)
    return windows


def _render_board(
    n_room_rows: int,
    n_room_cols: int,
    heights: list[int],
    widths: list[int],
    walls: list[dict[str, int]],
    windows: list[tuple[int, ...]],
) -> tuple[np.ndarray, list[int], list[int]]:
    wall_rows = _line_positions(heights)
    wall_cols = _line_positions(widths)
    board = np.full((wall_rows[-1] + 1, wall_cols[-1] + 1), BG_COLOR, dtype=int)
    board[wall_rows, :] = WALL_COLOR
    board[:, wall_cols] = WALL_COLOR

    for wall, holes in zip(walls, windows, strict=False):
        if wall["kind"] == 0:
            wall_c = wall_cols[wall["col"] + 1]
            start_r = wall_rows[wall["row"]] + 1
            for offset in holes:
                board[start_r + offset, wall_c] = BG_COLOR
        else:
            wall_r = wall_rows[wall["row"] + 1]
            start_c = wall_cols[wall["col"]] + 1
            for offset in holes:
                board[wall_r, start_c + offset] = BG_COLOR

    return board, wall_rows, wall_cols


def _graph_from_selected(room_count: int, walls: list[dict[str, int]], selected: set[int]) -> list[set[int]]:
    graph = [set() for _ in range(room_count)]
    for wall_id in selected:
        wall = walls[wall_id]
        a = wall["a"]
        b = wall["b"]
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _component_count(graph: list[set[int]]) -> int:
    seen: set[int] = set()
    count = 0
    for start in range(len(graph)):
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return count


def _expected_output(
    board: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    graph: list[set[int]],
    legend_colors: list[int],
) -> np.ndarray:
    rooms, _n_room_rows, _n_room_cols = _room_rects(wall_rows, wall_cols)
    fill_colors = _room_fill_colors(rooms, graph, legend_colors)
    out = board.copy()
    for room_id, color in fill_colors.items():
        r0, c0, r1, c1 = rooms[room_id]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _legend_row(width: int, legend_colors: list[int]) -> np.ndarray:
    row = np.full(width, BG_COLOR, dtype=int)
    positions = sorted(random.sample(range(width), len(legend_colors)))
    for pos, color in zip(positions, legend_colors, strict=False):
        row[pos] = int(color)
    return row


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        layout = _sample_layout()
        if layout is None:
            continue

        n_room_rows, n_room_cols, heights, widths = layout
        room_count = n_room_rows * n_room_cols
        walls = _all_walls(n_room_rows, n_room_cols, heights, widths)

        need_hub = random.choice([True, False])
        need_both_orientations = random.choice([True, False])
        need_multiwindow = random.choice([True, False])

        selected = _selected_view_walls(room_count, walls, need_hub, need_both_orientations)
        if selected is None:
            continue

        counts = _window_counts(walls, selected, need_multiwindow)
        if counts is None:
            continue

        windows = _wall_windows(walls, counts)
        if windows is None:
            continue

        graph = _graph_from_selected(room_count, walls, selected)
        components = _component_count(graph)
        if not (2 <= components <= len(LEGEND_COLOR_POOL)):
            continue

        degrees = [len(neighbors) for neighbors in graph]
        if max(degrees) == 0:
            continue
        if need_multiwindow and all(count <= 1 for count in counts):
            continue
        if need_hub and max(degrees) < 2:
            continue
        if need_both_orientations and len({walls[wall_id]["kind"] for wall_id in selected}) < 2:
            continue

        board, wall_rows, wall_cols = _render_board(n_room_rows, n_room_cols, heights, widths, walls, windows)
        legend_colors = random.sample(LEGEND_COLOR_POOL, components)
        inp = np.vstack([_legend_row(board.shape[1], legend_colors), board])

        expected_board = _expected_output(board, wall_rows, wall_cols, graph, legend_colors)
        expected = np.vstack([inp[0:1, :], expected_board])

        actual = main(inp)
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
