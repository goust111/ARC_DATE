from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
WALL = 5
COLOR_CYCLE = [2, 1, 3, 4, 7, 6, 8, 9]  # red, blue, green, yellow, orange, magenta, azure, maroon
FADE_COLOR = {
    color: COLOR_CYCLE[(idx + 1) % len(COLOR_CYCLE)]
    for idx, color in enumerate(COLOR_CYCLE)
}

ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (2, 5)
MAX_GRID_SIZE = 30
MAX_MARGIN_TOTAL = 4
GEN_ATTEMPTS = 600


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _cluster_lines(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _find_wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    wall_pts = np.argwhere(grid == WALL)
    assert wall_pts.size > 0, "expected grey walls"
    rmin = int(wall_pts[:, 0].min())
    rmax = int(wall_pts[:, 0].max())
    cmin = int(wall_pts[:, 1].min())
    cmax = int(wall_pts[:, 1].max())
    sub = grid[rmin : rmax + 1, cmin : cmax + 1]

    row_counts = np.count_nonzero(sub == WALL, axis=1)
    col_counts = np.count_nonzero(sub == WALL, axis=0)
    row_threshold = max(2, int(np.ceil(0.6 * sub.shape[1])))
    col_threshold = max(2, int(np.ceil(0.6 * sub.shape[0])))

    row_hits = np.flatnonzero(row_counts >= row_threshold).tolist()
    col_hits = np.flatnonzero(col_counts >= col_threshold).tolist()
    assert row_hits and col_hits, "expected wall lattice lines"

    wall_rows = [
        rmin + max(group, key=lambda idx: int(row_counts[idx]))
        for group in _cluster_lines([int(v) for v in row_hits])
    ]
    wall_cols = [
        cmin + max(group, key=lambda idx: int(col_counts[idx]))
        for group in _cluster_lines([int(v) for v in col_hits])
    ]

    assert len(wall_rows) >= 2 and len(wall_cols) >= 2
    assert wall_rows[0] == rmin and wall_rows[-1] == rmax
    assert wall_cols[0] == cmin and wall_cols[-1] == cmax
    assert all(b - a >= 2 for a, b in zip(wall_rows, wall_rows[1:], strict=False))
    assert all(b - a >= 2 for a, b in zip(wall_cols, wall_cols[1:], strict=False))
    return wall_rows, wall_cols


def _room_boxes(wall_rows: list[int], wall_cols: list[int]) -> dict[int, tuple[int, int, int, int]]:
    room_cols = len(wall_cols) - 1
    boxes: dict[int, tuple[int, int, int, int]] = {}
    room_id = 0
    for rr in range(len(wall_rows) - 1):
        for cc in range(room_cols):
            boxes[room_id] = (
                int(wall_rows[rr] + 1),
                int(wall_rows[rr + 1] - 1),
                int(wall_cols[cc] + 1),
                int(wall_cols[cc + 1] - 1),
            )
            room_id += 1
    return boxes


def _room_neighbors(room_rows: int, room_cols: int, room_id: int) -> list[int]:
    rr, cc = divmod(room_id, room_cols)
    out: list[int] = []
    if rr > 0:
        out.append(room_id - room_cols)
    if rr + 1 < room_rows:
        out.append(room_id + room_cols)
    if cc > 0:
        out.append(room_id - 1)
    if cc + 1 < room_cols:
        out.append(room_id + 1)
    return out


def _build_room_graph(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
) -> tuple[list[list[int]], dict[tuple[int, int], tuple[int, int]]]:
    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    adjacency = [set() for _ in range(room_rows * room_cols)]
    doors: dict[tuple[int, int], tuple[int, int]] = {}

    for rr in range(room_rows):
        top = int(wall_rows[rr] + 1)
        bottom = int(wall_rows[rr + 1])
        for cc in range(room_cols - 1):
            divider_col = int(wall_cols[cc + 1])
            segment = grid[top:bottom, divider_col]
            gaps = np.flatnonzero(segment == BLACK)
            assert np.all((segment == WALL) | (segment == BLACK))
            assert len(gaps) <= 1, "each shared wall may contain at most one door"
            if len(gaps) == 1:
                a = rr * room_cols + cc
                b = rr * room_cols + cc + 1
                key = _edge_key(a, b)
                door = (top + int(gaps[0]), divider_col)
                adjacency[a].add(b)
                adjacency[b].add(a)
                doors[key] = door

    for rr in range(room_rows - 1):
        divider_row = int(wall_rows[rr + 1])
        for cc in range(room_cols):
            left = int(wall_cols[cc] + 1)
            right = int(wall_cols[cc + 1])
            segment = grid[divider_row, left:right]
            gaps = np.flatnonzero(segment == BLACK)
            assert np.all((segment == WALL) | (segment == BLACK))
            assert len(gaps) <= 1, "each shared wall may contain at most one door"
            if len(gaps) == 1:
                a = rr * room_cols + cc
                b = (rr + 1) * room_cols + cc
                key = _edge_key(a, b)
                door = (divider_row, left + int(gaps[0]))
                adjacency[a].add(b)
                adjacency[b].add(a)
                doors[key] = door

    return [sorted(neighbors) for neighbors in adjacency], doors


def _extract_marker_pairs(
    grid: np.ndarray,
    boxes: dict[int, tuple[int, int, int, int]],
) -> dict[int, tuple[int, int]]:
    markers: dict[int, list[int]] = {}
    for room_id, (r0, r1, c0, c1) in boxes.items():
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(v) for v in np.unique(patch) if int(v) not in (BLACK, WALL)]
        assert len(colors) <= 1, "each room may contain at most one marker color"
        if not colors:
            continue
        color = colors[0]
        assert color in FADE_COLOR, "marker color must come from the visible color cycle"
        assert int(np.count_nonzero(patch == color)) == 1, "marker must be a single pixel"
        markers.setdefault(color, []).append(room_id)

    pairs: dict[int, tuple[int, int]] = {}
    for color, rooms in markers.items():
        assert len(rooms) == 2, "every marker color must appear in exactly two rooms"
        pairs[color] = (rooms[0], rooms[1])
    assert pairs, "expected at least one marker pair"
    return pairs


def _shortest_path(adjacency: list[list[int]], start: int, goal: int) -> list[int]:
    parent = {int(start): -1}
    queue = deque([int(start)])
    while queue and goal not in parent:
        room_id = queue.popleft()
        for neighbor in adjacency[room_id]:
            if neighbor in parent:
                continue
            parent[neighbor] = room_id
            queue.append(neighbor)
    assert goal in parent, "paired rooms must be connected by doors"

    path = [int(goal)]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored pair of marked rooms by coloring the doors on their unique shortest room path.

    Input structure:
    - The background is black and the walls are grey one-pixel lattice lines that partition the grid into rectangular rooms.
    - Adjacent rooms may have a single black gap on their shared wall; each such gap is a door.
    - Every room interior is otherwise black, except that some rooms contain exactly one colored marker pixel.
    - Marker colors come in pairs, so exactly two rooms share each non-black, non-grey marker color.

    Rule:
    - Recover the room lattice from the grey wall rows and columns, then build the room-adjacency graph from the black door gaps.
    - Pair the two rooms that contain the same marker color and find their shortest door-by-door room path.
    - Recolor every door on that path with the marker color.
    - For every intermediate room on that path, recolor only its black background cells with the next color in the visible cycle [red, blue, green, yellow, orange, magenta, azure, maroon], wrapping around at the end.

    Key insight:
    - The puzzle is solved at the room-graph level: wall gaps define connectivity, paired markers choose the endpoints, and the path interior uses a separate “next color” mapping rather than the endpoint color itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_rows, wall_cols = _find_wall_lines(g)
    boxes = _room_boxes(wall_rows, wall_cols)
    adjacency, doors = _build_room_graph(g, wall_rows, wall_cols)
    pairs = _extract_marker_pairs(g, boxes)

    out = g.copy()
    for color in sorted(pairs):
        start, goal = pairs[color]
        path = _shortest_path(adjacency, start, goal)
        for a, b in zip(path, path[1:], strict=False):
            out[doors[_edge_key(a, b)]] = int(color)
        fade = FADE_COLOR[color]
        for room_id in path[1:-1]:
            r0, r1, c0, c1 = boxes[room_id]
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[patch == BLACK] = fade
    return out


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

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
        return True


def _sample_sizes(count: int) -> list[int]:
    lo, hi = ROOM_SIZE_RANGE
    return [random.randint(lo, hi) for _ in range(count)]


def _wall_positions(sizes: list[int], offset: int) -> list[int]:
    positions = [int(offset)]
    for size in sizes:
        positions.append(positions[-1] + int(size) + 1)
    return positions


def _grid_edges(room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for room_id in range(room_rows * room_cols):
        for neighbor in _room_neighbors(room_rows, room_cols, room_id):
            if room_id < neighbor:
                edges.append((room_id, neighbor))
    return edges


def _adjacency_from_edges(
    room_rows: int,
    room_cols: int,
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    adjacency = [set() for _ in range(room_rows * room_cols)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return [sorted(neighbors) for neighbors in adjacency]


def _bfs_path_allowed(
    room_rows: int,
    room_cols: int,
    allowed: set[int],
    start: int,
    goal: int,
) -> list[int] | None:
    if start not in allowed or goal not in allowed:
        return None
    parent = {int(start): -1}
    queue = deque([int(start)])
    while queue and goal not in parent:
        room_id = queue.popleft()
        for neighbor in _room_neighbors(room_rows, room_cols, room_id):
            if neighbor not in allowed or neighbor in parent:
                continue
            parent[neighbor] = room_id
            queue.append(neighbor)
    if goal not in parent:
        return None
    path = [int(goal)]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _available_adjacent_paths(
    room_rows: int,
    room_cols: int,
    available: set[int],
) -> list[list[int]]:
    paths: list[list[int]] = []
    for a, b in _grid_edges(room_rows, room_cols):
        if a in available and b in available:
            paths.append([a, b])
    return paths


def _candidate_long_paths(
    room_rows: int,
    room_cols: int,
    available: set[int],
    min_len: int,
) -> list[list[int]]:
    rooms = sorted(available)
    paths: list[list[int]] = []
    for idx, start in enumerate(rooms):
        for goal in rooms[idx + 1 :]:
            path = _bfs_path_allowed(room_rows, room_cols, available, start, goal)
            if path is None or len(path) - 1 < min_len:
                continue
            paths.append(path)
    return paths


def _pick_weighted_path(paths: list[list[int]]) -> list[int]:
    weights = [max(1, len(path) - 1) for path in paths]
    return random.choices(paths, weights=weights, k=1)[0]


def _choose_pair_paths(room_rows: int, room_cols: int, mode: str) -> list[list[int]] | None:
    available = set(range(room_rows * room_cols))

    if mode == "adjacent":
        paths: list[list[int]] = []
        for _ in range(2):
            candidates = _available_adjacent_paths(room_rows, room_cols, available)
            if not candidates:
                return None
            path = random.choice(candidates)
            paths.append(path)
            available.difference_update(path)
        return paths

    if mode == "long":
        min_len = 3 if room_rows * room_cols >= 6 and random.random() < 0.5 else 2
        candidates = _candidate_long_paths(room_rows, room_cols, available, min_len)
        if not candidates:
            return None
        return [_pick_weighted_path(candidates)]

    if mode == "mixed":
        candidates = _candidate_long_paths(room_rows, room_cols, available, 2)
        if not candidates:
            return None
        long_path = _pick_weighted_path(candidates)
        available.difference_update(long_path)
        adjacent = _available_adjacent_paths(room_rows, room_cols, available)
        if not adjacent:
            return None
        paths = [long_path, random.choice(adjacent)]
        random.shuffle(paths)
        return paths

    raise ValueError(mode)


def _choose_pair_colors(count: int) -> list[int] | None:
    colors = COLOR_CYCLE[:]
    random.shuffle(colors)
    chosen: list[int] = []
    blocked: set[int] = set()
    for color in colors:
        if color in blocked or FADE_COLOR[color] in blocked:
            continue
        chosen.append(color)
        blocked.add(color)
        blocked.add(FADE_COLOR[color])
        if len(chosen) == count:
            return chosen
    return None


def _random_spanning_tree(
    room_rows: int,
    room_cols: int,
    mandatory_paths: list[list[int]],
) -> list[tuple[int, int]] | None:
    n_rooms = room_rows * room_cols
    all_edges = _grid_edges(room_rows, room_cols)
    mandatory_edges: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    for path in mandatory_paths:
        for a, b in zip(path, path[1:], strict=False):
            key = _edge_key(a, b)
            if key not in seen_edges:
                seen_edges.add(key)
                mandatory_edges.append(key)

    dsu = _DSU(n_rooms)
    tree_edges: list[tuple[int, int]] = []
    for a, b in mandatory_edges:
        if not dsu.union(a, b):
            return None
        tree_edges.append((a, b))

    others = [edge for edge in all_edges if _edge_key(*edge) not in seen_edges]
    random.shuffle(others)
    for a, b in others:
        if dsu.union(a, b):
            tree_edges.append(_edge_key(a, b))
            if len(tree_edges) == n_rooms - 1:
                break

    if len(tree_edges) != n_rooms - 1:
        return None
    return tree_edges


def _sample_margins(bbox_size: int) -> tuple[int, int] | None:
    slack = MAX_GRID_SIZE - bbox_size
    if slack < 0:
        return None
    total = random.randint(0, min(MAX_MARGIN_TOTAL, slack))
    lead = random.randint(0, total)
    return lead, total - lead


def _build_input(
    row_sizes: list[int],
    col_sizes: list[int],
    tree_edges: list[tuple[int, int]],
    pair_paths: list[list[int]],
    pair_colors: list[int],
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], dict[tuple[int, int], tuple[int, int]]] | None:
    bbox_height = sum(row_sizes) + len(row_sizes) + 1
    bbox_width = sum(col_sizes) + len(col_sizes) + 1
    vert_margins = _sample_margins(bbox_height)
    horiz_margins = _sample_margins(bbox_width)
    if vert_margins is None or horiz_margins is None:
        return None
    top_margin, bottom_margin = vert_margins
    left_margin, right_margin = horiz_margins

    wall_rows = _wall_positions(row_sizes, top_margin)
    wall_cols = _wall_positions(col_sizes, left_margin)
    height = wall_rows[-1] + bottom_margin + 1
    width = wall_cols[-1] + right_margin + 1
    if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for row in wall_rows:
        grid[row, wall_cols[0] : wall_cols[-1] + 1] = WALL
    for col in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, col] = WALL

    room_rows = len(row_sizes)
    room_cols = len(col_sizes)
    door_positions: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b in tree_edges:
        ra, ca = divmod(a, room_cols)
        rb, cb = divmod(b, room_cols)
        key = _edge_key(a, b)
        if ra == rb:
            divider_col = int(wall_cols[max(ca, cb)])
            door_row = random.randint(int(wall_rows[ra] + 1), int(wall_rows[ra + 1] - 1))
            grid[door_row, divider_col] = BLACK
            door_positions[key] = (door_row, divider_col)
        else:
            divider_row = int(wall_rows[max(ra, rb)])
            door_col = random.randint(int(wall_cols[ca] + 1), int(wall_cols[ca + 1] - 1))
            grid[divider_row, door_col] = BLACK
            door_positions[key] = (divider_row, door_col)

    boxes = _room_boxes(wall_rows, wall_cols)
    endpoints = Counter(room_id for path in pair_paths for room_id in (path[0], path[-1]))
    assert endpoints and all(count == 1 for count in endpoints.values())

    for path, color in zip(pair_paths, pair_colors, strict=False):
        for room_id in (path[0], path[-1]):
            r0, r1, c0, c1 = boxes[room_id]
            marker_r = random.randint(r0, r1)
            marker_c = random.randint(c0, c1)
            grid[marker_r, marker_c] = int(color)

    return grid, boxes, door_positions


def _expected_output(
    grid: np.ndarray,
    boxes: dict[int, tuple[int, int, int, int]],
    door_positions: dict[tuple[int, int], tuple[int, int]],
    pair_paths: list[list[int]],
    pair_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for path, color in zip(pair_paths, pair_colors, strict=False):
        for a, b in zip(path, path[1:], strict=False):
            out[door_positions[_edge_key(a, b)]] = int(color)
        fade = FADE_COLOR[color]
        for room_id in path[1:-1]:
            r0, r1, c0, c1 = boxes[room_id]
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[patch == BLACK] = fade
    return out


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    row_sizes = _sample_sizes(room_rows)
    col_sizes = _sample_sizes(room_cols)
    mode = random.choice(["adjacent", "long", "mixed"])

    pair_paths = _choose_pair_paths(room_rows, room_cols, mode)
    if pair_paths is None:
        return None
    pair_colors = _choose_pair_colors(len(pair_paths))
    if pair_colors is None:
        return None
    tree_edges = _random_spanning_tree(room_rows, room_cols, pair_paths)
    if tree_edges is None:
        return None

    built = _build_input(row_sizes, col_sizes, tree_edges, pair_paths, pair_colors)
    if built is None:
        return None
    grid, boxes, door_positions = built

    adjacency = _adjacency_from_edges(room_rows, room_cols, tree_edges)
    for path in pair_paths:
        actual = _shortest_path(adjacency, path[0], path[-1])
        if actual != path:
            return None

    expected = _expected_output(grid, boxes, door_positions, pair_paths, pair_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-pair corridor puzzle")
