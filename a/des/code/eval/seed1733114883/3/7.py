import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
MAROON = 9

NON_RED_BASE_COLORS = [1, 3, 5, 6, 7, 8]
ALL_BASE_COLORS = [1, 2, 3, 5, 6, 7, 8]

MIN_ROOM_SIDE = 4
GEN_RETRIES = 500


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _replace_marker_cells(grid: np.ndarray) -> np.ndarray:
    analysis = np.asarray(grid, dtype=int).copy()
    h, w = analysis.shape
    for marker in (MAROON, YELLOW):
        pts = np.argwhere(analysis == marker)
        assert pts.shape == (1, 2), f"expected exactly one marker of color {marker}"
        r, c = map(int, pts[0])
        colors = [
            int(analysis[nr, nc])
            for nr, nc in _neighbors8(r, c, h, w)
            if int(analysis[nr, nc]) not in {BLACK, MAROON, YELLOW}
        ]
        assert colors, "marker must sit inside a room"
        analysis[r, c] = Counter(colors).most_common(1)[0][0]
    return analysis


def _extract_rooms(analysis: np.ndarray) -> list[dict]:
    rooms = []
    for color in sorted(int(c) for c in np.unique(analysis) if int(c) != BLACK):
        coords = np.argwhere(analysis == color)
        row_counts = Counter(int(r) for r, _ in coords)
        col_counts = Counter(int(c) for _, c in coords)
        dense_rows = sorted(r for r, count in row_counts.items() if count >= MIN_ROOM_SIDE)
        dense_cols = sorted(c for c, count in col_counts.items() if count >= MIN_ROOM_SIDE)
        assert dense_rows and dense_cols, f"failed to recover room for color {color}"
        r0, r1 = dense_rows[0], dense_rows[-1]
        c0, c1 = dense_cols[0], dense_cols[-1]
        assert dense_rows == list(range(r0, r1 + 1))
        assert dense_cols == list(range(c0, c1 + 1))
        patch = analysis[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "room interior must be a solid rectangle"
        rooms.append({"index": len(rooms), "color": color, "bbox": (r0, c0, r1, c1)})
    assert 4 <= len(rooms) <= 7
    return rooms


def _shared_wall_segment(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]):
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b

    if ac1 + 2 == bc0:
        lo, hi = max(ar0, br0), min(ar1, br1)
        if lo <= hi:
            return [(r, ac1 + 1) for r in range(lo, hi + 1)]
    if bc1 + 2 == ac0:
        lo, hi = max(ar0, br0), min(ar1, br1)
        if lo <= hi:
            return [(r, bc1 + 1) for r in range(lo, hi + 1)]
    if ar1 + 2 == br0:
        lo, hi = max(ac0, bc0), min(ac1, bc1)
        if lo <= hi:
            return [(ar1 + 1, c) for c in range(lo, hi + 1)]
    if br1 + 2 == ac0:
        lo, hi = max(ac0, bc0), min(ac1, bc1)
        if lo <= hi:
            return [(br1 + 1, c) for c in range(lo, hi + 1)]
    return None


def _room_containing(rooms: list[dict], point: tuple[int, int]) -> int:
    r, c = point
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return int(room["index"])
    raise AssertionError("point is not inside any room")


def _build_channel_graph(analysis: np.ndarray, rooms: list[dict]) -> dict[int, dict[int, list[tuple[int, int]]]]:
    graph = {int(room["index"]): {} for room in rooms}
    for i, room_a in enumerate(rooms):
        for room_b in rooms[i + 1 :]:
            segment = _shared_wall_segment(room_a["bbox"], room_b["bbox"])
            if segment is None:
                continue
            a_to_b = [(r, c) for r, c in segment if int(analysis[r, c]) == int(room_b["color"])]
            b_to_a = [(r, c) for r, c in segment if int(analysis[r, c]) == int(room_a["color"])]
            if a_to_b and b_to_a:
                cells = a_to_b + b_to_a
                ia = int(room_a["index"])
                ib = int(room_b["index"])
                graph[ia][ib] = cells
                graph[ib][ia] = cells
    return graph


def _path_between(graph: dict[int, dict[int, list[tuple[int, int]]]], start: int, goal: int) -> list[int]:
    if start == goal:
        return [start]
    prev = {start: None}
    queue = deque([start])
    while queue and goal not in prev:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt in prev:
                continue
            prev[nxt] = node
            queue.append(nxt)
    assert goal in prev, "goal room must be reachable from the traveler room"
    path = [goal]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build channels from reciprocal wall windows, then mark the unique room-to-room path from traveler to goal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    traveler = np.argwhere(g == MAROON)
    target = np.argwhere(g == YELLOW)
    assert traveler.shape == (1, 2)
    assert target.shape == (1, 2)

    analysis = _replace_marker_cells(g)
    rooms = _extract_rooms(analysis)
    graph = _build_channel_graph(analysis, rooms)

    start_room = _room_containing(rooms, tuple(map(int, traveler[0])))
    goal_room = _room_containing(rooms, tuple(map(int, target[0])))
    room_path = _path_between(graph, start_room, goal_room)

    out = g.copy()
    for a, b in zip(room_path, room_path[1:], strict=False):
        for r, c in graph[a][b]:
            out[r, c] = RED
    return out


def _rect_size(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _candidate_orientations(rect: tuple[int, int, int, int]) -> list[str]:
    height, width = _rect_size(rect)
    options = []
    if width >= 2 * MIN_ROOM_SIDE + 1:
        options.append("v")
    if height >= 2 * MIN_ROOM_SIDE + 1:
        options.append("h")
    return options


def _split_rect(
    rect: tuple[int, int, int, int], orient: str, rng: random.Random
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect
    if orient == "v":
        wall = rng.randint(c0 + MIN_ROOM_SIDE, c1 - MIN_ROOM_SIDE)
        return (r0, c0, r1, wall - 1), (r0, wall + 1, r1, c1)
    wall = rng.randint(r0 + MIN_ROOM_SIDE, r1 - MIN_ROOM_SIDE)
    return (r0, c0, wall - 1, c1), (wall + 1, c0, r1, c1)


def _partition_rooms(rng: random.Random, h: int, w: int, room_count: int):
    rects = [(1, 1, h - 2, w - 2)]
    for _ in range(room_count - 1):
        candidates = []
        for idx, rect in enumerate(rects):
            area = (_rect_size(rect)[0] * _rect_size(rect)[1])
            for orient in _candidate_orientations(rect):
                candidates.extend([(idx, orient)] * area)
        if not candidates:
            return None
        idx, orient = rng.choice(candidates)
        rect = rects.pop(idx)
        first, second = _split_rect(rect, orient, rng)
        rects.extend([first, second])
    return rects


def _adjacency_data(rects: list[tuple[int, int, int, int]]):
    graph = {i: {} for i in range(len(rects))}
    edges = []
    for i, rect_a in enumerate(rects):
        for j in range(i + 1, len(rects)):
            segment = _shared_wall_segment(rect_a, rects[j])
            if segment is None:
                continue
            graph[i][j] = segment
            graph[j][i] = segment
            edges.append((i, j, segment))
    return graph, edges


def _graph_distances(graph: dict[int, dict[int, list[tuple[int, int]]]], start: int) -> dict[int, int]:
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


def _interior_cell(rect: tuple[int, int, int, int], rng: random.Random) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return rng.randint(r0 + 1, r1 - 1), rng.randint(c0 + 1, c1 - 1)


def _choose_room_colors(
    room_count: int, graph: dict[int, dict[int, list[tuple[int, int]]]], rng: random.Random
) -> list[int]:
    if room_count < 7:
        return rng.sample(NON_RED_BASE_COLORS, room_count)

    leaves = [room for room, nbrs in graph.items() if len(nbrs) == 1]
    assert leaves, "a tree with more than one node must have a leaf"
    red_leaf = rng.choice(leaves)
    remaining = rng.sample(NON_RED_BASE_COLORS, room_count - 1)
    colors = [None] * room_count
    colors[red_leaf] = RED
    slots = [idx for idx in range(room_count) if idx != red_leaf]
    rng.shuffle(slots)
    for idx, color in zip(slots, remaining, strict=False):
        colors[idx] = color
    return [int(color) for color in colors]


def _choose_start_goal(
    colors: list[int], graph: dict[int, dict[int, list[tuple[int, int]]]], rng: random.Random
) -> tuple[int, int]:
    candidates = [idx for idx, color in enumerate(colors) if color != RED]
    pairs = []
    long_pairs = []
    for i, start in enumerate(candidates):
        dist = _graph_distances(graph, start)
        for goal in candidates[i + 1 :]:
            if goal not in dist:
                continue
            pair = (start, goal)
            pairs.append(pair)
            if dist[goal] >= 2:
                long_pairs.append(pair)
    assert pairs, "need at least one valid start-goal pair"
    pool = long_pairs if long_pairs and rng.random() < 0.7 else pairs
    return rng.choice(pool)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    room_count = rng.choice([4, 5, 6, 7])
    if room_count == 7 and (h < 23 or w < 23):
        room_count = rng.choice([4, 5, 6])

    rects = _partition_rooms(rng, h, w, room_count)
    if rects is None:
        return None

    graph, edges = _adjacency_data(rects)
    degrees = [len(graph[idx]) for idx in range(room_count)]
    if any(deg < 1 or deg > 3 for deg in degrees):
        return None
    if len(edges) != room_count - 1:
        return None
    if any(len(segment) < 4 for _, _, segment in edges):
        return None

    colors = _choose_room_colors(room_count, graph, rng)
    grid = np.zeros((h, w), dtype=int)
    for rect, color in zip(rects, colors, strict=False):
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    for a, b, segment in edges:
        interior = segment[1:-1]
        if len(interior) < 2:
            return None
        pos_a, pos_b = rng.sample(interior, 2)
        grid[pos_a] = colors[b]
        grid[pos_b] = colors[a]

    start_room, goal_room = _choose_start_goal(colors, graph, rng)
    start_pos = _interior_cell(rects[start_room], rng)
    goal_pos = _interior_cell(rects[goal_room], rng)
    grid[start_pos] = MAROON
    grid[goal_pos] = YELLOW

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 7 input")
