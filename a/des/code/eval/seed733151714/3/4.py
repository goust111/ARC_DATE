from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (15, 24)
GRID_WIDTH_RANGE = (15, 24)
SEGMENT_COUNT_CHOICES = [3, 4]
SEGMENT_LENGTH_RANGE = (3, 8)
GEN_RETRIES = 500


def _in_bounds(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _component(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(coords)
    assert pts, "expected a non-empty component"
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        cell = stack.pop()
        r, c = cell
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _segment_endpoints(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg = sum((r + dr, c + dc) in pts for dr, dc in DIRS)
        assert deg in (1, 2), "segment must be a simple 4-connected polyline"
        degrees[(r, c)] = deg
    endpoints = {cell for cell, deg in degrees.items() if deg == 1}
    assert len(endpoints) == 2, "segment must have exactly two endpoints"
    return endpoints


def _parse_segments(g: np.ndarray) -> tuple[dict[int, dict[str, object]], dict[tuple[int, int], int]]:
    segments: dict[int, dict[str, object]] = {}
    segment_by_cell: dict[tuple[int, int], int] = {}
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, YELLOW)]
    counts = {color: int(np.sum(g == color)) for color in colors}

    for color, count in counts.items():
        if count == 2:
            continue
        assert count >= 3, "non-gate foreground colors must be path segments"
        coords = [tuple(map(int, cell)) for cell in np.argwhere(g == color)]
        assert _component(coords) == set(coords), "each segment color must form one component"
        endpoints = _segment_endpoints(coords)
        segments[color] = {"coords": coords, "endpoints": endpoints}
        for cell in coords:
            assert cell not in segment_by_cell, "segment cells must be disjoint"
            segment_by_cell[cell] = color

    assert len(segments) >= 2, "expected at least two path segments"
    return segments, segment_by_cell


def _parse_gates(
    g: np.ndarray,
    segments: dict[int, dict[str, object]],
    segment_by_cell: dict[tuple[int, int], int],
) -> tuple[dict[int, list[dict[str, object]]], dict[tuple[int, int], int], dict[int, tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = g.shape
    adjacency: dict[int, list[dict[str, object]]] = {color: [] for color in segments}
    endpoint_gate_uses: dict[tuple[int, int], int] = {
        cell: 0 for data in segments.values() for cell in data["endpoints"]  # type: ignore[index]
    }
    gate_pixels: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}

    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, YELLOW)]
    counts = {color: int(np.sum(g == color)) for color in colors}
    gate_colors = [color for color, count in counts.items() if count == 2]
    assert len(gate_colors) == len(segments) - 1, "a linear chain with n segments needs n-1 gates"

    seen_gaps: set[tuple[int, int]] = set()
    for gate_color in gate_colors:
        pts = [tuple(map(int, cell)) for cell in np.argwhere(g == gate_color)]
        assert len(pts) == 2
        p0, p1 = pts
        gate_pixels[gate_color] = (p0, p1)

        if p0[0] == p1[0]:
            assert abs(p0[1] - p1[1]) == 2, "horizontal gate posts must straddle one gap cell"
            gap = (p0[0], (p0[1] + p1[1]) // 2)
            path_cells = [(gap[0] - 1, gap[1]), (gap[0] + 1, gap[1])]
        else:
            assert p0[1] == p1[1], "gate posts must align horizontally or vertically"
            assert abs(p0[0] - p1[0]) == 2, "vertical gate posts must straddle one gap cell"
            gap = ((p0[0] + p1[0]) // 2, p0[1])
            path_cells = [(gap[0], gap[1] - 1), (gap[0], gap[1] + 1)]

        assert _in_bounds(h, w, gap)
        assert g[gap] == BG, "gate midpoint must be the black gap cell"
        assert gap not in seen_gaps, "each gate must define a unique gap"
        seen_gaps.add(gap)

        left, right = path_cells
        assert left in segment_by_cell and right in segment_by_cell, "gate gap must connect two segments"
        left_seg = segment_by_cell[left]
        right_seg = segment_by_cell[right]
        assert left_seg != right_seg, "a gate must connect distinct segments"
        assert left in segments[left_seg]["endpoints"]  # type: ignore[operator]
        assert right in segments[right_seg]["endpoints"]  # type: ignore[operator]

        adjacency[left_seg].append(
            {"neighbor": right_seg, "gate_color": gate_color, "gap": gap, "endpoint": left}
        )
        adjacency[right_seg].append(
            {"neighbor": left_seg, "gate_color": gate_color, "gap": gap, "endpoint": right}
        )
        endpoint_gate_uses[left] += 1
        endpoint_gate_uses[right] += 1

    assert all(uses <= 1 for uses in endpoint_gate_uses.values()), "each endpoint can meet at most one gate"
    return adjacency, endpoint_gate_uses, gate_pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the colored path fragments into one chain by reading each gate across its 1-cell gap,
    then recolor the ordered segments so the path starts yellow and switches to each gate's color
    after passing through that gate. Gate pixels stay unchanged and gap cells are filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    yellow_cells = np.argwhere(g == YELLOW)
    assert len(yellow_cells) == 1, "expected one yellow start marker"
    yellow = tuple(map(int, yellow_cells[0]))

    segments, segment_by_cell = _parse_segments(g)
    adjacency, endpoint_gate_uses, gate_pixels = _parse_gates(g, segments, segment_by_cell)

    start_touching = [cell for cell in _neighbors4(yellow, h, w) if cell in segment_by_cell]
    assert len(start_touching) == 1, "yellow marker must touch exactly one start segment cell"
    start_endpoint = start_touching[0]
    start_segment = segment_by_cell[start_endpoint]
    assert start_endpoint in segments[start_segment]["endpoints"]  # type: ignore[operator]

    degrees = {color: len(edges) for color, edges in adjacency.items()}
    assert all(deg in (1, 2) for deg in degrees.values()), "segment graph must be a simple chain"
    ends = [color for color, deg in degrees.items() if deg == 1]
    assert len(ends) == 2, "segment graph must have exactly two ends"
    assert start_segment in ends, "yellow must mark one end of the chain"

    for color, data in segments.items():
        endpoints = data["endpoints"]  # type: ignore[assignment]
        gate_count = sum(endpoint_gate_uses[cell] for cell in endpoints)
        if color == start_segment:
            assert gate_count == 1, "start segment should meet exactly one gate"
        elif degrees[color] == 1:
            assert gate_count == 1, "terminal segment should meet exactly one gate"
        else:
            assert gate_count == 2, "middle segments should meet two gates"

    order = [start_segment]
    ordered_edges: list[dict[str, object]] = []
    seen_segments = {start_segment}
    prev: int | None = None
    cur = start_segment
    while True:
        candidates = [edge for edge in adjacency[cur] if edge["neighbor"] != prev]
        if not candidates:
            break
        assert len(candidates) == 1, "path order must be unique"
        edge = candidates[0]
        ordered_edges.append(edge)
        prev = cur
        cur = int(edge["neighbor"])
        assert cur not in seen_segments, "segment chain must not loop"
        seen_segments.add(cur)
        order.append(cur)

    assert len(order) == len(segments), "all segments must belong to the same chain"

    out = np.full_like(g, BG)
    out[yellow] = YELLOW
    for gate_color, posts in gate_pixels.items():
        for r, c in posts:
            out[r, c] = gate_color

    recolor = {order[0]: YELLOW}
    for i in range(1, len(order)):
        recolor[order[i]] = int(ordered_edges[i - 1]["gate_color"])

    for color, data in segments.items():
        coords = data["coords"]  # type: ignore[assignment]
        rows, cols = zip(*coords)
        out[rows, cols] = recolor[color]

    for edge in ordered_edges:
        r, c = edge["gap"]  # type: ignore[misc]
        out[r, c] = int(edge["gate_color"])

    return out


def _valid_path_cell(
    cell: tuple[int, int],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    gaps: set[tuple[int, int]],
    *,
    allowed_occupied: set[tuple[int, int]],
    allowed_gaps: set[tuple[int, int]],
) -> bool:
    if not _in_bounds(h, w, cell) or cell in occupied or cell in gaps:
        return False
    for nbr in _neighbors4(cell, h, w):
        if nbr in occupied and nbr not in allowed_occupied:
            return False
        if nbr in gaps and nbr not in allowed_gaps:
            return False
    return True


def _grow_segment(
    start: tuple[int, int],
    length: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    gaps: set[tuple[int, int]],
    *,
    allowed_occupied_start: set[tuple[int, int]],
    allowed_gaps_start: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    local_occupied = set(occupied)
    if not _valid_path_cell(
        start,
        h,
        w,
        local_occupied,
        gaps,
        allowed_occupied=allowed_occupied_start,
        allowed_gaps=allowed_gaps_start,
    ):
        return None

    path = [start]
    local_occupied.add(start)

    def dfs() -> bool:
        if len(path) == length:
            return True

        cur = path[-1]
        choices = DIRS.copy()
        random.shuffle(choices)
        if len(path) >= 2 and random.random() < 0.6:
            prev = path[-2]
            straight = (cur[0] - prev[0], cur[1] - prev[1])
            choices.sort(key=lambda step: step != straight)

        for dr, dc in choices:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not _valid_path_cell(
                nxt,
                h,
                w,
                local_occupied,
                gaps,
                allowed_occupied={cur},
                allowed_gaps=set(),
            ):
                continue
            path.append(nxt)
            local_occupied.add(nxt)
            if dfs():
                return True
            local_occupied.remove(nxt)
            path.pop()
        return False

    if dfs():
        return path
    return None


def _choose_gap(
    endpoint: tuple[int, int],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    gaps: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    choices = DIRS.copy()
    random.shuffle(choices)
    for dr, dc in choices:
        gap = (endpoint[0] + dr, endpoint[1] + dc)
        nxt = (endpoint[0] + 2 * dr, endpoint[1] + 2 * dc)
        if not _in_bounds(h, w, gap) or not _in_bounds(h, w, nxt):
            continue
        if gap in occupied or gap in gaps or nxt in occupied or nxt in gaps:
            continue

        gap_neighbors = set(_neighbors4(gap, h, w))
        if {nbr for nbr in gap_neighbors if nbr in occupied} != {endpoint}:
            continue
        if any(nbr in gaps for nbr in gap_neighbors):
            continue

        if dr != 0:
            posts = [(gap[0], gap[1] - 1), (gap[0], gap[1] + 1)]
        else:
            posts = [(gap[0] - 1, gap[1]), (gap[0] + 1, gap[1])]

        if any(not _in_bounds(h, w, post) or post in occupied or post in gaps for post in posts):
            continue
        if any(
            any(nbr != gap and (nbr in occupied or nbr in gaps) for nbr in _neighbors4(post, h, w))
            for post in posts
        ):
            continue

        if not _valid_path_cell(
            nxt,
            h,
            w,
            occupied,
            gaps | {gap},
            allowed_occupied=set(),
            allowed_gaps={gap},
        ):
            continue

        return gap, nxt, posts
    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    segment_count = random.choice(SEGMENT_COUNT_CHOICES)
    lengths = [random.randint(*SEGMENT_LENGTH_RANGE) for _ in range(segment_count)]
    colors = random.sample(COLOR_POOL, 2 * segment_count - 1)
    segment_colors = colors[:segment_count]
    gate_colors = colors[segment_count:]

    grid = np.full((h, w), BG, dtype=int)
    yellow = (0, 0)
    grid[yellow] = YELLOW

    occupied: set[tuple[int, int]] = {yellow}
    gaps: set[tuple[int, int]] = set()
    segments: list[list[tuple[int, int]]] = []
    gates: list[tuple[int, tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]] = []

    start = random.choice([(0, 1), (1, 0)])
    incoming_gap: tuple[int, int] | None = None

    for i, length in enumerate(lengths):
        path = _grow_segment(
            start,
            length,
            h,
            w,
            occupied,
            gaps,
            allowed_occupied_start={yellow} if i == 0 else set(),
            allowed_gaps_start={incoming_gap} if incoming_gap is not None else set(),
        )
        if path is None:
            return None

        segments.append(path)
        occupied.update(path)

        if i == segment_count - 1:
            continue

        chosen = _choose_gap(path[-1], h, w, occupied, gaps)
        if chosen is None:
            return None
        gap, nxt, posts = chosen
        gaps.add(gap)
        occupied.update(posts)
        gates.append((gate_colors[i], (posts[0], posts[1]), gap))
        start = nxt
        incoming_gap = gap

    for color, path in zip(segment_colors, segments):
        for r, c in path:
            grid[r, c] = color
    for gate_color, posts, _gap in gates:
        for r, c in posts:
            grid[r, c] = gate_color

    try:
        out = main(grid)
    except AssertionError:
        return None
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "recoloring should change the scene"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-and-gates puzzle")
