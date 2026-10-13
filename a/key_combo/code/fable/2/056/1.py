from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, _ = grid.shape
    col = grid[:, -1]
    assert int(col[-1]) != bg, "legend must touch the bottom of the right edge"

    start = h - 1
    while start >= 0 and int(col[start]) != bg:
        start -= 1
    rows = np.arange(start + 1, h, dtype=int)
    colors = [int(col[r]) for r in rows]

    assert rows.size > 0
    assert len(set(colors)) == len(colors), "legend colors must be distinct"

    mask = np.zeros(grid.shape, dtype=bool)
    mask[rows, -1] = True
    return colors, mask


def _find_door_mask(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    open_mask = ~blocked
    doors = np.zeros((h, w), dtype=bool)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if blocked[r, c]:
                continue
            vertical = blocked[r - 1, c] and blocked[r + 1, c] and open_mask[r, c - 1] and open_mask[r, c + 1]
            horizontal = blocked[r, c - 1] and blocked[r, c + 1] and open_mask[r - 1, c] and open_mask[r + 1, c]
            if vertical ^ horizontal:
                doors[r, c] = True

    return doors


def _label_components(open_mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = open_mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if not open_mask[sr, sc] or labels[sr, sc] != -1:
                continue

            cid = len(components)
            queue = deque([(sr, sc)])
            labels[sr, sc] = cid
            coords = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        queue.append((nr, nc))

            components.append(np.asarray(coords, dtype=int))

    return labels, components


def _door_pair(r: int, c: int, blocked: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    if blocked[r - 1, c] and blocked[r + 1, c]:
        a = int(labels[r, c - 1])
        b = int(labels[r, c + 1])
    else:
        a = int(labels[r - 1, c])
        b = int(labels[r + 1, c])

    assert a >= 0 and b >= 0 and a != b
    return a, b


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each room by the legend color indexed by its minimum number of door crossings to the outdoors.

    Input structure:
    - A background canvas contains one building made from a single repeated wall color.
    - The building is partitioned into room interiors by 1-cell-thick walls; connections
      between rooms or between a room and the outdoors happen only through single-cell wall gaps.
    - The rightmost column contains a bottom-aligned vertical legend whose colors are all distinct.

    Rule:
    - Read the right-edge legend from bottom to top as the lookup table for door depth 1, 2, 3, ...
    - Temporarily treat every door gap as closed, so each room becomes its own 4-connected component.
    - Build the room graph by reopening the doors, compute each room's minimum number of crossed doors
      needed to reach the outdoor background, and fill the entire room with the corresponding legend color.
    - Keep the walls, door cells, outdoor background, and legend unchanged.

    Key insight:
    - A room's target color is not determined by how many doors it owns or by whether it touches the
      outer wall, but by the shortest path through the door graph from that room to the outdoor component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend_top_to_bottom, legend_mask = _extract_legend(g, bg)
    legend_bottom_to_top = list(reversed(legend_top_to_bottom))

    counts = Counter(map(int, g[~legend_mask].ravel()))
    wall_candidates = [color for color, count in counts.items() if color != bg and count > 1]
    assert wall_candidates, "expected a repeated wall color"
    wall = max(wall_candidates, key=lambda color: counts[color])

    blocked = (g == wall) | legend_mask
    doors = _find_door_mask(blocked)
    closed_open = ~(blocked | doors)

    labels, components = _label_components(closed_open)
    assert components, "expected an outdoor component"
    outdoor = int(labels[0, 0])
    assert outdoor >= 0

    graph = [set() for _ in range(len(components))]
    for r, c in np.argwhere(doors):
        a, b = _door_pair(int(r), int(c), blocked, labels)
        graph[a].add(b)
        graph[b].add(a)

    depth = {outdoor: 0}
    queue = deque([outdoor])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                queue.append(nxt)

    out = g.copy()
    for cid, coords in enumerate(components):
        if cid == outdoor:
            continue
        d = depth.get(cid)
        assert d is not None and 1 <= d <= len(legend_bottom_to_top)
        out[coords[:, 0], coords[:, 1]] = legend_bottom_to_top[d - 1]

    return out


def _room_neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _boundary_sides(node: tuple[int, int], rows: int, cols: int) -> list[str]:
    r, c = node
    sides = []
    if r == 0:
        sides.append("top")
    if r == rows - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == cols - 1:
        sides.append("right")
    return sides


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a < b else (b, a)


def _compute_depths(
    nodes: list[tuple[int, int]],
    internal_doors: set[tuple[tuple[int, int], tuple[int, int]]],
    external_doors: dict[tuple[int, int], str],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    graph = {node: set() for node in nodes}
    incident = {node: 1 if node in external_doors else 0 for node in nodes}

    for a, b in internal_doors:
        graph[a].add(b)
        graph[b].add(a)
        incident[a] += 1
        incident[b] += 1

    queue = deque()
    depth: dict[tuple[int, int], int] = {}
    for node in external_doors:
        depth[node] = 1
        queue.append(node)

    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                queue.append(nxt)

    return depth, incident


def _door_count_counterexample(
    nodes: list[tuple[int, int]],
    depth: dict[tuple[int, int], int],
    incident: dict[tuple[int, int], int],
) -> bool:
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if incident[a] == incident[b] and depth[a] != depth[b]:
                return True
            if incident[a] != incident[b] and depth[a] == depth[b]:
                return True
    return False


def _sample_room_graph(rows: int, cols: int):
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    boundary_nodes = [node for node in nodes if _boundary_sides(node, rows, cols)]
    root_count = random.choice([1, 1, 2])
    roots = random.sample(boundary_nodes, root_count)

    external_doors = {node: random.choice(_boundary_sides(node, rows, cols)) for node in roots}
    connected = set(roots)
    internal_doors: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while len(connected) < len(nodes):
        frontier = []
        for node in list(connected):
            for nxt in _room_neighbors(node, rows, cols):
                if nxt not in connected:
                    frontier.append((node, nxt))
        if not frontier:
            return None
        a, b = random.choice(frontier)
        internal_doors.add(_edge_key(a, b))
        connected.add(b)

    extra_internal = set()
    for node in nodes:
        for nxt in _room_neighbors(node, rows, cols):
            edge = _edge_key(node, nxt)
            if edge not in internal_doors:
                extra_internal.add(edge)
    extra_internal = list(extra_internal)
    random.shuffle(extra_internal)
    for edge in extra_internal:
        if random.random() < 0.3:
            internal_doors.add(edge)

    extra_boundary = [node for node in boundary_nodes if node not in external_doors]
    random.shuffle(extra_boundary)
    for node in extra_boundary:
        if random.random() < 0.18:
            external_doors[node] = random.choice(_boundary_sides(node, rows, cols))

    depth, incident = _compute_depths(nodes, internal_doors, external_doors)
    if len(depth) != len(nodes):
        return None

    boundary_without_exit = [node for node in boundary_nodes if node not in external_doors]
    max_depth = max(depth.values())

    if max_depth < 3 or max_depth > 8:
        return None
    if len(set(depth.values())) < 3:
        return None
    if not boundary_without_exit or not any(depth[node] > 1 for node in boundary_without_exit):
        return None
    interior_nodes = [node for node in nodes if node not in boundary_nodes]
    if interior_nodes:
        shallow_interior = min(depth[node] for node in interior_nodes)
        deep_boundary = max(depth[node] for node in boundary_without_exit)
        if shallow_interior >= deep_boundary:
            return None
    elif max(depth[node] for node in boundary_without_exit) < 3:
        return None
    if not _door_count_counterexample(nodes, depth, incident):
        return None
    if len(external_doors) == len(boundary_nodes):
        return None

    return nodes, internal_doors, external_doors, depth


def _cell_bounds(sizes: list[int]) -> list[tuple[int, int]]:
    bounds = []
    cur = 1
    for size in sizes:
        bounds.append((cur, cur + size - 1))
        cur += size + 1
    return bounds


def _render_case(
    row_sizes: list[int],
    col_sizes: list[int],
    wall: int,
    bg: int,
    internal_doors: set[tuple[tuple[int, int], tuple[int, int]]],
    external_doors: dict[tuple[int, int], str],
    depth: dict[tuple[int, int], int],
    legend_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(row_sizes)
    cols = len(col_sizes)
    row_bounds = _cell_bounds(row_sizes)
    col_bounds = _cell_bounds(col_sizes)
    building_h = sum(row_sizes) + rows + 1
    building_w = sum(col_sizes) + cols + 1

    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    gap_to_legend = random.randint(2, 4)
    height = max(top_margin + building_h + bottom_margin, len(legend_colors) + random.randint(0, 3))
    width = left_margin + building_w + gap_to_legend + 1
    if height > 30 or width > 30:
        raise ValueError("rendered grid exceeds ARC limits")

    inp = np.full((height, width), bg, dtype=int)
    out = inp.copy()

    br0 = top_margin
    bc0 = left_margin
    inp[br0 : br0 + building_h, bc0 : bc0 + building_w] = wall
    out[br0 : br0 + building_h, bc0 : bc0 + building_w] = wall

    for r in range(rows):
        r0, r1 = row_bounds[r]
        for c in range(cols):
            c0, c1 = col_bounds[c]
            rr0 = br0 + r0
            rr1 = br0 + r1 + 1
            cc0 = bc0 + c0
            cc1 = bc0 + c1 + 1
            inp[rr0:rr1, cc0:cc1] = bg
            out[rr0:rr1, cc0:cc1] = legend_colors[depth[(r, c)] - 1]

    for (a, b) in internal_doors:
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            wall_col = bc0 + col_bounds[min(c1, c2)][1] + 1
            door_row = br0 + random.randint(*row_bounds[r1])
            inp[door_row, wall_col] = bg
            out[door_row, wall_col] = bg
        else:
            wall_row = br0 + row_bounds[min(r1, r2)][1] + 1
            door_col = bc0 + random.randint(*col_bounds[c1])
            inp[wall_row, door_col] = bg
            out[wall_row, door_col] = bg

    for (r, c), side in external_doors.items():
        r0, r1 = row_bounds[r]
        c0, c1 = col_bounds[c]
        if side == "top":
            pos = (br0, bc0 + random.randint(c0, c1))
        elif side == "bottom":
            pos = (br0 + building_h - 1, bc0 + random.randint(c0, c1))
        elif side == "left":
            pos = (br0 + random.randint(r0, r1), bc0)
        else:
            pos = (br0 + random.randint(r0, r1), bc0 + building_w - 1)
        inp[pos] = bg
        out[pos] = bg

    legend_col = width - 1
    start = height - len(legend_colors)
    for idx, color in enumerate(reversed(legend_colors)):
        inp[start + idx, legend_col] = color
        out[start + idx, legend_col] = color

    return inp, out


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    if rows * cols < 5 or rows * cols > 10:
        return None

    row_sizes = [random.randint(2, 4) for _ in range(rows)]
    col_sizes = [random.randint(2, 4) for _ in range(cols)]

    graph = _sample_room_graph(rows, cols)
    if graph is None:
        return None
    nodes, internal_doors, external_doors, depth = graph
    del nodes

    max_depth = max(depth.values())
    colors = list(range(10))
    bg, wall = random.sample(colors, 2)
    remaining = [color for color in colors if color not in {bg, wall}]
    legend_colors = random.sample(remaining, max_depth)

    try:
        inp, expected = _render_case(row_sizes, col_sizes, wall, bg, internal_doors, external_doors, depth, legend_colors)
    except ValueError:
        return None

    if not np.array_equal(main(inp), expected):
        return None

    solved_colors = set(map(int, np.unique(expected)))
    if len(solved_colors) < 4:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        case = _sample_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid several-doors puzzle")
