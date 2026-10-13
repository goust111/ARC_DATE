from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, neighbors4


TRACK_COLOR = int(Color.GREY)
MARGIN_CHOICES = (1, 2)
NODE_SIZE_CHOICES = (2, 3)
GAP_CHOICES = (2, 3, 4)
BACKGROUND_CHOICES = (0, 4, 6, 7, 8, 9)
PATH_NODE_COUNT_RANGE = (4, 6)
MAX_GENERATION_ATTEMPTS = 500
NON_TERMINAL_COLORS = [c for c in range(10) if c not in {int(Color.BLUE), int(Color.RED), TRACK_COLOR}]


def _canonical_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _node_origin(cell: tuple[int, int], node_size: int, gap: int, margin: int) -> tuple[int, int]:
    r, c = cell
    step = int(node_size + gap)
    return int(margin + r * step), int(margin + c * step)


def _paint_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(color)


def _node_bbox_from_cell(
    cell: tuple[int, int], node_size: int, gap: int, margin: int
) -> tuple[int, int, int, int]:
    r0, c0 = _node_origin(cell, node_size, gap, margin)
    return int(r0), int(c0), int(r0 + node_size - 1), int(c0 + node_size - 1)


def _solid_square_components(grid: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in components_mask(grid == int(color), connectivity=4):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 2 or h != w or int(coords.shape[0]) != h * w:
                continue
            out.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                    "size": int(h),
                }
            )
    return out


def _parse_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    square_candidates = _solid_square_components(g, bg)
    assert square_candidates, "expected solid station squares"

    size_counts = Counter(comp["size"] for comp in square_candidates)
    node_size = max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0]
    nodes = [comp for comp in square_candidates if int(comp["size"]) == int(node_size)]
    nodes.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    assert len(nodes) >= 4, "expected at least four stations"

    red_ids = [idx for idx, node in enumerate(nodes) if int(node["color"]) == int(Color.RED)]
    blue_ids = [idx for idx, node in enumerate(nodes) if int(node["color"]) == int(Color.BLUE)]
    assert len(red_ids) == 1 and len(blue_ids) == 1, "expected one red source and one blue sink"

    node_mask = np.zeros_like(g, dtype=bool)
    pix2node: dict[tuple[int, int], int] = {}
    parsed_nodes: list[dict] = []
    for node_id, node in enumerate(nodes):
        coords = {(int(r), int(c)) for r, c in node["coords"]}
        for r, c in coords:
            node_mask[int(r), int(c)] = True
            pix2node[(int(r), int(c))] = int(node_id)
        parsed_nodes.append(
            {
                "id": int(node_id),
                "color": int(node["color"]),
                "bbox": node["bbox"],
                "coords": coords,
            }
        )

    track_mask = (g != int(bg)) & ~node_mask
    track_components = components_mask(track_mask, connectivity=4)

    edges: dict[int, dict] = {}
    adjacency: dict[int, list[int]] = {node["id"]: [] for node in parsed_nodes}
    for edge_id, coords in enumerate(track_components):
        pts = [(int(r), int(c)) for r, c in coords]
        touched: set[int] = set()
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                node_id = pix2node.get((int(rr), int(cc)))
                if node_id is not None:
                    touched.add(int(node_id))
        assert len(touched) == 2, "each track segment must connect exactly two stations"
        a, b = sorted(touched)
        edges[int(edge_id)] = {"nodes": (int(a), int(b)), "coords": pts}
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    for node_id in adjacency:
        adjacency[node_id].sort()
        assert adjacency[node_id], "station graph must be connected through tracks"

    return {
        "grid": g,
        "bg": int(bg),
        "node_size": int(node_size),
        "nodes": parsed_nodes,
        "track_mask": track_mask,
        "edges": edges,
        "adjacency": adjacency,
        "source": int(red_ids[0]),
        "sink": int(blue_ids[0]),
    }


def _shortest_path_with_counts(adjacency: dict, start, goal):
    queue = deque([start])
    dist = {start: 0}
    count = {start: 1}
    parent = {start: None}

    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            next_dist = int(dist[current] + 1)
            if nxt not in dist:
                dist[nxt] = next_dist
                count[nxt] = int(count[current])
                parent[nxt] = current
                queue.append(nxt)
            elif next_dist == int(dist[nxt]):
                count[nxt] = int(count[nxt] + count[current])

    assert goal in dist, "source and sink must be connected"
    return dist, count, parent


def _reconstruct_path(parent: dict, start, goal) -> list:
    path = [goal]
    current = goal
    while current != start:
        current = parent[current]
        assert current is not None, "failed to reconstruct shortest path"
        path.append(current)
    path.reverse()
    return path


def _count_simple_paths(adjacency: dict, start, goal, limit: int = 2) -> int:
    seen = {start}

    def dfs(node) -> int:
        if node == goal:
            return 1
        total = 0
        for nxt in adjacency[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            total += dfs(nxt)
            seen.remove(nxt)
            if total >= int(limit):
                return int(limit)
        return int(total)

    return int(dfs(start))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique shortest source-to-sink station path and shift station colors forward by one step along it.

    Input structure:
    - The background contains several solid monochrome square stations connected by one-pixel-wide track segments.
    - Exactly one station is red (the source) and exactly one station is blue (the sink); all other stations may use other colors.
    - Track pixels and station squares together encode a graph, with each track segment joining two stations.

    Rule:
    - Interpret the station squares as graph nodes and the track segments as graph edges.
    - Find the unique shortest path from the red source station to the blue sink station, measured by station-to-station edge count.
    - Keep the source red; for every later station on that path, replace its color with the input color of the previous station.
      The sink's original blue is therefore overwritten, while stations off the path, all tracks, and the background stay unchanged.

    Key insight:
    - The required output is not based on geometric proximity in the raw pixels; the track layout must first be abstracted into a graph,
      and the ordered nodes of the shortest path then act as the slots that receive the one-step color shift.
    """
    parsed = _parse_graph(grid)
    dist, count, parent = _shortest_path_with_counts(
        parsed["adjacency"], parsed["source"], parsed["sink"]
    )
    assert int(count[parsed["sink"]]) == 1, "shortest source-to-sink path must be unique"
    path_ids = _reconstruct_path(parent, parsed["source"], parsed["sink"])
    assert len(path_ids) >= 2

    out = parsed["grid"].copy()
    input_colors = {node["id"]: int(node["color"]) for node in parsed["nodes"]}
    for idx, node_id in enumerate(path_ids):
        new_color = int(Color.RED) if idx == 0 else int(input_colors[int(path_ids[idx - 1])])
        _paint_bbox(out, parsed["nodes"][int(node_id)]["bbox"], new_color)
    return out


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _random_self_avoiding_path(
    rows: int, cols: int, length: int
) -> list[tuple[int, int]] | None:
    for _ in range(120):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        used = {start}

        def dfs() -> list[tuple[int, int]] | None:
            if len(path) == int(length):
                return list(path)
            nbrs = [nbr for nbr in _cell_neighbors(path[-1], rows, cols) if nbr not in used]
            random.shuffle(nbrs)
            nbrs.sort(key=lambda cell: sum(n not in used for n in _cell_neighbors(cell, rows, cols)), reverse=True)
            for nxt in nbrs:
                used.add(nxt)
                path.append(nxt)
                result = dfs()
                if result is not None:
                    return result
                path.pop()
                used.remove(nxt)
            return None

        result = dfs()
        if result is not None:
            return result
    return None


def _shortest_free_cell_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked = set(blocked)
    blocked.discard(start)
    blocked.discard(goal)
    queue = deque([start])
    parent = {start: None}

    while queue:
        cell = queue.popleft()
        if cell == goal:
            return _reconstruct_path(parent, start, goal)
        nbrs = [nbr for nbr in _cell_neighbors(cell, rows, cols) if nbr not in blocked and nbr not in parent]
        random.shuffle(nbrs)
        for nbr in nbrs:
            parent[nbr] = cell
            queue.append(nbr)
    return None


def _build_cell_adjacency(
    cells: set[tuple[int, int]], edges: set[tuple[tuple[int, int], tuple[int, int]]]
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adjacency = {cell: [] for cell in cells}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for cell in adjacency:
        adjacency[cell].sort()
    return adjacency


def _choose_canvas_params() -> tuple[int, int, int, int, int] | None:
    node_size = int(random.choice(NODE_SIZE_CHOICES))
    gap = int(random.choice(GAP_CHOICES))
    margin = int(random.choice(MARGIN_CHOICES))
    max_dim = int((30 - 2 * margin + gap) // (node_size + gap))
    if max_dim < 4:
        return None
    rows = random.randint(4, min(6, max_dim))
    cols = random.randint(4, min(6, max_dim))
    return rows, cols, node_size, gap, margin


def _find_detour(
    main_path: list[tuple[int, int]], rows: int, cols: int
) -> tuple[list[tuple[int, int]], int, int] | None:
    pairs = [(i, j) for i in range(len(main_path)) for j in range(i + 2, len(main_path))]
    random.shuffle(pairs)
    for i, j in pairs:
        start = main_path[i]
        goal = main_path[j]
        blocked = set(main_path) - {start, goal}
        detour = _shortest_free_cell_path(start, goal, rows, cols, blocked)
        if detour is None:
            continue
        if len(detour) - 1 > j - i:
            return detour, i, j
    return None


def _add_leaf_branches(
    cells: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    rows: int,
    cols: int,
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]]:
    cells = set(cells)
    edges = set(edges)
    for _ in range(random.randint(0, 2)):
        options: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for parent in sorted(cells):
            for nbr in _cell_neighbors(parent, rows, cols):
                if nbr not in cells:
                    options.append((parent, nbr))
        if not options:
            break
        parent, leaf = random.choice(options)
        cells.add(leaf)
        edges.add(_canonical_edge(parent, leaf))
        if random.random() < 0.4:
            ext = [nbr for nbr in _cell_neighbors(leaf, rows, cols) if nbr not in cells]
            if ext:
                leaf2 = random.choice(ext)
                cells.add(leaf2)
                edges.add(_canonical_edge(leaf, leaf2))
    return cells, edges


def _assign_node_colors(
    cells: set[tuple[int, int]],
    shortest_path: list[tuple[int, int]],
    bg: int,
) -> dict[tuple[int, int], int]:
    palette = [color for color in NON_TERMINAL_COLORS if color != int(bg)]
    assert len(palette) >= 4
    colors: dict[tuple[int, int], int] = {
        shortest_path[0]: int(Color.RED),
        shortest_path[-1]: int(Color.BLUE),
    }

    internal = shortest_path[1:-1]
    shuffled = palette[:]
    random.shuffle(shuffled)
    while len(shuffled) < len(internal):
        shuffled.extend(random.sample(palette, len(palette)))
    for cell, color in zip(internal, shuffled, strict=False):
        colors[cell] = int(color)

    off_path = sorted(cells - set(shortest_path))
    for cell in off_path:
        colors[cell] = int(random.choice(palette))
    return colors


def _render_graph(
    rows: int,
    cols: int,
    node_size: int,
    gap: int,
    margin: int,
    bg: int,
    node_colors: dict[tuple[int, int], int],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]]]:
    height = int(2 * margin + rows * node_size + (rows - 1) * gap)
    width = int(2 * margin + cols * node_size + (cols - 1) * gap)
    grid = np.full((height, width), int(bg), dtype=int)
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for cell, color in node_colors.items():
        bbox = _node_bbox_from_cell(cell, node_size, gap, margin)
        _paint_bbox(grid, bbox, color)
        bboxes[cell] = bbox

    port = int(node_size // 2)
    for a, b in sorted(edges):
        ar, ac = a
        br, bc = b
        ar0, ac0 = _node_origin(a, node_size, gap, margin)
        br0, bc0 = _node_origin(b, node_size, gap, margin)
        if ar == br:
            row = int(ar0 + port)
            left = int(min(ac0, bc0) + node_size)
            right = int(max(ac0, bc0) - 1)
            grid[row, left : right + 1] = TRACK_COLOR
        elif ac == bc:
            col = int(ac0 + port)
            top = int(min(ar0, br0) + node_size)
            bottom = int(max(ar0, br0) - 1)
            grid[top : bottom + 1, col] = TRACK_COLOR
        else:
            raise AssertionError("only orthogonal cell-adjacent edges are supported")

    return grid, bboxes


def _expected_output(
    grid: np.ndarray,
    shortest_path: list[tuple[int, int]],
    node_colors: dict[tuple[int, int], int],
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for idx, cell in enumerate(shortest_path):
        color = int(Color.RED) if idx == 0 else int(node_colors[shortest_path[idx - 1]])
        _paint_bbox(out, bboxes[cell], color)
    return out


def _parsed_path_as_cells(
    parsed: dict, node_size: int, gap: int, margin: int
) -> list[tuple[int, int]]:
    dist, count, parent = _shortest_path_with_counts(parsed["adjacency"], parsed["source"], parsed["sink"])
    assert int(count[parsed["sink"]]) == 1
    path_ids = _reconstruct_path(parent, parsed["source"], parsed["sink"])
    cells: list[tuple[int, int]] = []
    step = int(node_size + gap)
    for node_id in path_ids:
        r0, c0, _, _ = parsed["nodes"][int(node_id)]["bbox"]
        assert (int(r0) - margin) % step == 0 and (int(c0) - margin) % step == 0
        cells.append(((int(r0) - margin) // step, (int(c0) - margin) // step))
    return cells


def _validate_instance(
    grid: np.ndarray,
    shortest_path: list[tuple[int, int]],
    cell_colors: dict[tuple[int, int], int],
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
    cell_adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    node_size: int,
    gap: int,
    margin: int,
) -> bool:
    dist, count, parent = _shortest_path_with_counts(cell_adjacency, shortest_path[0], shortest_path[-1])
    if int(count[shortest_path[-1]]) != 1:
        return False
    if _reconstruct_path(parent, shortest_path[0], shortest_path[-1]) != shortest_path:
        return False
    if _count_simple_paths(cell_adjacency, shortest_path[0], shortest_path[-1], limit=2) < 2:
        return False
    if len(cell_colors) <= len(shortest_path):
        return False

    parsed = _parse_graph(grid)
    if len(parsed["nodes"]) != len(cell_colors):
        return False
    parsed_cells = _parsed_path_as_cells(parsed, node_size, gap, margin)
    if parsed_cells != shortest_path:
        return False
    if _count_simple_paths(parsed["adjacency"], parsed["source"], parsed["sink"], limit=2) < 2:
        return False

    expected = _expected_output(grid, shortest_path, cell_colors, bboxes)
    actual = main(grid)
    if np.array_equal(actual, grid) or not np.array_equal(actual, expected):
        return False
    if not np.array_equal(actual[parsed["track_mask"]], grid[parsed["track_mask"]]):
        return False

    for cell, bbox in bboxes.items():
        before = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        after = actual[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        if cell in set(shortest_path):
            expected_color = int(Color.RED)
            idx = shortest_path.index(cell)
            if idx > 0:
                expected_color = int(cell_colors[shortest_path[idx - 1]])
            if not np.all(after == expected_color):
                return False
        elif not np.array_equal(before, after):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    params = _choose_canvas_params()
    if params is None:
        return None
    rows, cols, node_size, gap, margin = params

    max_path_nodes = min(PATH_NODE_COUNT_RANGE[1], rows * cols - 2)
    if max_path_nodes < PATH_NODE_COUNT_RANGE[0]:
        return None
    main_node_count = random.randint(PATH_NODE_COUNT_RANGE[0], max_path_nodes)
    main_path = _random_self_avoiding_path(rows, cols, main_node_count)
    if main_path is None:
        return None

    detour_info = _find_detour(main_path, rows, cols)
    if detour_info is None:
        return None
    detour_path, _i, _j = detour_info

    cells = set(main_path)
    edges = {_canonical_edge(a, b) for a, b in zip(main_path, main_path[1:], strict=False)}
    cells.update(detour_path)
    edges.update(
        _canonical_edge(a, b) for a, b in zip(detour_path, detour_path[1:], strict=False)
    )
    cells, edges = _add_leaf_branches(cells, edges, rows, cols)

    cell_adjacency = _build_cell_adjacency(cells, edges)
    bg = int(random.choice([color for color in BACKGROUND_CHOICES if color != TRACK_COLOR]))
    cell_colors = _assign_node_colors(cells, main_path, bg)
    grid, bboxes = _render_graph(rows, cols, node_size, gap, margin, bg, cell_colors, edges)

    if _validate_instance(grid, main_path, cell_colors, bboxes, cell_adjacency, node_size, gap, margin):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid along-the-route color-shift instance")
