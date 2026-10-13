import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

NODE_SIZE = 4
CAVITY_SIZE = 2
MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(int(sr), int(c)) for c in range(int(sc), int(ec) + step, step)]
    if sc == ec:
        step = 1 if er >= sr else -1
        return [(int(r), int(sc)) for r in range(int(sr), int(er) + step, step)]
    raise ValueError("line must be axis-aligned")


def _ring_border_cells(top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(top, top + NODE_SIZE):
        for c in range(left, left + NODE_SIZE):
            if r in (top, top + NODE_SIZE - 1) or c in (left, left + NODE_SIZE - 1):
                cells.add((int(r), int(c)))
    return cells


def _cavity_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [(top + 1, left + 1), (top + 1, left + 2), (top + 2, left + 1), (top + 2, left + 2)]


def _detect_nodes(grid: np.ndarray) -> list[dict]:
    nodes: list[dict] = []
    for color in (BLUE, GREEN):
        for comp in _components_of_mask(grid == color):
            if len(comp) != 12:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != (NODE_SIZE, NODE_SIZE):
                continue
            border = _ring_border_cells(r0, c0)
            if set(comp) != border:
                continue
            cavity = _cavity_cells(r0, c0)
            if any(int(grid[r, c]) == color for r, c in cavity):
                continue
            nodes.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color": int(color),
                    "border": border,
                    "cavity": cavity,
                    "weight": 0 if color == GREEN else int(sum(int(grid[r, c]) == YELLOW for r, c in cavity)),
                }
            )
    nodes.sort(key=lambda node: (node["bbox"][0], node["bbox"][1]))
    for idx, node in enumerate(nodes):
        node["id"] = int(idx)
    return nodes


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN, YELLOW}

    nodes = _detect_nodes(g)
    assert 6 <= len(nodes) <= 10, "expected 6-10 node squares"
    terminals = [node["id"] for node in nodes if node["color"] == GREEN]
    assert len(terminals) == 2, "expected exactly two green terminals"

    border_owner: dict[tuple[int, int], int] = {}
    for node in nodes:
        for cell in node["border"]:
            border_owner[cell] = int(node["id"])

    edges: dict[tuple[int, int], dict] = {}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for comp in _components_of_mask(g == RED):
        touched: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                owner = border_owner.get((int(nr), int(nc)))
                if owner is not None:
                    touched.add(int(owner))
        assert len(touched) == 2, "each red track component must connect exactly two nodes"
        a, b = sorted(touched)
        key = _edge_key(a, b)
        assert key not in edges, "multiple track components between the same node pair are not supported"
        edges[key] = {"nodes": key, "pixels": [tuple(map(int, cell)) for cell in comp]}
        adjacency[a].add(b)
        adjacency[b].add(a)

    start, goal = map(int, terminals)
    all_paths = _all_simple_paths(adjacency, start, goal)
    assert len(all_paths) >= 2, "graph must offer at least two distinct terminal-to-terminal paths"
    best_path = min(all_paths, key=lambda path: _path_rank(path, nodes))

    return {
        "grid": g,
        "nodes": nodes,
        "edges": edges,
        "adjacency": adjacency,
        "start": start,
        "goal": goal,
        "paths": all_paths,
        "best_path": best_path,
    }


def _all_simple_paths(adjacency: dict[int, set[int]], start: int, goal: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    stack = [(int(start), [int(start)], {int(start)})]
    while stack:
        cur, path, seen = stack.pop()
        if cur == goal:
            out.append(tuple(path))
            continue
        for nxt in sorted(adjacency.get(cur, set()), reverse=True):
            if nxt in seen:
                continue
            stack.append((int(nxt), path + [int(nxt)], seen | {int(nxt)}))
    return out


def _path_cost(path: tuple[int, ...] | list[int], nodes: list[dict]) -> int:
    weights = {int(node["id"]): int(node["weight"]) for node in nodes}
    return sum(weights[int(a)] + weights[int(b)] for a, b in zip(path, path[1:]))


def _path_rank(path: tuple[int, ...] | list[int], nodes: list[dict]) -> tuple:
    rows = tuple(int(nodes[int(node_id)]["bbox"][0]) for node_id in path)
    cols = tuple(int(nodes[int(node_id)]["bbox"][1]) for node_id in path)
    return (_path_cost(path, nodes), len(path), rows, cols)


def _render_output(grid: np.ndarray, analysis: dict, path: tuple[int, ...] | list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    path_nodes = set(map(int, path))
    path_edges = {_edge_key(int(a), int(b)) for a, b in zip(path, path[1:])}

    for node in analysis["nodes"]:
        cavity_color = YELLOW if node["color"] == BLUE and int(node["id"]) in path_nodes else BLACK
        for r, c in node["cavity"]:
            out[r, c] = int(cavity_color)

    for key, edge in analysis["edges"].items():
        if key in path_edges:
            for r, c in edge["pixels"]:
                out[r, c] = GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count cavity markers to weight graph nodes, then recolor the minimum-cost terminal path.

    Input structure:
    - The background is black.
    - The grid contains 6-10 hollow 4x4 node squares with 2x2 cavities; exactly two have green borders and the rest have blue borders.
    - Red 1-pixel-wide track segments connect node borders into a graph, and blue-node cavities may contain 0-4 yellow marker pixels.

    Rule:
    - Treat each hollow square as one graph node and each red track component that directly links two squares as one graph edge.
    - Give every blue node weight W equal to the number of yellow pixels in its 2x2 cavity; green terminal nodes have weight 0.
    - For every edge, use cost weight(u) + weight(v), then find the minimum-total-cost path between the two green terminals.
    - If costs tie, prefer the path with fewer nodes; if it still ties, prefer the one whose first divergent node is higher in the grid.
    - Recolor every red track on the chosen path to green, fill every chosen blue node cavity completely with yellow, and erase all yellow markers from every non-path blue node cavity.

    Key insight:
    - The yellow cavity markers are not decoration: they encode traversal cost, so the correct route is determined by weighted graph search rather than by hop count alone.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int), analysis, analysis["best_path"])


TEMPLATES = [
    {
        "name": "full3",
        "positions": {(r, c) for r in range(3) for c in range(3)},
        "terminals": ((1, 0), (1, 2)),
        "mode": "detour",
    },
    {
        "name": "ring8",
        "positions": {(r, c) for r in range(3) for c in range(3)} - {(1, 1)},
        "terminals": ((1, 0), (1, 2)),
        "mode": "equal_hops",
    },
    {
        "name": "ladder6",
        "positions": {(r, c) for r in range(2) for c in range(3)},
        "terminals": ((0, 0), (0, 2)),
        "mode": "detour",
    },
    {
        "name": "ladder10",
        "positions": {(r, c) for r in range(2) for c in range(5)},
        "terminals": ((0, 0), (0, 4)),
        "mode": "detour",
    },
]


def _template_edges(positions: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for r, c in positions:
        for dr, dc in ((1, 0), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in positions:
                edges.append(((int(r), int(c)), (int(nxt[0]), int(nxt[1]))))
    return edges


def _abstract_paths(template: dict) -> tuple[dict, list[tuple[int, ...]], dict]:
    positions = sorted(template["positions"])
    pos_to_id = {pos: idx for idx, pos in enumerate(positions)}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a_pos, b_pos in _template_edges(template["positions"]):
        a = pos_to_id[a_pos]
        b = pos_to_id[b_pos]
        adjacency[a].add(b)
        adjacency[b].add(a)
    start = pos_to_id[template["terminals"][0]]
    goal = pos_to_id[template["terminals"][1]]
    return adjacency, _all_simple_paths(adjacency, start, goal), pos_to_id


def _sample_weights(template: dict, rng: random.Random) -> tuple[dict[int, int], tuple[int, ...]]:
    adjacency, paths, pos_to_id = _abstract_paths(template)
    terminal_ids = {pos_to_id[template["terminals"][0]], pos_to_id[template["terminals"][1]]}
    all_ids = sorted(adjacency.keys())
    for _ in range(400):
        weights = {node_id: 0 for node_id in all_ids}
        for node_id in all_ids:
            if node_id not in terminal_ids:
                weights[node_id] = rng.randint(0, 4)
        middle_weights = [weights[node_id] for node_id in all_ids if node_id not in terminal_ids]
        if len(set(middle_weights)) < 2:
            continue
        costs = [sum(weights[a] + weights[b] for a, b in zip(path, path[1:])) for path in paths]
        order = sorted(range(len(paths)), key=lambda i: (costs[i], len(paths[i]), paths[i]))
        best_idx = order[0]
        if costs[best_idx] == costs[order[1]]:
            continue
        best = paths[best_idx]
        hop_min = min(len(path) - 1 for path in paths)
        hop_ties = [path for path in paths if len(path) - 1 == hop_min]
        if template["mode"] == "detour" and len(best) - 1 == hop_min:
            continue
        if template["mode"] == "equal_hops" and len(hop_ties) < 2:
            continue
        if template["mode"] == "equal_hops" and len(best) - 1 != hop_min:
            continue
        best_middle = [node_id for node_id in best[1:-1]]
        if not best_middle:
            continue
        if not any(weights[node_id] < 4 for node_id in best_middle):
            continue
        if not any(weights[node_id] > 0 for node_id in all_ids if node_id not in terminal_ids and node_id not in best):
            continue
        if rng.random() < 0.55 and 0 not in middle_weights:
            continue
        return weights, best
    raise RuntimeError(f"failed to sample weights for template {template['name']}")


def _sample_layout(template: dict, rng: random.Random) -> tuple[dict[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    positions = template["positions"]
    max_row = max(r for r, _ in positions) + 1
    max_col = max(c for _, c in positions) + 1
    if max_row <= 2:
        row_gap_low, row_gap_high = 2, 4
        row_margin_low, row_margin_high = 0, 2
    else:
        row_gap_low, row_gap_high = 3, 6
        row_margin_low, row_margin_high = 1, 3

    if max_col >= 5:
        col_gap_low, col_gap_high = 1, 2
        col_margin_low, col_margin_high = 0, 1
    elif max_col <= 3:
        col_gap_low, col_gap_high = 3, 6
        col_margin_low, col_margin_high = 1, 3
    else:
        col_gap_low, col_gap_high = 2, 4
        col_margin_low, col_margin_high = 0, 2

    for _ in range(200):
        row_gaps = [rng.randint(row_gap_low, row_gap_high) for _ in range(max_row - 1)]
        col_gaps = [rng.randint(col_gap_low, col_gap_high) for _ in range(max_col - 1)]
        base_rows = [rng.randint(row_margin_low, row_margin_high)]
        base_cols = [rng.randint(col_margin_low, col_margin_high)]
        for gap in row_gaps:
            base_rows.append(base_rows[-1] + NODE_SIZE + gap)
        for gap in col_gaps:
            base_cols.append(base_cols[-1] + NODE_SIZE + gap)

        col_row_jitter = [rng.randint(0, 1) for _ in range(max_col)]
        row_col_jitter = [rng.randint(0, 1) for _ in range(max_row)]
        coords = {
            (r, c): (int(base_rows[r] + col_row_jitter[c]), int(base_cols[c] + row_col_jitter[r]))
            for r, c in positions
        }
        height = max(top for top, _ in coords.values()) + NODE_SIZE + rng.randint(row_margin_low, row_margin_high + 1)
        width = max(left for _, left in coords.values()) + NODE_SIZE + rng.randint(col_margin_low, col_margin_high + 1)
        if height > 30 or width > 30:
            continue

        boxes = [(*coords[pos], coords[pos][0] + NODE_SIZE - 1, coords[pos][1] + NODE_SIZE - 1) for pos in positions]
        ok = True
        for i in range(len(boxes)):
            r0a, c0a, r1a, c1a = boxes[i]
            for j in range(i + 1, len(boxes)):
                r0b, c0b, r1b, c1b = boxes[j]
                if not (r1a < r0b or r1b < r0a or c1a < c0b or c1b < c0a):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return coords, (int(height), int(width))
    raise RuntimeError(f"failed to place template {template['name']}")


def _route_edge(
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    box_a: tuple[int, int],
    box_b: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    top_a, left_a = box_a
    top_b, left_b = box_b
    if pos_a[0] == pos_b[0]:
        if pos_a[1] > pos_b[1]:
            pos_a, pos_b = pos_b, pos_a
            top_a, left_a, top_b, left_b = top_b, left_b, top_a, left_a
        start = (int(top_a + 1), int(left_a + NODE_SIZE))
        end = (int(top_b + 1), int(left_b - 1))
        if start[0] == end[0]:
            return _line_cells(start, end)
        cand1 = _line_cells(start, (start[0], end[1])) + _line_cells((start[0], end[1]), end)[1:]
        cand2 = _line_cells(start, (end[0], start[1])) + _line_cells((end[0], start[1]), end)[1:]
        return rng.choice([cand1, cand2])

    if pos_a[0] > pos_b[0]:
        pos_a, pos_b = pos_b, pos_a
        top_a, left_a, top_b, left_b = top_b, left_b, top_a, left_a
    start = (int(top_a + NODE_SIZE), int(left_a + 2))
    end = (int(top_b - 1), int(left_b + 2))
    if start[1] == end[1]:
        return _line_cells(start, end)
    cand1 = _line_cells(start, (end[0], start[1])) + _line_cells((end[0], start[1]), end)[1:]
    cand2 = _line_cells(start, (start[0], end[1])) + _line_cells((start[0], end[1]), end)[1:]
    return rng.choice([cand1, cand2])


def _draw_node(grid: np.ndarray, top: int, left: int, color: int, weight: int, rng: random.Random) -> None:
    for r, c in _ring_border_cells(top, left):
        grid[r, c] = int(color)
    cavity = _cavity_cells(top, left)
    if color == BLUE and weight > 0:
        for r, c in rng.sample(cavity, weight):
            grid[r, c] = YELLOW


def _build_expected(
    grid: np.ndarray,
    node_info: dict[int, dict],
    edge_pixels: dict[tuple[int, int], list[tuple[int, int]]],
    best_path: tuple[int, ...],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    path_nodes = set(best_path)
    path_edges = {_edge_key(a, b) for a, b in zip(best_path, best_path[1:])}
    for node_id, info in node_info.items():
        if info["color"] == BLUE:
            fill_color = YELLOW if node_id in path_nodes else BLACK
            for r, c in info["cavity"]:
                out[r, c] = int(fill_color)
    for key, pixels in edge_pixels.items():
        if key in path_edges:
            for r, c in pixels:
                out[r, c] = GREEN
    return out


def _signature(grid: np.ndarray) -> bytes:
    arr = np.asarray(grid, dtype=np.int8)
    return str(arr.shape).encode() + arr.tobytes()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template = rng.choice(TEMPLATES)
    adjacency, paths, pos_to_id = _abstract_paths(template)
    weights, best_path = _sample_weights(template, rng)
    try:
        coords, shape = _sample_layout(template, rng)
    except RuntimeError:
        return None

    grid = np.zeros(shape, dtype=int)
    node_info: dict[int, dict] = {}
    terminal_ids = {pos_to_id[template["terminals"][0]], pos_to_id[template["terminals"][1]]}

    for pos, node_id in pos_to_id.items():
        top, left = coords[pos]
        color = GREEN if node_id in terminal_ids else BLUE
        _draw_node(grid, top, left, color, weights[node_id], rng)
        node_info[node_id] = {
            "top": int(top),
            "left": int(left),
            "bbox": (int(top), int(left), int(top + NODE_SIZE - 1), int(left + NODE_SIZE - 1)),
            "cavity": _cavity_cells(top, left),
            "color": int(color),
            "weight": int(weights[node_id]),
        }

    edge_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pos_a, pos_b in _template_edges(template["positions"]):
        node_a = pos_to_id[pos_a]
        node_b = pos_to_id[pos_b]
        pixels = _route_edge(pos_a, pos_b, coords[pos_a], coords[pos_b], rng)
        if any(not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK for r, c in pixels):
            return None
        for r, c in pixels:
            grid[r, c] = RED
        edge_pixels[_edge_key(node_a, node_b)] = pixels

    expected = _build_expected(grid, node_info, edge_pixels, best_path)
    try:
        analysis = _analyze_grid(grid)
    except AssertionError:
        return None

    if len(analysis["paths"]) != len(paths):
        return None
    if analysis["best_path"] != min(analysis["paths"], key=lambda path: _path_rank(path, analysis["nodes"])):
        return None

    best_cost = _path_cost(analysis["best_path"], analysis["nodes"])
    all_costs = sorted(_path_cost(path, analysis["nodes"]) for path in analysis["paths"])
    if best_cost == all_costs[1]:
        return None

    hop_min = min(len(path) - 1 for path in analysis["paths"])
    hop_ties = [path for path in analysis["paths"] if len(path) - 1 == hop_min]
    if template["mode"] == "detour" and len(analysis["best_path"]) - 1 == hop_min:
        return None
    if template["mode"] == "equal_hops" and not (len(hop_ties) >= 2 and len(analysis["best_path"]) - 1 == hop_min):
        return None

    path_nodes = set(analysis["best_path"])
    path_blue_nodes = [node for node in analysis["nodes"] if node["color"] == BLUE and node["id"] in path_nodes]
    off_path_blue_nodes = [node for node in analysis["nodes"] if node["color"] == BLUE and node["id"] not in path_nodes]
    if not path_blue_nodes or not off_path_blue_nodes:
        return None
    if not any(node["weight"] < 4 for node in path_blue_nodes):
        return None
    if not any(node["weight"] > 0 for node in off_path_blue_nodes):
        return None
    if len({_path_cost(path, analysis["nodes"]) for path in analysis["paths"]}) < 2:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    out_sig = _signature(solved)
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
