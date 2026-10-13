import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
TEAL = 8
MAROON = 9

NODE_COLORS = (BLUE, RED, YELLOW, MAGENTA, ORANGE)
UNMATCHED_COLOR = MAROON
CELL_SIZE = 4
BORDER = 2
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_square(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h == w and h in (2, 3) and len(coords) == h * w


def _square_size(coords: list[tuple[int, int]]) -> int:
    r0, _c0, r1, _c1 = _bbox(coords)
    return r1 - r0 + 1


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    nodes = []
    loose_pixels = []
    colors = sorted(int(x) for x in np.unique(g) if int(x) not in (BLACK, GREY))
    for color in colors:
        for comp in _components(g == color):
            if len(comp) == 1:
                r, c = comp[0]
                loose_pixels.append({"color": color, "pos": (r, c)})
            elif _is_solid_square(comp):
                r0, c0, r1, c1 = _bbox(comp)
                cells = set(comp)
                nodes.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "size": _square_size(comp),
                        "cells": cells,
                    }
                )
            else:
                raise AssertionError("unexpected colored component")

    if len(nodes) < 2:
        raise AssertionError("expected at least two squares")
    node_colors = [node["color"] for node in nodes]
    if len(set(node_colors)) != len(node_colors):
        raise AssertionError("square colors must be unique")

    nodes.sort(key=lambda node: (node["bbox"][0], node["bbox"][1], node["color"]))
    for idx, node in enumerate(nodes):
        node["id"] = idx

    color_to_node = {node["color"]: node for node in nodes}
    cell_to_node = {}
    for node in nodes:
        for cell in node["cells"]:
            cell_to_node[cell] = node["id"]

    absorbed = []
    unmatched = []
    for px in loose_pixels:
        node = color_to_node.get(px["color"])
        if node is None:
            unmatched.append(px)
            continue
        direction = _absorption_direction(g, px["pos"], node["bbox"])
        if direction is None:
            unmatched.append(px)
            continue
        absorbed.append({"pixel": px, "node_id": node["id"], "direction": direction})

    track_components = []
    adjacency = {node["id"]: [] for node in nodes}
    for comp_id, comp in enumerate(_components(g == GREY)):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) != 1 and len(cols) != 1:
            raise AssertionError("each track segment must be straight")
        touched = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                node_id = cell_to_node.get((nr, nc))
                if node_id is not None:
                    touched.add(node_id)
        if len(touched) != 2:
            raise AssertionError("each track segment must touch exactly two squares")
        a, b = sorted(touched)
        track_components.append({"id": comp_id, "cells": list(comp), "nodes": (a, b)})
        adjacency[a].append((b, comp_id))
        adjacency[b].append((a, comp_id))

    terminal_ids = sorted({item["node_id"] for item in absorbed})
    if len(terminal_ids) != 2:
        raise AssertionError("expected exactly two terminal squares")

    start, goal = terminal_ids
    dist = {start: 0}
    count = {start: 1}
    prev = {start: None}
    prev_edge = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v, edge_id in adjacency[u]:
            nd = dist[u] + 1
            if v not in dist:
                dist[v] = nd
                count[v] = count[u]
                prev[v] = u
                prev_edge[v] = edge_id
                q.append(v)
            elif dist[v] == nd:
                count[v] += count[u]
    if goal not in dist:
        raise AssertionError("terminal squares must be connected")

    node_path = []
    edge_path = []
    cur = goal
    while cur is not None:
        node_path.append(cur)
        edge_id = prev_edge[cur]
        if edge_id is not None:
            edge_path.append(edge_id)
        cur = prev[cur]
    node_path.reverse()
    edge_path.reverse()

    return {
        "grid": g,
        "nodes": nodes,
        "loose_pixels": loose_pixels,
        "absorbed": absorbed,
        "unmatched": unmatched,
        "tracks": track_components,
        "adjacency": adjacency,
        "terminal_ids": terminal_ids,
        "shortest_count": count[goal],
        "node_path": node_path,
        "edge_path": edge_path,
    }


def _absorption_direction(grid: np.ndarray, pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r, c = pos
    r0, c0, r1, c1 = bbox
    options = []
    if r0 <= r <= r1:
        if c < c0 and np.all(grid[r, c + 1 : c0] == BLACK):
            options.append((0, 1))
        if c > c1 and np.all(grid[r, c1 + 1 : c] == BLACK):
            options.append((0, -1))
    if c0 <= c <= c1:
        if r < r0 and np.all(grid[r + 1 : r0, c] == BLACK):
            options.append((1, 0))
        if r > r1 and np.all(grid[r1 + 1 : r, c] == BLACK):
            options.append((-1, 0))
    if len(options) != 1:
        return None
    return options[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete loose pixels, use the two absorbed squares as terminals, and recolor the unique shortest graph path between them.

    Input structure:
    - A black background contains 4-5 solid colored squares, each a unique 2x2 or 3x3 block.
    - Grey 1-cell-wide straight segments connect square edges into a connected graph.
    - Several isolated loose pixels sit away from the graph; exactly two share colors with squares and align with them on a clear row or column.

    Rule:
    - Any loose pixel whose color matches a square slides along its clear row or column until it would touch that square, then disappears into it.
    - Any loose pixel whose color matches no square is deleted.
    - The two squares that absorb matching pixels become the terminals.
    - Find the shortest path between those terminals in the square-track graph, recolor non-terminal squares on that path to green and path tracks to teal, and leave everything else unchanged.

    Key insight:
    - The terminals are not pre-marked by position or size; they are defined only by which squares absorb same-colored loose pixels before the graph shortest-path step.
    """
    analysis = _parse_scene(np.asarray(grid, dtype=int))
    if analysis["shortest_count"] != 1:
        raise AssertionError("shortest path must be unique")

    out = np.asarray(grid, dtype=int).copy()
    for px in analysis["loose_pixels"]:
        r, c = px["pos"]
        out[r, c] = BLACK

    terminal_set = set(analysis["terminal_ids"])
    id_to_node = {node["id"]: node for node in analysis["nodes"]}
    for node_id in analysis["node_path"]:
        if node_id in terminal_set:
            continue
        for r, c in id_to_node[node_id]["cells"]:
            out[r, c] = GREEN

    tracks_by_id = {track["id"]: track for track in analysis["tracks"]}
    for edge_id in analysis["edge_path"]:
        for r, c in tracks_by_id[edge_id]["cells"]:
            out[r, c] = TEAL

    return out


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _pick_connected_cells(rows: int, cols: int, count: int, rng: random.Random) -> set[tuple[int, int]]:
    start = (rng.randrange(rows), rng.randrange(cols))
    picked = {start}
    while len(picked) < count:
        frontier = sorted({nb for cell in picked for nb in _cell_neighbors(cell, rows, cols) if nb not in picked})
        if not frontier:
            raise AssertionError("connected growth frontier vanished")
        picked.add(rng.choice(frontier))
    return picked


def _cell_edges(cells: set[tuple[int, int]]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges = set()
    for cell in cells:
        for nb in ((cell[0] + 1, cell[1]), (cell[0], cell[1] + 1)):
            if nb in cells:
                edges.add(tuple(sorted((cell, nb))))
    return edges


def _random_spanning_tree(cells: set[tuple[int, int]], full_edges: set, rng: random.Random) -> set:
    root = rng.choice(tuple(cells))
    seen = {root}
    tree = set()
    while seen != cells:
        frontier = [edge for edge in full_edges if (edge[0] in seen) ^ (edge[1] in seen)]
        if not frontier:
            raise AssertionError("could not build spanning tree")
        edge = rng.choice(frontier)
        tree.add(edge)
        seen.update(edge)
    return tree


def _adjacency_from_edges(edges: set[tuple[tuple[int, int], tuple[int, int]]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


def _lane_directions(cell: tuple[int, int], cells: set[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = cell[0] + dr, cell[1] + dc
        if (rr, cc) in cells:
            continue
        ok = True
        while 0 <= rr < rows and 0 <= cc < cols:
            if (rr, cc) in cells:
                ok = False
                break
            rr += dr
            cc += dc
        if ok:
            out.append((dr, dc))
    return out


def _shortest_path_info(
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[int, int, list[tuple[int, int]]]:
    adj = _adjacency_from_edges(edges)
    q = deque([start])
    dist = {start: 0}
    count = {start: 1}
    prev = {start: None}
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            nd = dist[u] + 1
            if v not in dist:
                dist[v] = nd
                count[v] = count[u]
                prev[v] = u
                q.append(v)
            elif dist[v] == nd:
                count[v] += count[u]
    if goal not in dist:
        raise AssertionError("graph must stay connected")
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[goal], count[goal], path


def _build_abstract_graph(rng: random.Random) -> dict | None:
    for _ in range(200):
        rows = rng.choice((3, 4))
        cols = rng.choice((3, 4))
        max_nodes = min(5, rows * cols)
        node_count = rng.randint(4, max_nodes)
        cells = _pick_connected_cells(rows, cols, node_count, rng)
        full_edges = _cell_edges(cells)
        if len(full_edges) < node_count - 1:
            continue

        tree_edges = _random_spanning_tree(cells, full_edges, rng)
        edges = set(tree_edges)
        extra_choices = list(full_edges - tree_edges)
        rng.shuffle(extra_choices)
        if extra_choices and rng.random() < 0.6:
            edges.add(extra_choices[0])

        lane_map = {cell: _lane_directions(cell, cells, rows, cols) for cell in cells}
        candidates = [cell for cell, dirs in lane_map.items() if dirs]
        if len(candidates) < 2:
            continue

        pair_options = []
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                dist, count, path = _shortest_path_info(edges, a, b)
                if count != 1 or not (2 <= dist <= 4):
                    continue
                path_edges = {tuple(sorted((path[k], path[k + 1]))) for k in range(len(path) - 1)}
                if len(set(path)) == len(cells) and len(path_edges) == len(edges):
                    continue
                pair_options.append((a, b, path))
        if not pair_options:
            continue

        adj = _adjacency_from_edges(edges)
        richer = [item for item in pair_options if len(adj[item[0]]) > 1 or len(adj[item[1]]) > 1]
        if richer and rng.random() < 0.6:
            terminals = rng.choice(richer)
        else:
            terminals = rng.choice(pair_options)

        return {
            "rows": rows,
            "cols": cols,
            "cells": cells,
            "edges": edges,
            "lane_map": lane_map,
            "terminals": terminals[:2],
            "path": terminals[2],
        }
    return None


def _square_bbox(cell: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    base_r = BORDER + cell[0] * CELL_SIZE
    base_c = BORDER + cell[1] * CELL_SIZE
    offset = (CELL_SIZE - size) // 2
    r0 = base_r + offset
    c0 = base_c + offset
    return r0, c0, r0 + size - 1, c0 + size - 1


def _edge_track_cells(
    a: tuple[int, int],
    b: tuple[int, int],
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    ra0, ca0, ra1, ca1 = bboxes[a]
    rb0, cb0, rb1, cb1 = bboxes[b]
    if a[0] == b[0]:
        left, right = (a, b) if a[1] < b[1] else (b, a)
        rl0, cl0, rl1, cl1 = bboxes[left]
        rr0, cr0, rr1, cr1 = bboxes[right]
        row = (max(rl0, rr0) + min(rl1, rr1)) // 2
        return [(row, c) for c in range(cl1 + 1, cr0)]
    if a[1] == b[1]:
        top, bottom = (a, b) if a[0] < b[0] else (b, a)
        rt0, ct0, rt1, ct1 = bboxes[top]
        rb0, cb0, rb1, cb1 = bboxes[bottom]
        col = (max(ct0, cb0) + min(ct1, cb1)) // 2
        return [(r, col) for r in range(rt1 + 1, rb0)]
    raise AssertionError("edges must connect cardinally adjacent cells")


def _lane_positions(
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    dr, dc = direction
    candidates = []
    if (dr, dc) == (-1, 0):
        for c in range(c0, c1 + 1):
            for r in range(0, r0 - 1):
                candidates.append((r, c))
    elif (dr, dc) == (1, 0):
        for c in range(c0, c1 + 1):
            for r in range(r1 + 2, h):
                candidates.append((r, c))
    elif (dr, dc) == (0, -1):
        for r in range(r0, r1 + 1):
            for c in range(0, c0 - 1):
                candidates.append((r, c))
    elif (dr, dc) == (0, 1):
        for r in range(r0, r1 + 1):
            for c in range(c1 + 2, w):
                candidates.append((r, c))
    return candidates


def _cell_is_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != BLACK:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != BLACK:
            return False
    return True


def _place_aligned_pixel(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    color: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    positions = _lane_positions(bbox, direction, grid.shape)
    rng.shuffle(positions)
    for r, c in positions:
        if _cell_is_isolated(grid, r, c):
            grid[r, c] = color
            return r, c
    return None


def _place_random_isolated_pixel(grid: np.ndarray, color: int, rng: random.Random) -> tuple[int, int] | None:
    cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    rng.shuffle(cells)
    for r, c in cells:
        if _cell_is_isolated(grid, r, c):
            grid[r, c] = color
            return r, c
    return None


def _generate_sizes(cells: set[tuple[int, int]], rng: random.Random) -> dict[tuple[int, int], int]:
    sizes = {cell: rng.choice((2, 3)) for cell in cells}
    if len(cells) >= 2 and len(set(sizes.values())) == 1 and rng.random() < 0.7:
        cell = rng.choice(tuple(cells))
        sizes[cell] = 5 - sizes[cell]
    return sizes


def _expected_output(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    terminals: tuple[tuple[int, int], tuple[int, int]],
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
    edge_pixels: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]],
    loose_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in loose_positions:
        out[r, c] = BLACK
    terminal_set = set(terminals)
    for cell in path:
        if cell in terminal_set:
            continue
        r0, c0, r1, c1 = bboxes[cell]
        out[r0 : r1 + 1, c0 : c1 + 1] = GREEN
    for i in range(len(path) - 1):
        edge = tuple(sorted((path[i], path[i + 1])))
        for r, c in edge_pixels[edge]:
            out[r, c] = TEAL
    return out


def _validate_generated(
    grid: np.ndarray,
    terminals: tuple[tuple[int, int], tuple[int, int]],
    cell_colors: dict[tuple[int, int], int],
    expected: np.ndarray,
    unmatched_count: int,
) -> bool:
    try:
        analysis = _parse_scene(grid)
    except AssertionError:
        return False
    if analysis["shortest_count"] != 1:
        return False
    if len(analysis["nodes"]) not in (4, 5):
        return False
    if len(analysis["tracks"]) < len(analysis["nodes"]) - 1:
        return False
    if len(analysis["absorbed"]) != 2:
        return False
    if len(analysis["unmatched"]) != unmatched_count:
        return False
    parsed_terminal_colors = {analysis["nodes"][node_id]["color"] for node_id in analysis["terminal_ids"]}
    expected_terminal_colors = {cell_colors[terminals[0]], cell_colors[terminals[1]]}
    if parsed_terminal_colors != expected_terminal_colors:
        return False
    if np.array_equal(grid, expected):
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    abstract = _build_abstract_graph(rng)
    if abstract is None:
        return None

    rows = abstract["rows"]
    cols = abstract["cols"]
    cells = abstract["cells"]
    edges = abstract["edges"]
    terminals = abstract["terminals"]
    path = abstract["path"]
    lane_map = abstract["lane_map"]

    h = 2 * BORDER + rows * CELL_SIZE
    w = 2 * BORDER + cols * CELL_SIZE
    grid = np.zeros((h, w), dtype=int)

    colors = rng.sample(list(NODE_COLORS), len(cells))
    ordered_cells = sorted(cells)
    cell_colors = {cell: color for cell, color in zip(ordered_cells, colors)}
    sizes = _generate_sizes(cells, rng)
    bboxes = {cell: _square_bbox(cell, sizes[cell]) for cell in cells}

    for cell in ordered_cells:
        r0, c0, r1, c1 = bboxes[cell]
        grid[r0 : r1 + 1, c0 : c1 + 1] = cell_colors[cell]

    edge_pixels = {}
    for edge in edges:
        pixels = _edge_track_cells(edge[0], edge[1], bboxes)
        if not pixels:
            return None
        for r, c in pixels:
            if grid[r, c] != BLACK:
                return None
        for r, c in pixels:
            grid[r, c] = GREY
        edge_pixels[tuple(sorted(edge))] = pixels

    loose_positions = []
    for terminal in terminals:
        directions = list(lane_map[terminal])
        rng.shuffle(directions)
        placed = None
        for direction in directions:
            placed = _place_aligned_pixel(grid, bboxes[terminal], direction, cell_colors[terminal], rng)
            if placed is not None:
                loose_positions.append(placed)
                break
        if placed is None:
            return None

    unmatched_count = rng.randint(1, 3)
    aligned_unmatched = min(unmatched_count, rng.randint(0, unmatched_count))
    lane_cells = [cell for cell in ordered_cells if lane_map[cell]]
    for _ in range(aligned_unmatched):
        rng.shuffle(lane_cells)
        placed = None
        for cell in lane_cells:
            dirs = list(lane_map[cell])
            rng.shuffle(dirs)
            for direction in dirs:
                placed = _place_aligned_pixel(grid, bboxes[cell], direction, UNMATCHED_COLOR, rng)
                if placed is not None:
                    loose_positions.append(placed)
                    break
            if placed is not None:
                break
        if placed is None:
            return None
    while len(loose_positions) < 2 + unmatched_count:
        placed = _place_random_isolated_pixel(grid, UNMATCHED_COLOR, rng)
        if placed is None:
            return None
        loose_positions.append(placed)

    expected = _expected_output(grid, path, terminals, bboxes, edge_pixels, loose_positions)
    if not _validate_generated(grid, terminals, cell_colors, expected, unmatched_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
