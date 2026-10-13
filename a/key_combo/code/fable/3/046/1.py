from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _center_offsets(size: int) -> list[tuple[int, int]]:
    if size % 2 == 1:
        m = size // 2
        return [(m, m)]
    a = size // 2 - 1
    b = size // 2
    return [(a, a), (a, b), (b, a), (b, b)]


def _node_candidate(g: np.ndarray, bg: int, color: int, coords: np.ndarray) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height != width or height < 2:
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    if int(np.count_nonzero(patch == int(color))) != int(coords.shape[0]):
        return None

    if bool(np.all(patch == int(color))):
        return {
            "color": int(color),
            "size": int(height),
            "bbox": (r0, c0, r1, c1),
            "coords": coords,
            "hollow": False,
        }

    if height < 3:
        return None
    if not bool(np.all((patch == int(color)) | (patch == int(bg)))):
        return None
    border = np.zeros(patch.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not bool(np.all(patch[border] == int(color))):
        return None
    if not all(int(patch[rr, cc]) == int(bg) for rr, cc in _center_offsets(height)):
        return None

    return {
        "color": int(color),
        "size": int(height),
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "hollow": True,
    }


def _find_nodes(g: np.ndarray, bg: int) -> list[dict]:
    nodes: list[dict] = []
    for color in _colors_present(g, exclude={bg}):
        color_nodes: list[dict] = []
        for coords in _components(g == int(color)):
            cand = _node_candidate(g, bg, int(color), coords)
            if cand is not None:
                color_nodes.append(cand)
        if color_nodes:
            assert len(color_nodes) == 1, "each node color must identify one square node"
            nodes.append(color_nodes[0])

    assert nodes, "expected square nodes"
    assert len({int(n["color"]) for n in nodes}) == len(nodes), "node colors must be unique"
    for idx, node in enumerate(nodes):
        node["id"] = int(idx)
    return nodes


def _track_graph(g: np.ndarray, bg: int, nodes: list[dict]):
    node_colors = {int(n["color"]) for n in nodes}
    track_colors = [c for c in _colors_present(g, exclude={bg}) if int(c) not in node_colors]
    assert len(track_colors) == 1, "expected exactly one track color"
    track_color = int(track_colors[0])

    pixel_to_node: dict[tuple[int, int], int] = {}
    for node in nodes:
        nid = int(node["id"])
        for r, c in node["coords"]:
            pixel_to_node[(int(r), int(c))] = nid

    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    edge_pixels: dict[int, list[tuple[int, int]]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    edge_id = 0
    for coords in _components(g == track_color):
        touched: set[int] = set()
        pts = [(int(r), int(c)) for r, c in coords]
        for r, c in pts:
            for nr, nc in _neighbors4(r, c):
                nid = pixel_to_node.get((nr, nc))
                if nid is not None:
                    touched.add(int(nid))
        assert len(touched) == 2, "each track segment must connect exactly two nodes"
        a, b = sorted(touched)
        pair = (int(a), int(b))
        assert pair not in seen_pairs, "parallel track segments would make the graph ambiguous"
        seen_pairs.add(pair)
        edge_id += 1
        edge_pixels[edge_id] = pts
        adj[a].append((b, edge_id))
        adj[b].append((a, edge_id))

    return adj, edge_pixels


def _unique_shortest_path(adj: dict[int, list[tuple[int, int]]], start: int, goal: int) -> list[int]:
    dist = {int(start): 0}
    count = {int(start): 1}
    prev: dict[int, int | None] = {int(start): None}
    q = deque([int(start)])

    while q:
        node = q.popleft()
        for nbr, _edge_id in adj.get(node, []):
            nbr = int(nbr)
            nd = dist[node] + 1
            if nbr not in dist:
                dist[nbr] = nd
                count[nbr] = count[node]
                prev[nbr] = node
                q.append(nbr)
            elif dist[nbr] == nd:
                count[nbr] += count[node]

    assert int(goal) in dist, "terminals must be connected"
    assert count[int(goal)] == 1, "shortest terminal path must be unique"

    path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        path.append(int(cur))
        cur = prev[cur]
    path.reverse()
    return path


def _ring_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw the shortest track path between hollow terminal squares as nested square rings.

    Input structure:
    - The background is the most common color.
    - Distinct node colors form square nodes; exactly two square nodes are hollow with
      background-colored centers, and those two terminal squares have different side lengths.
    - A single track color forms one-cell-wide segments between pairs of node squares,
      including branches and nodes that do not lie on the shortest terminal path.

    Rule:
    - Treat the square nodes as graph vertices and each track segment as an edge.
    - Find the unique path with the fewest visited nodes between the two hollow terminals.
    - Create a square output of side 2k-1 for k path nodes, painting nested rings from
      the larger terminal toward the smaller terminal in shortest-path order.

    Key insight:
    - The ring order is the graph shortest-path visitation order, not object size,
      color value, or spatial position; terminal size only chooses which end is outside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    nodes = _find_nodes(g, bg)
    hollow = [n for n in nodes if bool(n["hollow"])]
    assert len(hollow) == 2, "expected exactly two hollow terminal nodes"
    assert int(hollow[0]["size"]) != int(hollow[1]["size"]), "terminal sizes must differ"

    outer = max(hollow, key=lambda n: int(n["size"]))
    inner = min(hollow, key=lambda n: int(n["size"]))
    adj, _edge_pixels = _track_graph(g, bg, nodes)
    node_path = _unique_shortest_path(adj, int(outer["id"]), int(inner["id"]))

    by_id = {int(n["id"]): n for n in nodes}
    colors = [int(by_id[nid]["color"]) for nid in node_path]
    return _ring_output(colors)


# ---------------- generator ----------------

_MAX_ATTEMPTS = 500
_CELL_DIMS = ((3, 3), (3, 4), (4, 3), (4, 4))
_CELL_STEPS = (6, 7)
_NODE_SIZES = (3, 5)


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _has_bend(path: list[tuple[int, int]]) -> bool:
    axes = set()
    for a, b in zip(path, path[1:]):
        axes.add("r" if a[0] != b[0] else "c")
    return len(axes) == 2


def _random_path(rows: int, cols: int, edges: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(120):
        start = random.choice(cells)
        path = [start]
        used = {start}
        for _step in range(edges):
            choices = [n for n in _cell_neighbors(path[-1], rows, cols) if n not in used]
            if not choices:
                break
            nxt = random.choice(choices)
            path.append(nxt)
            used.add(nxt)
        if len(path) == edges + 1 and _has_bend(path):
            return path
    return None


def _build_tree(rows: int, cols: int, path: list[tuple[int, int]], total_nodes: int):
    used = set(path)
    graph_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for a, b in zip(path, path[1:]):
        graph_edges.add(tuple(sorted((a, b))))

    branch_options = []
    for cell in path[1:-1]:
        for nbr in _cell_neighbors(cell, rows, cols):
            if nbr not in used:
                branch_options.append((cell, nbr))
    if not branch_options:
        return None

    attach, first_branch = random.choice(branch_options)
    used.add(first_branch)
    graph_edges.add(tuple(sorted((attach, first_branch))))
    ordered_cells = list(path) + [first_branch]

    while len(ordered_cells) < total_nodes:
        candidates = []
        for cell in ordered_cells:
            for nbr in _cell_neighbors(cell, rows, cols):
                if nbr not in used:
                    candidates.append((cell, nbr))
        if not candidates:
            return None
        parent, child = random.choice(candidates)
        used.add(child)
        ordered_cells.append(child)
        graph_edges.add(tuple(sorted((parent, child))))

    degree = Counter()
    for a, b in graph_edges:
        degree[a] += 1
        degree[b] += 1
    if degree[attach] < 3:
        return None
    return ordered_cells, graph_edges


def _cell_centers(rows: int, cols: int, step: int) -> tuple[int, int, int, int]:
    min_h = (rows - 1) * step + 7
    min_w = (cols - 1) * step + 7
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    base_r = random.randint(3, height - 4 - (rows - 1) * step)
    base_c = random.randint(3, width - 4 - (cols - 1) * step)
    return height, width, base_r, base_c


def _draw_node(
    grid: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: int,
    bg: int,
    hollow: bool,
) -> tuple[int, int, int, int]:
    r, c = center
    half = size // 2
    r0, c0 = r - half, c - half
    r1, c1 = r + half, c + half
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    if hollow:
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
    return r0, c0, r1, c1


def _track_pixels(
    center_a: tuple[int, int],
    size_a: int,
    center_b: tuple[int, int],
    size_b: int,
) -> list[tuple[int, int]]:
    ra, ca = center_a
    rb, cb = center_b
    ha = size_a // 2
    hb = size_b // 2
    pixels: list[tuple[int, int]] = []
    if ra == rb:
        if ca < cb:
            c0, c1 = ca + ha + 1, cb - hb - 1
        else:
            c0, c1 = cb + hb + 1, ca - ha - 1
        pixels = [(ra, c) for c in range(c0, c1 + 1)]
    else:
        if ra < rb:
            r0, r1 = ra + ha + 1, rb - hb - 1
        else:
            r0, r1 = rb + hb + 1, ra - ha - 1
        pixels = [(r, ca) for r in range(r0, r1 + 1)]
    return pixels


def _known_shortest_count(
    cells: list[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[int, int]:
    adj: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    dist = {start: 0}
    count = {start: 1}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nbr in adj[cell]:
            nd = dist[cell] + 1
            if nbr not in dist:
                dist[nbr] = nd
                count[nbr] = count[cell]
                q.append(nbr)
            elif dist[nbr] == nd:
                count[nbr] += count[cell]
    return dist[goal], count[goal]


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(_CELL_DIMS)
    max_nodes = min(8, rows * cols)
    path_edges = random.randint(3, min(5, max_nodes - 2))
    min_nodes = path_edges + 2
    if min_nodes > max_nodes:
        return None
    total_nodes = random.randint(min_nodes, max_nodes)

    path = _random_path(rows, cols, path_edges)
    if path is None:
        return None
    tree = _build_tree(rows, cols, path, total_nodes)
    if tree is None:
        return None
    ordered_cells, graph_edges = tree

    step = random.choice(_CELL_STEPS)
    height, width, base_r, base_c = _cell_centers(rows, cols, step)

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    track_color = random.choice(palette)
    palette.remove(track_color)
    if len(palette) < total_nodes:
        return None
    node_colors = random.sample(palette, total_nodes)
    color_by_cell = dict(zip(ordered_cells, node_colors))

    big_at_path_start = bool(random.getrandbits(1))
    size_by_cell: dict[tuple[int, int], int] = {}
    size_by_cell[path[0]] = 5 if big_at_path_start else 3
    size_by_cell[path[-1]] = 3 if big_at_path_start else 5
    for cell in ordered_cells:
        if cell not in size_by_cell:
            size_by_cell[cell] = random.choice(_NODE_SIZES)
    if all(size_by_cell[cell] != 5 for cell in path[1:-1]):
        size_by_cell[random.choice(path[1:-1])] = 5

    center_by_cell = {
        cell: (base_r + cell[0] * step, base_c + cell[1] * step)
        for cell in ordered_cells
    }
    grid = np.full((height, width), int(bg), dtype=int)
    for cell in ordered_cells:
        _draw_node(
            grid,
            center_by_cell[cell],
            size_by_cell[cell],
            color_by_cell[cell],
            bg,
            hollow=cell in {path[0], path[-1]},
        )

    for a, b in graph_edges:
        pixels = _track_pixels(center_by_cell[a], size_by_cell[a], center_by_cell[b], size_by_cell[b])
        if not pixels:
            return None
        for r, c in pixels:
            if not (0 <= r < height and 0 <= c < width) or int(grid[r, c]) != int(bg):
                return None
        for r, c in pixels:
            grid[r, c] = int(track_color)

    if _get_bg(grid) != int(bg):
        return None

    outer_cell = path[0] if big_at_path_start else path[-1]
    inner_cell = path[-1] if big_at_path_start else path[0]
    dist, count = _known_shortest_count(ordered_cells, graph_edges, outer_cell, inner_cell)
    if dist != path_edges or count != 1:
        return None

    oriented_path = path if big_at_path_start else list(reversed(path))
    expected_colors = [int(color_by_cell[cell]) for cell in oriented_path]
    expected = _ring_output(expected_colors)

    visible_areas = []
    for cell in oriented_path:
        size = size_by_cell[cell]
        if cell in {path[0], path[-1]}:
            area = size * size - (size - 2) * (size - 2)
        else:
            area = size * size
        visible_areas.append(area)
    by_area = [
        color for _area, color in sorted(zip(visible_areas, expected_colors), key=lambda item: item[0], reverse=True)
    ]
    if by_area == expected_colors:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid graph-distance nested-ring input")
