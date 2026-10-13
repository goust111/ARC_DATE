from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = np.array([2, 3, 4, 6, 7, 8, 9], dtype=int)
_RNG = np.random.default_rng()


def _find_nodes(grid):
    h, w = grid.shape
    nodes = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color == BLACK:
                continue
            block = grid[r - 1 : r + 2, c - 1 : c + 2]
            if np.all(block == color):
                nodes.append((r, c, color))
    return nodes


def _add_edge(adjacency, edges, a, b):
    if a == b:
        return
    key = (a, b) if a < b else (b, a)
    if key in edges:
        return
    edges.add(key)
    adjacency[a].append(b)
    adjacency[b].append(a)


def _build_graph(grid, nodes):
    adjacency = [[] for _ in nodes]
    edges = set()
    by_row = defaultdict(list)
    by_col = defaultdict(list)
    for i, (r, c, _) in enumerate(nodes):
        by_row[r].append((c, i))
        by_col[c].append((r, i))

    for r, items in by_row.items():
        items.sort()
        for (c1, i1), (c2, i2) in zip(items, items[1:]):
            if c2 - c1 <= 3:
                continue
            segment = grid[r, c1 + 2 : c2 - 1]
            if segment.size and np.all(segment == GREY):
                _add_edge(adjacency, edges, i1, i2)

    for c, items in by_col.items():
        items.sort()
        for (r1, i1), (r2, i2) in zip(items, items[1:]):
            if r2 - r1 <= 3:
                continue
            segment = grid[r1 + 2 : r2 - 1, c]
            if segment.size and np.all(segment == GREY):
                _add_edge(adjacency, edges, i1, i2)

    return adjacency, edges


def _distances_from_color(adjacency, nodes, color):
    inf = len(nodes) + 1
    dist = [inf] * len(nodes)
    q = deque()
    for i, (_, _, node_color) in enumerate(nodes):
        if node_color == color:
            dist[i] = 0
            q.append(i)
    while q:
        node = q.popleft()
        for nxt in adjacency[node]:
            if dist[nxt] == inf:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return dist


def _assigned_node_colors(adjacency, nodes):
    colors = []
    for _, _, color in nodes:
        if color != GREY and color not in colors:
            colors.append(color)
    if not colors:
        return [GREY] * len(nodes)

    dist_by_color = {color: _distances_from_color(adjacency, nodes, color) for color in colors}
    assigned = []
    for i in range(len(nodes)):
        best = len(nodes) + 1
        best_colors = []
        for color in colors:
            dist = dist_by_color[color][i]
            if dist < best:
                best = dist
                best_colors = [color]
            elif dist == best:
                best_colors.append(color)
        assigned.append(best_colors[0] if len(best_colors) == 1 else GREY)
    return assigned


def _paint_node(out, node, color):
    r, c, _ = node
    out[r - 1 : r + 2, c - 1 : c + 2] = color


def _paint_edge(out, node_a, node_b, color):
    r1, c1, _ = node_a
    r2, c2, _ = node_b
    if r1 == r2:
        c_lo, c_hi = sorted((c1, c2))
        out[r1, c_lo + 2 : c_hi - 1] = color
    elif c1 == c2:
        r_lo, r_hi = sorted((r1, r2))
        out[r_lo + 2 : r_hi - 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Color the grey tree network by propagating colors inward from colored leaf nodes.

    Input structure:
    The grid has a black background, several solid 3x3 node blocks, and single-pixel grey
    horizontal or vertical line segments joining node centers into an acyclic path network.
    Colored 3x3 nodes are leaves of the network; all other nodes and all connectors are grey.

    Rule:
    Treat the nodes and straight connectors between adjacent aligned nodes as a tree graph.
    Each colored leaf spreads its color one graph edge at a time.  A grey node becomes colored
    when exactly one visible color is closest to it, including the case where several leaves of
    that same color reach it.  If different colors meet at a node, the node stays grey.  A
    connector is colored when it borders one color region or the same color at both ends; when
    different colors arrive from its two endpoint nodes, that connector remains grey.

    Key insight:
    The picture is a graph problem rather than a pixel flood fill: identify the 3x3 blocks as
    graph nodes, count connector-to-connector distance from the colored leaves, and leave grey
    boundaries exactly where competing color waves meet.
    """
    arr = np.asarray(grid)
    out = np.array(arr, copy=True)
    nodes = _find_nodes(arr)
    if not nodes:
        return out

    adjacency, edges = _build_graph(arr, nodes)
    assigned = _assigned_node_colors(adjacency, nodes)

    for a, b in edges:
        ca, cb = assigned[a], assigned[b]
        if ca != GREY and cb != GREY and ca == cb:
            edge_color = ca
        elif ca != GREY and cb == GREY:
            edge_color = ca
        elif cb != GREY and ca == GREY:
            edge_color = cb
        else:
            edge_color = GREY
        _paint_edge(out, nodes[a], nodes[b], edge_color)

    for node, color in zip(nodes, assigned):
        _paint_node(out, node, color)
    return out


def _randint(low, high):
    return int(_RNG.integers(low, high + 1))


def _grid_neighbors(cell, rows, cols):
    r, c = cell
    if r > 0:
        yield (r - 1, c)
    if r + 1 < rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < cols:
        yield (r, c + 1)


def _make_random_tree(rows, cols, target):
    start = (_randint(0, rows - 1), _randint(0, cols - 1))
    used = {start}
    edges = []
    while len(used) < target:
        frontier = []
        for cell in used:
            for nxt in _grid_neighbors(cell, rows, cols):
                if nxt not in used:
                    frontier.append((cell, nxt))
        if not frontier:
            break
        parent, child = frontier[_randint(0, len(frontier) - 1)]
        used.add(child)
        edges.append((parent, child))
    return used, edges


def _make_path(rows, cols, target):
    for _ in range(80):
        start = (_randint(0, rows - 1), _randint(0, cols - 1))
        path = [start]
        used = {start}
        while len(path) < target:
            options = [nxt for nxt in _grid_neighbors(path[-1], rows, cols) if nxt not in used]
            if not options:
                break
            nxt = options[_randint(0, len(options) - 1)]
            used.add(nxt)
            path.append(nxt)
        if len(path) == target:
            return set(path), list(zip(path, path[1:]))
    return _make_random_tree(rows, cols, target)


def _choose_layout():
    spacing = 4 if _randint(0, 1) == 0 else 5
    max_rows = (30 - 5) // spacing + 1
    max_cols = (30 - 5) // spacing + 1
    rows = _randint(4, max_rows)
    cols = _randint(4, max_cols)
    base_h = (rows - 1) * spacing + 5
    base_w = (cols - 1) * spacing + 5
    extra_h = _randint(0, 30 - base_h)
    extra_w = _randint(0, 30 - base_w)
    top_pad = _randint(0, extra_h)
    left_pad = _randint(0, extra_w)
    height = base_h + extra_h
    width = base_w + extra_w
    row0 = 2 + top_pad
    col0 = 2 + left_pad
    return rows, cols, spacing, height, width, row0, col0


def generate_input() -> np.ndarray:
    rows, cols, spacing, height, width, row0, col0 = _choose_layout()
    area = rows * cols
    target = _randint(5, min(area, 14))
    if _randint(0, 2) == 0:
        cells, edges = _make_path(rows, cols, target)
    else:
        cells, edges = _make_random_tree(rows, cols, target)

    degree = {cell: 0 for cell in cells}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    leaves = [cell for cell, deg in degree.items() if deg == 1]
    color_count = min(len(leaves), _randint(2, min(4, len(leaves))))
    chosen = list(_RNG.choice(PALETTE, size=color_count, replace=False))
    _RNG.shuffle(leaves)
    leaf_colors = {}
    for i, leaf in enumerate(leaves):
        leaf_colors[leaf] = int(chosen[i]) if i < color_count else int(chosen[_randint(0, color_count - 1)])

    grid = np.zeros((height, width), dtype=int)

    def center(cell):
        r, c = cell
        return row0 + r * spacing, col0 + c * spacing

    for a, b in edges:
        r1, c1 = center(a)
        r2, c2 = center(b)
        if r1 == r2:
            c_lo, c_hi = sorted((c1, c2))
            grid[r1, c_lo + 2 : c_hi - 1] = GREY
        else:
            r_lo, r_hi = sorted((r1, r2))
            grid[r_lo + 2 : r_hi - 1, c1] = GREY

    for cell in cells:
        r, c = center(cell)
        color = leaf_colors.get(cell, GREY)
        grid[r - 1 : r + 2, c - 1 : c + 2] = color

    return grid
