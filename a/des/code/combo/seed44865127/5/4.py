from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CANVAS_SIZE = 80
CANVAS_CENTER = 40
GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _is_node(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    size_r = int(r1 - r0 + 1)
    size_c = int(c1 - c0 + 1)
    if size_r != size_c or size_r not in (2, 3):
        return False
    if int(comp["area"]) != size_r * size_r:
        return False
    color = int(comp["color"])
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _follow_line(
    grid: np.ndarray,
    label_grid: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    line_color: int,
) -> int:
    h, w = grid.shape
    r, c = int(start_r), int(start_c)
    length = 0
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == line_color:
        length += 1
        r += dr
        c += dc
    assert length > 0, "expected a non-empty grey segment"
    assert 0 <= r < h and 0 <= c < w, "grey segment must terminate at another node"
    target = int(label_grid[r, c])
    assert target >= 0, "grey segment must end on a node"
    return target


def _extract_tree(grid: np.ndarray) -> tuple[int, int, list[dict], list[set[int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    components = _find_components(g, bg)
    nodes = [comp for comp in components if _is_node(g, comp)]
    assert len(nodes) >= 3, "expected at least three square nodes"

    node_color_set = {int(node["color"]) for node in nodes}
    non_bg_colors = set(map(int, np.unique(g))) - {bg}
    line_colors = non_bg_colors - node_color_set
    assert len(line_colors) == 1, "expected a unique grey line color"
    line_color = next(iter(line_colors))

    label_grid = np.full(g.shape, -1, dtype=int)
    out_nodes: list[dict] = []
    for idx, node in enumerate(nodes):
        coords = np.asarray(node["coords"], dtype=int)
        label_grid[coords[:, 0], coords[:, 1]] = idx
        r0, c0, r1, c1 = node["bbox"]
        out_nodes.append(
            {
                "color": int(node["color"]),
                "size": int(r1 - r0 + 1),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    edges: set[tuple[int, int]] = set()
    for idx, node in enumerate(nodes):
        r0, c0, r1, c1 = map(int, node["bbox"])
        starts = []
        if r0 > 0:
            for c in range(c0, c1 + 1):
                if int(g[r0 - 1, c]) == line_color:
                    starts.append((r0 - 1, c, -1, 0))
        if r1 + 1 < g.shape[0]:
            for c in range(c0, c1 + 1):
                if int(g[r1 + 1, c]) == line_color:
                    starts.append((r1 + 1, c, 1, 0))
        if c0 > 0:
            for r in range(r0, r1 + 1):
                if int(g[r, c0 - 1]) == line_color:
                    starts.append((r, c0 - 1, 0, -1))
        if c1 + 1 < g.shape[1]:
            for r in range(r0, r1 + 1):
                if int(g[r, c1 + 1]) == line_color:
                    starts.append((r, c1 + 1, 0, 1))

        for sr, sc, dr, dc in starts:
            target = _follow_line(g, label_grid, sr, sc, dr, dc, line_color)
            if target != idx:
                a, b = sorted((idx, target))
                edges.add((a, b))

    graph = [set() for _ in out_nodes]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)

    assert len(edges) == len(out_nodes) - 1, "connections must form a tree"
    seen = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert len(seen) == len(out_nodes), "tree must be connected"

    degrees = [len(neighbors) for neighbors in graph]
    max_degree = max(degrees)
    roots = [idx for idx, degree in enumerate(degrees) if degree == max_degree]
    assert len(roots) == 1, "expected a unique max-degree root"
    return bg, line_color, out_nodes, graph, roots[0]


def _root_tree(graph: list[set[int]], root: int) -> tuple[list[int], list[list[int]], list[int]]:
    parent = [-1] * len(graph)
    depth = [-1] * len(graph)
    parent[root] = root
    depth[root] = 0
    q = deque([root])
    order = [root]
    while q:
        node = q.popleft()
        for nxt in sorted(graph[node]):
            if depth[nxt] == -1:
                parent[nxt] = node
                depth[nxt] = depth[node] + 1
                q.append(nxt)
                order.append(nxt)
    assert len(order) == len(graph), "tree traversal must visit every node"
    children = [[] for _ in graph]
    for node in order[1:]:
        children[parent[node]].append(node)
    return parent, children, depth


def _ordered_children(children: list[list[int]], sizes: list[int], root: int) -> tuple[list[list[int]], tuple]:
    ordered = [[] for _ in children]

    def rec(node: int) -> tuple:
        child_pairs = [(rec(child), child) for child in children[node]]
        child_pairs.sort(key=lambda item: item[0])
        ordered[node] = [child for _sig, child in child_pairs]
        return (int(sizes[node]), tuple(sig for sig, _child in child_pairs))

    signature = rec(root)
    return ordered, signature


def _assign_x2(root: int, ordered: list[list[int]]) -> tuple[list[int], int]:
    x2 = [0] * len(ordered)

    def rec(node: int, leaf_index: int) -> int:
        if not ordered[node]:
            x2[node] = 2 * leaf_index
            return leaf_index + 1
        cur = leaf_index
        child_centers: list[int] = []
        for child in ordered[node]:
            cur = rec(child, cur)
            child_centers.append(x2[child])
        x2[node] = (child_centers[0] + child_centers[-1]) // 2
        return cur

    leaf_count = rec(root, 0)
    return x2, leaf_count


def _draw_tree_layout(bg: int, line_color: int, nodes: list[dict], graph: list[set[int]], root: int) -> np.ndarray:
    sizes = [int(node["size"]) for node in nodes]
    colors = [int(node["color"]) for node in nodes]
    _parent, children, depth = _root_tree(graph, root)
    ordered, _signature = _ordered_children(children, sizes, root)
    x2, _leaf_count = _assign_x2(root, ordered)

    max_size = max(sizes)
    half_step = max_size
    layer_step = max_size + 3
    margin = max_size + 1

    x = [margin + x2[idx] * half_step for idx in range(len(nodes))]
    top = [depth[idx] * layer_step for idx in range(len(nodes))]
    height = max(top[idx] + sizes[idx] for idx in range(len(nodes)))
    width = max(x[idx] + (sizes[idx] - 1) // 2 for idx in range(len(nodes))) + margin + 1
    out = np.full((height, width), bg, dtype=int)

    for idx, node in enumerate(nodes):
        size = sizes[idx]
        left = x[idx] - size // 2
        out[top[idx] : top[idx] + size, left : left + size] = colors[idx]

    for node in range(len(nodes)):
        if not ordered[node]:
            continue
        node_bottom = top[node] + sizes[node] - 1
        child_top = top[ordered[node][0]]
        assert child_top - node_bottom >= 3, "expected a gap for tree connectors"

        if len(ordered[node]) == 1 and x[ordered[node][0]] == x[node]:
            child = ordered[node][0]
            out[node_bottom + 1 : top[child], x[node]] = line_color
            continue

        bus_y = child_top - 2
        out[node_bottom + 1 : bus_y + 1, x[node]] = line_color
        span_left = min([x[node]] + [x[child] for child in ordered[node]])
        span_right = max([x[node]] + [x[child] for child in ordered[node]])
        out[bus_y, span_left : span_right + 1] = line_color
        for child in ordered[node]:
            out[bus_y : top[child], x[child]] = line_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the node-line tree as a canonical top-down hierarchy.

    The input contains small solid square nodes connected by one-cell-thick grey
    straight segments. The original embedding is discarded after the node graph is
    recovered. The unique max-degree node is treated as the root, children are
    ordered by a canonical size+subtree-shape signature, and the tree is redrawn
    from top to bottom on a clean canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg, line_color, nodes, graph, root = _extract_tree(g)
    return _draw_tree_layout(bg, line_color, nodes, graph, root)


def _chain_length_options(total: int) -> list[tuple[int, int, int]]:
    options = []
    for a in range(1, total - 1):
        for b in range(1, total - a):
            c = total - a - b
            counts = sorted((a, b, c))
            if counts[0] <= counts[1] <= counts[2] and max((a == b) + (b == c) + (a == c), 0) <= 1:
                options.append((a, b, c))
    # The previous check keeps "all equal" out and allows at most one duplicated value.
    dedup = sorted(set(options))
    return dedup


def _build_chain_topology(rng: random.Random) -> list[list[int]]:
    total_nodes = rng.randint(5, 8)
    lengths = list(rng.choice(_chain_length_options(total_nodes - 1)))
    rng.shuffle(lengths)

    children: list[list[int]] = [[] for _ in range(total_nodes)]
    next_id = 1
    for length in lengths:
        head = next_id
        children[0].append(head)
        next_id += 1
        prev = head
        for _ in range(length - 1):
            child = next_id
            children[prev].append(child)
            next_id += 1
            prev = child
    assert next_id == total_nodes
    return children


def _build_branch_topology() -> list[list[int]]:
    # Root has four distinct subtrees: leaf, leaf, chain(2), and a 2-leaf fork.
    children = [[] for _ in range(8)]
    children[0] = [1, 2, 3, 5]
    children[3] = [4]
    children[5] = [6, 7]
    return children


def _signature_unique(children: list[list[int]], sizes: list[int], node: int = 0) -> tuple | None:
    child_sigs: list[tuple] = []
    for child in children[node]:
        sig = _signature_unique(children, sizes, child)
        if sig is None:
            return None
        child_sigs.append(sig)
    child_sigs.sort()
    for prev, cur in zip(child_sigs, child_sigs[1:]):
        if prev == cur:
            return None
    return (int(sizes[node]), tuple(child_sigs))


def _sample_sizes(children: list[list[int]], rng: random.Random) -> list[int] | None:
    for _ in range(200):
        sizes = [rng.choice((2, 3)) for _ in range(len(children))]
        if len(set(sizes)) < 2:
            continue
        if _signature_unique(children, sizes) is not None:
            return sizes
    return None


def _subtree_sizes(children: list[list[int]]) -> list[int]:
    sizes = [0] * len(children)

    def rec(node: int) -> int:
        total = 1
        for child in children[node]:
            total += rec(child)
        sizes[node] = total
        return total

    rec(0)
    return sizes


def _place_topology(children: list[list[int]], rng: random.Random) -> dict[int, tuple[int, int]] | None:
    subtree = _subtree_sizes(children)
    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    used = {(0, 0)}

    def descendants(node: int) -> list[int]:
        out = [node]
        for child in children[node]:
            out.extend(descendants(child))
        return out

    desc_cache = {node: descendants(node) for node in range(len(children))}

    def rollback(node: int) -> None:
        for target in desc_cache[node]:
            pos = positions.pop(target, None)
            if pos is not None:
                used.remove(pos)

    def rec(node: int) -> bool:
        ordered_children = sorted(children[node], key=lambda child: (-subtree[child], child))

        def place_child(index: int) -> bool:
            if index == len(ordered_children):
                return True

            child = ordered_children[index]
            pr, pc = positions[node]
            directions = DIRS4[:]
            rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = pr + dr, pc + dc
                if not (-3 <= nr <= 3 and -3 <= nc <= 3):
                    continue
                if (nr, nc) in used:
                    continue
                positions[child] = (nr, nc)
                used.add((nr, nc))
                if rec(child) and place_child(index + 1):
                    return True
                rollback(child)
            return False

        return place_child(0)

    return positions if rec(0) else None


def _side_port(center: tuple[int, int], size: int, direction: tuple[int, int]) -> tuple[int, int]:
    r, c = center
    dr, dc = direction
    if dr == -1:
        return (r - size // 2, c)
    if dr == 1:
        return (r + (size - 1) // 2, c)
    if dc == -1:
        return (r, c - size // 2)
    return (r, c + (size - 1) // 2)


def _draw_input_from_topology(
    children: list[list[int]],
    positions: dict[int, tuple[int, int]],
    sizes: list[int],
    colors: list[int],
    bg: int,
    stride: int,
) -> np.ndarray:
    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), bg, dtype=int)
    centers: dict[int, tuple[int, int]] = {}

    for node, (cr, cc) in positions.items():
        center_r = CANVAS_CENTER + cr * stride
        center_c = CANVAS_CENTER + cc * stride
        centers[node] = (center_r, center_c)
        size = sizes[node]
        top = center_r - size // 2
        left = center_c - size // 2
        grid[top : top + size, left : left + size] = colors[node]

    for parent, child_list in enumerate(children):
        for child in child_list:
            pr, pc = positions[parent]
            cr, cc = positions[child]
            dr = cr - pr
            dc = cc - pc
            direction = (int(np.sign(dr)), int(np.sign(dc)))
            assert direction in {(-1, 0), (1, 0), (0, -1), (0, 1)}
            parent_port = _side_port(centers[parent], sizes[parent], direction)
            child_port = _side_port(centers[child], sizes[child], (-direction[0], -direction[1]))

            if direction[0] != 0:
                c = parent_port[1]
                r0 = min(parent_port[0], child_port[0]) + 1
                r1 = max(parent_port[0], child_port[0])
                grid[r0:r1, c] = GREY
            else:
                r = parent_port[0]
                c0 = min(parent_port[1], child_port[1]) + 1
                c1 = max(parent_port[1], child_port[1])
                grid[r, c0:c1] = GREY

    rows, cols = np.nonzero(grid != bg)
    r0 = max(0, int(rows.min()) - random.randint(2, 4))
    r1 = min(grid.shape[0] - 1, int(rows.max()) + random.randint(2, 4))
    c0 = max(0, int(cols.min()) - random.randint(2, 4))
    c1 = min(grid.shape[1] - 1, int(cols.max()) + random.randint(2, 4))
    cropped = grid[r0 : r1 + 1, c0 : c1 + 1]

    k = random.randrange(4)
    cropped = np.rot90(cropped, k).copy()
    if random.random() < 0.5:
        cropped = np.fliplr(cropped).copy()
    if random.random() < 0.5:
        cropped = np.flipud(cropped).copy()

    top_pad = random.randint(1, 4)
    bottom_pad = random.randint(1, 4)
    left_pad = random.randint(1, 4)
    right_pad = random.randint(1, 4)
    return np.pad(
        cropped,
        ((top_pad, bottom_pad), (left_pad, right_pad)),
        mode="constant",
        constant_values=bg,
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(("chains", "branch"))
    children = _build_chain_topology(rng) if mode == "chains" else _build_branch_topology()
    sizes = _sample_sizes(children, rng)
    if sizes is None:
        return None

    positions = _place_topology(children, rng)
    if positions is None:
        return None

    bg = rng.choice([color for color in range(10) if color != GREY])
    palette = [color for color in range(10) if color not in {bg, GREY}]
    assert len(children) <= len(palette)
    colors = rng.sample(palette, len(children))
    stride = rng.choice((5, 6, 7))
    grid = _draw_input_from_topology(children, positions, sizes, colors, bg, stride)

    if _get_bg(grid) != bg:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tree-layout ARC input")
