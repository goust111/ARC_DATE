import random
from itertools import product

import numpy as np


BG = 0
GREY = 5
MAROON = 9
NODE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _left_extent(size: int) -> int:
    return size // 2


def _up_extent(size: int) -> int:
    return size // 2


def _right_extent(size: int) -> int:
    return size - 1 - size // 2


def _down_extent(size: int) -> int:
    return size - 1 - size // 2


def _collect_same_color_component(
    grid: np.ndarray, start_r: int, start_c: int, visited: np.ndarray
) -> list[tuple[int, int]]:
    color = int(grid[start_r, start_c])
    stack = [(start_r, start_c)]
    visited[start_r, start_c] = True
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (
                _in_bounds(grid.shape, nr, nc)
                and not visited[nr, nc]
                and int(grid[nr, nc]) == color
            ):
                visited[nr, nc] = True
                stack.append((nr, nc))
    return coords


def _collect_mask_component(mask: np.ndarray, start_r: int, start_c: int, visited: np.ndarray) -> list[tuple[int, int]]:
    stack = [(start_r, start_c)]
    visited[start_r, start_c] = True
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))
    return coords


def _find_nodes(grid: np.ndarray) -> list[dict]:
    node_mask = ~np.isin(grid, (BG, GREY, MAROON))
    visited = np.zeros(grid.shape, dtype=bool)
    nodes: list[dict] = []
    for r, c in zip(*np.where(node_mask)):
        rr, cc = int(r), int(c)
        if visited[rr, cc]:
            continue
        coords = _collect_same_color_component(grid, rr, cc, visited)
        rows = [p[0] for p in coords]
        cols = [p[1] for p in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        assert size_r == size_c, "each node must have a square bbox"
        assert len(coords) == size_r * size_c, "each node must be a solid square"
        assert 2 <= size_r <= 4, "node size must be between 2 and 4"
        color = int(grid[rr, cc])
        nodes.append(
            {
                "id": len(nodes),
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "size": size_r,
            }
        )
    assert len(nodes) >= 4, "expected a closed ring with at least four nodes"
    return nodes


def _find_edges(grid: np.ndarray) -> list[dict]:
    edge_mask = np.isin(grid, (GREY, MAROON))
    visited = np.zeros(grid.shape, dtype=bool)
    edges: list[dict] = []
    for r, c in zip(*np.where(edge_mask)):
        rr, cc = int(r), int(c)
        if visited[rr, cc]:
            continue
        coords = _collect_mask_component(edge_mask, rr, cc, visited)
        rows = [p[0] for p in coords]
        cols = [p[1] for p in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        horizontal = r0 == r1
        vertical = c0 == c1
        assert horizontal or vertical, "each connector must be a straight segment"
        maroons = [(r, c) for r, c in coords if int(grid[r, c]) == MAROON]
        edges.append(
            {
                "id": len(edges),
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "orientation": "h" if horizontal else "v",
                "maroons": maroons,
            }
        )
    return edges


def _build_node_id_grid(shape: tuple[int, int], nodes: list[dict]) -> np.ndarray:
    node_ids = np.full(shape, -1, dtype=int)
    for node in nodes:
        nid = int(node["id"])
        for r, c in node["coords"]:
            node_ids[r, c] = nid
    return node_ids


def _attach_incident_nodes(edges: list[dict], node_ids: np.ndarray, nodes: list[dict]) -> list[list[int]]:
    adjacency = [[] for _ in nodes]
    for edge in edges:
        incident: set[int] = set()
        for r, c in edge["coords"]:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if _in_bounds(node_ids.shape, nr, nc) and int(node_ids[nr, nc]) != -1:
                    incident.add(int(node_ids[nr, nc]))
        assert len(incident) == 2, "each connector must join exactly two nodes"
        a, b = sorted(incident)
        edge["nodes"] = (a, b)
        adjacency[a].append(b)
        adjacency[b].append(a)

    for neighbors in adjacency:
        assert len(neighbors) == 2, "each node must have degree two in the ring"
    return adjacency


def _arrow_direction(edge: dict) -> tuple[int, int]:
    maroons = edge["maroons"]
    assert len(maroons) == 2, "expected exactly two maroon arrow pixels"
    (r1, c1), (r2, c2) = maroons
    if edge["orientation"] == "h":
        assert r1 == r2 and abs(c1 - c2) == 1, "horizontal arrow pixels must be adjacent"
        axis_mid = (edge["bbox"][1] + edge["bbox"][3]) / 2.0
        d1 = abs(c1 - axis_mid)
        d2 = abs(c2 - axis_mid)
        assert d1 != d2, "arrow tail must occupy the unique midpoint cell"
        if d1 < d2:
            return 0, 1 if c2 > c1 else -1
        return 0, 1 if c1 > c2 else -1

    assert c1 == c2 and abs(r1 - r2) == 1, "vertical arrow pixels must be adjacent"
    axis_mid = (edge["bbox"][0] + edge["bbox"][2]) / 2.0
    d1 = abs(r1 - axis_mid)
    d2 = abs(r2 - axis_mid)
    assert d1 != d2, "arrow tail must occupy the unique midpoint cell"
    if d1 < d2:
        return 1 if r2 > r1 else -1, 0
    return 1 if r1 > r2 else -1, 0


def _edge_end_nodes(edge: dict, nodes: list[dict]) -> tuple[int, int]:
    a, b = edge["nodes"]
    a_box = nodes[a]["bbox"]
    b_box = nodes[b]["bbox"]
    r0, c0, r1, c1 = edge["bbox"]
    if edge["orientation"] == "h":
        left = a if a_box[3] < c0 else b
        right = b if left == a else a
        left_box = nodes[left]["bbox"]
        right_box = nodes[right]["bbox"]
        assert left_box[3] < c0 and right_box[1] > c1, "horizontal edge must link left and right nodes"
        return left, right

    top = a if a_box[2] < r0 else b
    bottom = b if top == a else a
    top_box = nodes[top]["bbox"]
    bottom_box = nodes[bottom]["bbox"]
    assert top_box[2] < r0 and bottom_box[0] > r1, "vertical edge must link top and bottom nodes"
    return top, bottom


def _ordered_ring(adjacency: list[list[int]], start: int, nxt: int, n_nodes: int) -> list[int]:
    order = [start]
    prev = start
    cur = nxt
    while cur != start:
        order.append(cur)
        choices = [neighbor for neighbor in adjacency[cur] if neighbor != prev]
        assert len(choices) == 1, "ring traversal should have a unique continuation"
        prev, cur = cur, choices[0]
    assert len(order) == n_nodes, "traversal must visit every node exactly once"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate node colors one step forward along the arrow-marked connector ring.

    Input structure:
    - The background is black.
    - Foreground nodes are monochrome solid squares of side 2 to 4, with one
      distinct color per node.
    - Grey one-cell-wide horizontal or vertical connectors join the nodes into
      one closed cycle, and one connector contains two adjacent maroon pixels.

    Rule:
    - Identify the node cycle defined by the grey connectors.
    - Read the maroon pair as an arrow on its connector: the more central maroon
      pixel is the tail and the adjacent maroon pixel points toward the next node.
    - Starting from the node at the back of that arrow, traverse the full cycle
      in arrow direction and replace each node's color with the next node's color.
    - Keep every node's size and position unchanged, and preserve all grey
      connectors and maroon arrow pixels.

    Key insight:
    - The task separates topology from appearance: grey lines define the ring,
      the maroon pair fixes a traversal direction on that ring, and only the
      ordered node colors cycle while geometry stays fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    nodes = _find_nodes(g)
    edges = _find_edges(g)
    node_ids = _build_node_id_grid(g.shape, nodes)
    adjacency = _attach_incident_nodes(edges, node_ids, nodes)

    arrow_edges = [edge for edge in edges if edge["maroons"]]
    assert len(arrow_edges) == 1, "expected exactly one arrow-marked connector"
    arrow_edge = arrow_edges[0]
    dr, dc = _arrow_direction(arrow_edge)
    end_a, end_b = _edge_end_nodes(arrow_edge, nodes)

    if arrow_edge["orientation"] == "h":
        start, nxt = (end_a, end_b) if dc > 0 else (end_b, end_a)
    else:
        start, nxt = (end_a, end_b) if dr > 0 else (end_b, end_a)

    order = _ordered_ring(adjacency, start, nxt, len(nodes))
    out = g.copy()
    for idx, nid in enumerate(order):
        new_color = int(nodes[order[(idx + 1) % len(order)]]["color"])
        for r, c in nodes[nid]["coords"]:
            out[r, c] = new_color
    return out


# --------------- generator ---------------

LINE_LENGTH_CHOICES = (3, 5)
SIDE_NAMES = ("top", "right", "bottom", "left")
MAX_ATTEMPTS = 400


def _extra_side_counts() -> tuple[int, int, int, int]:
    total_extras = random.randint(0, 3)
    choices = [
        counts
        for counts in product(range(3), repeat=4)
        if sum(counts) == total_extras
    ]
    return random.choice(choices)


def _all_length_plans(gap_count: int) -> list[tuple[int, ...]]:
    if gap_count == 0:
        return [()]
    return list(product(LINE_LENGTH_CHOICES, repeat=gap_count))


def _axis_span(size_a: int, size_b: int, line_len: int, axis: str, forward: bool) -> int:
    if axis == "h" and forward:
        return _right_extent(size_a) + 1 + line_len + _left_extent(size_b)
    if axis == "h":
        return _left_extent(size_a) + 1 + line_len + _right_extent(size_b)
    if forward:
        return _down_extent(size_a) + 1 + line_len + _up_extent(size_b)
    return _up_extent(size_a) + 1 + line_len + _down_extent(size_b)


def _matching_length_plans(chain_a_sizes: list[int], chain_b_sizes: list[int], axis: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    plans_a = _all_length_plans(len(chain_a_sizes) - 1)
    plans_b = _all_length_plans(len(chain_b_sizes) - 1)

    spans_a: dict[int, list[tuple[int, ...]]] = {}
    for plan in plans_a:
        total = 0
        for i, line_len in enumerate(plan):
            total += _axis_span(chain_a_sizes[i], chain_a_sizes[i + 1], line_len, axis, True)
        spans_a.setdefault(total, []).append(plan)

    matches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for plan in plans_b:
        total = 0
        for i, line_len in enumerate(plan):
            total += _axis_span(chain_b_sizes[i], chain_b_sizes[i + 1], line_len, axis, False)
        for other in spans_a.get(total, []):
            matches.append((other, plan))
    if not matches:
        return None
    return random.choice(matches)


def _build_labels(extra_counts: tuple[int, int, int, int]) -> list[str]:
    top_extra, right_extra, bottom_extra, left_extra = extra_counts
    labels = ["tl"]
    labels.extend(f"t{i}" for i in range(top_extra))
    labels.append("tr")
    labels.extend(f"r{i}" for i in range(right_extra))
    labels.append("br")
    labels.extend(f"b{i}" for i in range(bottom_extra))
    labels.append("bl")
    labels.extend(f"l{i}" for i in range(left_extra))
    return labels


def _chain_indices(labels: list[str], extra_counts: tuple[int, int, int, int]) -> dict[str, list[int]]:
    top_extra, right_extra, bottom_extra, left_extra = extra_counts
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    top = [label_to_index["tl"]]
    top.extend(label_to_index[f"t{i}"] for i in range(top_extra))
    top.append(label_to_index["tr"])

    right = [label_to_index["tr"]]
    right.extend(label_to_index[f"r{i}"] for i in range(right_extra))
    right.append(label_to_index["br"])

    bottom = [label_to_index["br"]]
    bottom.extend(label_to_index[f"b{i}"] for i in range(bottom_extra))
    bottom.append(label_to_index["bl"])

    left = [label_to_index["bl"]]
    left.extend(label_to_index[f"l{i}"] for i in range(left_extra))
    left.append(label_to_index["tl"])

    return {"top": top, "right": right, "bottom": bottom, "left": left}


def _compute_centers(
    sizes: list[int],
    chains: dict[str, list[int]],
    top_plan: tuple[int, ...],
    right_plan: tuple[int, ...],
    bottom_plan: tuple[int, ...],
    left_plan: tuple[int, ...],
) -> list[tuple[int, int]]:
    centers: list[list[int | None]] = [[None, None] for _ in sizes]
    tl = chains["top"][0]
    centers[tl] = [0, 0]

    for i, line_len in enumerate(top_plan):
        a = chains["top"][i]
        b = chains["top"][i + 1]
        row = int(centers[a][0])
        col = int(centers[a][1]) + _axis_span(sizes[a], sizes[b], line_len, "h", True)
        centers[b] = [row, col]

    for i, line_len in enumerate(right_plan):
        a = chains["right"][i]
        b = chains["right"][i + 1]
        row = int(centers[a][0]) + _axis_span(sizes[a], sizes[b], line_len, "v", True)
        col = int(centers[a][1])
        centers[b] = [row, col]

    for i, line_len in enumerate(bottom_plan):
        a = chains["bottom"][i]
        b = chains["bottom"][i + 1]
        row = int(centers[a][0])
        col = int(centers[a][1]) - _axis_span(sizes[a], sizes[b], line_len, "h", False)
        centers[b] = [row, col]

    for i, line_len in enumerate(left_plan):
        a = chains["left"][i]
        b = chains["left"][i + 1]
        row = int(centers[a][0]) - _axis_span(sizes[a], sizes[b], line_len, "v", False)
        col = int(centers[a][1])
        if centers[b][0] is None:
            centers[b] = [row, col]
        else:
            assert centers[b] == [row, col], "side plans must close consistently"

    return [(int(r), int(c)) for r, c in centers]


def _node_bbox(center: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    r, c = center
    return (
        r - _up_extent(size),
        c - _left_extent(size),
        r + _down_extent(size),
        c + _right_extent(size),
    )


def _shifted_layout(
    centers: list[tuple[int, int]], sizes: list[int]
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    boxes = [_node_bbox(center, size) for center, size in zip(centers, sizes)]
    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    max_r = max(box[2] for box in boxes)
    max_c = max(box[3] for box in boxes)
    base_h = max_r - min_r + 1
    base_w = max_c - min_c + 1
    assert base_h <= 30 and base_w <= 30, "ring structure must fit ARC bounds before padding"

    extra_rows = random.randint(0, min(4, 30 - base_h))
    extra_cols = random.randint(0, min(4, 30 - base_w))
    top_pad = random.randint(0, extra_rows)
    left_pad = random.randint(0, extra_cols)
    row_shift = top_pad - min_r
    col_shift = left_pad - min_c
    shifted = [(r + row_shift, c + col_shift) for r, c in centers]
    final_h = base_h + extra_rows
    final_w = base_w + extra_cols
    return shifted, (final_h, final_w)


def _edge_cells(
    center_a: tuple[int, int], size_a: int, center_b: tuple[int, int], size_b: int
) -> list[tuple[int, int]]:
    ra, ca = center_a
    rb, cb = center_b
    if ra == rb:
        row = ra
        if ca < cb:
            start = ca + _right_extent(size_a) + 1
            end = cb - _left_extent(size_b) - 1
            return [(row, c) for c in range(start, end + 1)]
        start = cb + _right_extent(size_b) + 1
        end = ca - _left_extent(size_a) - 1
        return [(row, c) for c in range(end, start - 1, -1)]

    assert ca == cb
    col = ca
    if ra < rb:
        start = ra + _down_extent(size_a) + 1
        end = rb - _up_extent(size_b) - 1
        return [(r, col) for r in range(start, end + 1)]
    start = rb + _down_extent(size_b) + 1
    end = ra - _up_extent(size_a) - 1
    return [(r, col) for r in range(end, start - 1, -1)]


def _render_case(
    sizes: list[int],
    colors: list[int],
    centers: list[tuple[int, int]],
    arrow_edge_index: int,
    forward_arrow: bool,
) -> np.ndarray:
    shifted_centers, shape = _shifted_layout(centers, sizes)
    grid = np.full(shape, BG, dtype=int)

    for size, color, center in zip(sizes, colors, shifted_centers):
        r0, c0, r1, c1 = _node_bbox(center, size)
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    n = len(sizes)
    edge_lists: list[list[tuple[int, int]]] = []
    for i in range(n):
        j = (i + 1) % n
        cells = _edge_cells(shifted_centers[i], sizes[i], shifted_centers[j], sizes[j])
        assert len(cells) in LINE_LENGTH_CHOICES, "generator only uses validated odd line lengths"
        edge_lists.append(cells)
        for r, c in cells:
            grid[r, c] = GREY

    base_cells = edge_lists[arrow_edge_index]
    directed = base_cells if forward_arrow else list(reversed(base_cells))
    tail_idx = len(directed) // 2
    grid[directed[tail_idx]] = MAROON
    grid[directed[tail_idx + 1]] = MAROON
    return grid


def _gen_attempt() -> np.ndarray | None:
    extra_counts = _extra_side_counts()
    labels = _build_labels(extra_counts)
    n_nodes = len(labels)
    sizes = [random.randint(2, 4) for _ in range(n_nodes)]
    colors = random.sample(NODE_COLOR_POOL, n_nodes)
    chains = _chain_indices(labels, extra_counts)

    top_sizes = [sizes[idx] for idx in chains["top"]]
    right_sizes = [sizes[idx] for idx in chains["right"]]
    bottom_sizes = [sizes[idx] for idx in chains["bottom"]]
    left_sizes = [sizes[idx] for idx in chains["left"]]

    top_bottom = _matching_length_plans(top_sizes, bottom_sizes, "h")
    right_left = _matching_length_plans(right_sizes, left_sizes, "v")
    if top_bottom is None or right_left is None:
        return None

    top_plan, bottom_plan = top_bottom
    right_plan, left_plan = right_left
    centers = _compute_centers(sizes, chains, top_plan, right_plan, bottom_plan, left_plan)

    arrow_edge_index = random.randrange(n_nodes)
    forward_arrow = bool(random.getrandbits(1))
    grid = _render_case(sizes, colors, centers, arrow_edge_index, forward_arrow)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid ring input after {MAX_ATTEMPTS} attempts")
