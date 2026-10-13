from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]

MIN_OBJECT_AREA = 5
MIN_CONNECTOR_RUN = 3
GEN_RETRIES = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=int)
    dirs = DIR4 if connectivity == 4 else DIR8
    next_id = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if labels[rr, cc] != 0:
            continue
        next_id += 1
        labels[rr, cc] = next_id
        q = deque([(rr, cc)])
        while q:
            cr, cc = q.popleft()
            for dr, dc in dirs:
                nr, nc = cr + dr, cc + dc
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = next_id
                    q.append((nr, nc))
    return labels, next_id


def component_coords(labels: np.ndarray, n_labels: int) -> list[np.ndarray]:
    return [np.argwhere(labels == k) for k in range(1, n_labels + 1)]


def touches_mask8(mask: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in DIR8:
        nr, nc = r + dr, c + dc
        if in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
            return True
    return False


def edge_component_ids(labels: np.ndarray, n_labels: int) -> set[int]:
    edge_ids: set[int] = set()
    h, w = labels.shape
    for comp_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == comp_id)
        if np.any(
            (coords[:, 0] == 0)
            | (coords[:, 1] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == w - 1)
        ):
            edge_ids.add(comp_id)
    return edge_ids


def build_blue_graph(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[list[int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    index = {coord: i for i, coord in enumerate(coords)}
    adj = [[] for _ in coords]
    for i, (r, c) in enumerate(coords):
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            j = index.get((nr, nc))
            if j is None:
                continue
            adj[i].append(j)
            adj[j].append(i)
    return coords, adj


def bridge_edges(adj: list[list[int]]) -> set[tuple[int, int]]:
    n = len(adj)
    disc = [-1] * n
    low = [0] * n
    time = 0
    bridges: set[tuple[int, int]] = set()

    def dfs(u: int, parent: int) -> None:
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        for v in adj[u]:
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((u, v) if u < v else (v, u))
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for node in range(n):
        if disc[node] == -1:
            dfs(node, -1)
    return bridges


def non_bridge_components(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[np.ndarray], dict[tuple[int, int], int]]:
    coords, adj = build_blue_graph(mask)
    bridges = bridge_edges(adj)
    comp_id = [-1] * len(coords)
    comps: list[list[tuple[int, int]]] = []
    next_id = 0
    for start in range(len(coords)):
        if comp_id[start] != -1:
            continue
        comp_id[start] = next_id
        q = deque([start])
        comp: list[tuple[int, int]] = []
        while q:
            u = q.popleft()
            comp.append(coords[u])
            for v in adj[u]:
                edge = (u, v) if u < v else (v, u)
                if edge in bridges or comp_id[v] != -1:
                    continue
                comp_id[v] = next_id
                q.append(v)
        comps.append(comp)
        next_id += 1
    coord_to_comp = {coord: comp_id[i] for i, coord in enumerate(coords)}
    return coords, [np.array(comp, dtype=int) for comp in comps], coord_to_comp


def cycle_membership(graph: list[set[int]]) -> set[int]:
    degree = [len(neigh) for neigh in graph]
    in_core = [True] * len(graph)
    q = deque(i for i, deg in enumerate(degree) if deg < 2)
    while q:
        node = q.popleft()
        if not in_core[node]:
            continue
        in_core[node] = False
        for other in graph[node]:
            if in_core[other]:
                degree[other] -= 1
                if degree[other] == 1:
                    q.append(other)
    return {i for i, keep in enumerate(in_core) if keep}


def connector_interior_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    blue = g == BLUE
    h, w = g.shape
    mask = np.zeros_like(blue)

    for r in range(h):
        c = 0
        while c < w:
            is_h = (
                blue[r, c]
                and c - 1 >= 0
                and c + 1 < w
                and blue[r, c - 1]
                and blue[r, c + 1]
                and (r == 0 or not blue[r - 1, c])
                and (r == h - 1 or not blue[r + 1, c])
            )
            if not is_h:
                c += 1
                continue
            start = c
            while c < w:
                ok = (
                    blue[r, c]
                    and c - 1 >= 0
                    and c + 1 < w
                    and blue[r, c - 1]
                    and blue[r, c + 1]
                    and (r == 0 or not blue[r - 1, c])
                    and (r == h - 1 or not blue[r + 1, c])
                )
                if not ok:
                    break
                c += 1
            if c - start >= MIN_CONNECTOR_RUN:
                mask[r, start:c] = True

    for c in range(w):
        r = 0
        while r < h:
            is_v = (
                blue[r, c]
                and r - 1 >= 0
                and r + 1 < h
                and blue[r - 1, c]
                and blue[r + 1, c]
                and (c == 0 or not blue[r, c - 1])
                and (c == w - 1 or not blue[r, c + 1])
            )
            if not is_v:
                r += 1
                continue
            start = r
            while r < h:
                ok = (
                    blue[r, c]
                    and r - 1 >= 0
                    and r + 1 < h
                    and blue[r - 1, c]
                    and blue[r + 1, c]
                    and (c == 0 or not blue[r, c - 1])
                    and (c == w - 1 or not blue[r, c + 1])
                )
                if not ok:
                    break
                r += 1
            if r - start >= MIN_CONNECTOR_RUN:
                mask[start:r, c] = True

    return mask


def analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE}

    blue = g == BLUE
    blue_labels, n_blue = label_mask(blue, connectivity=4)
    assert n_blue == 1, "foreground must be a single connected blue network"

    line_mask = connector_interior_mask(g)
    object_labels, n_objects = label_mask(blue & ~line_mask, connectivity=4)
    object_blocks = component_coords(object_labels, n_objects)
    assert 4 <= len(object_blocks) <= 6, "expected 4-6 object nodes"
    assert all(len(block) >= MIN_OBJECT_AREA for block in object_blocks), "object too small"

    object_masks: list[np.ndarray] = []
    object_id = np.full(g.shape, -1, dtype=int)
    for idx, block in enumerate(object_blocks):
        mask = np.zeros(g.shape, dtype=bool)
        mask[block[:, 0], block[:, 1]] = True
        object_masks.append(mask)
        object_id[block[:, 0], block[:, 1]] = idx

    line_labels, n_lines = label_mask(line_mask, connectivity=4)
    line_components = component_coords(line_labels, n_lines)

    graph = [set() for _ in object_masks]
    connector_touch = np.zeros(g.shape, dtype=bool)
    line_touch_objects: list[tuple[int, int]] = []
    for coords in line_components:
        touched: set[int] = set()
        touch_pixels: set[tuple[int, int]] = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            connector_touch[rr, cc] = True
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if not in_bounds(g.shape, nr, nc):
                    continue
                oid = int(object_id[nr, nc])
                if oid >= 0:
                    touched.add(oid)
                    touch_pixels.add((nr, nc))
        assert len(touched) == 2, "every connector must join exactly two objects"
        for rr, cc in touch_pixels:
            connector_touch[rr, cc] = True
        a, b = sorted(touched)
        graph[a].add(b)
        graph[b].add(a)
        line_touch_objects.append((a, b))

    cycle_nodes = cycle_membership(graph)
    assert cycle_nodes, "expected at least one cycle"
    assert any(len(neigh) == 1 for neigh in graph), "expected at least one leaf object"

    bg_labels, n_bg = label_mask(g == BLACK, connectivity=4)
    edge_bg = edge_component_ids(bg_labels, n_bg)
    holes_by_object = {i: [] for i in range(len(object_masks))}
    for bg_id in range(1, n_bg + 1):
        if bg_id in edge_bg:
            continue
        coords = np.argwhere(bg_labels == bg_id)
        owners: set[int] = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if in_bounds(g.shape, nr, nc):
                    oid = int(object_id[nr, nc])
                    if oid >= 0:
                        owners.add(oid)
        if len(owners) == 1:
            holes_by_object[next(iter(owners))].append(coords)

    return {
        "grid": g,
        "object_masks": object_masks,
        "object_graph": graph,
        "cycle_nodes": cycle_nodes,
        "holes_by_object": holes_by_object,
        "bg_labels": bg_labels,
        "edge_bg": edge_bg,
        "connector_touch": connector_touch,
        "line_components": line_components,
        "line_touch_objects": line_touch_objects,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each blue object's boundary according to whether its connector-graph node belongs to a cycle.

    Input structure:
    - The grid has black background and one connected blue foreground network.
    - Compact blue objects act as graph nodes, and one-cell-wide straight blue segments connect some node pairs.
    - The object graph is connected, contains at least one cycle, and also has at least one leaf node outside every cycle.

    Rule:
    - Recover the node graph from the blue network and mark every object that belongs to the graph's 2-core as a cycle member.
    - For a cycle-member object, color exterior black cells that are 8-neighbor-adjacent to the object red, unless those black cells also touch any connector segment; if the object encloses a hole, color the hole-side 8-neighbor boundary green.
    - For a non-cycle object, color only its exterior non-connector-adjacent 8-neighbor boundary yellow, and leave any enclosed hole black.
    - Keep all original blue object pixels and blue connector pixels unchanged.

    Key insight:
    - The same geometric object can receive or skip green inner outlining depending on its role in the recovered connection graph, so topology and hole geometry must be combined.
    """
    info = analyze_scene(np.asarray(grid, dtype=int))
    g = info["grid"]
    out = g.copy()
    exterior_bg = np.isin(info["bg_labels"], list(info["edge_bg"]))
    exterior_coords = np.argwhere(exterior_bg)

    for obj_id, obj_mask in enumerate(info["object_masks"]):
        outer_color = RED if obj_id in info["cycle_nodes"] else YELLOW
        for r, c in exterior_coords:
            rr = int(r)
            cc = int(c)
            if touches_mask8(obj_mask, rr, cc) and not touches_mask8(info["connector_touch"], rr, cc):
                out[rr, cc] = outer_color

        if obj_id not in info["cycle_nodes"]:
            continue
        for hole in info["holes_by_object"][obj_id]:
            for r, c in hole:
                rr = int(r)
                cc = int(c)
                if touches_mask8(obj_mask, rr, cc):
                    out[rr, cc] = GREEN
    return out


def side_anchor(shape: np.ndarray, side: str) -> tuple[int, int]:
    h, w = shape.shape
    if side == "left":
        rows = [r for r in range(h) if shape[r, 0] == BLUE]
        return rows[len(rows) // 2], 0
    if side == "right":
        rows = [r for r in range(h) if shape[r, w - 1] == BLUE]
        return rows[len(rows) // 2], w - 1
    if side == "top":
        cols = [c for c in range(w) if shape[0, c] == BLUE]
        return 0, cols[len(cols) // 2]
    if side == "bottom":
        cols = [c for c in range(w) if shape[h - 1, c] == BLUE]
        return h - 1, cols[len(cols) // 2]
    raise ValueError(side)


def make_shape(name: str, data: list[list[int]], has_hole: bool) -> dict:
    arr = np.array(data, dtype=int)
    return {
        "name": name,
        "mask": arr,
        "has_hole": has_hole,
        "anchors": {side: side_anchor(arr, side) for side in ("left", "right", "top", "bottom")},
    }


SHAPES = [
    make_shape("rect_2x3", [[1, 1, 1], [1, 1, 1]], False),
    make_shape("rect_3x2", [[1, 1], [1, 1], [1, 1]], False),
    make_shape("rect_2x4", [[1, 1, 1, 1], [1, 1, 1, 1]], False),
    make_shape("rect_4x2", [[1, 1], [1, 1], [1, 1], [1, 1]], False),
    make_shape("rect_3x3", [[1, 1, 1], [1, 1, 1], [1, 1, 1]], False),
    make_shape("rect_3x4", [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], False),
    make_shape("ring_3x3", [[1, 1, 1], [1, 0, 1], [1, 1, 1]], True),
    make_shape("ring_3x4", [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], True),
    make_shape("ring_4x3", [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], True),
    make_shape("ring_4x4", [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], True),
]

SHAPE_BY_NAME = {shape["name"]: shape for shape in SHAPES}
HOLE_NAMES = [shape["name"] for shape in SHAPES if shape["has_hole"]]
SOLID_NAMES = [shape["name"] for shape in SHAPES if not shape["has_hole"]]
OPPOSITE = {"left": "right", "right": "left", "up": "bottom", "down": "top"}
NODE_OUTWARD = {
    "tl": ["left", "up"],
    "tr": ["up", "right"],
    "br": ["right", "down"],
    "bl": ["down", "left"],
}


def anchor_coord(shape: dict, pos: tuple[int, int], side: str) -> tuple[int, int]:
    local = shape["anchors"][side]
    return pos[0] + local[0], pos[1] + local[1]


def branch_side(direction: str) -> str:
    if direction == "left":
        return "left"
    if direction == "right":
        return "right"
    if direction == "up":
        return "top"
    if direction == "down":
        return "bottom"
    raise ValueError(direction)


def line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1] + step, b[1], step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0] + step, b[0], step)]
    raise ValueError("connectors must be straight")


def sample_shapes(n_objects: int) -> dict[str, dict]:
    cycle_names = [random.choice(HOLE_NAMES)]
    remaining = [name for name in SHAPE_BY_NAME if name not in cycle_names]
    cycle_names.extend(random.sample(remaining, 3))
    random.shuffle(cycle_names)

    extra_names: list[str] = []
    if n_objects > 4:
        remaining = [name for name in SHAPE_BY_NAME if name not in cycle_names]
        need_branch_hole = random.random() < 0.55 and any(name in HOLE_NAMES for name in remaining)
        if need_branch_hole:
            branch_hole = random.choice([name for name in remaining if name in HOLE_NAMES])
            extra_names.append(branch_hole)
            remaining.remove(branch_hole)
        extra_names.extend(random.sample(remaining, n_objects - 4 - len(extra_names)))
        random.shuffle(extra_names)

    names = cycle_names + extra_names
    return {
        "tl": SHAPE_BY_NAME[names[0]],
        "tr": SHAPE_BY_NAME[names[1]],
        "br": SHAPE_BY_NAME[names[2]],
        "bl": SHAPE_BY_NAME[names[3]],
        **({f"b{i}": SHAPE_BY_NAME[names[4 + i]] for i in range(n_objects - 4)}),
    }


def cycle_positions(shapes: dict[str, dict]) -> dict[str, tuple[int, int]]:
    tl, tr, br, bl = shapes["tl"], shapes["tr"], shapes["br"], shapes["bl"]
    min_col_gap = max(
        tr["anchors"]["top"][1] - tl["anchors"]["top"][1] + tl["mask"].shape[1] + 2,
        br["anchors"]["top"][1] - bl["anchors"]["top"][1] + bl["mask"].shape[1] + 2,
    )
    min_row_gap = max(
        bl["anchors"]["left"][0] - tl["anchors"]["left"][0] + tl["mask"].shape[0] + 2,
        br["anchors"]["left"][0] - tr["anchors"]["left"][0] + tr["mask"].shape[0] + 2,
    )

    c_left = 0
    r_top = 0
    c_right = c_left + min_col_gap + random.randint(1, 3)
    r_bottom = r_top + min_row_gap + random.randint(1, 3)

    return {
        "tl": (r_top - tl["anchors"]["right"][0], c_left - tl["anchors"]["bottom"][1]),
        "tr": (r_top - tr["anchors"]["left"][0], c_right - tr["anchors"]["bottom"][1]),
        "br": (r_bottom - br["anchors"]["left"][0], c_right - br["anchors"]["top"][1]),
        "bl": (r_bottom - bl["anchors"]["right"][0], c_left - bl["anchors"]["top"][1]),
    }


def attach_branch(
    positions: dict[str, tuple[int, int]],
    shapes: dict[str, dict],
    parent: str,
    child: str,
    direction: str,
    span: int,
) -> tuple[str, str, str, str]:
    side = branch_side(direction)
    parent_anchor = anchor_coord(shapes[parent], positions[parent], side)
    child_side = OPPOSITE[direction]
    child_anchor_local = shapes[child]["anchors"][child_side]

    if direction == "left":
        child_anchor = (parent_anchor[0], parent_anchor[1] - span)
    elif direction == "right":
        child_anchor = (parent_anchor[0], parent_anchor[1] + span)
    elif direction == "up":
        child_anchor = (parent_anchor[0] - span, parent_anchor[1])
    else:
        child_anchor = (parent_anchor[0] + span, parent_anchor[1])

    positions[child] = (child_anchor[0] - child_anchor_local[0], child_anchor[1] - child_anchor_local[1])
    return parent, side, child, child_side


def render_scene(
    shapes: dict[str, dict],
    positions: dict[str, tuple[int, int]],
    branch_edges: list[tuple[str, str, str, str]],
) -> np.ndarray | None:
    positions = dict(positions)
    edges = [
        ("tl", "right", "tr", "left"),
        ("tr", "bottom", "br", "top"),
        ("bl", "right", "br", "left"),
        ("tl", "bottom", "bl", "top"),
    ]
    edges.extend(branch_edges)

    all_nodes = list(positions)
    min_r = min(positions[node][0] for node in all_nodes)
    min_c = min(positions[node][1] for node in all_nodes)
    max_r = max(positions[node][0] + shapes[node]["mask"].shape[0] - 1 for node in all_nodes)
    max_c = max(positions[node][1] + shapes[node]["mask"].shape[1] - 1 for node in all_nodes)

    tight_h = max_r - min_r + 1
    tight_w = max_c - min_c + 1
    pad_top = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_bottom = random.randint(1, 2)
    pad_right = random.randint(1, 2)
    h = tight_h + pad_top + pad_bottom
    w = tight_w + pad_left + pad_right
    if h > 30 or w > 30:
        return None

    row_shift = pad_top - min_r
    col_shift = pad_left - min_c
    for node in positions:
        positions[node] = (positions[node][0] + row_shift, positions[node][1] + col_shift)

    grid = np.zeros((h, w), dtype=int)
    for node, pos in positions.items():
        shape = shapes[node]["mask"]
        r0, c0 = pos
        r1, c1 = r0 + shape.shape[0], c0 + shape.shape[1]
        if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
            return None
        if np.any((shape == BLUE) & (grid[r0:r1, c0:c1] != BLACK)):
            return None
        grid[r0:r1, c0:c1][shape == BLUE] = BLUE

    for src, src_side, dst, dst_side in edges:
        a = anchor_coord(shapes[src], positions[src], src_side)
        b = anchor_coord(shapes[dst], positions[dst], dst_side)
        cells = line_cells(a, b)
        if len(cells) < 2:
            return None
        for r, c in cells:
            if not in_bounds(grid.shape, r, c) or grid[r, c] != BLACK:
                return None
        for r, c in cells:
            grid[r, c] = BLUE
    return grid


def sample_branch_plan(
    n_objects: int,
) -> tuple[dict[str, tuple[int, int]], list[tuple[str, str, str, str]], dict[str, dict]]:
    shapes = sample_shapes(n_objects)
    positions = cycle_positions(shapes)
    slots = [(node, random.choice(NODE_OUTWARD[node])) for node in ("tl", "tr", "br", "bl")]
    random.shuffle(slots)
    edges: list[tuple[str, str, str, str]] = []

    if n_objects == 5:
        parent, direction = slots[0]
        edges.append(attach_branch(positions, shapes, parent, "b0", direction, random.randint(4, 6)))
    else:
        if random.random() < 0.5:
            parent, direction = slots[0]
            edges.append(attach_branch(positions, shapes, parent, "b0", direction, random.randint(4, 5)))
            edges.append(attach_branch(positions, shapes, "b0", "b1", direction, random.randint(4, 5)))
        else:
            (p0, d0), (p1, d1) = slots[:2]
            edges.append(attach_branch(positions, shapes, p0, "b0", d0, random.randint(4, 6)))
            edges.append(attach_branch(positions, shapes, p1, "b1", d1, random.randint(4, 6)))
    return positions, edges, shapes


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_objects = random.choice([5, 6])
        positions, branch_edges, shapes = sample_branch_plan(n_objects)
        grid = render_scene(shapes, positions, branch_edges)
        if grid is None:
            continue
        try:
            info = analyze_scene(grid)
            out = main(grid)
        except Exception:
            continue
        if len(info["object_masks"]) != n_objects:
            continue
        if len(info["cycle_nodes"]) < 4:
            continue
        if not any(info["holes_by_object"][obj] for obj in info["cycle_nodes"]):
            continue
        non_cycle = set(range(len(info["object_masks"]))) - set(info["cycle_nodes"])
        if not non_cycle:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("generate_input: exhausted attempts")
