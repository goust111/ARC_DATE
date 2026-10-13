from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_CANVAS = 30
GEN_ATTEMPTS = 400
GEN_SLOT = 7


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _find_components(grid: np.ndarray, colors: list[int]) -> list[dict]:
    out = []
    for color in sorted(set(int(c) for c in colors)):
        comps = _components_mask(grid == color)
        for coords in comps:
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    out.sort(key=lambda comp: (comp["bbox"], comp["color"], comp["area"]))
    return out


def _is_hollow_container(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    color = int(comp["color"])
    return (
        int(comp["area"]) == int(border.sum())
        and np.all(patch[border] == color)
        and np.all(patch[~border] == BLACK)
    )


def _adjacent_pipe_cells(
    shape: tuple[int, int],
    coords: set[tuple[int, int]],
    pipe_mask: np.ndarray,
) -> set[tuple[int, int]]:
    attached = set()
    for r, c in coords:
        for rr, cc in _neighbors4(shape, r, c):
            if bool(pipe_mask[rr, cc]):
                attached.add((rr, cc))
    return attached


def _infer_upstream(source_coord: tuple[int, int], containers: list[dict], grid: np.ndarray) -> dict[int, int]:
    pipe_mask = grid == GREY
    pipe_cells = [tuple(map(int, rc)) for rc in np.argwhere(pipe_mask)]
    assert pipe_cells, "expected a non-empty grey pipe network"

    node_coords = [{source_coord}]
    for comp in containers:
        node_coords.append({tuple(map(int, rc)) for rc in comp["coords"]})

    attachments = []
    for coords in node_coords:
        attached = _adjacent_pipe_cells(grid.shape, coords, pipe_mask)
        assert attached, "every station must touch the grey pipes"
        attachments.append(attached)

    source_degree = len(attachments[0])
    assert source_degree == 1, "source pixel must sit at a pipe end"

    adj: dict[tuple, set[tuple]] = defaultdict(set)
    for r, c in pipe_cells:
        pv = ("pipe", int(r), int(c))
        adj[pv]
        for rr, cc in _neighbors4(grid.shape, r, c):
            if bool(pipe_mask[rr, cc]):
                adj[pv].add(("pipe", int(rr), int(cc)))

    for idx, attached in enumerate(attachments):
        nv = ("node", int(idx))
        for r, c in attached:
            pv = ("pipe", int(r), int(c))
            adj[nv].add(pv)
            adj[pv].add(nv)
        assert adj[nv], "every station must connect into the graph"

    upstream: dict[int, int] = {}
    visited = set()
    stack = [(("node", 0), 0)]
    while stack:
        vertex, last_station = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)

        next_last = last_station
        if vertex[0] == "node":
            node_idx = int(vertex[1])
            if node_idx != 0:
                assert node_idx not in upstream, "station tree must be acyclic"
                upstream[node_idx] = int(last_station)
            next_last = node_idx

        for nb in adj[vertex]:
            if nb not in visited:
                stack.append((nb, next_last))

    assert visited == set(adj), "pipe network must be connected to the source"
    assert len(upstream) == len(containers), "every container must be reachable from the source"
    return upstream


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hollow container with the color of the previous station along the grey pipe tree.

    Input structure:
    - Black background contains several one-cell-thick colored hollow rectangles.
    - Grey one-cell-wide pipes connect those containers into a tree.
    - Exactly one non-grey singleton source pixel touches a pipe endpoint.

    Rule:
    - Keep every container border, every grey pipe, and the source pixel unchanged.
    - Traverse the pipe tree outward from the source to determine each container's unique upstream station.
    - Fill a container interior with the source color if the source is upstream, otherwise with its parent container's border color.

    Key insight:
    - The interior color is shifted by one station, so the task is to recover pipe direction from the rooted tree rather than match each container to its own border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(c) for c in np.unique(g) if int(c) not in {BLACK, GREY}]
    components = _find_components(g, colors)

    source_coord: tuple[int, int] | None = None
    source_color: int | None = None
    containers: list[dict] = []

    for comp in components:
        if _is_hollow_container(g, comp):
            containers.append(comp)
            continue

        if int(comp["area"]) != 1:
            raise AssertionError("non-container colored component must be the singleton source")
        coord = tuple(map(int, comp["coords"][0]))
        grey_neighbors = sum(1 for rr, cc in _neighbors4(g.shape, *coord) if int(g[rr, cc]) == GREY)
        assert grey_neighbors == 1, "source pixel must touch exactly one pipe cell"
        assert source_coord is None, "expected a unique source pixel"
        source_coord = coord
        source_color = int(comp["color"])

    assert source_coord is not None and source_color is not None
    assert containers, "expected at least one hollow container"

    containers.sort(key=lambda comp: comp["bbox"])
    upstream = _infer_upstream(source_coord, containers, g)

    out = g.copy()
    for node_idx, comp in enumerate(containers, start=1):
        parent_idx = int(upstream[node_idx])
        fill_color = source_color if parent_idx == 0 else int(containers[parent_idx - 1]["color"])
        r0, c0, r1, c1 = comp["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


def _draw_container(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_pipe_cell(grid: np.ndarray, r: int, c: int) -> None:
    assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
    assert int(grid[r, c]) in {BLACK, GREY}
    grid[r, c] = GREY


def _draw_vertical(grid: np.ndarray, c: int, r0: int, r1: int) -> None:
    lo = min(r0, r1)
    hi = max(r0, r1)
    for r in range(lo, hi + 1):
        _paint_pipe_cell(grid, r, c)


def _draw_horizontal(grid: np.ndarray, r: int, c0: int, c1: int) -> None:
    lo = min(c0, c1)
    hi = max(c0, c1)
    for c in range(lo, hi + 1):
        _paint_pipe_cell(grid, r, c)


def _sample_tree() -> tuple[list[int], dict[int, list[int]], list[int]]:
    n = random.choices([3, 4, 5, 6], weights=[1, 3, 3, 2])[0]
    parents = [-1, 0, 1]
    depths = [0, 1, 2]
    children = {0: [1], 1: [2], 2: []}

    while len(parents) < n:
        candidates = [i for i in range(len(parents)) if depths[i] < 2 and len(children[i]) < 2]
        weights = []
        for idx in candidates:
            weight = 1
            if len(children[idx]) == 1:
                weight += 3
            if depths[idx] == 1:
                weight += 2
            if depths[idx] == 0:
                weight += 1
            weights.append(weight)
        parent = random.choices(candidates, weights=weights)[0]
        node = len(parents)
        parents.append(parent)
        depths.append(depths[parent] + 1)
        children[node] = []
        children[parent].append(node)

    for idx, kids in children.items():
        if len(kids) == 2 and random.random() < 0.5:
            children[idx] = [kids[1], kids[0]]

    return parents, children, depths


def _assign_centers(children: dict[int, list[int]]) -> dict[int, int]:
    centers: dict[int, int] = {}
    next_leaf = [0]

    def dfs(node: int) -> None:
        kids = children[node]
        if not kids:
            centers[node] = next_leaf[0]
            next_leaf[0] += GEN_SLOT
            return
        if len(kids) == 1:
            dfs(kids[0])
            centers[node] = centers[kids[0]]
            return
        dfs(kids[0])
        dfs(kids[1])
        centers[node] = (centers[kids[0]] + centers[kids[1]]) // 2

    dfs(0)
    return centers


def _fill_from_metadata(grid: np.ndarray, source_color: int, nodes: list[dict]) -> np.ndarray:
    out = grid.copy()
    for idx, node in enumerate(nodes):
        parent = int(node["parent"])
        fill = source_color if parent == -1 else int(nodes[parent]["color"])
        r0, c0, r1, c1 = node["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def _fill_all_source(grid: np.ndarray, source_color: int, nodes: list[dict]) -> np.ndarray:
    out = grid.copy()
    for node in nodes:
        r0, c0, r1, c1 = node["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = source_color
    return out


def _fill_self_color(grid: np.ndarray, nodes: list[dict]) -> np.ndarray:
    out = grid.copy()
    for node in nodes:
        r0, c0, r1, c1 = node["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(node["color"])
    return out


def _crop_to_content(grid: np.ndarray, source_coord: tuple[int, int], nodes: list[dict]):
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top = random.randint(0, min(2, int(r0)))
    left = random.randint(0, min(2, int(c0)))
    bottom = random.randint(0, min(2, int(grid.shape[0] - 1 - r1)))
    right = random.randint(0, min(2, int(grid.shape[1] - 1 - c1)))

    rr0 = int(r0) - top
    cc0 = int(c0) - left
    rr1 = int(r1) + bottom
    cc1 = int(c1) + right
    cropped = grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()

    sr, sc = source_coord
    new_source = (sr - rr0, sc - cc0)
    new_nodes = []
    for node in nodes:
        nr0, nc0, nr1, nc1 = node["bbox"]
        new_nodes.append(
            {
                **node,
                "bbox": (nr0 - rr0, nc0 - cc0, nr1 - rr0, nc1 - cc0),
            }
        )

    return cropped, new_source, new_nodes


def _connect_source(grid: np.ndarray, source_coord: tuple[int, int], root_center: int, root_top: int) -> None:
    sr, sc = source_coord
    start_r = sr + 1
    end_r = root_top - 1
    if sc == root_center:
        _draw_vertical(grid, sc, start_r, end_r)
        return
    bend_r = random.randint(start_r, end_r - 1) if start_r < end_r else end_r
    _draw_vertical(grid, sc, start_r, bend_r)
    _draw_horizontal(grid, bend_r, sc, root_center)
    _draw_vertical(grid, root_center, bend_r, end_r)


def _connect_children(grid: np.ndarray, parent_node: dict, child_nodes: list[dict]) -> None:
    start_r = int(parent_node["bbox"][2]) + 1
    end_r = int(child_nodes[0]["bbox"][0]) - 1
    parent_c = int(parent_node["center"])
    child_centers = [int(node["center"]) for node in child_nodes]

    if len(child_centers) == 1:
        child_c = child_centers[0]
        if child_c == parent_c:
            _draw_vertical(grid, child_c, start_r, end_r)
            return
        bend_r = random.randint(start_r, end_r - 1) if start_r < end_r else end_r
        _draw_vertical(grid, parent_c, start_r, bend_r)
        _draw_horizontal(grid, bend_r, parent_c, child_c)
        _draw_vertical(grid, child_c, bend_r, end_r)
        return

    branch_r = random.randint(start_r, end_r - 1) if start_r < end_r else end_r
    _draw_vertical(grid, parent_c, start_r, branch_r)
    _draw_horizontal(grid, branch_r, min(child_centers), max(child_centers))
    for child_c in child_centers:
        _draw_vertical(grid, child_c, branch_r, end_r)


def _generate_case():
    parents, children, depths = _sample_tree()
    node_count = len(parents)
    centers_rel = _assign_centers(children)

    widths = [random.choice([3, 5]) for _ in range(node_count)]
    heights = [random.choice([3, 4, 5]) for _ in range(node_count)]

    max_height_by_depth: dict[int, int] = {}
    for idx, depth in enumerate(depths):
        max_height_by_depth[depth] = max(max_height_by_depth.get(depth, 0), heights[idx])

    top_by_depth = {0: random.randint(2, 3)}
    max_depth = max(depths)
    for depth in range(1, max_depth + 1):
        top_by_depth[depth] = top_by_depth[depth - 1] + max_height_by_depth[depth - 1] + random.randint(3, 5)

    min_left_rel = min(centers_rel[idx] - widths[idx] // 2 for idx in range(node_count))
    max_right_rel = max(centers_rel[idx] + widths[idx] // 2 for idx in range(node_count))
    min_shift = 1 - min_left_rel
    max_shift = GEN_CANVAS - 2 - max_right_rel
    if min_shift > max_shift:
        return None
    shift = random.randint(min_shift, max_shift)

    colors = list(PALETTE)
    random.shuffle(colors)
    source_color = colors[0]
    container_colors = colors[1 : 1 + node_count]

    nodes = []
    for idx in range(node_count):
        center = centers_rel[idx] + shift
        top = top_by_depth[depths[idx]]
        left = center - widths[idx] // 2
        bottom = top + heights[idx] - 1
        right = left + widths[idx] - 1
        if not (0 <= top <= bottom < GEN_CANVAS and 0 <= left <= right < GEN_CANVAS):
            return None
        nodes.append(
            {
                "bbox": (top, left, bottom, right),
                "color": int(container_colors[idx]),
                "center": int(center),
                "parent": int(parents[idx]),
                "depth": int(depths[idx]),
            }
        )

    root_center = int(nodes[0]["center"])
    root_top = int(nodes[0]["bbox"][0])
    source_row = random.randint(0, root_top - 2)
    offset_choices = [0] if root_top == 2 else [-2, -1, 0, 1, 2]
    source_col = max(1, min(GEN_CANVAS - 2, root_center + random.choice(offset_choices)))
    source_coord = (source_row, source_col)

    grid = np.zeros((GEN_CANVAS, GEN_CANVAS), dtype=int)
    for node in nodes:
        _draw_container(grid, node["bbox"], int(node["color"]))
    assert int(grid[source_coord]) == BLACK
    grid[source_coord] = source_color

    _connect_source(grid, source_coord, root_center, root_top)
    for idx, node in enumerate(nodes):
        kids = children[idx]
        if kids:
            _connect_children(grid, node, [nodes[k] for k in kids])

    cropped, cropped_source, cropped_nodes = _crop_to_content(grid, source_coord, nodes)
    expected = _fill_from_metadata(cropped, source_color, cropped_nodes)

    depth_two_nodes = [node for node in cropped_nodes if node["depth"] == 2]
    if not depth_two_nodes:
        return None
    if np.array_equal(cropped, expected):
        return None
    if np.array_equal(expected, _fill_all_source(cropped, source_color, cropped_nodes)):
        return None
    if np.array_equal(expected, _fill_self_color(cropped, cropped_nodes)):
        return None

    actual = main(cropped)
    if not np.array_equal(actual, expected):
        return None

    for node in cropped_nodes:
        r0, c0, r1, c1 = node["bbox"]
        patch = cropped[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[1:-1, 1:-1] == BLACK):
            return None

    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-relay input")
