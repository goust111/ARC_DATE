from __future__ import annotations

from collections import deque
import random

import numpy as np


ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BLACK = 0
GRAY = 5
NODE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
STEP = 5
MAX_SPAN = 4
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ORTHO4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _is_2x2_node(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return False
    return {(r, c) for r, c in coords} == {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    }


def _parse_tree(grid: np.ndarray) -> tuple[list[dict], list[set[int]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    nodes: list[dict] = []
    line_components: list[tuple[int, list[tuple[int, int]]]] = []
    for color in colors:
        for comp in _components_of_color(g, color):
            if _is_2x2_node(comp):
                nodes.append(
                    {
                        "color": int(color),
                        "coords": comp,
                        "bbox": _bbox(comp),
                    }
                )
            else:
                line_components.append((int(color), comp))

    assert nodes, "expected at least one 2x2 node"
    line_colors = {color for color, _ in line_components}
    assert len(line_colors) == 1, "expected exactly one gray line color"

    nodes.sort(key=lambda node: (node["bbox"][0], node["bbox"][1], node["color"]))
    cell_to_node: dict[tuple[int, int], int] = {}
    for idx, node in enumerate(nodes):
        for cell in node["coords"]:
            cell_to_node[cell] = idx

    adjacency = [set() for _ in nodes]
    seen_edges: set[tuple[int, int]] = set()
    for _line_color, comp in line_components:
        touching: set[int] = set()
        for r, c in comp:
            neighbors = 0
            for dr, dc in ORTHO4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_to_node:
                    touching.add(cell_to_node[(nr, nc)])
                    neighbors += 1
            assert neighbors <= 2, "line pixels should only meet node edges"
        assert len(touching) == 2, "each line segment must connect exactly two nodes"
        a, b = sorted(touching)
        edge = (a, b)
        assert edge not in seen_edges, "tree edges must be unique"
        seen_edges.add(edge)
        adjacency[a].add(b)
        adjacency[b].add(a)

    assert len(seen_edges) == len(nodes) - 1, "a tree with n nodes must have n-1 edges"
    return nodes, adjacency, bg


def _levels_from_root(nodes: list[dict], adjacency: list[set[int]]) -> dict[int, int]:
    leaves = [idx for idx, nbrs in enumerate(adjacency) if len(nbrs) == 1]
    assert leaves, "expected at least one leaf"
    root = min(leaves, key=lambda idx: (nodes[idx]["bbox"][0], nodes[idx]["bbox"][1]))

    levels = {root: 0}
    q = deque([root])
    while q:
        cur = q.popleft()
        for nxt in sorted(adjacency[cur]):
            if nxt in levels:
                continue
            levels[nxt] = levels[cur] + 1
            q.append(nxt)

    assert len(levels) == len(nodes), "tree must be connected"
    return levels


def _square_bbox(node_bbox: tuple[int, int, int, int], level: int) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = node_bbox
    # The square is centered on the junction at the middle of the 2x2 node.
    return r0 - level, c0 - level, r0 + level + 2, c0 + level + 2


def _render(nodes: list[dict], levels: dict[int, int], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    order = sorted(range(len(nodes)), key=lambda idx: (-levels[idx], nodes[idx]["bbox"][0], nodes[idx]["bbox"][1]))
    h, w = shape
    for idx in order:
        top, left, bottom, right = _square_bbox(nodes[idx]["bbox"], levels[idx])
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h - 1, bottom)
        cc1 = min(w - 1, right)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(nodes[idx]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the gray tree to get node depths, then paint depth-sized squares from deepest to shallowest.

    Input structure:
    - The background is black.
    - Several different-colored 2x2 solid rectangles act as nodes.
    - Single-pixel gray horizontal or vertical line segments connect node edges into one acyclic connected tree.
    - The root is the leaf node that sits highest in the grid.

    Rule:
    - Identify every 2x2 node and each gray line segment, then convert the line topology into a node graph.
    - Breadth-first search from the topmost leaf to assign levels, with the root at level 0.
    - For a node at level L, draw a solid square of side 3 + 2*L centered on the 2x2 node's middle junction.
    - Clear the gray lines and paint nodes from largest level to smallest level so shallower squares overwrite deeper ones.

    Key insight:
    The gray connectors are not decoration; they encode a rooted tree whose BFS depth controls both the square size and the final overpaint order.
    """
    g = np.asarray(grid, dtype=int)
    nodes, adjacency, bg = _parse_tree(g)
    levels = _levels_from_root(nodes, adjacency)
    return _render(nodes, levels, g.shape, bg)


def _occupied_neighbors(pos: tuple[int, int], occupied: dict[tuple[int, int], int]) -> set[int]:
    r, c = pos
    found = set()
    for dr, dc in ORTHO4:
        if (r + dr, c + dc) in occupied:
            found.add(occupied[(r + dr, c + dc)])
    return found


def _sample_tree() -> tuple[dict[int, tuple[int, int]], dict[int, int | None], dict[int, int]] | None:
    node_count = random.randint(4, 7)
    max_depth = random.choice((2, 3))
    mode = random.choice(("chainy", "branchy", "mixed"))

    positions: dict[int, tuple[int, int]] = {0: (0, 0), 1: (1, 0)}
    parents: dict[int, int | None] = {0: None, 1: 0}
    depths: dict[int, int] = {0: 0, 1: 1}
    children: dict[int, list[int]] = {0: [1], 1: []}
    occupied = {(0, 0): 0, (1, 0): 1}

    while len(positions) < node_count:
        candidates: list[tuple[int, tuple[int, int]]] = []
        for node_id, (r, c) in positions.items():
            if node_id == 0 or depths[node_id] >= max_depth:
                continue
            for dr, dc in ORTHO4:
                nr, nc = r + dr, c + dc
                if nr < 1 or (nr, nc) in occupied:
                    continue
                if _occupied_neighbors((nr, nc), occupied) != {node_id}:
                    continue
                rows = [pos[0] for pos in positions.values()] + [nr]
                cols = [pos[1] for pos in positions.values()] + [nc]
                if max(rows) - min(rows) > MAX_SPAN or max(cols) - min(cols) > MAX_SPAN:
                    continue
                candidates.append((node_id, (nr, nc)))
        if not candidates:
            return None

        if mode == "chainy":
            leaf_candidates = [cand for cand in candidates if len(children[cand[0]]) == 0]
            if leaf_candidates:
                candidates = leaf_candidates
        elif mode == "branchy":
            split_candidates = [cand for cand in candidates if len(children[cand[0]]) <= 1]
            if split_candidates:
                candidates = split_candidates

        parent_id, new_pos = random.choice(candidates)
        new_id = len(positions)
        positions[new_id] = new_pos
        parents[new_id] = parent_id
        depths[new_id] = depths[parent_id] + 1
        children.setdefault(parent_id, []).append(new_id)
        children[new_id] = []
        occupied[new_pos] = new_id

    if max(depths.values()) < 2:
        return None
    if not any(
        positions[node][0] == positions[parent][0]
        for node, parent in parents.items()
        if parent is not None
    ):
        return None
    return positions, parents, depths


def _edge_line_cells(
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    row_slot: int | None = None,
    col_slot: int | None = None,
) -> list[tuple[int, int]]:
    r1, c1 = pos_a
    r2, c2 = pos_b
    if r1 == r2:
        assert row_slot is not None
        left = min(c1, c2)
        right = max(c1, c2)
        return [(r1 + row_slot, c) for c in range(left + 2, right)]
    assert c1 == c2 and col_slot is not None
    top = min(r1, r2)
    bottom = max(r1, r2)
    return [(r, c1 + col_slot) for r in range(top + 2, bottom)]


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _footprint_checks(bboxes: dict[int, tuple[int, int, int, int]], depths: dict[int, int]) -> bool:
    has_cross_level_overlap = False
    ids = sorted(bboxes)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if not _overlap(bboxes[a], bboxes[b]):
                continue
            if depths[a] == depths[b]:
                return False
            has_cross_level_overlap = True
    return has_cross_level_overlap


def _build_candidate() -> np.ndarray | None:
    sampled = _sample_tree()
    if sampled is None:
        return None
    logical_positions, parents, depths = sampled

    colors = random.sample(list(NODE_COLORS), len(logical_positions))
    physical_positions = {
        node_id: (logical_r * STEP, logical_c * STEP)
        for node_id, (logical_r, logical_c) in logical_positions.items()
    }

    line_cells: list[tuple[int, int]] = []
    for node_id, parent_id in parents.items():
        if parent_id is None:
            continue
        row_slot = random.randint(0, 1) if physical_positions[node_id][0] == physical_positions[parent_id][0] else None
        col_slot = random.randint(0, 1) if physical_positions[node_id][1] == physical_positions[parent_id][1] else None
        line_cells.extend(_edge_line_cells(physical_positions[node_id], physical_positions[parent_id], row_slot, col_slot))

    node_boxes = {node_id: (r, c, r + 1, c + 1) for node_id, (r, c) in physical_positions.items()}
    square_boxes = {node_id: _square_bbox(node_boxes[node_id], depths[node_id]) for node_id in physical_positions}
    if not _footprint_checks(square_boxes, depths):
        return None

    min_r = min(
        min(r for r, _ in line_cells) if line_cells else 0,
        min(box[0] for box in square_boxes.values()),
        min(box[0] for box in node_boxes.values()),
    )
    min_c = min(
        min(c for _, c in line_cells) if line_cells else 0,
        min(box[1] for box in square_boxes.values()),
        min(box[1] for box in node_boxes.values()),
    )
    max_r = max(
        max(r for r, _ in line_cells) if line_cells else 1,
        max(box[2] for box in square_boxes.values()),
        max(box[2] for box in node_boxes.values()),
    )
    max_c = max(
        max(c for _, c in line_cells) if line_cells else 1,
        max(box[3] for box in square_boxes.values()),
        max(box[3] for box in node_boxes.values()),
    )

    raw_h = max_r - min_r + 1
    raw_w = max_c - min_c + 1
    if raw_h > 30 or raw_w > 30:
        return None

    extra_top = random.randint(0, 30 - raw_h)
    extra_bottom = random.randint(0, 30 - raw_h - extra_top)
    extra_left = random.randint(0, 30 - raw_w)
    extra_right = random.randint(0, 30 - raw_w - extra_left)
    height = raw_h + extra_top + extra_bottom
    width = raw_w + extra_left + extra_right

    shift_r = extra_top - min_r
    shift_c = extra_left - min_c
    grid = np.full((height, width), BLACK, dtype=int)

    occupied_cells: set[tuple[int, int]] = set()
    for node_id, (r, c) in physical_positions.items():
        rr = r + shift_r
        cc = c + shift_c
        color = int(colors[node_id])
        grid[rr : rr + 2, cc : cc + 2] = color
        occupied_cells.update(
            {
                (rr, cc),
                (rr + 1, cc),
                (rr, cc + 1),
                (rr + 1, cc + 1),
            }
        )

    for r, c in line_cells:
        rr = r + shift_r
        cc = c + shift_c
        if (rr, cc) in occupied_cells:
            return None
        if int(grid[rr, cc]) not in {BLACK, GRAY}:
            return None
        grid[rr, cc] = GRAY

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None

    nodes, adjacency, _bg = _parse_tree(grid)
    levels = _levels_from_root(nodes, adjacency)
    rendered = _render(nodes, levels, grid.shape, BLACK)
    if not np.array_equal(out, rendered):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rooted-tree input")
