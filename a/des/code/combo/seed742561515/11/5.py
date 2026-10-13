import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG4 = ((-1, -1), (-1, 1), (1, -1), (1, 1))
MISSING_TO_DIR = {
    (0, 0): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (1, 1): (1, 1),
}

LATTICE_H_RANGE = (4, 7)
LATTICE_W_RANGE = (4, 7)
SOURCE_COUNT_CHOICES = (1, 2, 2, 3)
BACKBONE_LEN_RANGE = (3, 5)
EXTRA_PATH_LEN_RANGE = (1, 3)
EXTRA_NODE_BUDGET = (0, 5)
MAX_DEPTH = 5
MAX_OUT_DEGREE = 3
WANT_BRANCH_PROB = 0.55
GEN_ATTEMPTS = 400


def _mode(arr: np.ndarray) -> int:
    vals, cnts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in coords]
            cs = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "size": len(coords),
                }
            )
    return comps


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _is_node_component(comp: dict) -> bool:
    return comp["size"] == 4 and _bbox_hw(comp["bbox"]) == (2, 2)


def _is_arrow_component(comp: dict) -> bool:
    return comp["size"] == 3 and _bbox_hw(comp["bbox"]) == (2, 2)


def _arrow_direction(comp: dict) -> tuple[int, int]:
    r0, c0, _, _ = comp["bbox"]
    local = {(r - r0, c - c0) for r, c in comp["coords"]}
    missing = [(rr, cc) for rr in (0, 1) for cc in (0, 1) if (rr, cc) not in local]
    assert len(missing) == 1, "arrow must be a 2x2 block with one missing corner"
    return MISSING_TO_DIR[missing[0]]


def _diag_gap(box_from: tuple[int, int, int, int], box_to: tuple[int, int, int, int], dr: int, dc: int) -> int | None:
    fr0, fc0, fr1, fc1 = box_from
    tr0, tc0, tr1, tc1 = box_to
    row_gap = tr0 - fr1 - 1 if dr == 1 else fr0 - tr1 - 1
    col_gap = tc0 - fc1 - 1 if dc == 1 else fc0 - tc1 - 1
    if row_gap < 0 or col_gap < 0 or row_gap != col_gap:
        return None
    return int(row_gap)


def _nearest_node(nodes: dict[int, dict], ref_box: tuple[int, int, int, int], dr: int, dc: int) -> int | None:
    best_id = None
    best_gap = None
    tie = False
    for node_id, node in nodes.items():
        gap = _diag_gap(ref_box, node["bbox"], dr, dc)
        if gap is None:
            continue
        if best_gap is None or gap < best_gap:
            best_id = node_id
            best_gap = gap
            tie = False
        elif gap == best_gap:
            tie = True
    assert not tie, "arrow direction must resolve to a unique nearest node"
    return best_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source block's color along arrow chains of blank 2x2 containers.

    Sources are solid 2x2 blocks with indegree 0. Containers are 2x2 blocks reached
    through the directed arrow graph reconstructed from 3-cell L markers. The first
    container on a source's chain becomes solid, the second becomes a checkerboard
    alternating source color with background, and later containers are cleared.
    All arrows are erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode(g)

    comps = _components_by_color(g, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    arrow_colors = [
        color
        for color, color_comps in by_color.items()
        if color_comps and all(_is_arrow_component(comp) for comp in color_comps)
    ]
    assert len(arrow_colors) == 1, "expected exactly one arrow color"
    arrow_color = int(arrow_colors[0])

    arrows = [comp for comp in by_color[arrow_color]]
    node_comps = [comp for comp in comps if int(comp["color"]) != arrow_color]
    assert node_comps and all(_is_node_component(comp) for comp in node_comps), "non-arrow objects must be 2x2 blocks"

    nodes: dict[int, dict] = {}
    for node_id, comp in enumerate(node_comps):
        nodes[node_id] = {
            "bbox": tuple(int(x) for x in comp["bbox"]),
            "coords": list(comp["coords"]),
            "color": int(comp["color"]),
        }

    children: dict[int, list[int]] = defaultdict(list)
    indeg = {node_id: 0 for node_id in nodes}
    for comp in arrows:
        dr, dc = _arrow_direction(comp)
        src = _nearest_node(nodes, comp["bbox"], -dr, -dc)
        dst = _nearest_node(nodes, comp["bbox"], dr, dc)
        assert src is not None and dst is not None and src != dst, "every arrow must connect two nodes"
        assert dst not in children[src], "duplicate arrow connection"
        children[src].append(dst)
        indeg[dst] += 1

    assert all(v <= 1 for v in indeg.values()), "containers must have a unique predecessor"
    sources = [node_id for node_id, deg in indeg.items() if deg == 0]
    assert sources, "expected at least one source block"

    lineage: dict[int, tuple[int, int]] = {}
    q = deque((src, int(nodes[src]["color"]), 0) for src in sources)
    while q:
        node_id, source_color, depth = q.popleft()
        prev = lineage.get(node_id)
        if prev is not None:
            assert prev == (source_color, depth), "node reached inconsistently"
            continue
        lineage[node_id] = (source_color, depth)
        for child in children[node_id]:
            q.append((child, source_color, depth + 1))

    assert len(lineage) == len(nodes), "graph must be acyclic and source-rooted"

    out = g.copy()
    out[g == arrow_color] = bg
    checker = np.array([[1, 0], [0, 1]], dtype=int)
    for node_id, node in nodes.items():
        source_color, depth = lineage[node_id]
        if depth == 0:
            continue
        r0, c0, r1, c1 = node["bbox"]
        if depth == 1:
            out[r0 : r1 + 1, c0 : c1 + 1] = source_color
        elif depth == 2:
            patch = np.where(checker == 1, source_color, bg)
            out[r0 : r1 + 1, c0 : c1 + 1] = patch
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = bg

    return out


def _midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _free_children(
    cell: tuple[int, int],
    lattice_h: int,
    lattice_w: int,
    used_nodes: set[tuple[int, int]],
    used_midpoints: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = []
    r, c = cell
    for dr, dc in DIAG4:
        nr, nc = r + dr, c + dc
        nxt = (nr, nc)
        if not (0 <= nr < lattice_h and 0 <= nc < lattice_w):
            continue
        if nxt in used_nodes:
            continue
        if _midpoint(cell, nxt) in used_midpoints:
            continue
        out.append(nxt)
    return out


def _extend_path(
    cur: tuple[int, int],
    steps_left: int,
    lattice_h: int,
    lattice_w: int,
    used_nodes: set[tuple[int, int]],
    used_midpoints: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if steps_left == 0:
        return [cur]
    candidates = _free_children(cur, lattice_h, lattice_w, used_nodes, used_midpoints)
    rng.shuffle(candidates)
    for nxt in candidates:
        mid = _midpoint(cur, nxt)
        used_nodes.add(nxt)
        used_midpoints.add(mid)
        tail = _extend_path(nxt, steps_left - 1, lattice_h, lattice_w, used_nodes, used_midpoints, rng)
        if tail is not None:
            return [cur] + tail
        used_nodes.remove(nxt)
        used_midpoints.remove(mid)
    return None


def _grid_pos(cell: tuple[int, int]) -> tuple[int, int]:
    return 1 + 4 * cell[0], 1 + 4 * cell[1]


def _draw_arrow(grid: np.ndarray, parent: tuple[int, int], child: tuple[int, int], arrow_color: int, bg: int) -> None:
    pr, pc = _grid_pos(parent)
    dr = 1 if child[0] > parent[0] else -1
    dc = 1 if child[1] > parent[1] else -1
    ar = pr + 2 * dr
    ac = pc + 2 * dc
    grid[ar : ar + 2, ac : ac + 2] = arrow_color
    grid[ar + (1 if dr == 1 else 0), ac + (1 if dc == 1 else 0)] = bg


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    lattice_h = rng.randint(*LATTICE_H_RANGE)
    lattice_w = rng.randint(*LATTICE_W_RANGE)

    used_nodes: set[tuple[int, int]] = set()
    used_midpoints: set[tuple[int, int]] = set()
    roots: list[tuple[int, int]] = []
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    children: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    depth: dict[tuple[int, int], int] = {}
    source_root: dict[tuple[int, int], tuple[int, int]] = {}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def add_path(path: list[tuple[int, int]]) -> None:
        root = path[0]
        if root not in depth:
            roots.append(root)
            depth[root] = 0
            source_root[root] = root
        for a, b in zip(path, path[1:]):
            parent[b] = a
            children[a].append(b)
            edges.append((a, b))
            depth[b] = depth[a] + 1
            source_root[b] = source_root[a]

    def try_new_path(length: int) -> bool:
        roots_order = [(r, c) for r in range(lattice_h) for c in range(lattice_w) if (r, c) not in used_nodes]
        rng.shuffle(roots_order)
        for root in roots_order:
            used_nodes.add(root)
            path = _extend_path(root, length, lattice_h, lattice_w, used_nodes, used_midpoints, rng)
            if path is not None:
                add_path(path)
                return True
            used_nodes.remove(root)
        return False

    backbone_len = rng.randint(*BACKBONE_LEN_RANGE)
    if not try_new_path(backbone_len):
        return None

    source_count = rng.choice(SOURCE_COUNT_CHOICES)
    for _ in range(source_count - 1):
        if not try_new_path(rng.randint(*EXTRA_PATH_LEN_RANGE)):
            return None

    want_branch = rng.random() < WANT_BRANCH_PROB
    if want_branch:
        branch_candidates = [
            node
            for node in used_nodes
            if len(children[node]) < MAX_OUT_DEGREE
            and depth[node] < MAX_DEPTH
            and _free_children(node, lattice_h, lattice_w, used_nodes, used_midpoints)
        ]
        if not branch_candidates:
            return None
        branch_parent = rng.choice(branch_candidates)
        branch_child = rng.choice(_free_children(branch_parent, lattice_h, lattice_w, used_nodes, used_midpoints))
        used_nodes.add(branch_child)
        used_midpoints.add(_midpoint(branch_parent, branch_child))
        parent[branch_child] = branch_parent
        children[branch_parent].append(branch_child)
        edges.append((branch_parent, branch_child))
        depth[branch_child] = depth[branch_parent] + 1
        source_root[branch_child] = source_root[branch_parent]

    extra_budget = rng.randint(*EXTRA_NODE_BUDGET)
    for _ in range(extra_budget):
        candidates = [
            node
            for node in used_nodes
            if len(children[node]) < MAX_OUT_DEGREE
            and depth[node] < MAX_DEPTH
            and _free_children(node, lattice_h, lattice_w, used_nodes, used_midpoints)
        ]
        if not candidates:
            break
        branch_parent = rng.choice(candidates)
        branch_child = rng.choice(_free_children(branch_parent, lattice_h, lattice_w, used_nodes, used_midpoints))
        used_nodes.add(branch_child)
        used_midpoints.add(_midpoint(branch_parent, branch_child))
        parent[branch_child] = branch_parent
        children[branch_parent].append(branch_child)
        edges.append((branch_parent, branch_child))
        depth[branch_child] = depth[branch_parent] + 1
        source_root[branch_child] = source_root[branch_parent]

    if max(depth.values()) < 3:
        return None

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors[0]
    arrow_color = colors[1]
    placeholder = colors[2]
    root_colors = {root: colors[3 + idx] for idx, root in enumerate(roots)}

    h = 4 * lattice_h - 1
    w = 4 * lattice_w - 1
    grid = np.full((h, w), bg, dtype=int)

    for node in used_nodes:
        gr, gc = _grid_pos(node)
        fill = root_colors[node] if node in roots else placeholder
        grid[gr : gr + 2, gc : gc + 2] = fill

    for par, child in edges:
        _draw_arrow(grid, par, child, arrow_color, bg)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
