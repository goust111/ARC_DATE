# Variant of 7b0280bc
# Dimension: D15 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码，终端通过颜色数量直接识别（恰好2个的颜色）... → 边缘像素编码路径长度

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D15 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   core_change:
#     original: the two terminals are still recognized by the color that forms
#       exactly two solid squares, and the solver always picks the unique
#       shortest terminal-to-terminal path.
#     variant: keep the same graph objects, but add one border-touching symbol:
#       a contiguous run of K same-colored pixels on one grid edge. K encodes
#       the required path length in graph edges. The solver must recolor the
#       unique simple path whose edge count is exactly K, even when a shorter
#       path exists.
#   concretization:
#     nodes: same as the source task, solid NxN squares with a uniform N in
#       {2,3}; one node color appears exactly twice (the terminals), another
#       node color appears 5-8 times (stations).
#     edges: same as the source task, one-color track components connecting
#       pairs of squares.
#     symbol: one extra non-background color, used only for a 1xK or Kx1 run on
#       the outer border. Because the graph is padded away from the border, this
#       border run is isolated and visually readable as an instruction token.
#   behavior_trace:
#     example:
#       graph_paths_between_terminals:
#         - length 2: terminal A -> X -> terminal B
#         - length 4: terminal A -> P -> Q -> R -> terminal B
#       symbol: 4 pixels on the top edge
#     original_solver: recolors X and the 2-edge track, because BFS stops at
#       the shortest path.
#     variant_solver: ignores the shorter route and recolors P,Q,R plus the
#       4-edge track, because the border symbol requests length 4.
#     difference: the path-selection stage changes; node detection, track
#       extraction, and recoloring semantics remain the same.
#   complexity_calibration:
#     why_not_too_simple:
#       - the graph still contains 7-10 nodes and 1-2 cycles, so there are
#         meaningful alternative routes rather than a single chain.
#       - the generator only accepts instances where the requested length is
#         unique and strictly larger than the shortest-path length, making the
#         D15 change visible.
#     why_not_too_complex:
#       - the symbol is only a count, not a color mapping table or multi-part
#         code.
#       - K is restricted to a small visible range {3,4,5}.
#   hidden_assumptions_in_original_main:
#     - "the remaining non-node color is the track color" no longer holds after
#       adding the border symbol. The fix is to detect the unique border-touching
#       non-node component first and treat its size as K; the remaining non-node
#       color is the track color.
#     - "BFS to goal is the right objective" no longer holds. The fix is to
#       enumerate simple terminal-to-terminal paths up to length K and require
#       that exactly one path has length K.
#     - "all relevant non-bg components are internal" no longer holds. The
#       generator therefore pads the graph away from the border so the symbol is
#       the only border-touching non-node component.
#   placement_legality:
#     - node and track placement still uses the original cell-grid embedding.
#     - compared with the source task, the graph gets an extra 1-cell border
#       buffer so tracks do not touch the outer frame; this keeps the symbol
#       isolated and solver-detectable.
#     - the symbol is written only on empty border cells.
#   visual_observability:
#     - the requested path is not merely "some other route": the generator
#       enforces shorter alternatives, so a viewer can see that the recolored
#       route is sometimes a deliberate detour whose length matches the border
#       count.
#     - the border symbol is repeated pixels, not an invisible numeric property.
#   cleanliness:
#     - no overlaps: nodes, tracks, and symbol occupy disjoint pixels.
#     - no border clutter: the graph stays off the border, so the legend remains
#       a clean isolated strip.
#   color_inference:
#     - only one extra color is added for the symbol.
#     - recolor outputs remain the original green-for-stations and grey-for-path
#       tracks, so the human-visible change is in path choice, not in a new
#       output color mapping.
#   new_solver_assumptions_and_generator_guarantees:
#     - assumption: exactly one border-touching non-node component exists.
#       guarantee: the generator places exactly one symbol run and keeps all
#       graph pixels off the border.
#     - assumption: exactly one simple terminal path has length K.
#       guarantee: the generator enumerates simple paths and rejects instances
#       where the K-edge path is absent or non-unique.
#     - assumption: K refers to edge count, not pixel count.
#       guarantee: K is chosen from the unique simple path lengths in the
#       abstract graph, then encoded as the border run length.


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """Find solid monochrome NxN squares (N>=min_size)."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    out: list[dict] = []
    for col in colors_present(grid, exclude={bg}):
        for coords in components_mask(grid == int(col), connectivity=int(connectivity)):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < int(min_size) or h != w:
                continue
            if int(coords.shape[0]) != int(h * w):
                continue
            out.append(
                {
                    "color": int(col),
                    "size": int(h),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                }
            )
    return out


def _touches_border(coords: np.ndarray | list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in coords:
        if int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1:
            return True
    return False


def _node_adjacency_from_edge_adjacency(
    adj: dict[int, list[tuple[int, int]]]
) -> dict[int, list[int]]:
    node_adj: dict[int, list[int]] = defaultdict(list)
    for u, nbrs in adj.items():
        for v, _eid in nbrs:
            node_adj[int(u)].append(int(v))
    for u in list(node_adj):
        node_adj[u] = sorted(set(node_adj[u]))
    return node_adj


def _collect_simple_paths_by_length(
    node_adj: dict[int, list[int]],
    start: int,
    goal: int,
    max_len: int,
) -> dict[int, list[list[int]]]:
    """Enumerate simple node paths from start to goal, grouped by edge count."""
    # Lower-bound distance to goal for pruning.
    dist_lb: dict[int, int] = {int(goal): 0}
    q = deque([int(goal)])
    while q:
        u = int(q.popleft())
        for v in node_adj.get(u, []):
            if int(v) in dist_lb:
                continue
            dist_lb[int(v)] = dist_lb[int(u)] + 1
            q.append(int(v))

    out: dict[int, list[list[int]]] = defaultdict(list)
    path = [int(start)]
    seen = {int(start)}

    def dfs(u: int):
        used = len(path) - 1
        lb = dist_lb.get(int(u))
        if lb is None or used + int(lb) > int(max_len):
            return
        if int(u) == int(goal):
            out[int(used)].append(path.copy())
            return
        if used == int(max_len):
            return
        for v in node_adj.get(int(u), []):
            if int(v) in seen:
                continue
            seen.add(int(v))
            path.append(int(v))
            dfs(int(v))
            path.pop()
            seen.remove(int(v))

    dfs(int(start))
    return dict(out)


def _parse_graph_problem(grid: np.ndarray) -> dict:
    """Parse nodes, track graph, and border symbol from one grid."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    node_size = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(node_size)]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "color": int(d["color"]), "pts": pts})

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))
    node_colors = sorted(nodes_by_color)
    assert len(node_colors) >= 2

    terminal_color = None
    station_color = None
    for col in node_colors:
        if len(nodes_by_color[int(col)]) == 2:
            terminal_color = int(col)
        else:
            station_color = int(col)
    assert terminal_color is not None and station_color is not None
    term_ids = list(map(int, nodes_by_color[int(terminal_color)]))

    non_bg_colors = colors_present(g, exclude={bg})
    other_colors = [int(c) for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))]
    assert len(other_colors) >= 2

    symbol_color = None
    target_len = None
    for col in other_colors:
        comps = components_mask(g == int(col), connectivity=4)
        border_comps = [coords for coords in comps if _touches_border(coords, (h, w))]
        if not border_comps:
            continue
        assert len(border_comps) == 1
        symbol_color = int(col)
        target_len = int(border_comps[0].shape[0])
        break
    assert symbol_color is not None and target_len is not None

    track_candidates = [int(c) for c in other_colors if int(c) != int(symbol_color)]
    assert len(track_candidates) == 1
    track_color = int(track_candidates[0])

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        pts = set(nd["pts"])
        id2pts[nid] = pts
        for p in pts:
            pix2node[p] = nid

    track_comps = components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    pair_to_edge_id: dict[tuple[int, int], int] = {}
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        for r, c in pts:
            for rr, cc in neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        pair = (int(a), int(b))
        assert pair not in pair_to_edge_id
        pair_to_edge_id[pair] = int(edge_id)
        track_comp_pts[int(edge_id)] = pts
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))

    return {
        "grid": g,
        "bg": bg,
        "node_size": node_size,
        "nodes": nodes,
        "id2pts": id2pts,
        "term_ids": term_ids,
        "track_color": track_color,
        "track_comp_pts": track_comp_pts,
        "adj": adj,
        "pair_to_edge_id": pair_to_edge_id,
        "target_len": target_len,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique terminal path whose edge count matches the border symbol.

    The border symbol is a contiguous run of K same-colored pixels on one outer
    edge. K is the required number of graph edges, so the chosen route need not
    be the shortest path.
    """
    problem = _parse_graph_problem(grid)
    g = problem["grid"]
    id2pts = problem["id2pts"]
    term_ids = list(map(int, problem["term_ids"]))
    adj = problem["adj"]
    pair_to_edge_id = problem["pair_to_edge_id"]
    track_comp_pts = problem["track_comp_pts"]
    target_len = int(problem["target_len"])

    start, goal = int(term_ids[0]), int(term_ids[1])
    node_adj = _node_adjacency_from_edge_adjacency(adj)
    paths_by_len = _collect_simple_paths_by_length(node_adj, start, goal, target_len)
    target_paths = paths_by_len.get(int(target_len), [])
    assert len(target_paths) == 1
    node_path = list(map(int, target_paths[0]))
    edge_path = [
        int(pair_to_edge_id[tuple(sorted((int(node_path[i]), int(node_path[i + 1]))))])
        for i in range(len(node_path) - 1)
    ]

    out = g.copy()
    term_set = set(term_ids)
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in track_comp_pts[int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]                # shortest-path length
SYMBOL_PATH_LEN_CHOICES = [3, 4, 5]     # requested exact-length path via border count
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]   # excludes green=3 and grey=5
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 600


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    edges = set()
    for i, j in picked:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (i + di, j + dj)
            if n in picked:
                edges.add(tuple(sorted([(i, j), n])))
        if allow_skip:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (i + di // 2, j + dj // 2)
                far = (i + di, j + dj)
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted([(i, j), far])))
        if use_diagonal:
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                n = (i + di, j + dj)
                corner1 = (i + di, j)
                corner2 = (i, j + dj)
                if n in picked and corner1 not in picked and corner2 not in picked:
                    edges.add(tuple(sorted([(i, j), n])))
    return edges


def _pick_cells(
    kr: int,
    kc: int,
    v_count: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
):
    cells = [(i, j) for i in range(kr) for j in range(kc)]
    if len(cells) < v_count:
        return None
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(cells)
        picked = set(cells[:v_count])
        edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
        adj = {c: [] for c in picked}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        root = next(iter(picked))
        seen = {root}
        stack = [root]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) == v_count:
            return picked
    return None


def _bfs_with_counts(src: int, allowed: set[tuple[int, int]], v_count: int):
    adj = {i: [] for i in range(v_count)}
    for a, b in allowed:
        adj[a].append(b)
        adj[b].append(a)
    dist = {int(src): 0}
    cnt = {int(src): 1}
    q = deque([int(src)])
    while q:
        u = int(q.popleft())
        for v in adj[u]:
            if int(v) not in dist:
                dist[int(v)] = dist[int(u)] + 1
                cnt[int(v)] = cnt[int(u)]
                q.append(int(v))
            elif dist[int(v)] == dist[int(u)] + 1:
                cnt[int(v)] += cnt[int(u)]
    return dist, cnt, adj


def _build_graph_spatial(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    path_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    """
    Build a connected graph with a unique shortest path between the terminals.

    The D15 variant keeps this original topology generation stage, then later
    chooses a different longer path by border symbol.
    """
    v_count = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in cell_edges}

    ids = list(range(v_count))
    rng.shuffle(ids)
    t0, t1 = ids[0], ids[1]

    dist, cnt, adj = _bfs_with_counts(t0, full_edges, v_count)
    if t1 not in dist or dist[t1] != path_len:
        return None

    parent = {int(t0): None}
    q = deque([int(t0)])
    seen = {int(t0)}
    while q:
        u = int(q.popleft())
        if u == int(t1):
            break
        for v in adj[u]:
            if int(v) not in seen:
                seen.add(int(v))
                parent[int(v)] = int(u)
                q.append(int(v))
    cur = int(t1)
    sp_nodes = []
    while cur is not None:
        sp_nodes.append(int(cur))
        cur = parent[cur]
    sp_nodes.reverse()
    sp_edges = {
        tuple(sorted((int(sp_nodes[i]), int(sp_nodes[i + 1]))))
        for i in range(len(sp_nodes) - 1)
    }

    tree_edges = set(sp_edges)
    in_tree = set(sp_nodes)
    frontier = list(sp_nodes)
    while in_tree != set(range(v_count)):
        next_frontier = []
        for u in frontier:
            for v in adj[u]:
                if int(v) not in in_tree:
                    tree_edges.add(tuple(sorted((int(u), int(v)))))
                    in_tree.add(int(v))
                    next_frontier.append(int(v))
        if not next_frontier:
            return None
        frontier = next_frontier

    extra_candidates = list(full_edges - tree_edges)
    rng.shuffle(extra_candidates)
    final_edges = set(tree_edges)
    added = 0
    for e in extra_candidates:
        if added >= n_cycles:
            break
        cand = final_edges | {e}
        dist2, cnt2, _ = _bfs_with_counts(t0, cand, v_count)
        if dist2.get(int(t1)) == path_len and cnt2.get(int(t1)) == 1:
            final_edges.add(e)
            added += 1
    if added != n_cycles:
        return None

    dist3, cnt3, _ = _bfs_with_counts(t0, final_edges, v_count)
    if dist3.get(int(t1)) != path_len or cnt3.get(int(t1)) != 1:
        return None
    if len(final_edges) != v_count - 1 + n_cycles:
        return None

    id_map = {int(t0): 0, int(t1): v_count - 1}
    for idx, i in enumerate(i for i in range(v_count) if i != t0 and i != t1):
        id_map[int(i)] = int(idx + 1)
    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for a, b in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    new_path_nodes = [id_map[n] for n in sp_nodes]
    new_path_edges = [
        tuple(sorted((int(new_path_nodes[i]), int(new_path_nodes[i + 1]))))
        for i in range(len(new_path_nodes) - 1)
    ]

    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _cell_to_square_pos(cell: tuple[int, int], n_size: int, cell_size: int, origin: tuple[int, int]):
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - n_size) // 2
    return (or_r + i * cell_size + margin, or_c + j * cell_size + margin)


def _dock_tl(sq_r: int, sq_c: int, n_size: int, track_w: int, si: int, sj: int):
    if si == -1:
        return sq_r - track_w, sq_c
    if si == 1:
        return sq_r + n_size, sq_c + n_size - track_w
    if sj == -1:
        return sq_r + n_size - track_w, sq_c - track_w
    if sj == 1:
        return sq_r, sq_c + n_size
    return None


def _walk_footprints(sr: int, sc: int, er: int, ec: int):
    pts = [(int(sr), int(sc))]
    cr, cc = int(sr), int(sc)
    while cr != int(er) and cc != int(ec):
        cr += 1 if int(er) > cr else -1
        cc += 1 if int(ec) > cc else -1
        pts.append((int(cr), int(cc)))
    while cr != int(er):
        cr += 1 if int(er) > cr else -1
        pts.append((int(cr), int(cc)))
    while cc != int(ec):
        cc += 1 if int(ec) > cc else -1
        pts.append((int(cr), int(cc)))
    return pts


def _track_between_cells(
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    n_size: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    del rng
    track_w = n_size - 1
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = cell_b[0] - cell_a[0], cell_b[1] - cell_a[1]
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(ra, ca, n_size, track_w, si, sj)
    er, ec = _dock_tl(rb, cb, n_size, track_w, -si, -sj)

    pts_tl = _walk_footprints(sr, sc, er, ec)
    pixels = set()
    for r, c in pts_tl:
        for dr in range(track_w):
            for dc in range(track_w):
                pixels.add((int(r + dr), int(c + dc)))
    return list(pixels)


def _abstract_adj(v_count: int, edges: list[tuple[int, int]]):
    adj = {i: [] for i in range(v_count)}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    for i in range(v_count):
        adj[i].sort()
    return adj


def _choose_target_path(
    v_count: int,
    edges: list[tuple[int, int]],
    shortest_len: int,
    rng: random.Random,
):
    max_len = max(SYMBOL_PATH_LEN_CHOICES)
    node_adj = _abstract_adj(v_count, edges)
    paths_by_len = _collect_simple_paths_by_length(node_adj, 0, v_count - 1, max_len)
    candidates = [
        int(path_len)
        for path_len in SYMBOL_PATH_LEN_CHOICES
        if int(path_len) > int(shortest_len) and len(paths_by_len.get(int(path_len), [])) == 1
    ]
    if not candidates:
        return None
    target_len = int(rng.choice(candidates))
    target_nodes = list(map(int, paths_by_len[target_len][0]))
    target_edges = [
        tuple(sorted((int(target_nodes[i]), int(target_nodes[i + 1]))))
        for i in range(len(target_nodes) - 1)
    ]
    return target_len, target_nodes, target_edges


def _place_symbol_segment(
    canvas: np.ndarray,
    bg: int,
    symbol_color: int,
    target_len: int,
    rng: random.Random,
):
    s = int(canvas.shape[0])
    placements: list[list[tuple[int, int]]] = []
    for start in range(1, s - target_len):
        placements.append([(0, start + k) for k in range(target_len)])
        placements.append([(s - 1, start + k) for k in range(target_len)])
        placements.append([(start + k, 0) for k in range(target_len)])
        placements.append([(start + k, s - 1) for k in range(target_len)])
    rng.shuffle(placements)
    for pts in placements:
        if all(int(canvas[r, c]) == int(bg) for r, c in pts):
            for r, c in pts:
                canvas[r, c] = int(symbol_color)
            return pts
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_size = rng.choice(NODE_SIZE_CHOICES)
    v_count = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    shortest_len = rng.choice(PATH_LEN_CHOICES)

    gap = rng.choice(GAP_CHOICES)
    cell_size = n_size + gap

    grid_options = []
    for kr in CELL_GRID_DIMS:
        for kc in CELL_GRID_DIMS:
            if kr * kc < v_count:
                continue
            min_s = max(kr, kc) * cell_size + 4
            if min_s > GRID_SIZE_RANGE[1]:
                continue
            grid_options.append((kr, kc, min_s))
    if not grid_options:
        return None
    kr, kc, min_s = rng.choice(grid_options)
    s = rng.randint(max(GRID_SIZE_RANGE[0], min_s), GRID_SIZE_RANGE[1])

    use_diagonal = False
    allow_skip = True
    picked = _pick_cells(kr, kc, v_count, rng, use_diagonal=use_diagonal, allow_skip=allow_skip)
    if picked is None:
        return None

    picked_list = list(picked)
    rng.shuffle(picked_list)
    cell_to_id = {cell: i for i, cell in enumerate(picked_list)}

    gr = _build_graph_spatial(
        picked,
        cell_to_id,
        rng,
        shortest_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if gr is None:
        return None
    edges, shortest_nodes, shortest_edges, cell_to_id = gr
    del shortest_edges
    id_to_cell = {v: k for k, v in cell_to_id.items()}
    v_count = len(picked)

    chosen = _choose_target_path(v_count, edges, len(shortest_nodes) - 1, rng)
    if chosen is None:
        return None
    target_len, target_nodes, target_edges = chosen

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    palette = pool[:5]
    bg = int(rng.choice(palette))
    rest = [int(c) for c in palette if int(c) != int(bg)]
    rng.shuffle(rest)
    terminal_color, station_color, track_color, symbol_color = rest

    canvas = np.full((s, s), bg, dtype=int)
    node_mask = np.zeros((s, s), dtype=bool)
    slack_r = s - kr * cell_size - 4
    slack_c = s - kc * cell_size - 4
    origin = (
        2 + rng.randint(0, max(0, slack_r)),
        2 + rng.randint(0, max(0, slack_c)),
    )

    id_to_pos: dict[int, tuple[int, int]] = {}
    for i in range(v_count):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, n_size, cell_size, origin)
        id_to_pos[i] = pos
        color = terminal_color if i in (0, v_count - 1) else station_color
        r0, c0 = pos
        canvas[r0 : r0 + n_size, c0 : c0 + n_size] = int(color)
        node_mask[r0 : r0 + n_size, c0 : c0 + n_size] = True

    track_mask = np.zeros((s, s), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], n_size, rng)
        for r, c in pixels:
            if not (0 <= r < s and 0 <= c < s):
                return None
            if r == 0 or c == 0 or r == s - 1 or c == s - 1:
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for r, c in pixels:
            canvas[r, c] = int(track_color)
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels

    symbol_pts = _place_symbol_segment(canvas, bg, symbol_color, target_len, rng)
    if symbol_pts is None:
        return None
    del symbol_pts

    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = canvas.copy()
    for node_id in target_nodes:
        if int(node_id) in (0, v_count - 1):
            continue
        r0, c0 = id_to_pos[int(node_id)]
        expected[r0 : r0 + n_size, c0 : c0 + n_size] = int(Color.GREEN)
    for edge in target_edges:
        for r, c in edge_track_pixels[tuple(sorted(edge))]:
            expected[r, c] = int(Color.GREY)

    # D15-specific non-degeneration check: the requested path must differ from
    # the shortest path chosen in the source task.
    if len(target_nodes) - 1 <= len(shortest_nodes) - 1:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        out = _gen_attempt(rng)
        if out is not None:
            return out
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
