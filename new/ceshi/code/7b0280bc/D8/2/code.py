from __future__ import annotations

# Variant of 7b0280bc
# Dimension: D8 — 多步处理序列 (multi_step)
# Change: 单步处理：找最短路径并一次性涂色... → 双终端独立路径合并
#
# Design analysis for this variant
# --------------------------------
# spec_variant:
#   input:
#     - The scene is still a graph embedded as solid NxN squares plus track components.
#     - There are now 3 terminal squares:
#         * A/source color appears exactly once.
#         * B/C target color appears exactly twice.
#         * Station color appears on the remaining 4-7 squares.
#     - Track color is the remaining non-background color.
#     - We intentionally add one extra node color versus the original task so that A is
#       visually distinguishable from B/C. Without that change, "which terminal is the
#       shared start?" would be invisible to the human solver.
#   output:
#     - Compute the unique shortest path A->B.
#     - Compute the unique shortest path A->C.
#     - Take the union of those two paths.
#     - Recolor every non-terminal square on that union to green(3).
#     - Recolor every track component on that union to grey(5).
#     - A, B, C themselves keep their original colors; this preserves the original
#       task's coloring semantics and changes only the D8 core behavior.
#
# behavior_trace:
#   original:
#     - locate 2 terminals of one color
#     - BFS once from terminal_1 to terminal_2
#     - recolor stations/edges on that single path
#   variant:
#     - locate source A by "count==1" node color
#     - locate targets B,C by "count==2" node color
#     - BFS from A, reconstruct path(A,B) and path(A,C)
#     - union the node ids and edge ids from both paths
#     - recolor the union
#   difference:
#     - processing is now explicitly multi-step: one shortest path does not complete
#       the task; the solver must repeat the procedure for a second target and merge.
#
# complexity_calibration:
#   - We keep the original graph embedding, square size, edge routing, and cycle counts.
#   - To make the new rule visually obvious, the generator varies whether the two shortest
#     paths share a prefix (Y-shaped union) or diverge immediately at A.
#   - Distances are still small (2-3 edges per target) so the puzzle stays legible.
#
# implicit_assumption_review:
#   - Old code assumed "the terminal color occurs exactly twice". This is no longer true.
#   - Old code assumed there was only one goal terminal; `term_ids[0], term_ids[1]` was
#     sufficient. Now we must identify 1 source id and 2 target ids by node-color counts.
#   - Old coloring loop skipped only two terminals; now it must skip three terminals.
#   - A single `prev/prev_edge` reconstruction is no longer enough unless we reconstruct
#     both source-target paths and union them.
#   - Generator must guarantee that the count pattern {1,2,>=4} over node colors exists,
#     that both source-target shortest paths are unique, and that routing still yields
#     separate track components touching exactly two nodes each.
#
# placement_legality_and_cleanliness:
#   - Squares remain uniform-size, non-overlapping, and separated by background.
#   - Track routing reuses the original footprint-based router.
#   - After routing, the generator verifies that track components are still separable and
#     each touches exactly two squares, avoiding visually merged or ambiguous edges.
#
# observability:
#   - Across examples, humans can see one uniquely colored source square and two same-
#     colored targets. Output highlights two branches that originate from the unique source.
#   - Because the generator sometimes creates a shared prefix and sometimes does not, the
#     "two paths then union" rule is supported by repeated structural evidence rather than
#     a single fixed picture.

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, components_mask, colors_present, get_bg, neighbors8


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


def _reconstruct_path(
    start: int,
    goal: int,
    prev: dict[int, int | None],
    prev_edge: dict[int, int | None],
) -> tuple[list[int], list[int]]:
    """Reconstruct a BFS path from start to goal using predecessor maps."""
    assert int(goal) in prev
    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        eid = prev_edge.get(int(cur))
        if eid is not None:
            edge_path.append(int(eid))
        par = prev.get(int(cur))
        cur = int(par) if par is not None else None
    node_path.reverse()
    edge_path.reverse()
    assert node_path and int(node_path[0]) == int(start)
    return node_path, edge_path


def _parse_graph(grid: np.ndarray) -> dict:
    """Parse node squares and track components into a graph representation."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(N)]
    assert nodes_raw

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "color": int(d["color"]), "pts": pts})

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    source_color = None
    target_color = None
    station_color = None
    for col, ids in nodes_by_color.items():
        count = len(ids)
        if count == 1:
            source_color = int(col)
        elif count == 2:
            target_color = int(col)
        elif count >= 1:
            station_color = int(col)

    assert source_color is not None
    assert target_color is not None
    assert station_color is not None

    source_id = int(nodes_by_color[int(source_color)][0])
    target_ids = list(map(int, nodes_by_color[int(target_color)]))
    assert len(target_ids) == 2

    non_bg_colors = colors_present(g, exclude={bg})
    node_colors = {int(source_color), int(target_color), int(station_color)}
    track_candidates = [c for c in non_bg_colors if int(c) not in node_colors]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

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
        track_comp_pts[int(edge_id)] = pts
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))

    return {
        "grid": g,
        "bg": bg,
        "size": N,
        "source_id": source_id,
        "target_ids": target_ids,
        "id2pts": id2pts,
        "track_comp_pts": track_comp_pts,
        "adj": adj,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the union of the two shortest paths from source A to targets B and C.
    """
    parsed = _parse_graph(np.array(grid, dtype=int))
    g = parsed["grid"]
    source_id = int(parsed["source_id"])
    target_ids = list(map(int, parsed["target_ids"]))
    id2pts = parsed["id2pts"]
    track_comp_pts = parsed["track_comp_pts"]
    adj = parsed["adj"]

    q = deque([source_id])
    prev: dict[int, int | None] = {source_id: None}
    prev_edge: dict[int, int | None] = {source_id: None}

    while q:
        u = int(q.popleft())
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))

    union_nodes: set[int] = {source_id}
    union_edges: set[int] = set()
    for goal in target_ids:
        node_path, edge_path = _reconstruct_path(source_id, int(goal), prev, prev_edge)
        union_nodes.update(map(int, node_path))
        union_edges.update(map(int, edge_path))

    out = g.copy()
    terminal_ids = {source_id, *target_ids}
    for nid in union_nodes:
        if int(nid) in terminal_ids:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in union_edges:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# --------------- generator ---------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)               # 1 source + 2 targets + 4-7 stations
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]                # per-target shortest-path edge count
SHARED_PREFIX_CHOICES = [0, 1]           # 0: branch immediately, 1: Y-shape
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]   # input excludes green(3)/grey(5)
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 600


def _compute_adjacency(
    picked: set[tuple[int, int]],
    use_diagonal: bool,
    allow_skip: bool,
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    """Compute valid undirected edges between picked cells on a grid."""
    edges = set()
    for (i, j) in picked:
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (i + di, j + dj)
            if n in picked:
                edges.add(tuple(sorted([(i, j), n])))
        if allow_skip:
            for (di, dj) in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (i + di // 2, j + dj // 2)
                far = (i + di, j + dj)
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted([(i, j), far])))
        if use_diagonal:
            for (di, dj) in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                n = (i + di, j + dj)
                corner1 = (i + di, j)
                corner2 = (i, j + dj)
                if n in picked and corner1 not in picked and corner2 not in picked:
                    edges.add(tuple(sorted([(i, j), n])))
    return edges


def _pick_cells(
    Kr: int,
    Kc: int,
    V: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
):
    """Pick V cells whose induced adjacency graph is connected."""
    cells = [(i, j) for i in range(Kr) for j in range(Kc)]
    if len(cells) < V:
        return None
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(cells)
        picked = set(cells[:V])
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
        if len(seen) == V:
            return picked
    return None


def _adj_from_edges(edges: set[tuple[int, int]], V: int) -> dict[int, list[int]]:
    adj = {i: [] for i in range(V)}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def _bfs_with_counts(src: int, edges: set[tuple[int, int]], V: int):
    """Return shortest-path distances and counts from src."""
    adj = _adj_from_edges(edges, V)
    dist = {int(src): 0}
    cnt = {int(src): 1}
    prev = {int(src): None}
    prev_edge = {int(src): None}
    q = deque([int(src)])
    while q:
        u = int(q.popleft())
        for v in adj[u]:
            e = tuple(sorted((int(u), int(v))))
            if int(v) not in dist:
                dist[int(v)] = int(dist[int(u)] + 1)
                cnt[int(v)] = int(cnt[int(u)])
                prev[int(v)] = int(u)
                prev_edge[int(v)] = e
                q.append(int(v))
            elif int(dist[int(v)]) == int(dist[int(u)] + 1):
                cnt[int(v)] = int(cnt[int(v)] + cnt[int(u)])
    return dist, cnt, adj, prev, prev_edge


def _enumerate_paths_exact(
    adj: dict[int, list[int]],
    start: int,
    steps: int,
    banned: set[int] | None = None,
) -> list[list[int]]:
    """Enumerate simple paths of exactly `steps` edges from start."""
    banned = set() if banned is None else set(map(int, banned))
    out: list[list[int]] = []

    def rec(cur: int, rem: int, path: list[int], used: set[int]):
        if rem == 0:
            out.append(list(path))
            return
        nxts = list(adj[int(cur)])
        for nxt in nxts:
            if int(nxt) in used or int(nxt) in banned:
                continue
            used.add(int(nxt))
            path.append(int(nxt))
            rec(int(nxt), rem - 1, path, used)
            path.pop()
            used.remove(int(nxt))

    rec(int(start), int(steps), [int(start)], {int(start)})
    return out


def _find_dual_path_core(
    full_adj: dict[int, list[int]],
    V: int,
    rng: random.Random,
    len_b: int,
    len_c: int,
    shared_len: int,
):
    """Find a source and two target paths with a shared prefix of fixed length."""
    sources = list(range(V))
    rng.shuffle(sources)
    for source in sources:
        prefix_paths = [[int(source)]] if shared_len == 0 else _enumerate_paths_exact(
            full_adj, int(source), int(shared_len)
        )
        rng.shuffle(prefix_paths)
        for prefix in prefix_paths:
            branch = int(prefix[-1])
            banned_prefix = set(map(int, prefix[:-1]))
            rem_b = int(len_b - shared_len)
            rem_c = int(len_c - shared_len)
            if rem_b <= 0 or rem_c <= 0:
                continue
            branch_paths_b = _enumerate_paths_exact(full_adj, branch, rem_b, banned_prefix)
            branch_paths_c = _enumerate_paths_exact(full_adj, branch, rem_c, banned_prefix)
            rng.shuffle(branch_paths_b)
            rng.shuffle(branch_paths_c)
            for pb in branch_paths_b:
                for pc in branch_paths_c:
                    if int(pb[-1]) == int(pc[-1]):
                        continue
                    if set(map(int, pb[1:])) & set(map(int, pc[1:])):
                        continue
                    full_b = list(prefix[:-1]) + list(pb)
                    full_c = list(prefix[:-1]) + list(pc)
                    assert len(full_b) == len_b + 1
                    assert len(full_c) == len_c + 1
                    edge_union = {
                        tuple(sorted((int(path[i]), int(path[i + 1]))))
                        for path in (full_b, full_c)
                        for i in range(len(path) - 1)
                    }
                    return {
                        "source": int(source),
                        "target_b": int(full_b[-1]),
                        "target_c": int(full_c[-1]),
                        "path_b": full_b,
                        "path_c": full_c,
                        "edge_union": edge_union,
                    }
    return None


def _path_edges_from_nodes(path: list[int]) -> list[tuple[int, int]]:
    return [tuple(sorted((int(path[i]), int(path[i + 1])))) for i in range(len(path) - 1)]


def _build_graph_spatial(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    len_b: int,
    len_c: int,
    shared_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    """
    Build a connected graph whose source-target shortest paths satisfy the dual-path rule.
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in cell_edges}
    full_adj = _adj_from_edges(full_edges, V)

    core = _find_dual_path_core(full_adj, V, rng, len_b, len_c, shared_len)
    if core is None:
        return None

    source = int(core["source"])
    target_b = int(core["target_b"])
    target_c = int(core["target_c"])
    tree_edges = set(core["edge_union"])
    in_tree = {
        *map(int, core["path_b"]),
        *map(int, core["path_c"]),
    }
    frontier = list(in_tree)

    while in_tree != set(range(V)):
        next_frontier: list[int] = []
        rng.shuffle(frontier)
        for u in frontier:
            nxts = list(full_adj[int(u)])
            rng.shuffle(nxts)
            for v in nxts:
                if int(v) in in_tree:
                    continue
                tree_edges.add(tuple(sorted((int(u), int(v)))))
                in_tree.add(int(v))
                next_frontier.append(int(v))
        if not next_frontier:
            return None
        frontier = next_frontier

    final_edges = set(tree_edges)
    extra_candidates = list(full_edges - tree_edges)
    rng.shuffle(extra_candidates)
    added = 0
    for e in extra_candidates:
        if added >= n_cycles:
            break
        cand = final_edges | {e}
        dist, cnt, _adj, _prev, _prev_edge = _bfs_with_counts(source, cand, V)
        if (
            dist.get(target_b) == len_b
            and cnt.get(target_b) == 1
            and dist.get(target_c) == len_c
            and cnt.get(target_c) == 1
        ):
            final_edges.add(e)
            added += 1
    if added != n_cycles:
        return None

    dist, cnt, _adj, prev, prev_edge = _bfs_with_counts(source, final_edges, V)
    if (
        dist.get(target_b) != len_b
        or cnt.get(target_b) != 1
        or dist.get(target_c) != len_c
        or cnt.get(target_c) != 1
    ):
        return None

    path_b, _ = _reconstruct_graph_path(source, target_b, prev, prev_edge)
    path_c, _ = _reconstruct_graph_path(source, target_c, prev, prev_edge)
    path_union_nodes = set(map(int, path_b)) | set(map(int, path_c))
    path_union_edges = set(_path_edges_from_nodes(path_b)) | set(_path_edges_from_nodes(path_c))

    id_map = {source: 0, target_b: V - 2, target_c: V - 1}
    next_id = 1
    for old in range(V):
        if old in id_map:
            continue
        id_map[old] = int(next_id)
        next_id += 1

    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for a, b in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    remapped_union_nodes = [id_map[n] for n in sorted(path_union_nodes)]
    remapped_union_edges = [tuple(sorted((id_map[a], id_map[b]))) for a, b in sorted(path_union_edges)]

    return {
        "edges": list(remapped_edges),
        "path_union_nodes": remapped_union_nodes,
        "path_union_edges": remapped_union_edges,
        "cell_to_id": remapped_cell_to_id,
        "shared_len": shared_len,
        "len_b": len_b,
        "len_c": len_c,
    }


def _reconstruct_graph_path(
    start: int,
    goal: int,
    prev: dict[int, int | None],
    prev_edge: dict[int, tuple[int, int] | None],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Reconstruct a graph path where prev_edge stores edge tuples."""
    assert int(goal) in prev
    nodes: list[int] = []
    edges: list[tuple[int, int]] = []
    cur: int | None = int(goal)
    while cur is not None:
        nodes.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
        par = prev.get(int(cur))
        cur = int(par) if par is not None else None
    nodes.reverse()
    edges.reverse()
    assert nodes and int(nodes[0]) == int(start)
    return nodes, edges


def _cell_to_square_pos(
    cell: tuple[int, int],
    N: int,
    cell_size: int,
    origin: tuple[int, int],
):
    """Top-left corner of the NxN square centered in the given cell."""
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return or_r + i * cell_size + margin, or_c + j * cell_size + margin


def _dock_tl(sq_r: int, sq_c: int, N: int, Wt: int, si: int, sj: int):
    """Top-left of the Wt x Wt footprint docked to one face of an NxN square."""
    if si == -1:
        return sq_r - Wt, sq_c
    if si == 1:
        return sq_r + N, sq_c + N - Wt
    if sj == -1:
        return sq_r + N - Wt, sq_c - Wt
    if sj == 1:
        return sq_r, sq_c + N
    return None


def _walk_footprints(sr: int, sc: int, er: int, ec: int):
    """Generate footprint top-left positions from start to end."""
    pts = [(sr, sc)]
    cr, cc = sr, sc
    while cr != er and cc != ec:
        cr += 1 if er > cr else -1
        cc += 1 if ec > cc else -1
        pts.append((cr, cc))
    while cr != er:
        cr += 1 if er > cr else -1
        pts.append((cr, cc))
    while cc != ec:
        cc += 1 if ec > cc else -1
        pts.append((cr, cc))
    return pts


def _track_between_cells(
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    N: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Route a (N-1)-wide track strip between two node squares."""
    del rng
    Wt = N - 1
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = cell_b[0] - cell_a[0], cell_b[1] - cell_a[1]
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    start = _dock_tl(ra, ca, N, Wt, si, sj)
    end = _dock_tl(rb, cb, N, Wt, -si, -sj)
    if start is None or end is None:
        return []
    sr, sc = start
    er, ec = end
    pts_tl = _walk_footprints(sr, sc, er, ec)

    pixels = set()
    for r, c in pts_tl:
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((r + dr, c + dc))
    return list(pixels)


def _verify_track_components(
    canvas: np.ndarray,
    node_colors: set[int],
    track_color: int,
) -> bool:
    """Ensure routed tracks still form one component per edge touching exactly two nodes."""
    bg = int(get_bg(canvas))
    square_cands = find_solid_squares(canvas, bg=bg, min_size=2, connectivity=8)
    pix2node: dict[tuple[int, int], int] = {}
    nid = 0
    for d in square_cands:
        if int(d["color"]) not in node_colors:
            continue
        for r, c in d["coords"]:
            pix2node[(int(r), int(c))] = int(nid)
        nid += 1
    track_comps = components_mask(canvas == int(track_color), connectivity=8)
    for coords in track_comps:
        touched: set[int] = set()
        for r, c in coords:
            for rr, cc in neighbors8(int(r), int(c)):
                hit = pix2node.get((int(rr), int(cc)))
                if hit is not None:
                    touched.add(int(hit))
        if len(touched) != 2:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    len_b = rng.choice(PATH_LEN_CHOICES)
    len_c = rng.choice(PATH_LEN_CHOICES)
    shared_options = [s for s in SHARED_PREFIX_CHOICES if s < min(len_b, len_c)]
    if not shared_options:
        return None
    shared_len = rng.choice(shared_options)

    gap = rng.choice(GAP_CHOICES)
    cell_size = N + gap

    KrKc_options = []
    for Kr in CELL_GRID_DIMS:
        for Kc in CELL_GRID_DIMS:
            if Kr * Kc < V:
                continue
            min_S = max(Kr, Kc) * cell_size + 2
            if min_S > GRID_SIZE_RANGE[1]:
                continue
            KrKc_options.append((Kr, Kc, min_S))
    if not KrKc_options:
        return None
    Kr, Kc, min_S = rng.choice(KrKc_options)
    S = rng.randint(max(GRID_SIZE_RANGE[0], min_S), GRID_SIZE_RANGE[1])

    use_diagonal = False
    allow_skip = True
    picked = _pick_cells(Kr, Kc, V, rng, use_diagonal=use_diagonal, allow_skip=allow_skip)
    if picked is None:
        return None

    picked_list = list(picked)
    rng.shuffle(picked_list)
    cell_to_id = {cell: i for i, cell in enumerate(picked_list)}

    graph = _build_graph_spatial(
        picked,
        cell_to_id,
        rng,
        len_b,
        len_c,
        shared_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if graph is None:
        return None

    edges = list(graph["edges"])
    path_union_nodes = list(graph["path_union_nodes"])
    path_union_edges = list(graph["path_union_edges"])
    cell_to_id = graph["cell_to_id"]
    id_to_cell = {v: k for k, v in cell_to_id.items()}

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    five = pool[:5]
    bg = rng.choice(five)
    rest = [c for c in five if c != bg]
    rng.shuffle(rest)
    source_color, target_color, station_color, track_color = rest

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))
    id_to_pos: dict[int, tuple[int, int]] = {}

    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[i] = pos
        if i == 0:
            color = source_color
        elif i in (V - 2, V - 1):
            color = target_color
        else:
            color = station_color
        r0, c0 = pos
        canvas[r0:r0 + N, c0:c0 + N] = color
        node_mask[r0:r0 + N, c0:c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N, rng)
        if not pixels:
            return None
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for r, c in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    node_colors = {int(source_color), int(target_color), int(station_color)}
    if not _verify_track_components(canvas, node_colors, int(track_color)):
        return None

    expected = canvas.copy()
    terminal_ids = {0, V - 2, V - 1}
    for nid in path_union_nodes:
        if int(nid) in terminal_ids:
            continue
        r0, c0 = id_to_pos[int(nid)]
        expected[r0:r0 + N, c0:c0 + N] = int(Color.GREEN)
    for e in path_union_edges:
        for r, c in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = int(Color.GREY)

    # Strong anti-degeneration check: the union must use more than one target branch.
    if len(path_union_edges) <= max(int(graph["len_b"]), int(graph["len_c"])):
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
