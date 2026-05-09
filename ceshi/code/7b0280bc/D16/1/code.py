# Variant of 7b0280bc
# Dimension: D16 — 障碍物/轨道交互 (obstacle)
# Change: 无显式障碍物，所有轨道段均可通行... → 阻塞节点

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D16 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   input_objects:
#     terminals:
#       shape: solid 3x3 squares of one color
#       count: exactly 2
#     stations:
#       shape: 3x3 squares of another color
#       count: 5-8
#       subtypes:
#         normal: fully filled 3x3
#         blocked: same 3x3 outline/color but center pixel is background
#     tracks:
#       shape: 2-pixel-wide 8-connected strips of a third color
#       role: graph edges between node squares
#   output_rule:
#     passable_graph: all terminals + all normal stations
#     blocked_graph_nodes: every station whose 3x3 center pixel is background
#     transform:
#       - run shortest-path search between the two terminals while forbidding blocked stations
#       - recolor only the used non-terminal stations to green(3)
#       - recolor only the used track components to grey(5)
#       - blocked stations and tracks leading only through blocked shortcuts stay unchanged
#
# behavior_trace:
#   original:
#     - detect only solid same-color squares as nodes
#     - BFS may traverse every detected station
#   variant:
#     - detect both solid 3x3 stations and 3x3 stations with a single bg hole at center
#     - the center-hole stations still count as nodes for edge attachment, but are removed
#       from the BFS state space
#   concrete_example:
#     - graph contains terminal A -> blocked station X -> terminal B as the 2-edge shortcut
#     - it also contains A -> s1 -> s2 -> s3 -> B as a longer unblocked detour
#     - original logic would choose A-X-B
#     - new logic must skip X and choose the detour instead
#
# complexity_calibration:
#   - The marker is structural and visible: a 3x3 station with a missing center is easy to
#     spot in 2-4 examples.
#   - To make the center-hole unambiguous, node size is fixed to 3 in the generator.
#     This is the only deliberate non-D16 adaptation.
#   - Blocked and unblocked stations share the same outer color, so the rule depends on the
#     visible hole, not on hidden color ids.
#
# implicit_assumptions_review:
#   - Original `find_solid_squares` assumed every node is fully filled. This fails for blocked
#     stations, so node detection is generalized to accept either a full NxN square or an
#     NxN square with exactly the center pixel missing.
#   - Original size inference used counts of detected solid squares. That would undercount
#     blocked nodes. The new detector returns both kinds before inferring N.
#   - Original BFS assumed every detected station is traversable. The new BFS must skip nodes
#     flagged as blocked, otherwise the output collapses back to the source task.
#   - Original recoloring loop wrote every path node's existing pixels. For blocked stations,
#     the center bg pixel is not part of the node pixel set, so even if one slipped into the
#     path it would retain its hole. The generator guarantees blocked stations are never on the
#     chosen path, but the solver still represents nodes by their actual occupied pixels only.
#
# placement_legality_and_visual_cleanliness:
#   - The generator keeps the original graph-first placement/routing scheme so squares and
#     tracks remain non-overlapping and in-bounds.
#   - The blocked marker uses the square interior only; it never conflicts with track routing.
#   - Every generated sample is accepted only if:
#       1. removing one interior shortest-path station still leaves a unique terminal path
#       2. that rerouted path is strictly longer than the unblocked shortest path
#       3. `main(input)` exactly matches the oracle output
#
# generator_assumptions_for_new_main:
#   - all nodes have odd side length 3, so the center-hole test is well defined
#   - terminals are always solid, never blocked
#   - at least one blocked station exists and it lies on the original unblocked shortest path
#   - after removing blocked stations, the remaining shortest terminal path is unique


def find_node_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """
    Find node candidates that are either:
    - solid monochrome NxN squares
    - monochrome NxN squares with exactly the center pixel replaced by bg
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    out: list[dict] = []
    for col in colors_present(grid, exclude={bg}):
        mask = grid == int(col)
        for coords in components_mask(mask, connectivity=int(connectivity)):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < int(min_size) or h != w:
                continue

            patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            area = int(coords.shape[0])
            blocked = False

            if area == int(h * w) and np.all(patch == int(col)):
                blocked = False
            elif h % 2 == 1 and h >= 3 and area == int(h * w - 1):
                expected = np.full((h, w), int(col), dtype=int)
                mid = h // 2
                expected[mid, mid] = int(bg)
                if np.array_equal(patch, expected):
                    blocked = True
                else:
                    continue
            else:
                continue

            out.append(
                {
                    "color": int(col),
                    "size": int(h),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                    "blocked": bool(blocked),
                }
            )
    return out


def _build_graph_from_grid(
    g: np.ndarray,
) -> tuple[list[dict], dict[int, set[tuple[int, int]]], dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]], list[int]]:
    bg = int(get_bg(g))

    node_cands = find_node_squares(g, bg=bg, min_size=2, connectivity=8)
    assert node_cands

    size_counts = Counter(int(d["size"]) for d in node_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in node_cands if int(d["size"]) == N]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append(
            {
                "id": int(i),
                "color": int(d["color"]),
                "pts": pts,
                "blocked": bool(d["blocked"]),
            }
        )

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    node_colors = sorted(nodes_by_color.keys())
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
    track_candidates = [c for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))]
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

    return nodes, id2pts, track_comp_pts, adj, term_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - detect node squares as solid squares or center-hole squares
    - treat center-hole stations as blocked graph nodes
    - compute the shortest terminal path while avoiding blocked stations
    - recolor traversed non-terminal stations to green and traversed track segments to grey
    """
    g = np.array(grid, dtype=int)
    nodes, id2pts, track_comp_pts, adj, term_ids = _build_graph_from_grid(g)

    blocked = {int(nd["id"]) for nd in nodes if bool(nd["blocked"])}
    assert not (set(term_ids) & blocked)

    start, goal = int(term_ids[0]), int(term_ids[1])
    q = deque([start])
    prev: dict[int, int | None] = {start: None}
    prev_edge: dict[int, int | None] = {start: None}

    while q:
        u = int(q.popleft())
        if u == goal:
            break
        for v, eid in adj.get(u, []):
            if int(v) in blocked or int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))

    assert goal in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        parent = prev.get(int(cur))
        cur = int(parent) if parent is not None else None
    node_path.reverse()
    edge_path.reverse()

    out = g.copy()
    term_set = set(map(int, term_ids))
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# ---------------------------------------------------------------------------
# Generator configuration
# ---------------------------------------------------------------------------

NODE_SIZE_CHOICES = [3]                  # fixed for a visible center-hole marker
NODE_COUNT_RANGE = (7, 10)              # V = total graph nodes
CYCLE_COUNT_CHOICES = [1, 2]            # independent cycles in the graph
PATH_LEN_CHOICES = [2, 3]               # original unblocked shortest-path edge count
GAP_CHOICES = [2, 3]                    # gap between node square edge and cell boundary
CELL_GRID_DIMS = (3, 4, 5)             # candidate Kr, Kc values for the cell grid
GRID_SIZE_RANGE = (15, 30)             # canvas side length S bounds
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]  # excludes green=3, grey=5
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 400


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
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
    cells = [(i, j) for i in range(Kr) for j in range(Kc)]
    if len(cells) < V:
        return None
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(cells)
        picked = set(cells[:V])
        edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
        adj = {c: [] for c in picked}
        for (a, b) in edges:
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


def _bfs_with_counts(src: int, allowed: set[tuple[int, int]], V: int):
    adj = {i: [] for i in range(V)}
    for (a, b) in allowed:
        adj[a].append(b)
        adj[b].append(a)
    dist = {src: 0}
    cnt = {src: 1}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                cnt[v] = cnt[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                cnt[v] += cnt[u]
    return dist, cnt, adj


def _shortest_path_info(
    edges: list[tuple[int, int]] | set[tuple[int, int]],
    V: int,
    start: int,
    goal: int,
    blocked: set[int] | None = None,
):
    blocked = set() if blocked is None else set(map(int, blocked))
    if int(start) in blocked or int(goal) in blocked:
        return None

    adj = {i: [] for i in range(V)}
    for (a, b) in edges:
        aa = int(a)
        bb = int(b)
        adj[aa].append(bb)
        adj[bb].append(aa)

    q = deque([int(start)])
    dist = {int(start): 0}
    cnt = {int(start): 1}
    parent = {int(start): None}

    while q:
        u = int(q.popleft())
        for v in adj[u]:
            if int(v) in blocked:
                continue
            if int(v) not in dist:
                dist[int(v)] = dist[u] + 1
                cnt[int(v)] = cnt[u]
                parent[int(v)] = int(u)
                q.append(int(v))
            elif dist[int(v)] == dist[u] + 1:
                cnt[int(v)] += cnt[u]

    if int(goal) not in dist:
        return None

    path_nodes = None
    path_edges = None
    if cnt[int(goal)] == 1:
        cur: int | None = int(goal)
        path_nodes = []
        path_edges = []
        while cur is not None:
            path_nodes.append(int(cur))
            par = parent.get(int(cur))
            if par is not None:
                path_edges.append(tuple(sorted((int(par), int(cur)))))
            cur = int(par) if par is not None else None
        path_nodes.reverse()
        path_edges.reverse()

    return {
        "dist": int(dist[int(goal)]),
        "count": int(cnt[int(goal)]),
        "path_nodes": path_nodes,
        "path_edges": path_edges,
    }


def _edges_from_adj(adj: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int]]:
    edges = set()
    for a, nbrs in adj.items():
        for b, _eid in nbrs:
            edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def _build_graph_spatial(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    path_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for (a, b) in cell_edges}

    ids = list(range(V))
    rng.shuffle(ids)
    t0, t1 = ids[0], ids[1]

    dist, cnt, adj = _bfs_with_counts(t0, full_edges, V)
    if t1 not in dist or dist[t1] != path_len:
        return None

    parent = {t0: None}
    q = deque([t0])
    seen = {t0}
    while q:
        u = q.popleft()
        if u == t1:
            break
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v] = u
                q.append(v)
    cur = t1
    sp_nodes = []
    while cur is not None:
        sp_nodes.append(cur)
        cur = parent[cur]
    sp_nodes.reverse()
    sp_edges = {tuple(sorted((sp_nodes[i], sp_nodes[i + 1]))) for i in range(len(sp_nodes) - 1)}

    tree_edges = set(sp_edges)
    in_tree = set(sp_nodes)
    frontier = list(sp_nodes)
    while in_tree != set(range(V)):
        next_frontier = []
        for u in frontier:
            for v in adj[u]:
                if v not in in_tree:
                    tree_edges.add(tuple(sorted((u, v))))
                    in_tree.add(v)
                    next_frontier.append(v)
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
        dist2, cnt2, _ = _bfs_with_counts(t0, cand, V)
        if dist2.get(t1) == path_len and cnt2.get(t1) == 1:
            final_edges.add(e)
            added += 1
    if added != n_cycles:
        return None

    dist3, cnt3, _ = _bfs_with_counts(t0, final_edges, V)
    if dist3.get(t1) != path_len or cnt3.get(t1) != 1:
        return None
    if len(final_edges) != V - 1 + n_cycles:
        return None

    id_map = {t0: 0, t1: V - 1}
    for idx, i in enumerate(i for i in range(V) if i != t0 and i != t1):
        id_map[i] = idx + 1
    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for (a, b) in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    new_path_nodes = [id_map[n] for n in sp_nodes]
    new_path_edges = [tuple(sorted((new_path_nodes[i], new_path_nodes[i + 1]))) for i in range(len(new_path_nodes) - 1)]

    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _choose_blocked_node(
    edges: list[tuple[int, int]],
    path_nodes: list[int],
    V: int,
    rng: random.Random,
):
    base_len = len(path_nodes) - 1
    candidates = []
    for blocked_node in path_nodes[1:-1]:
        info = _shortest_path_info(edges, V, 0, V - 1, blocked={int(blocked_node)})
        if info is None:
            continue
        if int(info["count"]) != 1:
            continue
        if int(info["dist"]) <= int(base_len):
            continue
        candidates.append((int(blocked_node), info))
    if not candidates:
        return None

    best_len = min(int(info["dist"]) for _, info in candidates)
    best = [(bn, info) for bn, info in candidates if int(info["dist"]) == best_len]
    return rng.choice(best)


def _cell_to_square_pos(cell: tuple[int, int], N: int, cell_size: int, origin: tuple[int, int]):
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return (or_r + i * cell_size + margin, or_c + j * cell_size + margin)


def _dock_tl(sq_r: int, sq_c: int, N: int, Wt: int, si: int, sj: int):
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
    del rng
    Wt = N - 1
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = cell_b[0] - cell_a[0], cell_b[1] - cell_a[1]
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(ra, ca, N, Wt, si, sj)
    er, ec = _dock_tl(rb, cb, N, Wt, -si, -sj)
    pts_tl = _walk_footprints(sr, sc, er, ec)

    pixels = set()
    for r, c in pts_tl:
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((r + dr, c + dc))
    return list(pixels)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    path_len = rng.choice(PATH_LEN_CHOICES)

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
        path_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if graph is None:
        return None
    edges, path_nodes, _path_edges, cell_to_id = graph
    id_to_cell = {v: k for k, v in cell_to_id.items()}
    V = len(picked)

    blocked_choice = _choose_blocked_node(edges, path_nodes, V, rng)
    if blocked_choice is None:
        return None
    blocked_node, reroute = blocked_choice
    reroute_nodes = list(reroute["path_nodes"])
    reroute_edges = list(reroute["path_edges"])

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    four = pool[:4]
    bg = rng.choice(four)
    rest = [c for c in four if c != bg]
    terminal_color = rng.choice(rest)
    rest2 = [c for c in rest if c != terminal_color]
    station_color = rng.choice(rest2)
    track_color = [c for c in rest2 if c != station_color][0]

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))
    id_to_pos = {}

    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[i] = pos
        color = terminal_color if i in (0, V - 1) else station_color
        r0, c0 = pos
        canvas[r0 : r0 + N, c0 : c0 + N] = color
        if i == int(blocked_node):
            canvas[r0 + N // 2, c0 + N // 2] = bg
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (a, b) in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N, rng)
        for (r, c) in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for (r, c) in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    detected_nodes, _id2pts2, _track_pts2, detected_adj, detected_terms = _build_graph_from_grid(canvas)
    detected_blocked = {int(nd["id"]) for nd in detected_nodes if bool(nd["blocked"])}
    if len(detected_blocked) != 1:
        return None
    detected_edges = _edges_from_adj(detected_adj)
    base_info = _shortest_path_info(
        detected_edges,
        len(detected_nodes),
        int(detected_terms[0]),
        int(detected_terms[1]),
        blocked=set(),
    )
    blocked_info = _shortest_path_info(
        detected_edges,
        len(detected_nodes),
        int(detected_terms[0]),
        int(detected_terms[1]),
        blocked=detected_blocked,
    )
    if base_info is None or blocked_info is None:
        return None
    if int(base_info["count"]) != 1 or int(blocked_info["count"]) != 1:
        return None
    if not (detected_blocked & set(base_info["path_nodes"])):
        return None
    if detected_blocked & set(blocked_info["path_nodes"]):
        return None
    if int(blocked_info["dist"]) <= int(base_info["dist"]):
        return None

    expected = canvas.copy()
    for gnode in reroute_nodes:
        if gnode in (0, V - 1):
            continue
        r0, c0 = id_to_pos[gnode]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for e in reroute_edges:
        for (r, c) in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = int(Color.GREY)

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
