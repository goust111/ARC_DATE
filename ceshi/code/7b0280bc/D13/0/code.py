from __future__ import annotations

# Variant of 7b0280bc
# Dimension: D13 — 终端-中间站基数关系 (meta)
# Change: 2个终端（1:1配对），其余全为中间站... → 多对多终端
#
# Design analysis for this variant
# --------------------------------
# a) Concrete pixel behavior and execution delta
# - Original input had exactly one terminal color with exactly 2 solid NxN squares.
#   All other node squares shared one station color, and one extra color was the track color.
# - This variant keeps the same visual vocabulary (solid squares = graph nodes, 8-connected
#   track components = graph edges), but now there are 4 terminal squares split into 2 pairs.
#   The two pairs are visibly encoded by using two distinct terminal colors, each appearing
#   exactly twice. Stations still share a single color. Tracks still share one color.
# - Output recolors the union of two shortest paths: for each same-colored terminal pair,
#   recolor the intermediate station squares on its unique shortest path to green and the
#   track components on that path to grey. The 4 terminals themselves remain unchanged.
# - Small execution trace:
#   original: node colors = {terminal: 2 copies, station: many copies}; solver picks that
#   unique 2-copy color and runs one BFS.
#   variant: node colors = {terminal_A: 2 copies, terminal_B: 2 copies, station: many};
#   solver must first group by color into two terminal pairs, then run BFS twice and recolor
#   the union of both results. The changed step is the color-to-role inference plus repeated
#   shortest-path extraction.
#
# b) Complexity calibration
# - The variant needs to be visibly different from the original, so each generated grid always
#   contains exactly 2 terminal colors with multiplicity 2. This gives repeated support inside
#   one example: the solver cannot ignore pairing because there are 4 terminals, not 2.
# - To avoid a visually ambiguous “all terminals look unrelated” situation, pairing uses only
#   visible color equality, not hidden ordering or coordinates.
# - To keep the task learnable from 2-4 examples, the number of terminal subclasses is fixed
#   at 2 rather than varying.
#
# c) Hidden-assumption audit of the original main()
# - `for col in node_colors: if len(nodes_by_color[col]) == 2: terminal_color = col`
#   assumed there is exactly one such color. In the variant, this would silently overwrite the
#   first terminal color with the second and lose one pair.
# - `assert not (terminal_color is None or station_color is None)` only checked existence of
#   one terminal color and one station color, not cardinality. The variant requires exactly
#   2 terminal-pair colors and exactly 1 station color.
# - `start, goal = term_ids[0], term_ids[1]` assumed there is only one pair to solve.
#   The variant must iterate over both same-colored terminal pairs.
# - Recoloring used `term_set` from a single pair. In the variant the exclusion set must be
#   the union of all 4 terminal nodes, otherwise a terminal from the other pair could be
#   mistakenly recolored if a path passed through it. The generator forbids that, but the
#   solver still guards against it.
#
# d) Placement legality
# - The generator still uses the original “graph first, then embed spatially” pattern.
# - It explicitly picks two node-disjoint target terminal paths in the abstract graph, connects
#   them into one connected tree using an extra connector path, then adds 1-2 cycle edges only
#   if both terminal-pair shortest paths remain unique and unchanged in length.
# - After spatial routing, the generator validates that every track component is still a single
#   8-connected edge touching exactly 2 node squares, so the solver sees the same abstract graph.
#
# e) Visual observability
# - The new rule is directly visible: there are 4 terminal squares, but they are split into
#   two same-colored pairs. Outputs consistently highlight two routes, not one.
# - Pairing is inferable without any hidden convention because “same color terminals belong
#   together” is repeated twice in the same grid and across examples.
#
# f) Visual cleanliness
# - The generator requires the two designated shortest paths to be node-disjoint, which makes
#   the two highlighted routes legible instead of collapsing into one tangled shared corridor.
# - Spatial routing rejects merged track components, preventing messy accidental unions.
#
# g) Color-change inferability
# - Input uses 5 colors total: background, terminal pair A, terminal pair B, stations, tracks.
#   Output still uses the same structured recolors as the original task: green only marks
#   intermediate stations on selected routes; grey only marks selected route tracks.
# - No sorting or pairing depends on numeric color ids; only visible equality classes matter.

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


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


def _shortest_node_edge_path(
    adj: dict[int, list[tuple[int, int]]], start: int, goal: int
) -> tuple[list[int], list[int]]:
    """Unique BFS tree path by edge count from start to goal."""
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}

    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))

    assert int(goal) in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        cur2 = prev.get(int(cur))
        cur = int(cur2) if cur2 is not None else None

    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest paths for two same-colored terminal pairs in a square-and-track graph.

    Input structure:
    - bg = most common color.
    - Three non-bg colors form solid NxN squares (N>=2):
      - two colors each mark exactly 2 terminals (2 terminal pairs),
      - one color marks all stations.
    - A fourth non-bg color forms 8-connected track components, each touching exactly 2 squares.

    Rule:
    - Pair terminals by color equality.
    - For each pair, find the unique shortest path (fewest graph edges) between its terminals.
    - Recolor every non-terminal square on either selected path to green.
    - Recolor every track component on either selected path to grey.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(N)]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "color": int(d["color"]), "pts": pts})

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    node_colors = sorted(nodes_by_color.keys())
    assert len(node_colors) >= 3

    terminal_pair_colors = sorted(
        int(col) for col in node_colors if len(nodes_by_color[int(col)]) == 2
    )
    station_colors = sorted(
        int(col) for col in node_colors if len(nodes_by_color[int(col)]) != 2
    )
    assert len(terminal_pair_colors) == 2
    assert len(station_colors) == 1

    terminal_pairs = [
        tuple(sorted(map(int, nodes_by_color[int(col)]))) for col in terminal_pair_colors
    ]

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [
        int(c) for c in non_bg_colors if int(c) not in set(map(int, node_colors))
    ]
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

    out = g.copy()
    all_terminal_ids = {int(nid) for pair in terminal_pairs for nid in pair}
    selected_nodes: set[int] = set()
    selected_edges: set[int] = set()

    for start, goal in terminal_pairs:
        node_path, edge_path = _shortest_node_edge_path(adj, int(start), int(goal))
        selected_nodes.update(int(nid) for nid in node_path)
        selected_edges.update(int(eid) for eid in edge_path)

    for nid in selected_nodes:
        if int(nid) in all_terminal_ids:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in selected_edges:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# --------------- generator ---------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (8, 10)               # 4 terminals + 4-6 stations
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]                # each terminal pair path length
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]   # excludes green=3, grey=5
CELL_PICK_RETRIES = 80
GEN_ATTEMPT_RETRIES = 800


def _compute_adjacency(
    picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    """Compute valid undirected cell-grid adjacencies."""
    edges = set()
    for i, j in picked:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (i + di, j + dj)
            if n in picked:
                edges.add(tuple(sorted(((i, j), n))))
        if allow_skip:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (i + di // 2, j + dj // 2)
                far = (i + di, j + dj)
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted(((i, j), far))))
        if use_diagonal:
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                n = (i + di, j + dj)
                corner1 = (i + di, j)
                corner2 = (i, j + dj)
                if n in picked and corner1 not in picked and corner2 not in picked:
                    edges.add(tuple(sorted(((i, j), n))))
    return edges


def _pick_cells(
    Kr: int,
    Kc: int,
    V: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
) -> set[tuple[int, int]] | None:
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


def _bfs_with_counts(
    src: int, allowed: set[tuple[int, int]], V: int
) -> tuple[dict[int, int], dict[int, int], dict[int, list[int]]]:
    """BFS from src; returns shortest distances, shortest-path counts, and adjacency list."""
    adj = {i: [] for i in range(V)}
    for a, b in allowed:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    dist = {int(src): 0}
    cnt = {int(src): 1}
    q = deque([int(src)])
    while q:
        u = int(q.popleft())
        for v in adj[u]:
            if int(v) not in dist:
                dist[int(v)] = int(dist[u]) + 1
                cnt[int(v)] = int(cnt[u])
                q.append(int(v))
            elif int(dist[v]) == int(dist[u]) + 1:
                cnt[int(v)] += int(cnt[u])
    return dist, cnt, adj


def _path_edges(path_nodes: tuple[int, ...] | list[int]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(path_nodes[i]), int(path_nodes[i + 1]))))
        for i in range(len(path_nodes) - 1)
    }


def _enumerate_simple_paths_of_length(
    adj: dict[int, list[int]], V: int, length: int
) -> list[tuple[int, ...]]:
    """Enumerate unique simple paths with exactly `length` edges."""
    out: set[tuple[int, ...]] = set()

    def dfs(path: list[int]) -> None:
        if len(path) - 1 == int(length):
            if int(path[0]) < int(path[-1]):
                out.add(tuple(map(int, path)))
            return
        u = int(path[-1])
        for v in adj[u]:
            if int(v) in path:
                continue
            path.append(int(v))
            dfs(path)
            path.pop()

    for start in range(V):
        dfs([int(start)])
    return list(out)


def _find_connector_path(
    adj: dict[int, list[int]],
    tree_nodes: set[int],
    other_path_nodes: set[int],
    rng: random.Random,
) -> list[int] | None:
    """
    Connect one node from `other_path_nodes` to any node in `tree_nodes`.

    Internal connector nodes must avoid both path node sets to keep the two target
    routes visually separate and to preserve the seeded tree structure.
    """
    starts = list(other_path_nodes)
    rng.shuffle(starts)
    for start in starts:
        q = deque([int(start)])
        prev: dict[int, int | None] = {int(start): None}
        while q:
            u = int(q.popleft())
            nbrs = list(adj[u])
            rng.shuffle(nbrs)
            for v in nbrs:
                vv = int(v)
                if vv in prev:
                    continue
                if vv in tree_nodes:
                    prev[vv] = int(u)
                    path: list[int] = []
                    cur: int | None = vv
                    while cur is not None:
                        path.append(int(cur))
                        cur = prev[int(cur)]
                    path.reverse()
                    if len(path) >= 2:
                        return path
                    continue
                if vv in other_path_nodes:
                    continue
                prev[vv] = int(u)
                q.append(vv)
    return None


def _pair_constraints_hold(
    allowed: set[tuple[int, int]], V: int, pair_specs: list[dict]
) -> bool:
    """Check both terminal pairs keep their intended unique shortest paths."""
    for spec in pair_specs:
        start, goal = map(int, spec["terminals"])
        dist, cnt, _ = _bfs_with_counts(int(start), allowed, int(V))
        if dist.get(int(goal)) != int(spec["path_len"]):
            return False
        if cnt.get(int(goal)) != 1:
            return False
    return True


def _build_graph_spatial(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    pair_path_lens: tuple[int, int],
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    """
    Build a connected graph with two visible terminal pairs and two disjoint target paths.

    Returns:
    - edges
    - pair_specs: [{terminals, path_nodes, path_edges, path_len}, ...]
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {
        tuple(sorted((int(cell_to_id[a]), int(cell_to_id[b])))) for a, b in cell_edges
    }
    adj = {i: [] for i in range(V)}
    for a, b in full_edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    path_cands_by_len = {
        int(L): _enumerate_simple_paths_of_length(adj, int(V), int(L))
        for L in sorted(set(map(int, pair_path_lens)))
    }
    for cands in path_cands_by_len.values():
        rng.shuffle(cands)

    len1, len2 = map(int, pair_path_lens)
    cands1 = path_cands_by_len[int(len1)]
    cands2 = path_cands_by_len[int(len2)]
    if not cands1 or not cands2:
        return None

    attempts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for p1 in cands1:
        s1 = set(map(int, p1))
        for p2 in cands2:
            s2 = set(map(int, p2))
            if s1 & s2:
                continue
            attempts.append((p1, p2))
    if not attempts:
        return None
    rng.shuffle(attempts)

    for path1, path2 in attempts:
        path1_nodes = tuple(map(int, path1))
        path2_nodes = tuple(map(int, path2))
        connector = _find_connector_path(
            adj,
            set(map(int, path1_nodes)),
            set(map(int, path2_nodes)),
            rng,
        )
        if connector is None:
            connector = _find_connector_path(
                adj,
                set(map(int, path2_nodes)),
                set(map(int, path1_nodes)),
                rng,
            )
            if connector is None:
                continue
            path1_nodes, path2_nodes = path2_nodes, path1_nodes

        tree_edges = set()
        tree_edges.update(_path_edges(path1_nodes))
        tree_edges.update(_path_edges(path2_nodes))
        tree_edges.update(_path_edges(connector))

        in_tree = set(map(int, path1_nodes)) | set(map(int, path2_nodes)) | set(map(int, connector))
        while len(in_tree) < V:
            boundary = [
                (int(u), int(v))
                for u in in_tree
                for v in adj[int(u)]
                if int(v) not in in_tree
            ]
            if not boundary:
                break
            u, v = rng.choice(boundary)
            tree_edges.add(tuple(sorted((int(u), int(v)))))
            in_tree.add(int(v))
        if len(in_tree) != V:
            continue

        pair_specs = [
            {
                "terminals": (int(path1_nodes[0]), int(path1_nodes[-1])),
                "path_nodes": list(map(int, path1_nodes)),
                "path_edges": sorted(_path_edges(path1_nodes)),
                "path_len": int(len(path1_nodes) - 1),
            },
            {
                "terminals": (int(path2_nodes[0]), int(path2_nodes[-1])),
                "path_nodes": list(map(int, path2_nodes)),
                "path_edges": sorted(_path_edges(path2_nodes)),
                "path_len": int(len(path2_nodes) - 1),
            },
        ]

        if not _pair_constraints_hold(tree_edges, int(V), pair_specs):
            continue

        extra_candidates = list(full_edges - tree_edges)
        rng.shuffle(extra_candidates)
        final_edges = set(tree_edges)
        added = 0
        for e in extra_candidates:
            if added >= int(n_cycles):
                break
            cand = set(final_edges)
            cand.add(tuple(sorted((int(e[0]), int(e[1])))))
            if _pair_constraints_hold(cand, int(V), pair_specs):
                final_edges = cand
                added += 1
        if added != int(n_cycles):
            continue

        if not _pair_constraints_hold(final_edges, int(V), pair_specs):
            continue
        if len(final_edges) != len(tree_edges) + int(n_cycles):
            continue

        return sorted(final_edges), pair_specs
    return None


def _cell_to_square_pos(
    cell: tuple[int, int], N: int, cell_size: int, origin: tuple[int, int]
) -> tuple[int, int]:
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return int(or_r + i * cell_size + margin), int(or_c + j * cell_size + margin)


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


def _walk_footprints(sr: int, sc: int, er: int, ec: int) -> list[tuple[int, int]]:
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
    N: int,
) -> list[tuple[int, int]]:
    """Route an (N-1)-wide 8-connected track between two square nodes."""
    Wt = int(N - 1)
    ra, ca = map(int, pos_a)
    rb, cb = map(int, pos_b)
    dci, dcj = int(cell_b[0] - cell_a[0]), int(cell_b[1] - cell_a[1])
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(ra, ca, int(N), int(Wt), int(si), int(sj))
    er, ec = _dock_tl(rb, cb, int(N), int(Wt), -int(si), -int(sj))

    pts_tl = _walk_footprints(int(sr), int(sc), int(er), int(ec))
    pixels = set()
    for r, c in pts_tl:
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((int(r + dr), int(c + dc)))
    return list(pixels)


def _validate_embedded_tracks(
    track_mask: np.ndarray,
    id_to_pos: dict[int, tuple[int, int]],
    N: int,
    edges: set[tuple[int, int]],
) -> bool:
    """Ensure each track component touches exactly one intended node pair."""
    pix2node: dict[tuple[int, int], int] = {}
    for nid, (r0, c0) in id_to_pos.items():
        for r in range(int(r0), int(r0) + int(N)):
            for c in range(int(c0), int(c0) + int(N)):
                pix2node[(int(r), int(c))] = int(nid)

    seen_edges: set[tuple[int, int]] = set()
    comps = components_mask(track_mask, connectivity=8)
    for coords in comps:
        touched: set[int] = set()
        for r, c in coords:
            rr0, cc0 = int(r), int(c)
            for rr, cc in neighbors8(int(rr0), int(cc0)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            return False
        e = tuple(sorted(touched))
        if e not in edges or e in seen_edges:
            return False
        seen_edges.add(e)
    return seen_edges == set(map(tuple, edges))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = int(rng.choice(NODE_SIZE_CHOICES))
    V = int(rng.randint(*NODE_COUNT_RANGE))
    n_cycles = int(rng.choice(CYCLE_COUNT_CHOICES))
    pair_path_lens = (
        int(rng.choice(PATH_LEN_CHOICES)),
        int(rng.choice(PATH_LEN_CHOICES)),
    )

    gap = int(rng.choice(GAP_CHOICES))
    cell_size = int(N + gap)

    KrKc_options = []
    for Kr in CELL_GRID_DIMS:
        for Kc in CELL_GRID_DIMS:
            if int(Kr * Kc) < int(V):
                continue
            min_S = int(max(Kr, Kc) * cell_size + 2)
            if min_S > GRID_SIZE_RANGE[1]:
                continue
            KrKc_options.append((int(Kr), int(Kc), int(min_S)))
    if not KrKc_options:
        return None
    Kr, Kc, min_S = rng.choice(KrKc_options)
    S = int(rng.randint(max(GRID_SIZE_RANGE[0], int(min_S)), GRID_SIZE_RANGE[1]))

    use_diagonal = False
    allow_skip = True
    picked = _pick_cells(
        int(Kr), int(Kc), int(V), rng, use_diagonal=use_diagonal, allow_skip=allow_skip
    )
    if picked is None:
        return None

    picked_list = list(picked)
    rng.shuffle(picked_list)
    cell_to_id = {cell: int(i) for i, cell in enumerate(picked_list)}

    graph = _build_graph_spatial(
        picked,
        cell_to_id,
        rng,
        pair_path_lens=pair_path_lens,
        n_cycles=int(n_cycles),
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if graph is None:
        return None
    edges, pair_specs = graph

    id_to_cell = {int(v): cell for cell, v in cell_to_id.items()}
    pair_terminals = [tuple(map(int, spec["terminals"])) for spec in pair_specs]
    all_terminal_ids = {int(nid) for pair in pair_terminals for nid in pair}

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    five = pool[:5]
    bg = int(rng.choice(five))
    rest = [int(c) for c in five if int(c) != int(bg)]
    rng.shuffle(rest)
    terminal_color_a, terminal_color_b, station_color, track_color = map(int, rest[:4])

    canvas = np.full((S, S), int(bg), dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = int(S - Kr * cell_size - 2)
    slack_c = int(S - Kc * cell_size - 2)
    origin = (
        int(1 + rng.randint(0, max(0, slack_r))),
        int(1 + rng.randint(0, max(0, slack_c))),
    )

    term_color_by_id: dict[int, int] = {}
    for nid in pair_terminals[0]:
        term_color_by_id[int(nid)] = int(terminal_color_a)
    for nid in pair_terminals[1]:
        term_color_by_id[int(nid)] = int(terminal_color_b)

    id_to_pos: dict[int, tuple[int, int]] = {}
    for i in range(V):
        cell = id_to_cell[int(i)]
        pos = _cell_to_square_pos(cell, int(N), int(cell_size), origin)
        id_to_pos[int(i)] = pos
        color = int(term_color_by_id.get(int(i), int(station_color)))
        r0, c0 = map(int, pos)
        canvas[r0 : r0 + N, c0 : c0 + N] = int(color)
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(
            id_to_cell[int(a)],
            id_to_cell[int(b)],
            id_to_pos[int(a)],
            id_to_pos[int(b)],
            int(N),
        )
        for r, c in pixels:
            if not (0 <= int(r) < S and 0 <= int(c) < S):
                return None
            if node_mask[int(r), int(c)] or track_mask[int(r), int(c)]:
                return None
        for r, c in pixels:
            canvas[int(r), int(c)] = int(track_color)
            track_mask[int(r), int(c)] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = list(pixels)

    if not _validate_embedded_tracks(track_mask, id_to_pos, int(N), set(edges)):
        return None

    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = canvas.copy()
    selected_edge_set: set[tuple[int, int]] = set()
    for spec in pair_specs:
        for nid in spec["path_nodes"]:
            if int(nid) in all_terminal_ids:
                continue
            r0, c0 = id_to_pos[int(nid)]
            expected[int(r0) : int(r0) + N, int(c0) : int(c0) + N] = int(Color.GREEN)
        selected_edge_set.update(tuple(sorted(map(int, e))) for e in spec["path_edges"])

    for e in selected_edge_set:
        for r, c in edge_track_pixels[tuple(sorted((int(e[0]), int(e[1]))))]:
            expected[int(r), int(c)] = int(Color.GREY)

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
