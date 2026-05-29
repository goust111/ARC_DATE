# Variant of 7b0280bc
# Dimension: D5 — 终点/停止条件 (dst)
# Change: 另一个少数色终端方块为终点，最短路径到达即停止... → 颜色匹配终点

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   core_change:
#     original: exactly two same-color terminal squares; BFS connects that fixed pair
#     variant: one marked start square must connect to the goal-candidate square whose
#       embedded center pixel has the same color as the start's embedded center pixel
#   visible_encoding:
#     node_size: fixed to 3x3
#     reason: the new rule needs a genuinely embedded pixel; 2x2 has no clear interior
#     start: one 3x3 square of a unique body color with a 1x1 center marker
#     goals: three 3x3 squares of another body color, each with a 1x1 center marker
#     stations: several solid 3x3 squares, unchanged from the original task spirit
#     tracks: unchanged; 2-pixel-wide routed segments linking node squares
#   behavior_trace:
#     original_flow:
#       - detect solid same-size squares
#       - choose the only color that has count 2 as the terminal color
#       - BFS between those two terminals
#     variant_flow:
#       - still detect same-size node squares, but terminals are now "almost solid":
#         8 body pixels + 1 center marker pixel
#       - identify the unique marked start-body square
#       - identify the goal-body squares
#       - read the start marker color and select the unique goal whose center marker
#         matches it
#       - BFS only to that matched goal
#     concrete_difference:
#       - if the start's marker is red and the three goal candidates carry markers
#         [blue, red, blue], the solver must ignore the closer blue-marked goals and
#         stop at the red-marked one
#   complexity_calibration:
#     - three goal candidates keep the color-match rule visible and non-trivial
#     - two marker colors are reused across candidates, so the rule is inferable from
#       a few examples instead of depending on many unrelated colors
#     - generator enforces at least one distractor goal that is strictly closer to the
#       start than the matched goal; this prevents degeneration into "pick any goal"
#   implicit_assumption_audit:
#     - original find_solid_squares was safe only because every node was monochrome;
#       now it can detect stations but not terminals, so marked terminals must be
#       scanned separately from raw 3x3 windows
#     - original terminal-color inference relied on "one node color appears exactly 2
#       times"; that is false here because the goal body color appears 3 times and the
#       start body color appears once
#     - original track-color inference excluded the two node colors; that is false here
#       because marker colors are additional non-bg colors, so track color must be read
#       from non-node pixels only
#     - recoloring all non-terminal nodes on the chosen path is still safe only if
#       distractor goals are guaranteed off the matched path; the generator enforces it
#   placement_legality:
#     - all nodes occupy disjoint 3x3 cells on a coarse grid
#     - tracks are routed exactly as in the original task and rejected if they overlap
#       nodes, overlap other tracks, or leave the canvas
#     - goal candidates are chosen from off-path graph nodes, so their unchanged bodies
#       do not visually break the highlighted shortest path
#   visual_observability:
#     - the only new signal is the center pixel, but fixing nodes to 3x3 makes it
#       large and unmistakable
#     - start and goal candidates have different body colors, so the solver's source
#       and the set of possible destinations are easy to read by eye
#   color_change_inferability:
#     - marker colors are drawn from just two repeated colors
#     - exactly one goal repeats the start marker color
#     - output changes only along the path to that matched goal, making the criterion
#       recoverable from 2-3 examples


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """Find solid monochrome NxN squares."""
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


def _find_marked_squares(grid: np.ndarray, N: int, bg: int, occupied: np.ndarray) -> list[dict]:
    """
    Detect 3x3 terminal squares of the form:
    - 8 cells of one body color
    - 1 center cell of a marker color
    """
    H, W = grid.shape
    center = (N // 2, N // 2)
    out: list[dict] = []
    used = occupied.copy()

    for r0 in range(H - N + 1):
        for c0 in range(W - N + 1):
            if bool(used[r0 : r0 + N, c0 : c0 + N].any()):
                continue
            patch = grid[r0 : r0 + N, c0 : c0 + N]
            if np.any(patch == int(bg)):
                continue
            marker = int(patch[center])
            if int((patch == marker).sum()) != 1:
                continue
            body_vals = [int(v) for v in patch.flatten() if int(v) != marker]
            if len(set(body_vals)) != 1:
                continue
            body = int(body_vals[0])
            coords = np.array(
                [(rr, cc) for rr in range(r0, r0 + N) for cc in range(c0, c0 + N)],
                dtype=int,
            )
            out.append(
                {
                    "color": body,
                    "marker_color": marker,
                    "bbox": (int(r0), int(c0), int(r0 + N - 1), int(c0 + N - 1)),
                    "coords": coords,
                }
            )
            used[r0 : r0 + N, c0 : c0 + N] = True
    return out


def _trace_path(src: int, dst: int, adj: dict[int, list[int]]) -> list[int] | None:
    parent: dict[int, int | None] = {int(src): None}
    q = deque([int(src)])
    while q:
        u = int(q.popleft())
        if u == int(dst):
            break
        for v in adj.get(int(u), []):
            if int(v) in parent:
                continue
            parent[int(v)] = int(u)
            q.append(int(v))
    if int(dst) not in parent:
        return None
    path: list[int] = []
    cur: int | None = int(dst)
    while cur is not None:
        path.append(int(cur))
        prev = parent.get(int(cur))
        cur = int(prev) if prev is not None else None
    path.reverse()
    return path


def _bfs_distances(adj: dict[int, list[int]], src: int) -> dict[int, int]:
    """Unweighted BFS distances on a simple adjacency list."""
    dist = {int(src): 0}
    q = deque([int(src)])
    while q:
        u = int(q.popleft())
        for v in adj.get(int(u), []):
            if int(v) in dist:
                continue
            dist[int(v)] = int(dist[int(u)] + 1)
            q.append(int(v))
    return dist


def _build_instance_graph(grid: np.ndarray) -> dict:
    """
    Parse nodes, identify the marker-matched destination, and build the node graph.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    solid = find_solid_squares(g, bg=bg, min_size=3, connectivity=8)
    assert solid, "no solid station squares found"
    size_counts = Counter(int(d["size"]) for d in solid)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    assert N >= 3, "terminal markers require an interior pixel"

    solid_nodes = [d for d in solid if int(d["size"]) == N]
    station_color_counts = Counter(int(d["color"]) for d in solid_nodes)
    station_color = int(station_color_counts.most_common(1)[0][0])
    station_nodes = [d for d in solid_nodes if int(d["color"]) == station_color]
    assert len(station_nodes) >= 1, "station nodes missing"

    occupied = np.zeros_like(g, dtype=bool)
    for d in station_nodes:
        r0, c0, r1, c1 = d["bbox"]
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    marked_nodes = _find_marked_squares(g, N, bg, occupied)
    assert marked_nodes, "no marked terminals found"

    marked_by_body: dict[int, list[dict]] = defaultdict(list)
    for d in marked_nodes:
        marked_by_body[int(d["color"])].append(d)
    assert len(marked_by_body) == 2, "expected one start body color and one goal body color"

    start_body_colors = [col for col, items in marked_by_body.items() if len(items) == 1]
    goal_body_colors = [col for col, items in marked_by_body.items() if len(items) >= 2]
    assert len(start_body_colors) == 1 and len(goal_body_colors) == 1, "terminal role counts invalid"

    start_raw = marked_by_body[int(start_body_colors[0])][0]
    goal_raw = list(marked_by_body[int(goal_body_colors[0])])
    start_marker = int(start_raw["marker_color"])
    matched_goals = [d for d in goal_raw if int(d["marker_color"]) == start_marker]
    assert len(matched_goals) == 1, "expected a unique marker-matched goal"
    matched_raw = matched_goals[0]

    nodes: list[dict] = []

    def _append_node(d: dict, role: str) -> int:
        nid = len(nodes)
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        center = (int(d["bbox"][0] + N // 2), int(d["bbox"][1] + N // 2))
        nodes.append(
            {
                "id": int(nid),
                "role": role,
                "body_color": int(d["color"]),
                "marker_color": int(d.get("marker_color")) if "marker_color" in d else None,
                "bbox": tuple(map(int, d["bbox"])),
                "center": center,
                "pts": pts,
            }
        )
        return int(nid)

    for d in station_nodes:
        _append_node(d, "station")
    start_id = _append_node(start_raw, "start")
    goal_candidate_ids: list[int] = []
    goal_match_id: int | None = None
    matched_bbox = tuple(map(int, matched_raw["bbox"]))
    for d in goal_raw:
        nid = _append_node(d, "goal")
        goal_candidate_ids.append(int(nid))
        if tuple(map(int, d["bbox"])) == matched_bbox:
            goal_match_id = int(nid)
    assert goal_match_id is not None, "matched goal node missing after assembly"

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    node_mask = np.zeros_like(g, dtype=bool)
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        for p in nd["pts"]:
            pix2node[p] = nid
            node_mask[p] = True

    non_bg = colors_present(g, exclude={bg})
    track_candidates = [int(c) for c in non_bg if np.any((g == int(c)) & (~node_mask))]
    assert len(track_candidates) == 1, "expected a unique non-node track color"
    track_color = int(track_candidates[0])

    track_comps = components_mask(g == track_color, connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    adj_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    simple_adj: dict[int, list[int]] = defaultdict(list)

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
        adj_edges[int(a)].append((int(b), int(edge_id)))
        adj_edges[int(b)].append((int(a), int(edge_id)))
        simple_adj[int(a)].append(int(b))
        simple_adj[int(b)].append(int(a))

    return {
        "grid": g,
        "bg": bg,
        "N": N,
        "nodes": nodes,
        "id2pts": id2pts,
        "track_color": track_color,
        "track_comp_pts": track_comp_pts,
        "adj": adj_edges,
        "simple_adj": simple_adj,
        "start_id": int(start_id),
        "goal_id": int(goal_match_id),
        "goal_candidate_ids": list(map(int, goal_candidate_ids)),
        "start_marker": int(start_marker),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique shortest path from the marked start square to the goal square
    whose embedded center color matches the start's center color.
    """
    info = _build_instance_graph(np.asarray(grid, dtype=int))
    g = info["grid"]
    adj = info["adj"]
    start = int(info["start_id"])
    goal = int(info["goal_id"])

    q = deque([start])
    prev: dict[int, int | None] = {start: None}
    prev_edge: dict[int, int | None] = {start: None}

    while q:
        u = int(q.popleft())
        if u == goal:
            break
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))

    assert goal in prev, "matched goal unreachable"

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
    for nid in node_path:
        role = str(info["nodes"][int(nid)]["role"])
        if role != "station":
            continue
        for r, c in info["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in info["track_comp_pts"].get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

NODE_SIZE_CHOICES = [3]                  # embedded center pixel must be visible
NODE_COUNT_RANGE = (8, 10)               # 1 start + 3 goals + 4-6 stations
GOAL_CANDIDATE_COUNT = 3
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]   # excludes green=3, grey=5
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 400


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    """Compute valid undirected edges between picked cells on a coarse grid."""
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


def _bfs_with_counts(src: int, allowed: set[tuple[int, int]], V: int):
    """BFS from src over allowed undirected edges; also count shortest paths."""
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
                dist[int(v)] = int(dist[int(u)] + 1)
                cnt[int(v)] = int(cnt[int(u)])
                q.append(int(v))
            elif int(dist[int(v)]) == int(dist[int(u)] + 1):
                cnt[int(v)] = int(cnt[int(v)] + cnt[int(u)])
    return dist, cnt, adj


def _search_start_goal_pair(
    full_edges: set[tuple[int, int]],
    V: int,
    path_len: int,
    rng: random.Random,
):
    dist_cache: dict[int, tuple[dict[int, int], dict[int, int], dict[int, list[int]]]] = {}
    ids = list(range(V))
    rng.shuffle(ids)
    for src in ids:
        if int(src) not in dist_cache:
            dist_cache[int(src)] = _bfs_with_counts(int(src), full_edges, V)
        dist, cnt, adj = dist_cache[int(src)]
        dsts = [v for v in ids if v != src and dist.get(int(v)) == int(path_len) and cnt.get(int(v)) == 1]
        rng.shuffle(dsts)
        for dst in dsts:
            path = _trace_path(int(src), int(dst), adj)
            if not path:
                continue
            off_path = [v for v in range(V) if v not in set(path)]
            closer = [v for v in off_path if dist.get(int(v), 10**9) < int(path_len)]
            if len(off_path) >= GOAL_CANDIDATE_COUNT - 1 and closer:
                return int(src), int(dst), list(map(int, path))
    return None


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
    Build a connected graph whose marked-goal path is unique, and with off-path nodes
    available as distractor goals.
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in cell_edges}

    pair = _search_start_goal_pair(full_edges, V, path_len, rng)
    if pair is None:
        return None
    start, matched, sp_nodes = pair

    _, _, full_adj = _bfs_with_counts(start, full_edges, V)
    sp_edges = {
        tuple(sorted((int(sp_nodes[i]), int(sp_nodes[i + 1]))))
        for i in range(len(sp_nodes) - 1)
    }

    tree_edges = set(sp_edges)
    in_tree = set(sp_nodes)
    frontier = list(sp_nodes)
    while in_tree != set(range(V)):
        next_frontier = []
        for u in frontier:
            nbrs = list(full_adj[int(u)])
            rng.shuffle(nbrs)
            for v in nbrs:
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
        cand = final_edges | {tuple(sorted((int(e[0]), int(e[1]))))}
        dist2, cnt2, _ = _bfs_with_counts(start, cand, V)
        if dist2.get(matched) == path_len and cnt2.get(matched) == 1:
            final_edges = cand
            added += 1
    if added != n_cycles:
        return None

    dist3, cnt3, final_adj = _bfs_with_counts(start, final_edges, V)
    if dist3.get(matched) != path_len or cnt3.get(matched) != 1:
        return None
    if len(final_edges) != V - 1 + n_cycles:
        return None

    final_path = _trace_path(start, matched, final_adj)
    if not final_path or len(final_path) - 1 != path_len:
        return None
    path_set = set(final_path)
    off_path = [v for v in range(V) if v not in path_set]
    closer = [v for v in off_path if dist3.get(int(v), 10**9) < path_len]
    if not closer or len(off_path) < GOAL_CANDIDATE_COUNT - 1:
        return None

    rng.shuffle(closer)
    distractors = [int(closer[0])]
    rest = [v for v in off_path if v != distractors[0]]
    rng.shuffle(rest)
    for v in rest:
        distractors.append(int(v))
        if len(distractors) == GOAL_CANDIDATE_COUNT - 1:
            break
    if len(distractors) != GOAL_CANDIDATE_COUNT - 1:
        return None

    id_map = {int(start): 0, int(matched): V - 1}
    next_ids = iter(range(1, V - 1))
    for old in range(V):
        if old in id_map:
            continue
        id_map[int(old)] = int(next(next_ids))

    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for a, b in final_edges}
    remapped_cell_to_id = {cell: int(id_map[cell_to_id[cell]]) for cell in cell_to_id}
    new_path_nodes = [int(id_map[n]) for n in final_path]
    new_path_edges = [
        tuple(sorted((int(new_path_nodes[i]), int(new_path_nodes[i + 1]))))
        for i in range(len(new_path_nodes) - 1)
    ]
    new_distractors = [int(id_map[d]) for d in distractors]

    return (
        list(remapped_edges),
        new_path_nodes,
        new_path_edges,
        remapped_cell_to_id,
        int(id_map[matched]),
        new_distractors,
    )


def _cell_to_square_pos(cell: tuple[int, int], N: int, cell_size: int, origin: tuple[int, int]):
    """Top-left corner of the NxN square centered in the given cell."""
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return (or_r + i * cell_size + margin, or_c + j * cell_size + margin)


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
    """Generate the sequence of footprint top-lefts from (sr,sc) to (er,ec)."""
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
    """Route a width-(N-1) track strip between two node squares."""
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
                pixels.add((int(r + dr), int(c + dc)))
    return list(pixels)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
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
    edges, path_nodes, path_edges, cell_to_id, matched_goal, distractor_goals = graph
    id_to_cell = {v: k for k, v in cell_to_id.items()}

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    bg, start_body, goal_body, station_color, track_color, marker_a, marker_b = colors[:7]
    shared_marker = rng.choice([marker_a, marker_b])
    distractor_marker = marker_b if shared_marker == marker_a else marker_a

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))
    id_to_pos: dict[int, tuple[int, int]] = {}

    goal_set = set(map(int, [matched_goal, *distractor_goals]))
    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[int(i)] = pos
        r0, c0 = pos

        if i == 0:
            canvas[r0 : r0 + N, c0 : c0 + N] = start_body
            canvas[r0 + N // 2, c0 + N // 2] = shared_marker
        elif i in goal_set:
            canvas[r0 : r0 + N, c0 : c0 + N] = goal_body
            marker = shared_marker if i == matched_goal else distractor_marker
            canvas[r0 + N // 2, c0 + N // 2] = marker
        else:
            canvas[r0 : r0 + N, c0 : c0 + N] = station_color
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    if any(n in goal_set for n in path_nodes[1:-1]):
        return None

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N, rng)
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if bool(node_mask[r, c]) or bool(track_mask[r, c]):
                return None
        for r, c in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels

    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = canvas.copy()
    for nid in path_nodes:
        if nid in (0, matched_goal):
            continue
        r0, c0 = id_to_pos[int(nid)]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for e in path_edges:
        for r, c in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = int(Color.GREY)

    try:
        actual = main(canvas)
        parsed = _build_instance_graph(canvas)
    except Exception:
        return None

    if len(parsed["goal_candidate_ids"]) != GOAL_CANDIDATE_COUNT:
        return None
    if len(parsed["nodes"]) != V:
        return None
    dist = _bfs_distances(parsed["simple_adj"], int(parsed["start_id"]))
    if int(dist.get(int(parsed["goal_id"]), -1)) != int(path_len):
        return None
    closer = [
        g
        for g in parsed["goal_candidate_ids"]
        if int(g) != int(parsed["goal_id"]) and int(dist.get(int(g), 10**9)) < int(dist[int(parsed["goal_id"])])
    ]
    if not closer:
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
