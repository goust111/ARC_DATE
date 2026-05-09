# Variant of 7b0280bc
# Dimension: D2 — 路径导航方式 (path)
# Change: 沿轨道像素通过8-邻接连接的方块节点间行进... → 方向约束导航
#
# Design analysis
# ===============
# variant_spec:
#   input_objects:
#     grid: square, 26..30 on a side
#     nodes:
#       shape: 3x3 squares only
#       colors:
#         terminal_fill: exactly 2 nodes
#         station_fill: remaining 5..7 nodes
#         marker: one shared special color used on exactly one pixel inside every node
#       marker_positions:
#         U: (0,1)  top-middle
#         D: (2,1)  bottom-middle
#         L: (1,0)  middle-left
#         R: (1,2)  middle-right
#       semantics:
#         the marker says which face the path is allowed to leave from
#     tracks:
#       color: one shared track color
#       geometry: same graph embedding idea as the original task
#       width: 1-pixel centerline
#       connectivity: 8-connected components, each touching exactly 2 node squares
#   output_rule:
#     terminals remain unchanged
#     stations on the chosen directed terminal-to-terminal path become green
#     traversed track components on that path become grey
#
#   behavior_trace:
#     original:
#       - every track touching two nodes is an undirected graph edge
#       - BFS between the two terminals ignores how each node looks internally
#     variant:
#       - a track is still an undirected physical connection
#       - but it can only be traversed out of node A if A's marker faces the side touched by that track
#       - therefore the graph used by BFS is directed, derived from node markers + track/node incidence
#     concrete_example:
#       - if terminal T touches tracks on its left and right, and its marker is on the right edge,
#         only the right track is usable when leaving T
#       - a visually shorter left-side route may exist but must be ignored
#
#   complexity_calibration:
#     - Node size is forced to 3 instead of {2,3}. This is the minimal adaptation needed to make
#       four departure directions visually unambiguous; 2x2 nodes cannot host a clear side-center marker.
#     - The marker color is shared across all nodes, so humans can notice it as a system-level cue.
#     - Track width is reduced to 1 pixel. This is a compatibility adjustment: with 3x3 directed nodes,
#       the old 2-pixel staggered rails too often fuse into the same 8-connected component, which hides
#       the intended graph structure instead of clarifying the new direction rule.
#     - Generator enforces that the directed solution path is strictly longer than the undirected
#       shortest path. This guarantees the new D2 behavior materially changes the answer.
#
#   implicit_assumption_audit:
#     original_code_lines_affected:
#       - find_solid_squares(...) assumed nodes were monochrome solids. This is no longer true because
#         every node contains one marker pixel of a different color.
#       - node color classification still works, but must now use the majority color inside a node.
#       - edge construction previously stored only which two nodes a track touched; now it must also
#         record which face of each node the component touches.
#       - shortest-path BFS previously ran on an undirected adjacency list; now it must run on a
#         directed adjacency list filtered by each node's marker direction.
#     new_solver_assumptions:
#       - every node is a 3x3 two-color patch with counts 8+1
#       - the minority pixel is on one of four side-center cells
#       - exactly one marker color is used across all nodes
#       - each incident track touches a node on exactly one face
#     generator_guarantees:
#       - all nodes satisfy the 3x3 / 8+1 pattern
#       - terminals share one fill color, stations share another
#       - every used edge has a well-defined touched face at each endpoint
#       - exactly one terminal can reach the other through the directed graph
#
#   placement_legality_and_visual_cleanliness:
#     - nodes remain separated by background cells
#     - tracks never overlap nodes or each other
#     - markers stay inside node squares and are overwritten only when a station is recolored green
#     - only five input colors are used: bg, terminal, station, marker, track
#     - no invisible color ordering or numeric-value logic is used

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


NODE_SIZE = 3
NODE_COUNT_RANGE = (7, 9)
CYCLE_COUNT_CHOICES = [1]
DIRECTED_PATH_LEN_CHOICES = [3]
GAP_CHOICES = [3]
CELL_GRID_DIMS = (4,)
GRID_SIZE_RANGE = (26, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
CELL_PICK_RETRIES = 60
DIR_ASSIGN_RETRIES = 80
GRAPH_ATTEMPT_RETRIES = 80
GEN_ATTEMPT_RETRIES = 400

DIR4 = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
DIR_TO_MARKER = {
    "U": (0, 1),
    "D": (2, 1),
    "L": (1, 0),
    "R": (1, 2),
}
MARKER_TO_DIR = {v: k for k, v in DIR_TO_MARKER.items()}


def _bbox_points(r0: int, c0: int, n: int = NODE_SIZE) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(n) for dc in range(n)}


def _detect_nodes(grid: np.ndarray, bg: int) -> tuple[list[dict], int, int, int]:
    """Detect 3x3 node squares with an 8+1 fill/marker pattern."""
    H, W = grid.shape
    candidates: list[dict] = []
    for r0 in range(H - NODE_SIZE + 1):
        for c0 in range(W - NODE_SIZE + 1):
            patch = np.array(grid[r0:r0 + NODE_SIZE, c0:c0 + NODE_SIZE], dtype=int)
            vals, counts = np.unique(patch, return_counts=True)
            vals = [int(v) for v in vals]
            counts = [int(v) for v in counts]
            if bg in vals or len(vals) != 2 or sorted(counts) != [1, 8]:
                continue
            marker_idx = 0 if counts[0] == 1 else 1
            fill_idx = 1 - marker_idx
            marker_rel = tuple(int(x) for x in np.argwhere(patch == vals[marker_idx])[0])
            if marker_rel not in MARKER_TO_DIR:
                continue
            candidates.append(
                {
                    "bbox": (r0, c0, r0 + NODE_SIZE - 1, c0 + NODE_SIZE - 1),
                    "fill_color": int(vals[fill_idx]),
                    "marker_color": int(vals[marker_idx]),
                    "direction": MARKER_TO_DIR[marker_rel],
                    "pts": _bbox_points(r0, c0),
                }
            )

    grouped: dict[int, list[dict]] = defaultdict(list)
    for cand in candidates:
        grouped[int(cand["marker_color"])].append(cand)

    best_group: list[dict] | None = None
    best_meta: tuple[int, int, int] | None = None
    for marker_color, group in grouped.items():
        fill_counts = Counter(int(d["fill_color"]) for d in group)
        terminal_cols = [col for col, cnt in fill_counts.items() if int(cnt) == 2]
        station_cols = [col for col, cnt in fill_counts.items() if int(cnt) >= 5]
        if len(terminal_cols) != 1 or len(station_cols) != 1:
            continue
        terminal_color = int(terminal_cols[0])
        station_color = int(station_cols[0])
        filtered = [
            d for d in group
            if int(d["fill_color"]) in {terminal_color, station_color}
        ]
        if not (NODE_COUNT_RANGE[0] <= len(filtered) <= NODE_COUNT_RANGE[1]):
            continue
        score = (len(filtered), int(fill_counts[station_color]), int(marker_color))
        if best_meta is None or score > best_meta:
            best_meta = score
            best_group = filtered

    assert best_group is not None and best_meta is not None

    terminal_color = next(
        int(col) for col, cnt in Counter(int(d["fill_color"]) for d in best_group).items()
        if int(cnt) == 2
    )
    station_color = next(
        int(col) for col, cnt in Counter(int(d["fill_color"]) for d in best_group).items()
        if int(cnt) >= 5
    )
    marker_color = int(best_group[0]["marker_color"])

    nodes: list[dict] = []
    for i, d in enumerate(sorted(best_group, key=lambda x: x["bbox"])):
        nodes.append(
            {
                "id": int(i),
                "fill_color": int(d["fill_color"]),
                "direction": str(d["direction"]),
                "bbox": tuple(int(v) for v in d["bbox"]),
                "pts": set(d["pts"]),
            }
        )
    return nodes, terminal_color, station_color, marker_color


def _track_touch_side(track_pt: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = track_pt
    r0, c0, r1, c1 = bbox
    if r < r0 and c0 <= c <= c1:
        return "U"
    if r > r1 and c0 <= c <= c1:
        return "D"
    if c < c0 and r0 <= r <= r1:
        return "L"
    if c > c1 and r0 <= r <= r1:
        return "R"
    return None


def _reconstruct_path(
    goal: int,
    prev: dict[int, int | None],
    prev_edge: dict[int, int | None],
) -> tuple[list[int], list[int]]:
    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        edge_id = prev_edge.get(int(cur))
        if edge_id is not None:
            edge_path.append(int(edge_id))
        parent = prev.get(int(cur))
        cur = int(parent) if parent is not None else None
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def _parse_graph(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    nodes, terminal_color, station_color, marker_color = _detect_nodes(g, bg)
    term_ids = [int(nd["id"]) for nd in nodes if int(nd["fill_color"]) == terminal_color]
    assert len(term_ids) == 2

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [
        int(c) for c in non_bg_colors
        if int(c) not in {terminal_color, station_color, marker_color}
    ]
    assert len(track_candidates) == 1
    track_color = int(track_candidates[0])

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    id2bbox: dict[int, tuple[int, int, int, int]] = {}
    id2dir: dict[int, str] = {}
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        id2bbox[nid] = tuple(int(v) for v in nd["bbox"])
        id2dir[nid] = str(nd["direction"])
        for p in nd["pts"]:
            pix2node[p] = nid

    track_comps = components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    directed_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    undirected_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue

        touched: set[int] = set()
        side_hits: dict[int, set[str]] = defaultdict(set)
        for r, c in pts:
            for rr, cc in neighbors8(r, c):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is None:
                    continue
                side = _track_touch_side((r, c), id2bbox[int(nid)])
                if side is None:
                    continue
                touched.add(int(nid))
                side_hits[int(nid)].add(side)

        if len(touched) != 2:
            continue
        if any(len(side_hits[int(nid)]) != 1 for nid in touched):
            continue

        a, b = sorted(touched)
        side_a = next(iter(side_hits[a]))
        side_b = next(iter(side_hits[b]))
        track_comp_pts[int(edge_id)] = pts
        undirected_adj[int(a)].append((int(b), int(edge_id)))
        undirected_adj[int(b)].append((int(a), int(edge_id)))
        if side_a == id2dir[int(a)]:
            directed_adj[int(a)].append((int(b), int(edge_id)))
        if side_b == id2dir[int(b)]:
            directed_adj[int(b)].append((int(a), int(edge_id)))

    return {
        "grid": g,
        "bg": bg,
        "nodes": nodes,
        "term_ids": term_ids,
        "id2pts": id2pts,
        "track_comp_pts": track_comp_pts,
        "directed_adj": directed_adj,
        "undirected_adj": undirected_adj,
    }


def _find_terminal_candidate_paths(
    term_ids: list[int],
    adj: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int, int, list[int], list[int]]]:
    candidate_paths: list[tuple[int, int, int, list[int], list[int]]] = []
    for start in term_ids:
        goal = term_ids[0] if int(start) == int(term_ids[1]) else term_ids[1]
        q = deque([int(start)])
        prev: dict[int, int | None] = {int(start): None}
        prev_edge: dict[int, int | None] = {int(start): None}
        while q:
            u = int(q.popleft())
            if u == int(goal):
                break
            for v, eid in adj.get(int(u), []):
                if int(v) in prev:
                    continue
                prev[int(v)] = int(u)
                prev_edge[int(v)] = int(eid)
                q.append(int(v))
        if int(goal) in prev:
            node_path, edge_path = _reconstruct_path(int(goal), prev, prev_edge)
            candidate_paths.append((len(edge_path), int(start), int(goal), node_path, edge_path))
    return candidate_paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique directed terminal-to-terminal path in a square-and-track graph.

    A track component still encodes a physical connection between two node squares, but the
    path may leave a node only through the face indicated by that node's special marker pixel.
    """
    parsed = _parse_graph(grid)
    g = np.array(parsed["grid"], copy=False)
    term_ids = list(parsed["term_ids"])
    id2pts = dict(parsed["id2pts"])
    track_comp_pts = dict(parsed["track_comp_pts"])
    directed_adj = dict(parsed["directed_adj"])

    candidate_paths = _find_terminal_candidate_paths(term_ids, directed_adj)
    assert candidate_paths
    best_len = min(int(item[0]) for item in candidate_paths)
    best = [item for item in candidate_paths if int(item[0]) == best_len]
    assert len(best) == 1
    _path_len, start, goal, node_path, edge_path = best[0]

    out = g.copy()
    term_set = {int(start), int(goal)}
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in track_comp_pts[int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)

    return out


def _compute_adjacency(picked: set[tuple[int, int]], allow_skip: bool = True) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges = set()
    for i, j in picked:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (i + di, j + dj)
            if nxt in picked:
                edges.add(tuple(sorted(((i, j), nxt))))
        if allow_skip:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (i + di // 2, j + dj // 2)
                far = (i + di, j + dj)
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted(((i, j), far))))
    return edges


def _pick_cells(Kr: int, Kc: int, V: int, rng: random.Random) -> set[tuple[int, int]] | None:
    cells = [(i, j) for i in range(Kr) for j in range(Kc)]
    if len(cells) < V:
        return None
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(cells)
        picked = set(cells[:V])
        cell_edges = _compute_adjacency(picked, allow_skip=True)
        adj = {c: [] for c in picked}
        for a, b in cell_edges:
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


def _sample_simple_path(adj: dict[int, list[int]], path_len: int, rng: random.Random) -> list[int] | None:
    starts = list(adj.keys())
    rng.shuffle(starts)
    candidates: list[list[int]] = []

    def dfs(path: list[int]) -> None:
        if len(candidates) >= 80:
            return
        if len(path) - 1 == int(path_len):
            candidates.append(list(path))
            return
        nbrs = list(adj[int(path[-1])])
        rng.shuffle(nbrs)
        for v in nbrs:
            if int(v) in path:
                continue
            path.append(int(v))
            dfs(path)
            path.pop()

    for start in starts:
        dfs([int(start)])
        if candidates:
            break
    if not candidates:
        return None
    return list(rng.choice(candidates))


def _bfs_distance(adj: dict[int, list[int]], start: int, goal: int) -> int | None:
    q = deque([int(start)])
    dist = {int(start): 0}
    while q:
        u = int(q.popleft())
        if u == int(goal):
            return int(dist[u])
        for v in adj[int(u)]:
            if int(v) in dist:
                continue
            dist[int(v)] = int(dist[u]) + 1
            q.append(int(v))
    return None


def _bfs_distance_labeled(adj: dict[int, list[tuple[int, int]]], start: int, goal: int) -> int | None:
    q = deque([int(start)])
    dist = {int(start): 0}
    while q:
        u = int(q.popleft())
        if u == int(goal):
            return int(dist[u])
        for v, _eid in adj.get(int(u), []):
            if int(v) in dist:
                continue
            dist[int(v)] = int(dist[u]) + 1
            q.append(int(v))
    return None


def _bfs_shortest_path(adj: dict[int, list[int]], start: int, goal: int) -> list[int] | None:
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v in adj[int(u)]:
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            q.append(int(v))
    if int(goal) not in prev:
        return None
    path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        path.append(int(cur))
        parent = prev[int(cur)]
        cur = int(parent) if parent is not None else None
    path.reverse()
    return path


def _build_graph(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    directed_path_len: int,
    n_cycles: int,
) -> tuple[list[tuple[int, int]], list[int], dict[int, tuple[int, int]]] | None:
    V = len(picked)
    cell_edges = _compute_adjacency(picked, allow_skip=True)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in cell_edges}
    full_adj = _adj_from_edges(full_edges, V)

    for _ in range(GRAPH_ATTEMPT_RETRIES):
        path_nodes = _sample_simple_path(full_adj, directed_path_len, rng)
        if path_nodes is None:
            return None
        start = int(path_nodes[0])
        goal = int(path_nodes[-1])
        short_path = _bfs_shortest_path(full_adj, start, goal)
        if short_path is None or len(short_path) >= len(path_nodes):
            continue

        long_edges = {
            tuple(sorted((int(path_nodes[i]), int(path_nodes[i + 1]))))
            for i in range(len(path_nodes) - 1)
        }
        short_edges = {
            tuple(sorted((int(short_path[i]), int(short_path[i + 1]))))
            for i in range(len(short_path) - 1)
        }
        base_edges = set(long_edges) | set(short_edges)
        in_tree = set(int(v) for v in path_nodes) | set(int(v) for v in short_path)
        cycle_count = len(base_edges) - len(in_tree) + 1
        if cycle_count < 1 or cycle_count > int(n_cycles):
            continue

        while len(in_tree) < V:
            frontier = [
                (int(u), int(v))
                for u in in_tree
                for v in full_adj[int(u)]
                if int(v) not in in_tree
            ]
            if not frontier:
                break
            u, v = rng.choice(frontier)
            base_edges.add(tuple(sorted((int(u), int(v)))))
            in_tree.add(int(v))
        if len(in_tree) != V:
            continue

        extra_needed = int(n_cycles) - cycle_count
        extra_candidates = list(full_edges - base_edges)
        if len(extra_candidates) < extra_needed:
            continue
        rng.shuffle(extra_candidates)
        final_edges = set(base_edges)
        for edge in extra_candidates[:extra_needed]:
            final_edges.add(tuple(sorted((int(edge[0]), int(edge[1])))))

        id_map = {int(start): 0, int(goal): V - 1}
        next_id = 1
        for old in range(V):
            if old in id_map:
                continue
            id_map[int(old)] = int(next_id)
            next_id += 1

        remapped_edges = {
            tuple(sorted((int(id_map[a]), int(id_map[b]))))
            for a, b in final_edges
        }
        remapped_path = [int(id_map[v]) for v in path_nodes]
        remapped_id_to_cell = {
            int(id_map[cell_to_id[cell]]): cell
            for cell in cell_to_id
        }
        return sorted(remapped_edges), remapped_path, remapped_id_to_cell
    return None


def _dir_from_cells(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = int(dst[0]) - int(src[0])
    dc = int(dst[1]) - int(src[1])
    if dr < 0 and dc == 0:
        return "U"
    if dr > 0 and dc == 0:
        return "D"
    if dc < 0 and dr == 0:
        return "L"
    if dc > 0 and dr == 0:
        return "R"
    raise ValueError((src, dst))


def _follow_directed(
    start: int,
    goal: int,
    outgoing: dict[int, tuple[int, tuple[int, int]]],
) -> tuple[list[int], list[tuple[int, int]]] | None:
    seen = set()
    cur = int(start)
    node_path = [int(cur)]
    edge_path: list[tuple[int, int]] = []
    while cur not in seen:
        if cur == int(goal):
            return node_path, edge_path
        seen.add(int(cur))
        step = outgoing.get(int(cur))
        if step is None:
            return None
        nxt, edge = step
        edge_path.append(tuple(sorted((int(edge[0]), int(edge[1])))))
        node_path.append(int(nxt))
        cur = int(nxt)
    if cur == int(goal):
        return node_path, edge_path
    return None


def _assign_directions(
    edges: list[tuple[int, int]],
    path_nodes: list[int],
    id_to_cell: dict[int, tuple[int, int]],
    rng: random.Random,
) -> dict[int, str] | None:
    V = len(id_to_cell)
    edge_sides: dict[tuple[int, int], dict[int, str]] = {}
    node_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        edge = tuple(sorted((int(a), int(b))))
        side_a = _dir_from_cells(id_to_cell[int(a)], id_to_cell[int(b)])
        side_b = OPPOSITE[side_a]
        edge_sides[edge] = {int(a): side_a, int(b): side_b}
        node_edges[int(a)].append(edge)
        node_edges[int(b)].append(edge)

    path_edges = [
        tuple(sorted((int(path_nodes[i]), int(path_nodes[i + 1]))))
        for i in range(len(path_nodes) - 1)
    ]
    fixed_dirs: dict[int, str] = {}
    for i, edge in enumerate(path_edges):
        node = int(path_nodes[i])
        fixed_dirs[int(node)] = str(edge_sides[edge][int(node)])

    goal = int(path_nodes[-1])
    goal_forbidden = str(edge_sides[path_edges[-1]][goal])
    goal_options = [
        str(edge_sides[edge][goal])
        for edge in node_edges[goal]
        if str(edge_sides[edge][goal]) != goal_forbidden
    ]
    if not goal_options:
        return None

    for _ in range(DIR_ASSIGN_RETRIES):
        dirs = dict(fixed_dirs)
        dirs[goal] = str(rng.choice(goal_options))
        for node in range(V):
            if int(node) in dirs:
                continue
            options = [str(edge_sides[edge][int(node)]) for edge in node_edges[int(node)]]
            if not options:
                return None
            dirs[int(node)] = str(rng.choice(options))

        outgoing: dict[int, tuple[int, tuple[int, int]]] = {}
        for node in range(V):
            usable = []
            for edge in node_edges[int(node)]:
                if str(edge_sides[edge][int(node)]) != dirs[int(node)]:
                    continue
                a, b = edge
                nxt = int(b) if int(a) == int(node) else int(a)
                usable.append((nxt, edge))
            if len(usable) != 1:
                break
            outgoing[int(node)] = usable[0]
        else:
            fwd = _follow_directed(0, V - 1, outgoing)
            rev = _follow_directed(V - 1, 0, outgoing)
            if rev is not None:
                continue
            if fwd is None:
                continue
            node_path, edge_path = fwd
            if [int(v) for v in node_path] != [int(v) for v in path_nodes]:
                continue
            if [tuple(sorted(e)) for e in edge_path] != [tuple(sorted(e)) for e in path_edges]:
                continue
            return dirs
    return None


def _cell_to_square_pos(cell: tuple[int, int], cell_size: int, origin: tuple[int, int]) -> tuple[int, int]:
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - NODE_SIZE) // 2
    return or_r + i * cell_size + margin, or_c + j * cell_size + margin


def _dock_tl(sq_r: int, sq_c: int, si: int, sj: int, width: int) -> tuple[int, int]:
    if si == -1:
        return sq_r - width, sq_c
    if si == 1:
        return sq_r + NODE_SIZE, sq_c + NODE_SIZE - width
    if sj == -1:
        return sq_r + NODE_SIZE - width, sq_c - width
    if sj == 1:
        return sq_r, sq_c + NODE_SIZE
    raise ValueError((si, sj))


def _walk_footprints(sr: int, sc: int, er: int, ec: int) -> list[tuple[int, int]]:
    pts = [(int(sr), int(sc))]
    cr, cc = int(sr), int(sc)
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
) -> list[tuple[int, int]]:
    ra, ca = pos_a
    rb, cb = pos_b
    si = 0 if int(cell_b[0]) == int(cell_a[0]) else (1 if int(cell_b[0]) > int(cell_a[0]) else -1)
    sj = 0 if int(cell_b[1]) == int(cell_a[1]) else (1 if int(cell_b[1]) > int(cell_a[1]) else -1)
    if si == -1:
        sr, sc = ra - 1, ca + 1
    elif si == 1:
        sr, sc = ra + NODE_SIZE, ca + 1
    elif sj == -1:
        sr, sc = ra + 1, ca - 1
    elif sj == 1:
        sr, sc = ra + 1, ca + NODE_SIZE
    else:
        raise ValueError((cell_a, cell_b))

    if -si == -1:
        er, ec = rb - 1, cb + 1
    elif -si == 1:
        er, ec = rb + NODE_SIZE, cb + 1
    elif -sj == -1:
        er, ec = rb + 1, cb - 1
    elif -sj == 1:
        er, ec = rb + 1, cb + NODE_SIZE
    else:
        raise ValueError((cell_a, cell_b))

    return _walk_footprints(sr, sc, er, ec)


TEMPLATE_CELL_GRID = 4
TEMPLATE_CELLS = {
    "S": (1, 0),
    "A": (1, 1),
    "B": (1, 2),
    "G": (0, 2),
    "Q": (0, 0),
    "L1": (2, 0),
    "L2": (2, 2),
    "L3": (3, 2),
    "L4": (0, 3),
}


def _rotate_cell(cell: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = cell
    for _ in range(int(turns) % 4):
        r, c = c, TEMPLATE_CELL_GRID - 1 - r
    return int(r), int(c)


def _rotate_dir(direction: str, turns: int) -> str:
    order = ["U", "R", "D", "L"]
    idx = order.index(str(direction))
    return order[(idx + int(turns)) % 4]


def _transform_template(
    cell: tuple[int, int],
    direction: str,
    turns: int,
    flip_h: bool,
) -> tuple[tuple[int, int], str]:
    new_cell = _rotate_cell(cell, turns)
    new_dir = _rotate_dir(direction, turns)
    if flip_h:
        r, c = new_cell
        new_cell = (int(r), TEMPLATE_CELL_GRID - 1 - int(c))
        if new_dir == "L":
            new_dir = "R"
        elif new_dir == "R":
            new_dir = "L"
    return new_cell, new_dir


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    gap = rng.choice(GAP_CHOICES)
    cell_size = NODE_SIZE + gap
    Kr = Kc = TEMPLATE_CELL_GRID
    min_S = TEMPLATE_CELL_GRID * cell_size + 2
    S = rng.randint(max(GRID_SIZE_RANGE[0], min_S), GRID_SIZE_RANGE[1])

    extras_by_v = {
        7: ["L1", "L2"],
        8: rng.choice([["L1", "L2", "L3"], ["L1", "L2", "L4"]]),
        9: ["L1", "L2", "L3", "L4"],
    }
    V = rng.randint(*NODE_COUNT_RANGE)
    included = ["S", "A", "B", "G", "Q"] + list(extras_by_v[int(V)])

    template_edges = [("S", "A"), ("A", "B"), ("B", "G"), ("S", "Q"), ("Q", "G")]
    if "L1" in included:
        template_edges.append(("S", "L1"))
    if "L2" in included:
        template_edges.append(("B", "L2"))
    if "L3" in included:
        template_edges.append(("L2", "L3"))
    if "L4" in included:
        template_edges.append(("G", "L4"))

    template_dirs = {
        "S": "R",
        "A": "R",
        "B": "U",
        "Q": "R",
        "L1": "U",
        "L4": "L",
    }
    if "L3" in included:
        template_dirs["L2"] = "D"
        template_dirs["L3"] = "U"
    elif "L2" in included:
        template_dirs["L2"] = "U"
    if "L4" in included:
        template_dirs["G"] = "R"
    else:
        template_dirs["G"] = "L"

    turns = rng.randint(0, 3)
    flip_h = bool(rng.randint(0, 1))
    transformed_cells: dict[str, tuple[int, int]] = {}
    transformed_dirs: dict[str, str] = {}
    for name in included:
        cell, direction = _transform_template(TEMPLATE_CELLS[name], template_dirs[name], turns, flip_h)
        transformed_cells[name] = cell
        transformed_dirs[name] = direction

    mid_names = [name for name in included if name not in {"S", "G"}]
    id_map = {"S": 0, "G": V - 1}
    for idx, name in enumerate(mid_names, start=1):
        id_map[name] = int(idx)

    path_nodes = [int(id_map[name]) for name in ["S", "A", "B", "G"]]
    edges = sorted(
        {
            tuple(sorted((int(id_map[a]), int(id_map[b]))))
            for a, b in template_edges
        }
    )
    dirs = {int(id_map[name]): str(transformed_dirs[name]) for name in included}
    id_to_cell = {int(id_map[name]): transformed_cells[name] for name in included}

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    bg, terminal_color, station_color, marker_color, track_color = pool[:5]

    canvas = np.full((S, S), int(bg), dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - TEMPLATE_CELL_GRID * cell_size - 2
    slack_c = S - TEMPLATE_CELL_GRID * cell_size - 2
    origin = (
        1 + rng.randint(0, max(0, slack_r)),
        1 + rng.randint(0, max(0, slack_c)),
    )

    id_to_pos: dict[int, tuple[int, int]] = {}
    for node in range(V):
        cell = id_to_cell[int(node)]
        pos = _cell_to_square_pos(cell, cell_size, origin)
        id_to_pos[int(node)] = pos
        fill = terminal_color if int(node) in {0, V - 1} else station_color
        r0, c0 = pos
        canvas[r0:r0 + NODE_SIZE, c0:c0 + NODE_SIZE] = int(fill)
        mr, mc = DIR_TO_MARKER[str(dirs[int(node)])]
        canvas[r0 + mr, c0 + mc] = int(marker_color)
        node_mask[r0:r0 + NODE_SIZE, c0:c0 + NODE_SIZE] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(id_to_cell[int(a)], id_to_cell[int(b)], id_to_pos[int(a)], id_to_pos[int(b)])
        for r, c in pixels:
            if not (0 <= int(r) < S and 0 <= int(c) < S):
                return None
            if node_mask[int(r), int(c)] or track_mask[int(r), int(c)]:
                return None
        for r, c in pixels:
            canvas[int(r), int(c)] = int(track_color)
            track_mask[int(r), int(c)] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels

    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = np.array(canvas, copy=True)
    for node in path_nodes[1:-1]:
        r0, c0 = id_to_pos[int(node)]
        expected[r0:r0 + NODE_SIZE, c0:c0 + NODE_SIZE] = int(Color.GREEN)
    for i in range(len(path_nodes) - 1):
        edge = tuple(sorted((int(path_nodes[i]), int(path_nodes[i + 1]))))
        for r, c in edge_track_pixels[edge]:
            expected[int(r), int(c)] = int(Color.GREY)

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        parsed = _parse_graph(canvas)
    except Exception:
        return None
    parsed_candidates = _find_terminal_candidate_paths(parsed["term_ids"], parsed["directed_adj"])
    if len(parsed_candidates) != 1:
        return None
    directed_len, start, goal, _node_path, _edge_path = parsed_candidates[0]
    undirected_len = _bfs_distance_labeled(parsed["undirected_adj"], start, goal)
    if undirected_len is None or int(undirected_len) >= int(directed_len):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
