# Variant of 7b0280bc
# Dimension: D5 — 终点/停止条件 (dst)
# Change: 另一个少数色终端方块为终点，最短路径到达即停止... → 多终端选择最近
#
# Design analysis
# ===============
# variant_spec:
#   preserved_core:
#     - 仍然是由等大实心方块节点和轨道连边构成的图。
#     - 输出仍然只把被选中路径上的中间站点染成绿(3)，路径轨道染成灰(5)。
#     - 距离仍然按图上的边数计算，而不是几何直线距离。
#   necessary_adaptation:
#     - 原题两个终端同色且对称，作为“起点/终点”并不需要区分。
#     - 一旦改成“多个候选终点里选最近”，若仍让全部终端同色，就无法从像素网格中知道哪一个是起点。
#     - 因此做最小适配：起点改为唯一颜色的 1 个方块；候选终点改为同色的 3 个方块；
#       其余节点仍是同色站点。图结构、轨道形态、着色结果都保持原题风格。
#   concretization:
#     - 输入共 5 色：背景、起点、候选终点、站点、轨道。
#     - 起点: 恰好 1 个 NxN 实心方块。
#     - 候选终点: 恰好 3 个同色 NxN 实心方块。
#     - 站点: 4-6 个同色 NxN 实心方块。
#     - 轨道: 由 8 邻接连通块组成，每段轨道连接恰好两个方块节点。
#   behavior_trace:
#     - 例子: 起点 S 到三个终点 A/B/C 的最短图距离分别为 2/3/4。
#     - 原题逻辑: 若只知道一对固定终端，只会求 S→某个预先指定终点。
#     - 变体逻辑: 先对 S 到所有候选终点做 BFS 距离比较，选距离 2 的 A，
#       再只给 S→A 这条路径着色。
#     - 差异: “终点是谁”不再是输入中预先固定的一对，而是由多终点距离比较得出。
#   complexity_calibration:
#     - 候选终点固定为 3 个，足以强迫解题者比较多条路径，又不会让颜色角色过多。
#     - 站点数固定大于终点数，避免 solver 仅凭颜色计数时出现歧义。
#     - 最近终点保持唯一，避免引入未定义的同距 tie-break 规则。
#   implicit_assumption_audit:
#     - 原代码假设“存在且仅存在 1 种颜色出现 2 个节点”，这一点在变体中失效。
#     - 原代码 `start, goal = term_ids[0], term_ids[1]` 只支持固定两端，必须改成
#       `start = 唯一起点`, `goals = 所有候选终点`, `goal = 最近的那个`。
#     - 原代码着色时只跳过两个终端；变体里应跳过起点和全部候选终点，避免把终点染绿。
#     - 轨道组件识别、方块检测、边构图逻辑不依赖“终点数量=2”，因此保持不变。
#   placement_legality:
#     - 仍沿用原题的 graph-first 再嵌入到 cell grid 的做法，保证节点不重叠、轨道不重叠、
#       轨道每段只接触两个节点。
#     - 先生成一个“起点到某个近终点”的唯一最短路径，再把另外 2 个更远节点换色为终点，
#       从构造上保证“最近终点唯一”。
#   visual_observability:
#     - 输入里会明显看到 1 个单独颜色的起点方块，和 3 个同色候选终点方块。
#     - 输出只连接到最近的那一个，其余终点保持原色，规则差异可直接观察。
#   cleanliness_and_color_inference:
#     - 仍只用规则化的整块方块和连续轨道，不叠层、不加内部纹理。
#     - 颜色变化只有两种：路径上的站点→绿，路径上的轨道→灰；与原题一致，易于归纳。
#   post_change_solver_assumptions:
#     - main() 现在假设节点颜色恰好 3 类，且节点计数满足 1 < 3 < station_count。
#     - generate_input() 通过固定“1 起点 + 3 终点 + 4-6 站点”显式保证该假设成立。

from __future__ import annotations

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


def _extract_graph(grid: np.ndarray) -> dict:
    """Parse nodes, colors, and track edges from the grid."""
    g = np.array(grid, dtype=int)
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

    node_colors = sorted(nodes_by_color.keys())
    assert len(node_colors) == 3

    source_colors = [col for col in node_colors if len(nodes_by_color[int(col)]) == 1]
    assert len(source_colors) == 1
    source_color = int(source_colors[0])

    other_node_colors = [col for col in node_colors if int(col) != int(source_color)]
    assert len(other_node_colors) == 2
    terminal_color = int(min(other_node_colors, key=lambda col: len(nodes_by_color[int(col)])))
    station_color = int(max(other_node_colors, key=lambda col: len(nodes_by_color[int(col)])))

    start = int(nodes_by_color[int(source_color)][0])
    goal_ids = list(map(int, nodes_by_color[int(terminal_color)]))
    assert len(goal_ids) >= 3
    assert len(nodes_by_color[int(station_color)]) > len(goal_ids)

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [
        c
        for c in non_bg_colors
        if int(c) not in (int(source_color), int(terminal_color), int(station_color))
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

    return {
        "grid": g,
        "bg": bg,
        "nodes": nodes,
        "id2pts": id2pts,
        "adj": adj,
        "track_comp_pts": track_comp_pts,
        "start": start,
        "goal_ids": goal_ids,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path from the unique source square to the nearest terminal square.

    Input structure:
    - bg = most common color.
    - Three node colors:
      * one unique source square,
      * three same-colored destination squares,
      * several same-colored station squares.
    - A fourth non-bg color forms track segments that connect node pairs.

    Rule:
    - Build the graph induced by node squares and track components.
    - From the unique source, compute shortest-path distances to all destination squares.
    - Choose the uniquely nearest destination.
    - Recolor every station square on that chosen path to green.
    - Recolor every track segment on that chosen path to grey.
    """
    parsed = _extract_graph(np.array(grid, dtype=int))
    g = parsed["grid"]
    adj = parsed["adj"]
    start = int(parsed["start"])
    goal_ids = list(map(int, parsed["goal_ids"]))
    track_comp_pts = parsed["track_comp_pts"]
    id2pts = parsed["id2pts"]

    q = deque([start])
    prev: dict[int, int | None] = {start: None}
    prev_edge: dict[int, int | None] = {start: None}
    dist: dict[int, int] = {start: 0}

    while q:
        u = int(q.popleft())
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            dist[int(v)] = int(dist[int(u)] + 1)
            q.append(int(v))

    reachable_goals = [(int(dist[gid]), int(gid)) for gid in goal_ids if int(gid) in dist]
    assert reachable_goals
    best_dist = min(d for d, _gid in reachable_goals)
    nearest_goals = [gid for d, gid in reachable_goals if int(d) == int(best_dist)]
    assert len(nearest_goals) == 1
    goal = int(nearest_goals[0])

    node_path: list[int] = []
    edge_path: list[int] = []
    cur = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        edge_id = prev_edge.get(int(cur))
        if edge_id is not None:
            edge_path.append(int(edge_id))
        prev_node = prev.get(int(cur))
        cur = int(prev_node) if prev_node is not None else None
    node_path.reverse()
    edge_path.reverse()

    out = g.copy()
    protected_nodes = {int(start), *map(int, goal_ids)}
    for nid in node_path:
        if int(nid) in protected_nodes:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# --------------- generator ---------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (8, 10)              # 1 source + 3 terminals + 4-6 stations
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]               # source -> chosen nearest terminal
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]   # excludes green=3, grey=5
DESTINATION_COUNT = 3
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 300


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    """Compute candidate graph edges between picked cells."""
    edges = set()
    for (i, j) in picked:
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
    Kr: int,
    Kc: int,
    V: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
):
    """Pick V cells on a Kr x Kc grid whose induced adjacency graph is connected."""
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
    """BFS from src. Returns shortest distances, number of shortest paths, and adjacency."""
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
    Build a connected graph with a unique shortest path from source to the chosen nearest goal.

    Returns:
    - edges after remapping node ids so source=0 and chosen nearest goal=V-1
    - path nodes and path edges for that chosen source-goal pair
    - remapped cell_to_id
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in cell_edges}

    ids = list(range(V))
    rng.shuffle(ids)
    src, near_goal = ids[0], ids[1]

    dist, _cnt, adj = _bfs_with_counts(src, full_edges, V)
    if near_goal not in dist or int(dist[near_goal]) != int(path_len):
        return None

    parent = {int(src): None}
    q = deque([int(src)])
    seen = {int(src)}
    while q:
        u = int(q.popleft())
        if int(u) == int(near_goal):
            break
        for v in adj[u]:
            if int(v) not in seen:
                seen.add(int(v))
                parent[int(v)] = int(u)
                q.append(int(v))
    cur = int(near_goal)
    sp_nodes: list[int] = []
    while cur is not None:
        sp_nodes.append(int(cur))
        cur = parent[int(cur)] if cur in parent else None
    sp_nodes.reverse()
    sp_edges = {tuple(sorted((sp_nodes[i], sp_nodes[i + 1]))) for i in range(len(sp_nodes) - 1)}

    tree_edges = set(sp_edges)
    in_tree = set(sp_nodes)
    frontier = list(sp_nodes)
    while in_tree != set(range(V)):
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
        dist2, cnt2, _adj2 = _bfs_with_counts(src, cand, V)
        if int(dist2.get(near_goal, -1)) == int(path_len) and int(cnt2.get(near_goal, 0)) == 1:
            final_edges.add(e)
            added += 1
    if added != n_cycles:
        return None

    dist3, cnt3, _adj3 = _bfs_with_counts(src, final_edges, V)
    if int(dist3.get(near_goal, -1)) != int(path_len) or int(cnt3.get(near_goal, 0)) != 1:
        return None
    if len(final_edges) != V - 1 + n_cycles:
        return None

    id_map = {int(src): 0, int(near_goal): V - 1}
    for idx, old_id in enumerate(i for i in range(V) if i != src and i != near_goal):
        id_map[int(old_id)] = int(idx + 1)
    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for a, b in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    path_nodes = [id_map[n] for n in sp_nodes]
    path_edges = [
        tuple(sorted((path_nodes[i], path_nodes[i + 1])))
        for i in range(len(path_nodes) - 1)
    ]

    return list(remapped_edges), path_nodes, path_edges, remapped_cell_to_id


def _pick_extra_destinations(
    edges: list[tuple[int, int]],
    V: int,
    rng: random.Random,
    chosen_goal: int,
) -> list[int] | None:
    """Choose two additional destinations that are strictly farther from the source."""
    dist, _cnt, _adj = _bfs_with_counts(0, set(edges), V)
    near_dist = dist.get(int(chosen_goal))
    if near_dist is None:
        return None
    candidates = [
        i
        for i in range(1, V - 1)
        if i in dist and int(dist[i]) > int(near_dist)
    ]
    if len(candidates) < DESTINATION_COUNT - 1:
        return None
    rng.shuffle(candidates)
    extra = list(candidates[: DESTINATION_COUNT - 1])
    dvals = sorted(int(dist[i]) for i in [chosen_goal, *extra])
    if dvals[1] <= dvals[0]:
        return None
    return [int(chosen_goal), *map(int, extra)]


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
    """Generate footprint top-left positions from start dock to end dock."""
    pts = [(int(sr), int(sc))]
    cr, cc = int(sr), int(sc)
    while cr != er and cc != ec:
        cr += 1 if er > cr else -1
        cc += 1 if ec > cc else -1
        pts.append((int(cr), int(cc)))
    while cr != er:
        cr += 1 if er > cr else -1
        pts.append((int(cr), int(cc)))
    while cc != ec:
        cc += 1 if ec > cc else -1
        pts.append((int(cr), int(cc)))
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
    del rng  # deterministic given the embedded graph
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

    gr = _build_graph_spatial(
        picked,
        cell_to_id,
        rng,
        path_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if gr is None:
        return None
    edges, path_nodes, path_edges, cell_to_id = gr
    id_to_cell = {v: k for k, v in cell_to_id.items()}
    V = len(picked)

    goal_ids = _pick_extra_destinations(edges, V, rng, chosen_goal=V - 1)
    if goal_ids is None:
        return None
    goal_set = set(map(int, goal_ids))

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    five = pool[:5]
    bg = rng.choice(five)
    rest = [c for c in five if c != bg]
    source_color = rng.choice(rest)
    rest = [c for c in rest if c != source_color]
    terminal_color = rng.choice(rest)
    rest = [c for c in rest if c != terminal_color]
    station_color = rng.choice(rest)
    track_color = [c for c in rest if c != station_color][0]

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))
    id_to_pos: dict[int, tuple[int, int]] = {}

    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[int(i)] = pos
        if int(i) == 0:
            color = source_color
        elif int(i) in goal_set:
            color = terminal_color
        else:
            color = station_color
        r0, c0 = pos
        canvas[r0 : r0 + N, c0 : c0 + N] = int(color)
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N, rng)
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for r, c in pixels:
            canvas[r, c] = int(track_color)
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels

    if np.argmax(np.bincount(canvas.ravel())) != int(bg):
        return None

    expected = canvas.copy()
    for node_id in path_nodes:
        if int(node_id) in {0, V - 1}:
            continue
        r0, c0 = id_to_pos[int(node_id)]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for e in path_edges:
        for r, c in edge_track_pixels[tuple(sorted(e))]:
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
