# Variant of 7b0280bc
# Dimension: D4 — 源对象定义 (src)
# Change: 数量为2的少数色NxN实心方块作为终端节点对... → 边缘控制像素指定起点
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     original_src: 两个同色少数色 NxN 实心方块直接给出起点和终点
#     new_src: 起点不再直接由少数色方块给出，而是由网格边缘的 1 个控制像素指定；
#       该控制像素与某个普通节点方块 8 邻接，这个被贴边标记到的方块才是起点
#     dst: 终点仍保留为唯一的少数色 NxN 实心方块
#   concrete_input:
#     colors_total: 4
#     bg: 最常见色
#     goal_color: 唯一终点方块的颜色，同时复用为边缘控制像素颜色
#     station_color: 其余所有 NxN 节点方块颜色；其中一个通过边缘控制像素被指定为起点
#     track_color: 轨道颜色
#     control_pixel:
#       count: exactly 1
#       location: exactly on one border cell
#       relation: 8 邻接 exactly 1 个 NxN 方块，且该方块不是终点
#   behavior_trace:
#     old_logic:
#       - 扫描方块颜色
#       - 找到“恰好有 2 个方块的颜色”作为 terminal_color
#       - 在这 2 个方块之间做 BFS
#     new_logic:
#       - 仍先扫描所有 NxN 方块和轨道图
#       - 在边框上找到唯一非背景控制像素
#       - 用它 8 邻接到的唯一方块确定 start
#       - 用与控制像素同色的唯一 NxN 方块确定 goal
#       - 在 start -> goal 之间做 BFS
#     difference:
#       - 原题里“起点身份”来自方块颜色；变体里“起点身份”来自边缘像素和方块的空间邻接
#       - 因此 solver 不能再用“少数色恰有两个方块”这条假设
#   complexity_calibration:
#     why_not_add_fifth_color: 会让颜色机制额外变化；本变体只改 src 定义，故复用 goal_color 作为控制色，
#       保持总色数 4，不引入新的颜色映射学习负担
#     why_single_control_pixel: 1 个边缘单像素已经能稳定、明显地表达“从边缘指向起点”
#     why_goal_unique: 起点定义已复杂化，终点继续保持唯一少数色方块，避免两端都变得隐式
#   implicit_assumption_review:
#     old_breakage_points:
#       - 原 main() 中 `len(nodes_by_color[col]) == 2` 推 terminal_color：变体下不成立
#       - 原 main() 直接把 2 个终端方块拿来 BFS：变体下起点并非特殊颜色方块
#     new_solver_assumptions:
#       - 存在且仅存在 1 个边框非背景、非节点像素作为控制像素
#       - 控制像素 8 邻接恰好 1 个节点方块
#       - 控制像素颜色对应恰好 1 个 NxN 方块，那个方块是终点
#       - 轨道连通图中 start 到 goal 存在唯一最短路
#     generator_guarantees:
#       - 强制起点节点位于所选边的最外层单元格，便于把方块贴近边框并安放控制像素
#       - 控制边方向上不存在从起点继续向外的轨道，避免控制像素和轨道混淆
#       - 其余节点保持离边框至少 1 个背景像素，保证唯一边框控制像素清晰可见
#   placement_legality:
#     squares: 全部为统一尺寸 N in {2,3} 的实心方块，不重叠
#     control: 仅放在 border 上，与起点方块 8 邻接，不与轨道重叠
#     tracks: 沿原任务的宽度 N-1 轨道布线；不与方块、控制像素重叠
#   visual_observability:
#     human_view:
#       - 输入里只有 1 个边缘单像素异常标记，且它贴着某个普通方块
#       - 另有 1 个同色目标方块在图内
#       - 输出仍只灰化路径轨道、绿化路径中间站，因此“从控制像素指示的起点走向目标方块”可被肉眼归纳
#   cleanliness:
#     - 控制像素只占 1 格，不覆盖轨道或节点
#     - 起点贴边但不触边，避免把方块误看成控制标记本身

from __future__ import annotations

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
    """Find solid monochrome NxN squares (N >= min_size)."""
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


def _border_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = map(int, shape)
    pts: list[tuple[int, int]] = []
    for c in range(W):
        pts.append((0, int(c)))
        if H > 1:
            pts.append((int(H - 1), int(c)))
    for r in range(1, max(1, H - 1)):
        pts.append((int(r), 0))
        if W > 1:
            pts.append((int(r), int(W - 1)))
    return pts


def _parse_graph_objects(grid: np.ndarray):
    """Shared parser for solver and generator-side verification."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(N)]
    assert len(nodes_raw) >= 2

    nodes: list[dict] = []
    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        node = {
            "id": int(i),
            "color": int(d["color"]),
            "pts": pts,
            "bbox": tuple(map(int, d["bbox"])),
        }
        nodes.append(node)
        id2pts[int(i)] = pts
        for p in pts:
            pix2node[p] = int(i)

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))
    assert len(nodes_by_color) == 2

    border_non_node = []
    for r, c in _border_coords(g.shape):
        if int(g[int(r), int(c)]) == int(bg):
            continue
        if (int(r), int(c)) in pix2node:
            continue
        border_non_node.append((int(r), int(c)))
    assert len(border_non_node) == 1
    ctrl_r, ctrl_c = border_non_node[0]
    ctrl_color = int(g[int(ctrl_r), int(ctrl_c)])

    touched_nodes: set[int] = set()
    for rr, cc in neighbors8(int(ctrl_r), int(ctrl_c)):
        nid = pix2node.get((int(rr), int(cc)))
        if nid is not None:
            touched_nodes.add(int(nid))
    assert len(touched_nodes) == 1
    start_id = int(next(iter(touched_nodes)))

    goal_ids = [int(nd["id"]) for nd in nodes if int(nd["color"]) == int(ctrl_color)]
    assert len(goal_ids) == 1
    goal_id = int(goal_ids[0])
    assert start_id != goal_id

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [int(c) for c in non_bg_colors if int(c) not in nodes_by_color]
    assert len(track_candidates) == 1
    track_color = int(track_candidates[0])

    return {
        "grid": g,
        "bg": bg,
        "N": N,
        "nodes": nodes,
        "id2pts": id2pts,
        "pix2node": pix2node,
        "nodes_by_color": nodes_by_color,
        "ctrl_pixel": (int(ctrl_r), int(ctrl_c)),
        "ctrl_color": ctrl_color,
        "start_id": start_id,
        "goal_id": goal_id,
        "track_color": track_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique shortest path from an edge-indicated start node to the unique goal node.

    Input structure:
    - bg = most common color.
    - All graph nodes are solid NxN squares (N>=2, uniform size).
    - Exactly one square has the control/goal color; all remaining squares share the station color.
    - Exactly one non-bg border pixel uses the goal color and is 8-adjacent to the start square.
    - A third non-bg color forms track components connecting pairs of squares.

    Rule:
    - Use the border control pixel to identify the start node.
    - Use the unique square with the same color as the control pixel as the goal node.
    - Find the shortest path (fewest graph edges) from start to goal.
    - Recolor every intermediate station square on that path to green and every track segment on
      that path to grey. Start square, goal square, and everything else stay unchanged.
    """
    parsed = _parse_graph_objects(grid)
    g = parsed["grid"]
    start = int(parsed["start_id"])
    goal = int(parsed["goal_id"])
    track_color = int(parsed["track_color"])
    pix2node = parsed["pix2node"]
    id2pts = parsed["id2pts"]

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

    assert goal in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        eid = prev_edge.get(int(cur))
        if eid is not None:
            edge_path.append(int(eid))
        parent = prev.get(int(cur))
        cur = int(parent) if parent is not None else None
    node_path.reverse()
    edge_path.reverse()

    out = g.copy()
    for nid in node_path[1:-1]:
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)
    for eid in edge_path:
        for r, c in track_comp_pts[int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)
    return out


# --------------- generator ---------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
CONTROL_EDGE_CHOICES = ["top", "bottom", "left", "right"]
CELL_PICK_RETRIES = 80
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
    required_cell: tuple[int, int],
):
    """Pick V cells inducing a connected spatial graph and containing required_cell."""
    cells = [(i, j) for i in range(Kr) for j in range(Kc)]
    if len(cells) < V or required_cell not in cells:
        return None
    others = [c for c in cells if c != required_cell]
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(others)
        picked = {required_cell, *others[: max(0, V - 1)]}
        edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
        adj = {c: [] for c in picked}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        root = required_cell
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
                dist[int(v)] = int(dist[u] + 1)
                cnt[int(v)] = int(cnt[u])
                q.append(int(v))
            elif int(dist[v]) == int(dist[u] + 1):
                cnt[int(v)] += int(cnt[u])
    return dist, cnt, adj


def _build_graph_spatial(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    path_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
    start_id: int,
):
    """
    Build a graph whose start node is fixed.

    The goal is chosen among nodes at exact distance `path_len` from start in the full spatial
    adjacency graph, then the final graph is pruned/augmented to preserve a unique shortest path.
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in cell_edges}

    dist0, _cnt0, adj0 = _bfs_with_counts(int(start_id), full_edges, V)
    goal_candidates = [int(v) for v, d in dist0.items() if int(v) != int(start_id) and int(d) == int(path_len)]
    if not goal_candidates:
        return None
    rng.shuffle(goal_candidates)

    for goal_id in goal_candidates:
        parent = {int(start_id): None}
        q = deque([int(start_id)])
        seen = {int(start_id)}
        while q:
            u = int(q.popleft())
            if u == int(goal_id):
                break
            for v in adj0[u]:
                if int(v) not in seen:
                    seen.add(int(v))
                    parent[int(v)] = int(u)
                    q.append(int(v))
        if int(goal_id) not in parent:
            continue

        cur = int(goal_id)
        sp_nodes = []
        while cur is not None:
            sp_nodes.append(int(cur))
            prev = parent[cur]
            cur = int(prev) if prev is not None else None
        sp_nodes.reverse()
        sp_edges = {tuple(sorted((sp_nodes[i], sp_nodes[i + 1]))) for i in range(len(sp_nodes) - 1)}

        tree_edges = set(sp_edges)
        in_tree = set(sp_nodes)
        frontier = list(sp_nodes)
        while in_tree != set(range(V)):
            next_frontier = []
            for u in frontier:
                for v in adj0[u]:
                    if int(v) not in in_tree:
                        tree_edges.add(tuple(sorted((int(u), int(v)))))
                        in_tree.add(int(v))
                        next_frontier.append(int(v))
            if not next_frontier:
                break
            frontier = next_frontier
        if in_tree != set(range(V)):
            continue

        extra_candidates = list(full_edges - tree_edges)
        rng.shuffle(extra_candidates)
        final_edges = set(tree_edges)
        added = 0
        for e in extra_candidates:
            if added >= n_cycles:
                break
            cand = final_edges | {e}
            dist2, cnt2, _ = _bfs_with_counts(int(start_id), cand, V)
            if int(dist2.get(int(goal_id), -1)) == int(path_len) and int(cnt2.get(int(goal_id), 0)) == 1:
                final_edges.add(e)
                added += 1
        if added != n_cycles:
            continue

        dist3, cnt3, _ = _bfs_with_counts(int(start_id), final_edges, V)
        if int(dist3.get(int(goal_id), -1)) != int(path_len) or int(cnt3.get(int(goal_id), 0)) != 1:
            continue
        if len(final_edges) != int(V - 1 + n_cycles):
            continue

        id_map = {int(start_id): 0, int(goal_id): int(V - 1)}
        next_id = 1
        for old_id in range(V):
            if int(old_id) in id_map:
                continue
            id_map[int(old_id)] = int(next_id)
            next_id += 1

        remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for a, b in final_edges}
        remapped_cell_to_id = {cell: int(id_map[cell_to_id[cell]]) for cell in cell_to_id}
        new_path_nodes = [int(id_map[n]) for n in sp_nodes]
        new_path_edges = [
            tuple(sorted((int(new_path_nodes[i]), int(new_path_nodes[i + 1]))))
            for i in range(len(new_path_nodes) - 1)
        ]
        return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id

    return None


def _cell_to_square_pos(cell: tuple[int, int], N: int, cell_size: int, origin: tuple[int, int]):
    i, j = cell
    or_r, or_c = origin
    margin = (int(cell_size) - int(N)) // 2
    return int(or_r + i * cell_size + margin), int(or_c + j * cell_size + margin)


def _dock_tl(sq_r: int, sq_c: int, N: int, Wt: int, si: int, sj: int):
    if si == -1:
        return int(sq_r - Wt), int(sq_c)
    if si == 1:
        return int(sq_r + N), int(sq_c + N - Wt)
    if sj == -1:
        return int(sq_r + N - Wt), int(sq_c - Wt)
    if sj == 1:
        return int(sq_r), int(sq_c + N)
    raise AssertionError("invalid dock direction")


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
    N: int,
) -> list[tuple[int, int]]:
    Wt = int(N - 1)
    ra, ca = pos_a
    rb, cb = pos_b
    dci = int(cell_b[0] - cell_a[0])
    dcj = int(cell_b[1] - cell_a[1])
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(int(ra), int(ca), int(N), int(Wt), int(si), int(sj))
    er, ec = _dock_tl(int(rb), int(cb), int(N), int(Wt), int(-si), int(-sj))
    pts_tl = _walk_footprints(int(sr), int(sc), int(er), int(ec))

    pixels = set()
    for r, c in pts_tl:
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((int(r + dr), int(c + dc)))
    return list(pixels)


def _edge_cells(Kr: int, Kc: int, edge: str) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, int(j)) for j in range(Kc)]
    if edge == "bottom":
        return [(int(Kr - 1), int(j)) for j in range(Kc)]
    if edge == "left":
        return [(int(i), 0) for i in range(Kr)]
    if edge == "right":
        return [(int(i), int(Kc - 1)) for i in range(Kr)]
    raise AssertionError("bad edge")


def _pick_origin(
    S: int,
    Kr: int,
    Kc: int,
    N: int,
    cell_size: int,
    start_cell: tuple[int, int],
    control_edge: str,
    rng: random.Random,
):
    margin = (int(cell_size) - int(N)) // 2
    slack_r = int(S - Kr * cell_size - 2)
    slack_c = int(S - Kc * cell_size - 2)

    if control_edge == "top":
        origin_r = int(1 - margin - start_cell[0] * cell_size)
        origin_c = int(1 + rng.randint(0, max(0, slack_c)))
    elif control_edge == "bottom":
        target_top = int(S - N - 1)
        origin_r = int(target_top - start_cell[0] * cell_size - margin)
        origin_c = int(1 + rng.randint(0, max(0, slack_c)))
    elif control_edge == "left":
        origin_r = int(1 + rng.randint(0, max(0, slack_r)))
        origin_c = int(1 - margin - start_cell[1] * cell_size)
    elif control_edge == "right":
        target_left = int(S - N - 1)
        origin_r = int(1 + rng.randint(0, max(0, slack_r)))
        origin_c = int(target_left - start_cell[1] * cell_size - margin)
    else:
        return None

    for i in range(Kr):
        for j in range(Kc):
            r0, c0 = _cell_to_square_pos((int(i), int(j)), int(N), int(cell_size), (int(origin_r), int(origin_c)))
            if r0 < 1 or c0 < 1 or r0 + N > S - 1 or c0 + N > S - 1:
                # The actual picked nodes will be a subset, but keeping every cell in-bounds makes
                # the placement reasoning uniform and prevents accidental border touching.
                return None

    return int(origin_r), int(origin_c)


def _control_pixel_for_square(pos: tuple[int, int], N: int, S: int, edge: str):
    r0, c0 = pos
    if edge == "top":
        return 0, int(c0)
    if edge == "bottom":
        return int(S - 1), int(c0 + N - 1)
    if edge == "left":
        return int(r0 + N - 1), 0
    if edge == "right":
        return int(r0), int(S - 1)
    raise AssertionError("bad edge")


def _verify_variant_signature(grid: np.ndarray):
    parsed = _parse_graph_objects(grid)
    ctrl_r, ctrl_c = parsed["ctrl_pixel"]
    H, W = grid.shape
    assert ctrl_r in (0, H - 1) or ctrl_c in (0, W - 1)
    assert len(parsed["nodes_by_color"][parsed["ctrl_color"]]) == 1
    return parsed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = int(rng.choice(NODE_SIZE_CHOICES))
    V = int(rng.randint(*NODE_COUNT_RANGE))
    n_cycles = int(rng.choice(CYCLE_COUNT_CHOICES))
    path_len = int(rng.choice(PATH_LEN_CHOICES))
    gap = int(rng.choice(GAP_CHOICES))
    cell_size = int(N + gap)
    control_edge = str(rng.choice(CONTROL_EDGE_CHOICES))

    KrKc_options = []
    for Kr in CELL_GRID_DIMS:
        for Kc in CELL_GRID_DIMS:
            if int(Kr * Kc) < int(V):
                continue
            min_S = int(max(Kr, Kc) * cell_size + 2)
            if int(min_S) > int(GRID_SIZE_RANGE[1]):
                continue
            KrKc_options.append((int(Kr), int(Kc), int(min_S)))
    if not KrKc_options:
        return None

    rng.shuffle(KrKc_options)
    use_diagonal = False
    allow_skip = True

    for Kr, Kc, min_S in KrKc_options:
        start_cells = _edge_cells(int(Kr), int(Kc), str(control_edge))
        rng.shuffle(start_cells)
        for required_cell in start_cells:
            S = int(rng.randint(max(int(GRID_SIZE_RANGE[0]), int(min_S)), int(GRID_SIZE_RANGE[1])))
            picked = _pick_cells(
                int(Kr),
                int(Kc),
                int(V),
                rng,
                use_diagonal=use_diagonal,
                allow_skip=allow_skip,
                required_cell=required_cell,
            )
            if picked is None:
                continue

            picked_list = list(picked)
            rng.shuffle(picked_list)
            if required_cell in picked_list:
                picked_list.remove(required_cell)
            picked_list.insert(0, required_cell)
            cell_to_id = {cell: int(i) for i, cell in enumerate(picked_list)}

            gr = _build_graph_spatial(
                picked,
                cell_to_id,
                rng,
                path_len,
                n_cycles,
                use_diagonal=use_diagonal,
                allow_skip=allow_skip,
                start_id=int(cell_to_id[required_cell]),
            )
            if gr is None:
                continue
            edges, path_nodes, path_edges, cell_to_id = gr
            id_to_cell = {int(v): k for k, v in cell_to_id.items()}
            start_cell = id_to_cell[0]

            origin = _pick_origin(
                int(S),
                int(Kr),
                int(Kc),
                int(N),
                int(cell_size),
                start_cell,
                str(control_edge),
                rng,
            )
            if origin is None:
                continue

            pool = list(COLOR_POOL)
            rng.shuffle(pool)
            bg, goal_color, station_color, track_color = map(int, pool[:4])

            canvas = np.full((int(S), int(S)), int(bg), dtype=int)
            node_mask = np.zeros((int(S), int(S)), dtype=bool)
            id_to_pos: dict[int, tuple[int, int]] = {}
            for nid in range(V):
                cell = id_to_cell[int(nid)]
                pos = _cell_to_square_pos(cell, int(N), int(cell_size), origin)
                id_to_pos[int(nid)] = pos
                color = int(goal_color) if int(nid) == int(V - 1) else int(station_color)
                r0, c0 = pos
                if r0 < 1 or c0 < 1 or r0 + N > S - 1 or c0 + N > S - 1:
                    break
                canvas[r0 : r0 + N, c0 : c0 + N] = int(color)
                node_mask[r0 : r0 + N, c0 : c0 + N] = True
            else:
                track_mask = np.zeros((int(S), int(S)), dtype=bool)
                edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
                ok_tracks = True
                for a, b in edges:
                    pixels = _track_between_cells(
                        id_to_cell[int(a)],
                        id_to_cell[int(b)],
                        id_to_pos[int(a)],
                        id_to_pos[int(b)],
                        int(N),
                    )
                    for r, c in pixels:
                        if not (0 <= int(r) < int(S) and 0 <= int(c) < int(S)):
                            ok_tracks = False
                            break
                        if node_mask[int(r), int(c)] or track_mask[int(r), int(c)]:
                            ok_tracks = False
                            break
                    if not ok_tracks:
                        break
                    for r, c in pixels:
                        canvas[int(r), int(c)] = int(track_color)
                        track_mask[int(r), int(c)] = True
                    edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels
                if not ok_tracks:
                    continue

                ctrl_r, ctrl_c = _control_pixel_for_square(id_to_pos[0], int(N), int(S), str(control_edge))
                if canvas[int(ctrl_r), int(ctrl_c)] != int(bg):
                    continue
                if track_mask[int(ctrl_r), int(ctrl_c)] or node_mask[int(ctrl_r), int(ctrl_c)]:
                    continue
                canvas[int(ctrl_r), int(ctrl_c)] = int(goal_color)

                if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
                    continue

                expected = canvas.copy()
                for gnode in path_nodes[1:-1]:
                    r0, c0 = id_to_pos[int(gnode)]
                    expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
                for e in path_edges:
                    for r, c in edge_track_pixels[tuple(sorted(e))]:
                        expected[int(r), int(c)] = int(Color.GREY)

                try:
                    parsed = _verify_variant_signature(canvas)
                    actual = main(canvas)
                except Exception:
                    continue

                if int(parsed["start_id"]) != 0 or int(parsed["goal_id"]) != int(V - 1):
                    continue
                if not np.array_equal(actual, expected):
                    continue
                return canvas

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        out = _gen_attempt(rng)
        if out is not None:
            k = int(rng.randint(0, 3))
            if k:
                out = np.rot90(out, k=k).copy()
            return out
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
