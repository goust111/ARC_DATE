from __future__ import annotations

# Variant of 7b0280bc
# Dimension: D11 — 颜色方案 (meta)
# Change: 4种颜色（背景、终端、中间站、轨道），不含绿(3)和灰(5)，输出中绿和灰是固定的涂色... → 输出涂色由输入推断
#
# Design analysis
# ===============
# variant_spec:
#   preserved_core:
#     - 输入主体仍然是“方块节点 + 轨道边”的图，任务仍是找两个终端之间的唯一最短路。
#     - 节点检测、轨道组件检测、BFS 最短路都沿用原题。
#   d11_change:
#     - 输出不再固定染成 green(3) / grey(5)。
#     - 输入额外带两根互不接触的 1x2 水平双色图例条，格式固定为 [源色, 目标色]。
#     - 一根图例条以“中间站颜色”为源色，给出“路径中间站应该染成什么色”。
#     - 另一根图例条以“轨道颜色”为源色，给出“路径轨道应该染成什么色”。
#   concrete_example:
#     - 若图中中间站是红色、轨道是黄色，且图例条分别是 [红, 蓝] 与 [黄, 橙]，
#       则输出里路径上的中间站要染蓝，路径上的轨道要染橙。
#   behavior_trace:
#     - 原逻辑: 找到 node_path / edge_path 后，直接把路径站点无条件写成 3，把路径轨道无条件写成 5。
#     - 变体逻辑: 先从输入剩余的 4 个图例像素恢复两条 [源->目标] 映射，再按映射给 node_path / edge_path 上色。
#     - 差异: 最短路求法不变，真正变化的是“输出色值来自哪里”。
#   complexity_calibration:
#     - 只增加 2 根图例条，共 4 个像素，结构简单但信息足够。
#     - 规则档位只有两条映射，且每条都直接复用图主体里已有的源色，不会让配色关系过多或无支撑。
#   implicit_assumptions_audit:
#     - 原 main 第 177-185 行默认“路径站点一定写 3、路径轨道一定写 5”；该前提在变体中失效，必须改为动态推断。
#     - 原 main 第 103-107 行假设“非节点色里最大面积者是轨道色”；加入图例后仍成立，因为两种目标色只在图例中各出现 1 次。
#     - 新 main 额外假设“除节点与有效轨道外，剩余非背景像素恰好组成两根水平 1x2 图例条”；generator 会显式保证。
#   placement_legality:
#     - 图例条放在背景空位中，要求本体与 1-cell halo 都不接触节点、轨道、另一根图例条。
#     - 这样既不会被误识别成节点，也不会并入轨道组件。
#   visual_observability:
#     - 解题者能直接看见两根固定格式的图例条，而且它们在输入/输出中保持不变。
#     - 同时最短路上的大块站点与长轨道在输出中统一改成图例右半色，规则可从少量样例肉眼归纳。
#   cleanliness:
#     - 图例条很小、隔离放置，不叠压图主体，不制造额外路径或伪节点。
#   color_inference:
#     - 仍只用 ARC 常规颜色，且本变体固定不使用 green(3)/grey(5)，防止退化回原题固定涂色。
#     - 图例给出的是“结构性线索”而不是数值线索，不依赖颜色编号大小或任何不可见顺序。

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, neighbors8


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


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _infer_recolor_targets(
    g: np.ndarray,
    *,
    bg: int,
    station_color: int,
    track_color: int,
    id2pts: dict[int, set[tuple[int, int]]],
    track_comp_pts: dict[int, list[tuple[int, int]]],
) -> tuple[int, int]:
    """
    Infer repaint colors from the two isolated legend bars.

    After removing node pixels and valid track pixels, exactly four non-bg pixels remain.
    They form two horizontal 1x2 bars of the form [source, target].
    """
    graph_pixels: set[tuple[int, int]] = set()
    for pts in id2pts.values():
        graph_pixels.update(pts)
    for pts in track_comp_pts.values():
        graph_pixels.update(pts)

    legend_pixels = [
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in graph_pixels
    ]
    assert len(legend_pixels) == 4

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, c in legend_pixels:
        legend_mask[int(r), int(c)] = True

    legend_comps = components_mask(legend_mask, connectivity=4)
    assert len(legend_comps) == 2

    station_recolor = None
    track_recolor = None
    for coords in legend_comps:
        pts = sorted([(int(r), int(c)) for r, c in coords], key=lambda p: (p[0], p[1]))
        assert len(pts) == 2
        (r0, c0), (r1, c1) = pts
        assert r0 == r1 and c1 == c0 + 1
        src = int(g[r0, c0])
        dst = int(g[r1, c1])
        if src == int(station_color):
            station_recolor = dst
        elif src == int(track_color):
            track_recolor = dst

    assert station_recolor is not None and track_recolor is not None
    return int(station_recolor), int(track_recolor)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path between two terminal squares in a square-and-track graph.

    Variant rule:
    - The path itself is still found exactly as in the base task.
    - Repaint colors are not fixed. They are inferred from two small horizontal legend bars
      elsewhere in the input, each encoding [source_color -> target_color].
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

    start, goal = int(term_ids[0]), int(term_ids[1])
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
    cur = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        cur2 = prev.get(int(cur))
        cur = int(cur2) if cur2 is not None else None
    node_path.reverse()
    edge_path.reverse()

    station_recolor, track_recolor = _infer_recolor_targets(
        g,
        bg=bg,
        station_color=int(station_color),
        track_color=int(track_color),
        id2pts=id2pts,
        track_comp_pts=track_comp_pts,
    )

    out = g.copy()
    term_set = set(map(int, term_ids))
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(station_recolor)

    for eid in edge_path:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(track_recolor)

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
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 300


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
    new_path_edges = [
        tuple(sorted((new_path_nodes[i], new_path_nodes[i + 1])))
        for i in range(len(new_path_nodes) - 1)
    ]

    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _cell_to_square_pos(cell: tuple[int, int], N: int, cell_size: int, origin: tuple[int, int]):
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return or_r + i * cell_size + margin, or_c + j * cell_size + margin


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
) -> list[tuple[int, int]]:
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


def _domino_slot_clear(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    if not (0 <= int(r) < H and 0 <= int(c) and int(c) + 1 < W):
        return False
    rr0 = max(0, int(r) - 1)
    rr1 = min(H, int(r) + 2)
    cc0 = max(0, int(c) - 1)
    cc1 = min(W, int(c) + 3)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _reserve_domino(mask: np.ndarray, r: int, c: int):
    H, W = mask.shape
    rr0 = max(0, int(r) - 1)
    rr1 = min(H, int(r) + 2)
    cc0 = max(0, int(c) - 1)
    cc1 = min(W, int(c) + 3)
    mask[rr0:rr1, cc0:cc1] = True


def _place_legend_bars(
    canvas: np.ndarray,
    occupied: np.ndarray,
    bars: list[tuple[int, int]],
    rng: random.Random,
) -> bool:
    H, W = canvas.shape
    reserve = occupied.copy()

    for src, dst in bars:
        candidates = []
        for r in range(H):
            for c in range(W - 1):
                if _domino_slot_clear(reserve, r, c):
                    border_score = min(r, H - 1 - r, c, W - 2 - c)
                    candidates.append((border_score, rng.random(), r, c))
        if not candidates:
            return False
        candidates.sort()
        _score, _noise, r, c = candidates[0]
        canvas[int(r), int(c)] = int(src)
        canvas[int(r), int(c) + 1] = int(dst)
        _reserve_domino(reserve, int(r), int(c))
    return True


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

    bg, terminal_color, station_color, track_color, station_recolor, track_recolor = rng.sample(COLOR_POOL, 6)

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
        color = terminal_color if i in (0, V - 1) else station_color
        r0, c0 = pos
        canvas[r0 : r0 + N, c0 : c0 + N] = color
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (a, b) in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N)
        for (r, c) in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for (r, c) in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    if not _place_legend_bars(
        canvas,
        node_mask | track_mask,
        [(station_color, station_recolor), (track_color, track_recolor)],
        rng,
    ):
        return None

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    expected = canvas.copy()
    for gnode in path_nodes:
        if gnode in (0, V - 1):
            continue
        r0, c0 = id_to_pos[gnode]
        expected[r0 : r0 + N, c0 : c0 + N] = station_recolor
    for e in path_edges:
        for (r, c) in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = track_recolor

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
