# Variant of 7b0280bc
# Dimension: D1 — 路径选择策略 (path)
# Change: BFS最短路径（边数最少），且最短路径唯一... → 加权最短路径

from __future__ import annotations

# Design analysis for this D1 variant
# -----------------------------------
# a) Concrete behavior + trace
#    原题把每条轨道连通分量当作“等权边”，所以 terminals 之间只比较经过几段轨道。
#    本变体仍然是“方块节点 + 轨道边”的同一视觉语言，但每条边的权重改成该轨道连通分量的像素数。
#    因此人类不能只数“几段边”，还要估算“哪条轨道更长”。
#    生成器固定制造两类候选路线：
#    - 路线 A：边数更少（例如 2 段），但每段很长，总像素数大。
#    - 路线 B：边数更多（例如 4 段），但每段很短，总像素数小。
#    小例子：若 A 的两段轨道像素数是 15 和 15，总权重 30；B 的四段像素数是 4/4/6/4，总权重 18。
#    原 BFS 会选 A（2 边），新规则必须选 B（18 < 30）。输出因此会把 B 上的中间站染绿，把 B 的轨道染灰。
#
# b) Complexity calibration
#    只把 solver 从 BFS 改成 Dijkstra 还不够，因为随机图容易退化成“边数最少路径恰好也是最短权重路径”。
#    所以 generator 不再生成任意连通图，而是稳定生成“长桥路径 vs 短链路径”的竞争结构，再加 1-4 个叶子节点做干扰。
#    这样视觉上能直接看到：输出选中的不是更少边的路线，而是更短总长度的路线。
#    为了让“边权 = 轨道像素数”直接可见，生成器把节点固定为 2x2，于是轨道宽度固定为 1，权重就是肉眼可数的轨道长度。
#
# c) Implicit-assumption audit of original main()
#    - `track_comps = components_mask(...)`：原来只需要 edge_id；现在仍然成立，但每个连通分量还要记录像素数作为权重。
#    - `adj[node].append((neighbor, edge_id))`：原来隐含“所有边等价”；本变体不成立，必须改成 `(..., weight)`。
#    - `deque` BFS：原来正确，因为目标函数是边数。现在错误，因为先到达不代表总权重最小，必须改成 Dijkstra。
#    - 路径回溯逻辑本身仍可复用，只要 `prev/prev_edge` 改由 Dijkstra 更新。
#    - 颜色与对象检测逻辑不依赖权重定义，保持不变。
#
# d) Placement legality
#    所有节点仍是统一大小的实心 NxN 方块；轨道仍是单一颜色的连通分量，且每个分量只接触恰好两个节点。
#    生成器先定义粗网格上的节点模板，再把模板嵌入画布。放置时逐像素检查：
#    - 不越界
#    - 不覆盖节点
#    - 不覆盖已有轨道
#    然后再用 solver 自己重新抽图验证边数、最短路和期望输出。
#
# e) Visual observability
#    关键差异是“灰色路径并非边数最少，而是总长度更短”。模板中长桥路径通常是一条明显的长横线 + 长竖线，
#    而被选中的路径是数段较短的斜/折线串联。2-3 个样例就足以让人看到规则变化。
#
# f) Visual cleanliness
#    模板使用一条主对角短链、一条外围长桥和少量叶子支路；轨道不会相互覆盖。叶子只作为干扰，不参与 terminals 间最优路。
#    旋转/翻转只改变整体朝向，不会把局部挤成难以辨认的团块。
#
# g) Color inferability
#    颜色规则完全沿用原题：中间站 -> 绿(3)，被选中的轨道 -> 灰(5)。颜色种类仍控制在 4 种。
#
# h) Behavioral guard against regression
#    generator 显式检查：同一输入上，按边数的 BFS 路径必须与按像素权重的 Dijkstra 路径不同。
#    这避免样本退化成原题。
#
# i) New assumptions introduced by the new main()
#    - 每条轨道必须是独立连通分量，这样“轨道像素数 = 边权”才有定义。
#    - 终端间存在唯一的最小总权重路径。generator 通过固定模板和求解后校验来保证这一点。
#    - 轨道像素数必须能稳定反映“长/短”差异，因此模板中的长桥边和短链边在几何长度上刻意拉开。

from collections import Counter, defaultdict, deque
import heapq
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


def _extract_graph(grid: np.ndarray) -> dict:
    """Recover nodes, track components, and weighted adjacency from the grid."""
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
    edge_weights: dict[int, int] = {}
    edge_nodes: dict[int, tuple[int, int]] = {}
    adj: dict[int, list[tuple[int, int, int]]] = defaultdict(list)  # node -> [(neighbor, edge_id, weight)]

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
        w = int(len(pts))
        track_comp_pts[int(edge_id)] = pts
        edge_weights[int(edge_id)] = w
        edge_nodes[int(edge_id)] = (int(a), int(b))
        adj[int(a)].append((int(b), int(edge_id), int(w)))
        adj[int(b)].append((int(a), int(edge_id), int(w)))

    return {
        "grid": g,
        "bg": bg,
        "N": N,
        "term_ids": term_ids,
        "id2pts": id2pts,
        "track_comp_pts": track_comp_pts,
        "edge_weights": edge_weights,
        "edge_nodes": edge_nodes,
        "adj": adj,
    }


def _reconstruct_path(
    start: int,
    goal: int,
    prev: dict[int, int | None],
    prev_edge: dict[int, int | None],
) -> tuple[list[int], list[int]]:
    assert goal in prev
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


def _best_path_by_weight(graph: dict) -> tuple[list[int], list[int], int]:
    """Dijkstra on track-component pixel counts."""
    start, goal = map(int, graph["term_ids"])
    adj = graph["adj"]

    pq: list[tuple[int, int]] = [(0, int(start))]
    dist: dict[int, int] = {int(start): 0}
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}

    while pq:
        cur_dist, u = heapq.heappop(pq)
        u = int(u)
        if cur_dist != dist.get(u):
            continue
        if u == goal:
            break
        for v, eid, weight in adj.get(u, []):
            v = int(v)
            nd = int(cur_dist + int(weight))
            if nd < dist.get(v, 10**9):
                dist[v] = nd
                prev[v] = u
                prev_edge[v] = int(eid)
                heapq.heappush(pq, (nd, v))

    node_path, edge_path = _reconstruct_path(start, goal, prev, prev_edge)
    return node_path, edge_path, int(dist[goal])


def _best_path_by_edges(graph: dict) -> tuple[list[int], list[int], int]:
    """BFS helper used only by generator diagnostics."""
    start, goal = map(int, graph["term_ids"])
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}

    while q:
        u = int(q.popleft())
        if u == goal:
            break
        for v, eid, _weight in graph["adj"].get(u, []):
            v = int(v)
            if v in prev:
                continue
            prev[v] = u
            prev_edge[v] = int(eid)
            q.append(v)

    node_path, edge_path = _reconstruct_path(start, goal, prev, prev_edge)
    return node_path, edge_path, int(len(edge_path))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the minimum-total-track-length path between the two terminal squares.

    Edge weight = number of pixels in that track component.
    """
    graph = _extract_graph(np.array(grid, dtype=int))
    node_path, edge_path, _ = _best_path_by_weight(graph)

    out = np.array(graph["grid"], copy=True)
    term_set = set(map(int, graph["term_ids"]))
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in graph["id2pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in graph["track_comp_pts"][int(eid)]:
            out[int(r), int(c)] = int(Color.GREY)
    return out


# --------------------------- generator ---------------------------

NODE_SIZE = 2
CONTENT_SIZE = 22
GRID_SIZE_RANGE = (22, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
GEN_ATTEMPT_RETRIES = 300

SQUARES = {
    0: (1, 1),
    1: (1, 18),
    2: (6, 4),
    3: (10, 8),
    4: (14, 12),
    5: (18, 16),
    6: (8, 13),
    7: (9, 1),
}
EDGE_PIXELS = {
    (0, 1): [(1, c) for c in range(3, 18)],
    (1, 5): [(r, 18) for r in range(3, 18)],
    (0, 2): [(3, 3), (4, 4), (5, 5)],
    (2, 3): [(8, 6), (9, 7)],
    (3, 4): [(12, 10), (13, 11)],
    (4, 5): [(16, 14), (17, 15)],
    (3, 6): [(10, 10), (9, 11), (9, 12)],
    (2, 7): [(8, 3), (9, 3)],
}
WEIGHTED_PATH_EDGES = {(0, 2), (2, 3), (3, 4), (4, 5)}
WEIGHTED_PATH_STATIONS = {2, 3, 4}


def _choose_colors(rng: random.Random) -> tuple[int, int, int, int]:
    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    bg, terminal_color, station_color, track_color = pool[:4]
    return int(bg), int(terminal_color), int(station_color), int(track_color)


def _make_base_masks() -> dict[str, np.ndarray]:
    terminal = np.zeros((CONTENT_SIZE, CONTENT_SIZE), dtype=bool)
    station = np.zeros((CONTENT_SIZE, CONTENT_SIZE), dtype=bool)
    track = np.zeros((CONTENT_SIZE, CONTENT_SIZE), dtype=bool)
    path_station = np.zeros((CONTENT_SIZE, CONTENT_SIZE), dtype=bool)
    path_track = np.zeros((CONTENT_SIZE, CONTENT_SIZE), dtype=bool)

    for nid, (r0, c0) in SQUARES.items():
        target = terminal if nid in (0, 5) else station
        target[int(r0) : int(r0 + NODE_SIZE), int(c0) : int(c0 + NODE_SIZE)] = True
        if nid in WEIGHTED_PATH_STATIONS:
            path_station[int(r0) : int(r0 + NODE_SIZE), int(c0) : int(c0 + NODE_SIZE)] = True

    for edge, pts in EDGE_PIXELS.items():
        for r, c in pts:
            track[int(r), int(c)] = True
            if edge in WEIGHTED_PATH_EDGES:
                path_track[int(r), int(c)] = True

    return {
        "terminal": terminal,
        "station": station,
        "track": track,
        "path_station": path_station,
        "path_track": path_track,
    }


def _transform_masks(masks: dict[str, np.ndarray], rng: random.Random) -> dict[str, np.ndarray]:
    k = int(rng.randrange(4))
    do_flip = bool(rng.randrange(2))
    out: dict[str, np.ndarray] = {}
    for key, mask in masks.items():
        cur = np.rot90(mask, k)
        if do_flip:
            cur = np.fliplr(cur)
        out[key] = cur
    return out


def _build_expected_output(canvas: np.ndarray, path_station_mask: np.ndarray, path_track_mask: np.ndarray) -> np.ndarray:
    out = np.array(canvas, copy=True)
    out[np.array(path_station_mask, dtype=bool)] = int(Color.GREEN)
    out[np.array(path_track_mask, dtype=bool)] = int(Color.GREY)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    S = int(rng.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1]))
    off_r = int(rng.randint(0, S - CONTENT_SIZE))
    off_c = int(rng.randint(0, S - CONTENT_SIZE))
    masks = _transform_masks(_make_base_masks(), rng)

    bg, terminal_color, station_color, track_color = _choose_colors(rng)
    canvas = np.full((S, S), int(bg), dtype=int)
    terminal_mask = np.zeros((S, S), dtype=bool)
    station_mask = np.zeros((S, S), dtype=bool)
    track_mask = np.zeros((S, S), dtype=bool)
    path_station_mask = np.zeros((S, S), dtype=bool)
    path_track_mask = np.zeros((S, S), dtype=bool)

    r1, c1 = off_r + CONTENT_SIZE, off_c + CONTENT_SIZE
    terminal_mask[off_r:r1, off_c:c1] = masks["terminal"]
    station_mask[off_r:r1, off_c:c1] = masks["station"]
    track_mask[off_r:r1, off_c:c1] = masks["track"]
    path_station_mask[off_r:r1, off_c:c1] = masks["path_station"]
    path_track_mask[off_r:r1, off_c:c1] = masks["path_track"]

    if np.any(terminal_mask & station_mask):
        return None
    if np.any(track_mask & (terminal_mask | station_mask)):
        return None

    canvas[terminal_mask] = int(terminal_color)
    canvas[station_mask] = int(station_color)
    canvas[track_mask] = int(track_color)

    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = _build_expected_output(canvas, path_station_mask, path_track_mask)

    try:
        actual = main(canvas)
        graph = _extract_graph(canvas)
        node_path_w, edge_path_w, dist_w = _best_path_by_weight(graph)
        node_path_b, edge_path_b, _ = _best_path_by_edges(graph)
    except Exception:
        return None

    if len(graph["track_comp_pts"]) != len(EDGE_PIXELS):
        return None
    if not np.array_equal(actual, expected):
        return None
    if edge_path_w == edge_path_b:
        return None
    if len(edge_path_b) >= len(edge_path_w):
        return None

    weight_w = int(sum(graph["edge_weights"][eid] for eid in edge_path_w))
    weight_b = int(sum(graph["edge_weights"][eid] for eid in edge_path_b))
    if dist_w != weight_w or not (weight_w < weight_b):
        return None
    if len(node_path_w) <= len(node_path_b):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        out = _gen_attempt(rng)
        if out is not None:
            return out
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
