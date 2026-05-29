# Variant of 7b0280bc
# Dimension: D13 — 终端-中间站基数关系 (meta)
# Change: 2个终端（1:1配对），其余全为中间站... → 所有节点同色需推断终端

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


"""
Design analysis for D13
======================

spec:
  preserved_core:
    - 输入仍然是「实心方块节点 + 轨道组件」编码的图。
    - 输出仍然是：最短路上的中间站染绿，最短路上的轨道染灰。
    - 方块尺寸、轨道宽度、正方形画法、8 邻接触边判边的 solver 机制都保留。
  variant_core:
    - 原题用两种节点颜色区分 terminal/station。
    - 变体中所有节点都使用同一种颜色，terminal 不能再靠颜色识别。
    - terminal 改为：所有度数为 1 的叶子节点中，图距离唯一最远的一对。
  concrete_trace:
    - 原逻辑：`nodes_by_color` 中“恰好出现 2 次”的颜色直接被判成 terminal。
    - 新逻辑：先建图，再找所有叶子；对每对叶子做 BFS 距离比较，选唯一最远叶对。
    - 差异：terminal 的身份从“像素颜色标签”转成“图结构角色”。
  complexity_calibration:
    - 如果只有 2 个叶子，那么虽然同色，终端仍然几乎是显然的，变体不够明显。
    - 生成器因此保证总叶子数至少 3；额外叶子挂在主干中部，避免与 terminal-terminal 距离打平。
    - 为了避免自由度过高，图形采用固定长度主干 + 少量侧叶 + 可选单个 cycle gadget。
  implicit_assumptions_review:
    - 原 `main()` 假设“至少两种节点颜色且其中一种正好出现 2 次”；变体下失效。
    - 原 `main()` 通过排除 terminal/station 颜色来找 track 颜色；变体下改为排除唯一节点颜色。
    - 新 `main()` 新增假设：
      - 节点颜色中有一个主色，承载全部节点方块。
      - 图中至少 3 个叶子。
      - 叶子里最远的一对唯一。
    - 这些新假设都由生成器显式保证，并在 solver 中用 assert 保护。
  placement_legality:
    - 采用 5x6 的局部 cell 模板，主干横向 6 个节点，终端在两端。
    - 额外叶子只挂在主干中部的两个节点上，因此不会与终端形成同长度竞争。
    - 可选 cycle gadget 只绕开主干的一条边，不会缩短 terminal-terminal 的最短距离。
    - 所有轨道仍用原题的 footprint 路由器绘制，并检测越界/重叠。
  visual_observability:
    - 从输入看，所有节点完全同色，解题者必须观察“哪些是叶子、哪一对叶子最远”。
    - 从输出看，被染色的是贯穿主干的那条长路径；侧叶和 cycle gadget 不会被误染。
    - 这与原题“颜色直接指出两个端点”形成可见差异。
  cleanliness:
    - 输入只保留 3 种颜色：背景、节点、轨道；颜色关系更干净，结构线索更突出。
    - cycle gadget 数量限制为 0 或 1，避免画面过乱。
  color_reasoning:
    - 颜色不再承担 terminal/station 分类，只承担“节点 vs 轨道 vs 背景”分区。
    - 输出颜色依旧固定为 green(3)/grey(5)，因此人类能把结构变化与染色结果对应起来。
"""


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
                    "coords": coords,
                }
            )
    return out


def _build_graph_from_grid(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    dominant_size = [d for d in square_cands if int(d["size"]) == int(N)]

    squares_by_color: dict[int, list[dict]] = defaultdict(list)
    for d in dominant_size:
        squares_by_color[int(d["color"])].append(d)
    node_color = int(max(squares_by_color.items(), key=lambda kv: (len(kv[1]), -int(kv[0])))[0])
    node_entries = list(squares_by_color[int(node_color)])
    assert len(node_entries) >= 3

    non_bg = colors_present(g, exclude={bg})
    track_candidates = [c for c in non_bg if int(c) != int(node_color)]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    nodes: list[dict] = []
    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for i, d in enumerate(node_entries):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "pts": pts})
        id2pts[int(i)] = pts
        for p in pts:
            pix2node[p] = int(i)

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

    assert len(adj) >= 2
    return g, id2pts, track_comp_pts, adj


def _bfs_with_counts(start: int, adj: dict[int, list[tuple[int, int]]]):
    q = deque([int(start)])
    dist: dict[int, int] = {int(start): 0}
    count: dict[int, int] = {int(start): 1}
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}

    while q:
        u = int(q.popleft())
        for v, eid in adj.get(int(u), []):
            v = int(v)
            nd = int(dist[int(u)] + 1)
            if v not in dist:
                dist[v] = nd
                count[v] = int(count[int(u)])
                prev[v] = int(u)
                prev_edge[v] = int(eid)
                q.append(v)
            elif dist[v] == nd:
                count[v] += int(count[int(u)])
    return dist, count, prev, prev_edge


def _infer_terminal_pair(adj: dict[int, list[tuple[int, int]]]) -> tuple[int, int]:
    nodes = sorted({int(k) for k in adj} | {int(v) for vals in adj.values() for v, _ in vals})
    degrees = {nid: len(adj.get(int(nid), [])) for nid in nodes}
    leaves = [int(nid) for nid, deg in degrees.items() if int(deg) == 1]
    assert len(leaves) >= 3

    best_pair: tuple[int, int] | None = None
    best_dist = -1
    n_best = 0
    for a, b in combinations(leaves, 2):
        dist, _count, _prev, _prev_edge = _bfs_with_counts(int(a), adj)
        assert int(b) in dist
        dab = int(dist[int(b)])
        if dab > best_dist:
            best_pair = (int(a), int(b))
            best_dist = dab
            n_best = 1
        elif dab == best_dist:
            n_best += 1
    assert best_pair is not None and n_best == 1
    return best_pair


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path between the structurally inferred terminals.

    Variant rule:
    - All node squares have the same input color.
    - Terminal squares are the unique farthest-apart pair among all degree-1 leaf nodes.
    - The shortest path between those terminals is recolored exactly as in the source task:
      intermediate node squares -> green, traversed track components -> grey.
    """
    g, id2pts, track_comp_pts, adj = _build_graph_from_grid(grid)
    start, goal = _infer_terminal_pair(adj)

    dist, count, prev, prev_edge = _bfs_with_counts(int(start), adj)
    assert int(goal) in dist
    assert int(count[int(goal)]) == 1

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


NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
GAP_CHOICES = [2, 3]
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
GEN_ATTEMPT_RETRIES = 300

TEMPLATE_H = 5
TEMPLATE_W = 6
SPINE_CELLS = [(2, c) for c in range(6)]
LEAF_SLOTS = [
    (2, (1, 2)),
    (2, (3, 2)),
    (3, (1, 3)),
    (3, (3, 3)),
]
CYCLE_OPTIONS = [
    {"u": 1, "v": 2, "cells": [(1, 1), (1, 2)]},
    {"u": 1, "v": 2, "cells": [(3, 1), (3, 2)]},
    {"u": 3, "v": 4, "cells": [(1, 3), (1, 4)]},
    {"u": 3, "v": 4, "cells": [(3, 3), (3, 4)]},
]


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


def _transform_coord(r: int, c: int, rot: int, flip_h: bool):
    h, w = TEMPLATE_H, TEMPLATE_W
    rr, cc = int(r), int(c)
    if flip_h:
        cc = w - 1 - cc
    for _ in range(int(rot) % 4):
        rr, cc = cc, h - 1 - rr
        h, w = w, h
    return int(rr), int(cc)


def _transform_layout(cells_by_id: dict[int, tuple[int, int]], rng: random.Random):
    rot = rng.randrange(4)
    flip_h = bool(rng.randrange(2))
    out = {nid: _transform_coord(r, c, rot, flip_h) for nid, (r, c) in cells_by_id.items()}
    h, w = TEMPLATE_H, TEMPLATE_W
    for _ in range(rot % 4):
        h, w = w, h
    return out, int(h), int(w)


def _simple_graph_metrics(edges: list[tuple[int, int]], V: int):
    adj: dict[int, list[int]] = {i: [] for i in range(V)}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def _verify_template_graph(edges: list[tuple[int, int]], V: int):
    adj0 = _simple_graph_metrics(edges, V)
    labeled_adj = {u: [(v, 0) for v in vs] for u, vs in adj0.items()}
    start, goal = _infer_terminal_pair(labeled_adj)
    assert {int(start), int(goal)} == {0, 5}
    dist, count, _prev, _prev_edge = _bfs_with_counts(int(start), labeled_adj)
    assert int(dist[int(goal)]) == 5
    assert int(count[int(goal)]) == 1


def _build_template_graph(V: int, rng: random.Random):
    assert 7 <= V <= 10
    extra_nodes = V - 6

    layout_choices: list[dict | None] = [None]
    if extra_nodes >= 3:
        layout_choices.extend(CYCLE_OPTIONS)
    rng.shuffle(layout_choices)

    cycle = None
    n_leaves = -1
    available_leaf_slots: list[tuple[int, tuple[int, int]]] = []
    for cand in layout_choices:
        cycle_cost = 0 if cand is None else 2
        leaves = extra_nodes - cycle_cost
        if leaves < 1:
            continue
        blocked = set() if cand is None else set(cand["cells"])
        available = [(anchor, cell) for anchor, cell in LEAF_SLOTS if cell not in blocked]
        if leaves <= len(available):
            cycle = cand
            n_leaves = leaves
            available_leaf_slots = available
            break
    assert n_leaves >= 1

    cells_by_id = {i: SPINE_CELLS[i] for i in range(6)}
    edges = [(i, i + 1) for i in range(5)]
    path_nodes = list(range(6))
    path_edges = [tuple(sorted((i, i + 1))) for i in range(5)]
    next_id = 6

    if cycle is not None:
        a_id = next_id
        b_id = next_id + 1
        next_id += 2
        cells = list(cycle["cells"])
        cells_by_id[a_id] = cells[0]
        cells_by_id[b_id] = cells[1]
        edges.extend(
            [
                (int(cycle["u"]), int(a_id)),
                (int(a_id), int(b_id)),
                (int(b_id), int(cycle["v"])),
            ]
        )

    chosen_leaf_slots = rng.sample(available_leaf_slots, n_leaves)
    for anchor, cell in chosen_leaf_slots:
        cells_by_id[next_id] = cell
        edges.append((int(anchor), int(next_id)))
        next_id += 1

    assert next_id == V
    _verify_template_graph([tuple(sorted(e)) for e in edges], V)
    return cells_by_id, [tuple(sorted(e)) for e in edges], path_nodes, path_edges


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    gap = rng.choice(GAP_CHOICES)
    cell_size = N + gap

    cells_by_id, edges, path_nodes, path_edges = _build_template_graph(V, rng)
    cells_by_id, Kr, Kc = _transform_layout(cells_by_id, rng)

    min_h = Kr * cell_size
    min_w = Kc * cell_size
    min_side = max(min_h, min_w)
    if min_side > GRID_SIZE_RANGE[1]:
        return None
    S = rng.randint(max(GRID_SIZE_RANGE[0], min_side), GRID_SIZE_RANGE[1])

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    bg, node_color, track_color = pool[:3]

    canvas = np.full((S, S), int(bg), dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - min_h
    slack_c = S - min_w
    origin = (rng.randint(0, slack_r), rng.randint(0, slack_c))

    id_to_pos: dict[int, tuple[int, int]] = {}
    for i in range(V):
        pos = _cell_to_square_pos(cells_by_id[i], N, cell_size, origin)
        id_to_pos[i] = pos
        r0, c0 = pos
        if not (0 <= r0 and 0 <= c0 and r0 + N <= S and c0 + N <= S):
            return None
        if node_mask[r0 : r0 + N, c0 : c0 + N].any():
            return None
        canvas[r0 : r0 + N, c0 : c0 + N] = int(node_color)
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(cells_by_id[a], cells_by_id[b], id_to_pos[a], id_to_pos[b], N)
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[int(r), int(c)] or track_mask[int(r), int(c)]:
                return None
        for r, c in pixels:
            canvas[int(r), int(c)] = int(track_color)
            track_mask[int(r), int(c)] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = canvas.copy()
    for nid in path_nodes:
        if nid in (0, 5):
            continue
        r0, c0 = id_to_pos[nid]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for e in path_edges:
        for r, c in edge_track_pixels[tuple(sorted(e))]:
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
