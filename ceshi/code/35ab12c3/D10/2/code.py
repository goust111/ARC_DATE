# Variant of 35ab12c3
# Dimension: D10 — 形状种类与拓扑 (meta)
# Change: 六种形状类型（折线/三角形/矩形/菱形/五边形/六边形），通过加权随机选择，包含开放折线和闭合多边形两种拓扑... → 分支树拓扑

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

"""
Design analysis for the D10 variant
===================================

variant_spec:
  core_change:
    old:
      - 原 solver 把同色点按质心极角排序，然后连接成 open line / closed polygon。
      - 这等价于默认 "每个顶点度数 <= 2"，不会产生分叉。
    new:
      - 每个可解对象改成一棵 8 方向直线树。
      - 树中必须存在一个分叉点，度数 >= 3。
      - 输出仍然是把输入中的同色顶点用水平 / 垂直 / 45 度线段补全。
      - 阴影规则保留：若某个顶点旁有异色像素，则把整棵树按该偏移平移复制。

  concrete_pixel_behavior:
    input:
      - 形状色只保留树的顶点像素，不保留边中间的像素。
      - 常见局部形状是一个 hub 点，加上 3-4 个叶子点，偶尔一条枝带一个转折顶点。
      - 阴影色只保留 1 个锚点像素，并且该像素 8 邻域里恰好只接触一个形状顶点。
    output:
      - 用形状色填满整棵树的所有边。
      - 再用阴影色沿同一组边做平行复制；形状本体优先，阴影只能落在背景上。

  small_trace:
    old_logic:
      - 四个顶点若呈 T 形，极角排序会把它们排成一个环序列。
      - 连接相邻点会得到错误的折线/四边形，不会恢复 T 形的三叉结构。
    new_logic:
      - 枚举所有同色顶点对。
      - 只有 "8 方向对齐 + 原图中间无遮挡 + 线段内部没有别的同色顶点" 的点对才算原子边。
      - 若这些原子边构成连通无环图，且最大度数 >= 3，就把它识别为分支树并全部绘制。
    difference:
      - 关键变化发生在 solver 的 "边推断" 环节，以及 generator 的 "对象拓扑" 环节。

  complexity_calibration:
    - 不再混入 polygon / polyline；所有 shape 都显式带分叉，避免样本退化回原题。
    - 仍保留 2-4 个对象、0-3 个阴影、0-1 个干扰色，保持原题的多对象布局复杂度。
    - 树模板只允许少量自由度：3 或 4 个主枝，至多 1 条折弯枝，避免规则过散。

  old_main_assumptions_audited:
    - _angle_cycle(points_rc):
      assumption: 同色点存在可排序的单环/单链顺序。
      status: 在树里失效；必须删除。
    - cycle[start:] + cycle[:start]:
      assumption: "起点" 只影响从哪个端点开始沿链/环走。
      status: 在树里没有单一遍历环序；改为只用于找阴影锚点，不再决定边。
    - edges = consecutive pairs (or wraparound):
      assumption: 每个点只需连前后两个邻居。
      status: 在分叉点失效；改为从可见直线邻接图恢复全部边。
    - drawable_edges phase:
      assumption: 即使漏掉一个分支，其他边仍来自 cycle。
      status: 不成立；现在直接以推断图为唯一边集合。
    - carried shadows copied per edge:
      assumption: 阴影只需沿已知主边平移。
      status: 仍成立，只要主边集合正确。

  placement_legality:
    - 生成器直接构造树的顶点和显式边，不再从顶点环自动闭合。
    - 每个候选树都做几何验证：
      1. 每条边都严格 H/V/45°。
      2. 非相邻边不能共享网格单元。
      3. 由顶点反推的原子边集合必须恰好等于预设边集合。
      4. 图必须是树，且最大度数 >= 3。
    - 放到大画布时仍使用 output patch + padding，保证不同对象的输出区域彼此分离。

  visual_observability:
    - 肉眼可见的变化是：输出里出现 T / Y / 多叉树，而不是闭合多边形或单链。
    - 不使用颜色 id 做任何可见排序或映射；颜色只做对象区分。
    - 每个训练样例中会反复出现多个分叉对象，足以让人归纳 "同色点形成树"。

  cleanliness:
    - 通过 patch 放置和 >=1 背景边距，避免不同树或阴影互相贴边。
    - 每棵树最多一个折弯枝，防止局部过密和难辨认交叉。

  new_solver_assumptions:
    - 一个 shape 色的输入顶点集合，在 "可见直线相邻顶点" 规则下会形成唯一树。
    - 若存在阴影像素，它只会 8 邻接一个顶点，因此锚点无歧义。
    - generator 通过局部几何验证和 main(inp)==out oracle check 保证这些假设成立。
"""


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _has_interior_same_color_vertex(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        if int(g[rr, cc]) == col:
            return True
    return False


def _infer_tree_edges(g: np.ndarray, pts_rc: list[tuple[int, int]], col: int, bg: int):
    pts_rc = sorted(pts_rc)
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    degree = {p: 0 for p in pts_rc}
    adj = {p: set() for p in pts_rc}

    for i, a in enumerate(pts_rc):
        for b in pts_rc[i + 1 :]:
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            if not _clear_in_original(g, a[0], a[1], b[0], b[1], bg, allow={col}):
                continue
            if _has_interior_same_color_vertex(g, a[0], a[1], b[0], b[1], col):
                continue
            edges.append((a, b))
            degree[a] += 1
            degree[b] += 1
            adj[a].add(b)
            adj[b].add(a)

    if len(edges) != len(pts_rc) - 1:
        return []

    start = pts_rc[0]
    stack = [start]
    seen = {start}
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)

    if len(seen) != len(pts_rc):
        return []
    if max(degree.values()) < 3:
        return []
    return edges


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        seed_vertices = []
        for r, c in sorted(pts_rc):
            carried_here = []
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        carried_here.append((drn, dcn, v))
            if carried_here:
                seed_vertices.append(((r, c), carried_here))

        if seed_vertices:
            (_, _), carried = seed_vertices[0]
        else:
            carried = []

        drawable_edges = _infer_tree_edges(g, pts_rc, col, bg)
        if not drawable_edges:
            continue

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

TREE_BRANCH_CHOICES = [3, 4]
TREE_STRAIGHT_LEN_RANGE = (2, 7)
TREE_BEND_FIRST_RANGE = (2, 5)
TREE_BEND_SECOND_RANGE = (2, 5)
TREE_BEND_PROB = 0.4
TREE_BUILD_RETRIES = 100

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 300


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _canonical_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _local_inferred_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pts = sorted(vertices)
    pt_set = set(pts)
    edges = []
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            blocked = False
            for rr, cc in _segment_cells(a[0], a[1], b[0], b[1]):
                if (rr, cc) == a or (rr, cc) == b:
                    continue
                if (rr, cc) in pt_set:
                    blocked = True
                    break
            if not blocked:
                edges.append((a, b))
    return edges


def _validate_tree_geometry(vertices: list[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
    if len(set(vertices)) != len(vertices):
        return False
    if len(vertices) < 4:
        return False
    if len(edges) != len(vertices) - 1:
        return False

    edge_sets: list[set[tuple[int, int]]] = []
    degrees = {v: 0 for v in vertices}
    for a, b in edges:
        if not _aligned_8(a[0], a[1], b[0], b[1]):
            return False
        for rr, cc in _segment_cells(a[0], a[1], b[0], b[1]):
            if (rr, cc) != a and (rr, cc) != b and (rr, cc) in degrees:
                return False
        edge_sets.append(set(_gen_segment_cells(a, b)))
        degrees[a] += 1
        degrees[b] += 1

    for i in range(len(edges)):
        ai, bi = edges[i]
        allowed = {ai, bi}
        for j in range(i + 1, len(edges)):
            aj, bj = edges[j]
            shared = edge_sets[i] & edge_sets[j]
            common_endpoints = allowed & {aj, bj}
            if shared != common_endpoints:
                return False

    inferred = {_canonical_edge(a, b) for a, b in _local_inferred_edges(vertices)}
    intended = {_canonical_edge(a, b) for a, b in edges}
    if inferred != intended:
        return False
    if max(degrees.values()) < 3:
        return False
    return True


def _build_branch_tree() -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    for _ in range(TREE_BUILD_RETRIES):
        hub = (0, 0)
        vertices = [hub]
        edges_idx: list[tuple[int, int]] = []
        branch_count = int(np.random.choice(TREE_BRANCH_CHOICES))
        dir_indices = np.random.choice(len(DIR8), size=branch_count, replace=False)
        bend_branch = int(np.random.randint(branch_count)) if np.random.rand() < TREE_BEND_PROB else None

        failed = False
        for branch_i, dir_idx in enumerate(dir_indices):
            dr, dc = DIR8[int(dir_idx)]
            if bend_branch is not None and branch_i == bend_branch:
                leg1 = int(np.random.randint(*TREE_BEND_FIRST_RANGE))
                mid = (hub[0] + dr * leg1, hub[1] + dc * leg1)
                if mid in vertices:
                    failed = True
                    break
                vertices.append(mid)
                mid_idx = len(vertices) - 1
                edges_idx.append((0, mid_idx))

                turn_dirs = [d for d in DIR8 if d != (dr, dc) and d != (-dr, -dc)]
                turn_order = np.random.permutation(len(turn_dirs))
                placed_turn = False
                for turn_pick in turn_order:
                    tr, tc = turn_dirs[int(turn_pick)]
                    leg2 = int(np.random.randint(*TREE_BEND_SECOND_RANGE))
                    leaf = (mid[0] + tr * leg2, mid[1] + tc * leg2)
                    if leaf in vertices:
                        continue
                    trial_vertices = vertices + [leaf]
                    trial_edges = [(trial_vertices[a], trial_vertices[b]) for a, b in edges_idx + [(mid_idx, len(vertices))]]
                    if _validate_tree_geometry(trial_vertices, trial_edges):
                        vertices.append(leaf)
                        edges_idx.append((mid_idx, len(vertices) - 1))
                        placed_turn = True
                        break
                if not placed_turn:
                    failed = True
                    break
            else:
                leg = int(np.random.randint(*TREE_STRAIGHT_LEN_RANGE))
                leaf = (hub[0] + dr * leg, hub[1] + dc * leg)
                if leaf in vertices:
                    failed = True
                    break
                vertices.append(leaf)
                edges_idx.append((0, len(vertices) - 1))

        if failed:
            continue

        t_vertices = _transform_coords(vertices)
        t_edges = [(t_vertices[a], t_vertices[b]) for a, b in edges_idx]
        if _validate_tree_geometry(t_vertices, t_edges):
            return t_vertices, t_edges
    return None


def _shape_cells(edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in edges:
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in edges]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    directions = _edge_dirs(edges)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for anchor in vertices:
        for dr, dc in DIR8:
            if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
                continue
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in shape_cells:
                continue
            if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
                continue
            candidates.append((anchor, (dr, dc)))
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (_ar, _ac), (dr, dc) = shadow
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    built = _build_branch_tree()
    if built is None:
        return None
    vertices, edges = built
    cells = _shape_cells(edges)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, edges, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    shifted_edges = [
        ((a[0] - r0, a[1] - c0), (b[0] - r0, b[1] - c0))
        for a, b in edges
    ]

    for a, b in shifted_edges:
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, shape_color, int(shadow_color), offset)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(c) for c in palette[:shape_count]]
    shadow_colors = [int(c) for c in palette[shape_count : shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
