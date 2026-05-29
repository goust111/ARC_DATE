# Variant of 35ab12c3
# Dimension: D3 — 顶点排序与连接拓扑 (path)
# Change: 极角排序（angle_cycle）——所有同色顶点按围绕质心的极角排序，2个顶点连直线段，3+个顶点按顺序连成闭合多边形... → 最小生成树连接

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# a) Concrete behavior / trace
#    原题把同色顶点按极角排成 cycle，所以 4 个矩形角点会画出闭环；本变体改为：
#    先枚举所有 H/V/45° 且中间无遮挡的同色候选边，再按长度最短优先做确定性 Kruskal，
#    得到无环连通树。例子：角点 {(0,0),(0,4),(4,4),(4,0)} 在原题会画 4 条边闭合矩形；
#    本题会选 3 条长度为 4 的边形成“U”形树，不再闭环。
#    阴影逻辑不变：选带异色邻居的起始顶点，记录其 8 邻域偏移，并把整棵树沿该偏移平行复制。
#
# b) Complexity calibration
#    如果生成器只产出 2 点线段或纯直线链，MST 与原题视觉差异不明显。这里强制每个样例至少
#    有 1 个真正 branching 的同色组（输出中最大度数 >= 3），其余形状在链与分叉树之间混合。
#    同时把方向族限制在最多 3 类，给阴影留下可见的非共线偏移。
#
# c) Implicit-assumption review for main()
#    - _aligned_8 / _segment_cells / _clear_in_original 只依赖“边是 H/V/45° 直段”，这在变体中仍成立。
#    - 原来的 _angle_cycle 依赖“全部顶点能排成唯一环/链”；该前提已失效，必须替换为候选边图 + MST。
#    - start vertex 的用途只是确定 shadow offset 来源，不依赖成环顺序；保留。
#    - _draw_line_bg_only 依赖“主形状先画、阴影后画，主形状优先级更高”；MST 仍适用。
#
# d) Placement legality
#    生成器先构造局部树补丁，再用 random_free_location 按输出补丁放置，确保不同对象输出像素
#    间至少隔 1 个背景格。局部补丁在提交前额外做两层验证：
#    1. 局部输入上的 solver-MST 必须恰好等于生成器意图树；
#    2. main(local_input) 必须精确重建 local_output。
#
# e) Visual observability
#    人类能从 2-4 例中看出区别，因为至少一个 4-7 顶点对象会输出成树杈，而不是闭合多边形。
#    排序/选边完全基于几何可见量（直线对齐与长度），不使用颜色 id 做排序。
#
# f) Clean visuals
#    只保留单层阴影；对象间保持 padding；树边禁止自交/重叠，避免同一区域多层覆盖。
#
# g) Color inference
#    颜色规则未变：每个对象/阴影/干扰色都使用独立颜色，不引入新的颜色映射负担。
#
# h) Post-change solver assumptions
#    main() 现在假设：某个可解的同色对象，其所有真实边都包含在“对齐且无遮挡”的候选边集中，
#    且该候选图的确定性 MST 就是目标拓扑。generate_input() 通过 oracle 过滤精确保证这一点。
# ---------------------------------------------------------------------------

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR_FAMILIES = [
    [(0, 1), (0, -1)],
    [(1, 0), (-1, 0)],
    [(1, 1), (-1, -1)],
    [(1, -1), (-1, 1)],
]


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


def _canonical_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _edge_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _candidate_edges(points_rc: list[tuple[int, int]], g: np.ndarray, bg: int, col: int):
    edges = []
    n = len(points_rc)
    for i in range(n):
        for j in range(i + 1, n):
            a = points_rc[i]
            b = points_rc[j]
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            if not _clear_in_original(g, a[0], a[1], b[0], b[1], bg, allow={col}):
                continue
            aa, bb = _canonical_edge(a, b)
            edges.append((_edge_len(aa, bb), aa, bb))
    edges.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[2][0], x[2][1]))
    return edges


def _mst_edges(points_rc: list[tuple[int, int]], g: np.ndarray, bg: int, col: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(points_rc) < 2:
        return []

    pts = sorted(points_rc)
    parent = {p: p for p in pts}
    rank = {p: 0 for p in pts}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for _w, a, b in _candidate_edges(pts, g, bg, col):
        if union(a, b):
            chosen.append((a, b))
            if len(chosen) == len(pts) - 1:
                break
    return chosen


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

        seed_pts = []
        for r, c in pts_rc:
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_pts.append((r, c))
                        break
        start = min(seed_pts) if seed_pts else min(pts_rc)

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = start
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        drawable_edges = _mst_edges(pts_rc, g, bg, col)
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


# --------------- generator (D3 variant) ---------------

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

VERTEX_COUNT_RANGE = (2, 8)
SEG_LEN_RANGE = (2, 7)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TREE_BUILD_RETRIES = 100
BRANCH_SHAPE_PROB = 0.6

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 400


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


def _edge_coords(vertices: list[tuple[int, int]], edge_indices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[j]) for i, j in edge_indices]


def _tree_cells(vertices: list[tuple[int, int]], edge_indices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _edge_coords(vertices, edge_indices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _degrees(edge_indices: list[tuple[int, int]], n_vertices: int) -> list[int]:
    deg = [0] * n_vertices
    for i, j in edge_indices:
        deg[i] += 1
        deg[j] += 1
    return deg


def _max_degree(edge_indices: list[tuple[int, int]], n_vertices: int) -> int:
    return max(_degrees(edge_indices, n_vertices), default=0)


def _build_chain_spec() -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    n_vertices = int(np.random.randint(*VERTEX_COUNT_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        if gap_hi < LINE_MIN_GAP:
            return None
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    vertices = [(0, x) for x in pos]
    edge_indices = [(i, i + 1) for i in range(n_vertices - 1)]
    return vertices, edge_indices


def _build_branch_spec() -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    total_vertices = int(np.random.randint(4, VERTEX_COUNT_RANGE[1]))
    family_ids = np.random.choice(len(DIR_FAMILIES), size=3, replace=False).tolist()
    lengths = np.random.choice(np.arange(SEG_LEN_RANGE[0], SEG_LEN_RANGE[1]), size=3, replace=False).tolist()

    vertices = [(0, 0)]
    edge_indices: list[tuple[int, int]] = []
    branch_tips: list[tuple[int, tuple[int, int]]] = []

    for fam, length in zip(family_ids, lengths):
        direction = DIR_FAMILIES[int(fam)][int(np.random.randint(2))]
        new_pt = (direction[0] * int(length), direction[1] * int(length))
        vertices.append(new_pt)
        edge_indices.append((0, len(vertices) - 1))
        branch_tips.append((len(vertices) - 1, direction))

    while len(vertices) < total_vertices:
        tip_idx = int(np.random.randint(len(branch_tips)))
        parent_idx, direction = branch_tips[tip_idx]
        parent = vertices[parent_idx]
        length = int(np.random.randint(*SEG_LEN_RANGE))
        new_pt = (parent[0] + direction[0] * length, parent[1] + direction[1] * length)
        if new_pt in vertices:
            return None
        vertices.append(new_pt)
        edge_indices.append((parent_idx, len(vertices) - 1))
        branch_tips[tip_idx] = (len(vertices) - 1, direction)

    return vertices, edge_indices


def _spec_to_transformed(vertices: list[tuple[int, int]], edge_indices: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    transformed = _transform_coords(vertices)
    return transformed, edge_indices


def _mst_matches_intended(vertices: list[tuple[int, int]], edge_indices: list[tuple[int, int]]) -> bool:
    coords = np.array(vertices, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    inp = np.full((h, w), BG, dtype=int)
    shifted = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted:
        inp[r, c] = 1
    got = {_canonical_edge(a, b) for a, b in _mst_edges(shifted, inp, BG, 1)}
    want = {_canonical_edge(shifted[i], shifted[j]) for i, j in edge_indices}
    return got == want


def _make_tree_spec(force_branch: bool) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(TREE_BUILD_RETRIES):
        if force_branch or np.random.rand() < BRANCH_SHAPE_PROB:
            spec = _build_branch_spec()
        else:
            spec = _build_chain_spec()
        if spec is None:
            continue
        vertices, edge_indices = _spec_to_transformed(*spec)
        if force_branch and _max_degree(edge_indices, len(vertices)) < 3:
            continue
        if not _mst_matches_intended(vertices, edge_indices):
            continue
        return vertices, edge_indices
    return None


def _choose_shadow(
    vertices: list[tuple[int, int]],
    tree_cells: set[tuple[int, int]],
    edge_indices: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = []
    for a, b in _edge_coords(vertices, edge_indices):
        directions.append(_step(a, b))
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in tree_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    tree_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(tree_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in tree_cells)
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


def _make_shape_patch(shape_color: int, shadow_color: int | None, force_branch: bool) -> tuple[np.ndarray, np.ndarray, bool] | None:
    spec = _make_tree_spec(force_branch)
    if spec is None:
        return None
    vertices, edge_indices = spec
    tree_cells = _tree_cells(vertices, edge_indices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, tree_cells, edge_indices)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(tree_cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for i, j in edge_indices:
        a = shifted_vertices[i]
        b = shifted_vertices[j]
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
    branching = _max_degree(edge_indices, len(vertices)) >= 3
    return inp, out, branching


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
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
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
    required_branch_idx = int(np.random.randint(shape_count))
    saw_branch = False

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        force_branch = idx == required_branch_idx
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx), force_branch=force_branch)
            if patch is None:
                continue
            patch_in, patch_out, is_branch = patch
            if _try_place(inp, out, patch_in, patch_out):
                saw_branch = saw_branch or is_branch
                placed = True
                break
        if not placed:
            return None

    if not saw_branch:
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
