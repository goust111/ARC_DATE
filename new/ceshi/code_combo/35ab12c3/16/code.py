# Combo variant of 35ab12c3 (index 16)
# combo_spec: D3/1+D5/0+D10/2
# Name: 最小生成树连接+无阴影+分支树拓扑

from __future__ import annotations

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for combo D3/1 + D5/0 + D10/2
#
# a) Concrete behavior / trace
#    原题对同色点做极角排序，3+ 点通常形成闭环或折线；若某个起点旁有异色邻居，还会把整条
#    形状沿该偏移平行复制成“阴影”。融合后两处都改变：
#    1. 同色点先枚举所有 H/V/45° 且原图中间无遮挡的候选边，再做确定性 Kruskal，得到最小
#       生成树；因此四个矩形角点会输出成三条边的树，而不是四边闭环。
#    2. 邻接异色像素不再被复制，只保留为原地单像素标记。
#    具体小例子：若输入有红色顶点 {(1,3),(4,3),(1,7),(1,10)}，且蓝色单像素在 (0,3)。
#    - 原题可能把红点按极角连成折回/闭环，并把蓝色沿整条形状平移复制。
#    - 本题只会从可见候选边里选最短无环树，例如 (1,3)-(4,3)、(1,3)-(1,7)、(1,7)-(1,10)，
#      蓝色仍只是 (0,3) 这一个点。
#
# b) Complexity calibration
#    仅改成 MST 还不够明显，所以生成器把所有 shape 都做成显式 branching tree，至少一个顶点
#    度数 >= 3；同时仍保留 2-4 个对象和可选干扰色，避免退化成单一大图。D5 的“无阴影”也不
#    直接删空，而是保留 1-3 个邻接单像素 decoy，让样例里稳定出现“有邻接异色但不会扩散”的
#    可见对比。
#
# c) Implicit-assumption review for main()
#    - `_angle_cycle` / wraparound cycle 假设“每色点集能排成单链或单环”，在 branching tree 下失效，
#      必须删除。
#    - 原题 `seed` / `carried` 逻辑假设邻接异色像素一定驱动平行投影；D5/0 明确要求删掉整段传播。
#    - `_aligned_8` / `_segment_cells` / `_clear_in_original` 只依赖“边必须是 H/V/45° 直段”，融合后
#      仍成立。
#    - MST 本身新增假设：真实树边必须都出现在“对齐且无遮挡”的候选边里，并且按长度+字典序的
#      Kruskal 结果唯一对应目标树。生成器通过 `_mst_matches_intended` 和 `main(local_in)==local_out`
#      双重 oracle 保证这一点。
#
# d) Placement legality
#    生成器先在局部 patch 中显式构造树边、只在输入保留顶点与 decoy，再用 `random_free_location`
#    按输出 patch 放到大画布。这样不同对象的最终输出像素至少隔 1 个背景格。局部构造还验证：
#    1. 每条边都严格 H/V/45°；
#    2. 不同枝条除共享端点外不共享格子；
#    3. solver 的 MST 恰好等于意图树；
#    4. decoy 只贴着一个顶点且不落在树上。
#
# e) Visual observability
#    解题者会反复看到：同色点总是补成树，而不是闭环；同时邻边的其他颜色小点保持原状，没有
#    投影出第二层平行线。这个差异在 2-3 个样例里就可肉眼归纳，不依赖颜色 id 或不可见排序。
#
# f) Clean visuals
#    不使用阴影后叠层明显减少；branch tree 只允许有限方向族和同方向延枝，不做任意折返，避免
#    交叉。decoy 是孤立单像素，干扰色也要求彼此非 H/V/diag 对齐，输出保持清爽。
#
# g) Color inference
#    没有颜色映射规则。shape 色、decoy 色、distractor 色都只是“不同对象/标记的不同颜色”，且每
#    种颜色的几何角色由局部结构决定：多点同色形成 tree，单像素邻接异色不传播。
#
# h) Post-change solver assumptions to audit against generator
#    - 对于可解 shape 色，候选边图必须连通，否则 MST 不完整；生成器只产出通过 oracle 的树。
#    - 单像素颜色被视为 inert，这要求 decoy/distractor 不要误生成成多点同色组；palette 与 patch
#      构造保证每个 decoy 色只出现 1 次。
#    - 若某个多像素颜色不是 shape，它不能恰好形成可连通 MST；干扰色通过“任意两点都不 H/V/diag
#      对齐”来保证不会误触发。
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
    return dr != 0 and abs(dr) == abs(dc)


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _candidate_edges(
    points_rc: list[tuple[int, int]], g: np.ndarray, bg: int, col: int
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    edges = []
    pts = sorted(points_rc)
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            if not _clear_in_original(g, a[0], a[1], b[0], b[1], bg, allow={col}):
                continue
            aa, bb = _canonical_edge(a, b)
            edges.append((_edge_len(aa, bb), aa, bb))
    edges.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[2][0], x[2][1]))
    return edges


def _mst_edges(
    points_rc: list[tuple[int, int]], g: np.ndarray, bg: int, col: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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

    if len(chosen) != len(pts) - 1:
        return []
    return chosen


"""
Overview:
Connect each multi-pixel color by the deterministic minimum spanning tree over all
visible H/V/45° candidate edges, and do nothing special with adjacent foreign colors.

Input structure:
- Background is the dominant color.
- A shape color appears as 4-7 same-colored vertex pixels that geometrically define
  a branching tree.
- Some other colors may appear as single-pixel decoys touching one vertex, or as
  small distractor groups whose pixels are pairwise non-aligned.

Rule:
For each color with at least two pixels, enumerate all same-color pairs that are
H/V/45° aligned and have no blocking non-shape pixels between them. Choose the
deterministic Kruskal MST of those candidate edges and fill every cell on those
segments with that color. Single-pixel adjacent markers are left unchanged; no
parallel shadow copy is drawn.

Key insight:
The old task depended on cyclic vertex ordering plus shadow propagation. Here the
topology is recovered purely from shortest visible straight connections, so the
output is a tree, often with a degree-3 hub, and nearby other-color pixels are inert.
"""
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        drawable_edges = _mst_edges(pts_rc, g, bg, col)
        if not drawable_edges:
            continue

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_DECOYS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

VERTEX_COUNT_RANGE = (4, 8)
PRIMARY_BRANCH_CHOICES = [3, 4]
PRIMARY_SEG_LEN_RANGE = (2, 7)
EXTEND_SEG_LEN_RANGE = (2, 6)
TREE_BUILD_RETRIES = 120

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500


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


def _edge_coords(
    vertices: list[tuple[int, int]], edge_indices: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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


def _build_branch_spec() -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    family_count = int(np.random.choice(PRIMARY_BRANCH_CHOICES))
    total_vertices = int(np.random.randint(max(family_count + 1, VERTEX_COUNT_RANGE[0]), VERTEX_COUNT_RANGE[1]))
    family_ids = np.random.choice(len(DIR_FAMILIES), size=family_count, replace=False).tolist()

    vertices = [(0, 0)]
    edge_indices: list[tuple[int, int]] = []
    branch_tips: list[tuple[int, tuple[int, int]]] = []

    for fam in family_ids:
        direction = DIR_FAMILIES[int(fam)][int(np.random.randint(2))]
        length = int(np.random.randint(*PRIMARY_SEG_LEN_RANGE))
        new_pt = (direction[0] * length, direction[1] * length)
        if new_pt in vertices:
            return None
        vertices.append(new_pt)
        edge_indices.append((0, len(vertices) - 1))
        branch_tips.append((len(vertices) - 1, direction))

    while len(vertices) < total_vertices:
        tip_slot = int(np.random.randint(len(branch_tips)))
        parent_idx, direction = branch_tips[tip_slot]
        parent = vertices[parent_idx]
        length = int(np.random.randint(*EXTEND_SEG_LEN_RANGE))
        new_pt = (parent[0] + direction[0] * length, parent[1] + direction[1] * length)
        if new_pt in vertices:
            return None
        vertices.append(new_pt)
        edge_indices.append((parent_idx, len(vertices) - 1))
        branch_tips[tip_slot] = (len(vertices) - 1, direction)

    return vertices, edge_indices


def _spec_to_transformed(
    vertices: list[tuple[int, int]], edge_indices: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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


def _make_tree_spec() -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(TREE_BUILD_RETRIES):
        spec = _build_branch_spec()
        if spec is None:
            continue
        vertices, edge_indices = _spec_to_transformed(*spec)
        if _max_degree(edge_indices, len(vertices)) < 3:
            continue
        if not _mst_matches_intended(vertices, edge_indices):
            continue
        return vertices, edge_indices
    return None


def _choose_decoy_anchor(
    vertices: list[tuple[int, int]],
    tree_cells: set[tuple[int, int]],
    edge_indices: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = [_step(a, b) for a, b in _edge_coords(vertices, edge_indices)]
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in tree_cells:
            continue
        hits = sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1)
        if hits != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    tree_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    decoy: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(tree_cells) + vertices
    if decoy is not None:
        (ar, ac), (dr, dc) = decoy
        coords.append((ar + dr, ac + dc))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, decoy_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    spec = _make_tree_spec()
    if spec is None:
        return None
    vertices, edge_indices = spec
    tree_cells = _tree_cells(vertices, edge_indices)

    decoy = None
    if decoy_color is not None:
        decoy = _choose_decoy_anchor(vertices, tree_cells, edge_indices)
        if decoy is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(tree_cells, vertices, decoy)
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

    if decoy is not None:
        (ar, ac), offset = decoy
        marker = (ar - r0 + offset[0], ac - c0 + offset[1])
        if not (0 <= marker[0] < h and 0 <= marker[1] < w):
            return None
        if inp[marker] != BG or out[marker] != BG:
            return None
        inp[marker] = int(decoy_color)
        out[marker] = int(decoy_color)

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
    max_decoys = min(MAX_DECOYS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_decoys < 1:
        return None
    decoy_count = int(np.random.randint(1, max_decoys + 1))
    total = shape_count + decoy_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(c) for c in palette[:shape_count]]
    decoy_colors = [int(c) for c in palette[shape_count : shape_count + decoy_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, decoy_colors, distractor_color


def _assign_decoys(shape_count: int, decoy_colors: list[int]) -> dict[int, int]:
    targets = np.random.choice(shape_count, size=len(decoy_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), decoy_colors)}


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
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, decoy_colors, distractor_color = palette
    decoy_map = _assign_decoys(shape_count, decoy_colors)

    total_colors = shape_count + len(decoy_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), decoy_map.get(idx))
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
