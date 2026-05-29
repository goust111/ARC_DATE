# Combo variant of 35ab12c3 (index 4)
# combo_spec: D5/1+D6/2
# Name: 多层嵌套阴影+重叠区域擦除
"""
Design analysis for the combo variant
=====================================

[a] 具体化与行为追踪
- 原题里，一个 shape 颜色只给顶点；如果起始顶点旁边有一个异色像素，就把整条
  轮廓按这个单步偏移复制一次，而且重叠时仍保留 shape 本色。
- 组合后变成“链式 2-3 层阴影 + 擦除式合成”：
  1. 设 base 顶点集合为 S0，起始顶点为 p0。
  2. 第一层颜色 c1 在输入中只出现 1 个像素 p1，且 p1 与 p0 8 邻接；
     offset1 = p1 - p0。
  3. 第二层颜色 c2 在输入中只出现 1 个像素 p2，且 p2 与 p1 8 邻接；
     offset2 = p2 - p1。
  4. 若有第三层，同理得到 p3 与 offset3。
  5. 输出要画 S0、S0+offset1、S0+(offset1+offset2)、S0+(offset1+offset2+offset3)。
  6. 任一格子若被两层及以上轮廓同时覆盖，则该格擦成背景；仅被一层覆盖时保留
     该层颜色。这是多色集合上的“XOR 风格”空洞合成。
- 小例子：一条 2x7 的细矩形框，第一层累计偏移为 (1,1)，第二层为 (1,2)。
  原逻辑会得到 base 框 + 两层平行彩框，重叠处仍保留先画的颜色。
  新逻辑会把 base 与第一层的重叠横边、以及第一层与第二层的重叠边都擦成 0，
  所以输出出现明显的镂空缝隙。

[b] 复杂度校准
- 若保留原题 2-4 个 shape 且每个都强制 2-3 层阴影，会超出 {1,2,3,4,6,7,8}
  的 7 色预算，也会让图面过满。
- 因此组合版将 shape 数量收缩为 1-2 个；两 shape 时只允许 [2,2]、[2,3]、
  [3,2] 这类层数计划，单 shape 时允许 2 或 3 层，并且只在颜色预算宽松时允许
  distractor。
- 为了让 D6/2 的“擦除”肉眼可见，shape 强烈偏向 2 行/2 列的细矩形框；这种几何
  在对角或近对角累计偏移下会稳定地产生多格重叠空洞。

[c] 原 solver 的隐式假设审查
- 原 `seed_idxs` 假设“有 shadow 时，至少有一个 base 顶点旁边直接贴着异色像素”。
  组合版保留第一层仍贴着 base 起始顶点，所以这个前提仍成立。
- 原 `carried` 只读取起始顶点周围一圈异色像素，并把它们都当作“单层直接平移”。
  这在链式多层里不成立，因此 solver 必须改成从当前锚点递归追踪 2-3 层链。
- 原 `_draw_line_bg_only` 依赖“shape color wins”；一旦要求擦除重叠，就不能边画边决
  定颜色，而必须先收集所有层的 cell 集，再按覆盖次数合成。
- 原 `len(pts_rc) < 2: continue` 对组合版仍成立：shadow 颜色在输入里依旧只有一个
  锚点像素，不会被当作独立 shape 处理。
- 新冲突点：如果 generator 允许“洞”被别的对象占进去，solver 会把两个对象错误地
  视为彼此干扰。因此全局放置必须基于“所有触及过的 cell”占位，而不是仅基于最终
  可见输出。

[d] 放置合法性
- 每个 shape 先在局部 patch 中搜索一条简单锚点链；链上新锚点必须只接触前驱，不
  能贴到别的顶点或旧锚点，避免 solver 在 `_extract_shadow_chain` 时歧义分叉。
- 对每个累计偏移都要求不与任一主边方向共线，避免退化成“沿边滑动”的假平移。
- patch bbox 覆盖所有层的原始触及区域；全局放置时使用单独的 occupancy 画布，把
  被擦空的洞也当成已占用区域，从而防止对象穿洞重叠。

[e] 视觉可观察性
- 输入中，每个 shape 起点旁都能看到 2-3 个不同颜色的短链；输出中，这些颜色对应
  2-3 道嵌套轮廓，而且交叠处固定变成背景。
- 颜色角色完全由位置结构给出，不依赖颜色 id 排序。解题者只需看“链式单像素锚点”
  和“重叠后挖洞”的几何后果即可归纳规则。
- generator 还显式拒绝“组合后输出与旧的 shape-wins 输出完全一样”的样本，防止
  退化为只剩 D5/1 而没有 D6/2。

[f] 可视化整洁度
- 全局 padding 提高到 2，避免多层轮廓对象彼此贴边。
- 两 shape 同时出现时只允许有限层数计划，避免 7 色满负荷下的拥挤噪声。
- distractor 保持稀疏、非对齐、原样保留，不参与阴影链或擦除逻辑。

[g] 变化的可推断性
- 每个 shadow 颜色在输入中只出现一个链上锚点，在输出中扩展为对应那一层轮廓。
- 颜色总数被控制在 3-7 之间，并且每种颜色都有明确结构身份：base、某一层 shadow、
  或 distractor；不存在纯颜色重映射。

新增 solver 假设与 generator 保证
-----------------------------------
- solver 假设：shadow anchors 构成一条简单链；generator 保证每个新锚点只与前驱相邻。
- solver 假设：每个累计偏移对应的整层轮廓都可恢复；generator 保证累计偏移不共线、
  不越界，并且该层在 XOR 合成后仍保留可见像素。
- solver 假设：组合规则真的产生了擦除差异；generator 通过和旧 `shape-wins` 输出比对，
  强制至少出现若干差异像素与空洞像素。
"""

import math
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def _clear_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> bool:
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


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _collect_drawable_edges(
    g: np.ndarray,
    cycle: list[tuple[int, int]],
    col: int,
    bg: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(cycle) == 2:
        edges = [(cycle[0], cycle[1])]
    else:
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
            continue
        drawable_edges.append(((r0, c0), (r1, c1)))
    return drawable_edges


def _collect_cells_for_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in edges:
        cells.update(_segment_cells(r0, c0, r1, c1))
    return cells


def _collect_shifted_cells(
    g: np.ndarray,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    offset: tuple[int, int],
    bg: int,
) -> set[tuple[int, int]]:
    drn, dcn = offset
    h, w = g.shape
    cells: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in edges:
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
        cells.update(_segment_cells(rr0, cc0, rr1, cc1))
    return cells


def _extract_shadow_chain(
    g: np.ndarray,
    start_rc: tuple[int, int],
    base_col: int,
    bg: int,
    max_layers: int = 3,
) -> list[tuple[int, int, int]]:
    h, w = g.shape
    chain: list[tuple[int, int, int]] = []
    used_colors = {base_col}
    used_positions = {start_rc}
    prev: tuple[int, int] | None = None
    cur = start_rc

    for _ in range(max_layers):
        candidates: list[tuple[int, int, int, int, int]] = []
        for drn, dcn in DIR8:
            rr = cur[0] + drn
            cc = cur[1] + dcn
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if prev is not None and (rr, cc) == prev:
                continue
            if (rr, cc) in used_positions:
                continue
            v = int(g[rr, cc])
            if v == bg or v in used_colors:
                continue
            candidates.append((rr, cc, drn, dcn, v))
        if not candidates:
            break
        rr, cc, drn, dcn, v = min(candidates, key=lambda x: (x[0], x[1], x[4]))
        chain.append((drn, dcn, v))
        used_colors.add(v)
        used_positions.add((rr, cc))
        prev, cur = cur, (rr, cc)

    return chain


def _compose_layers(
    out: np.ndarray,
    layers: list[tuple[int, set[tuple[int, int]]]],
    bg: int,
    erase_overlaps: bool,
) -> None:
    if not layers:
        return
    if not erase_overlaps:
        for rr, cc in layers[0][1]:
            out[rr, cc] = layers[0][0]
        for col, cells in layers[1:]:
            for rr, cc in cells:
                if out[rr, cc] == bg:
                    out[rr, cc] = col
        return

    counts: Counter[tuple[int, int]] = Counter()
    owner: dict[tuple[int, int], int] = {}
    affected: set[tuple[int, int]] = set()

    for col, cells in layers:
        for cell in cells:
            affected.add(cell)
            counts[cell] += 1
            if counts[cell] == 1:
                owner[cell] = col

    for rr, cc in affected:
        out[rr, cc] = bg

    for (rr, cc), cnt in counts.items():
        if cnt == 1:
            out[rr, cc] = owner[(rr, cc)]


def _solve(grid: np.ndarray, *, erase_overlaps: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_idxs.append(i)
                        break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]
        drawable_edges = _collect_drawable_edges(g, cycle, col, bg)
        if not drawable_edges:
            continue

        layers: list[tuple[int, set[tuple[int, int]]]] = [
            (col, _collect_cells_for_edges(drawable_edges))
        ]
        cum_r = 0
        cum_c = 0
        for drn, dcn, ncol in _extract_shadow_chain(g, cycle[0], col, bg, max_layers=3):
            cum_r += drn
            cum_c += dcn
            shadow_cells = _collect_shifted_cells(g, drawable_edges, (cum_r, cum_c), bg)
            if shadow_cells:
                layers.append((ncol, shadow_cells))

        _compose_layers(out, layers, bg, erase_overlaps=erase_overlaps)

    return out


"""
Overview:
Expand each same-color vertex set into its straight outline, then follow a 2-3 step
shadow-anchor chain and draw cumulative parallel copies. Any cell covered by more than
one layer is erased to background.

Input structure:
- Background is the dominant color.
- A shape color appears on 2+ aligned vertices that define a polyline/polygon outline.
- Near the chosen start vertex there is a simple chain of 2-3 single-pixel colors, each
  8-neighboring the previous anchor and defining the next cumulative shadow offset.
- Optional distractor colors remain as-is.

Rule:
1. Recover drawable straight edges between the ordered vertices of each shape color.
2. Draw the base outline.
3. Follow the shadow chain; after each step, add the cumulative offset and draw another
   copy of the same outline in that layer's color.
4. Compose all layers together with XOR-style erasure: cells touched by exactly one layer
   keep that layer color; cells touched by multiple layers become background.

Key insight:
The single-pixel shadow colors are not standalone objects. They form an ordered anchor
chain whose cumulative offsets generate nested outline copies, and the visible novelty is
the hole pattern created where those copies overlap.
"""
def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, erase_overlaps=True)


# --------------- generator ---------------

BG = 0
OCC = 9
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_PLANS = [
    (2, [2, 2]),
    (2, [2, 3]),
    (2, [3, 2]),
    (1, [2]),
    (1, [3]),
]
SHAPE_PLAN_WEIGHTS = np.array([0.24, 0.18, 0.12, 0.22, 0.24], dtype=float)
DISTRACTOR_PROB = 0.30
PLACEMENT_PADDING = 2
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)
OVERLAP_RECT_W_RANGE = (5, 10)
OVERLAP_BIAS = 0.82
SHAPE_WEIGHTS = np.array([0.18, 0.12, 0.16, 0.14, 0.16, 0.10, 0.14], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

CHAIN_RETRIES = 140
SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500
MIN_HOLE_CELLS = 2
MIN_BASE_OVERLAP = 2
MIN_VISIBLE_PER_LAYER = 2
MIN_DIFF_VS_SHAPE_WINS = 2


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


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


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    hh = int(np.random.randint(*RECT_H_RANGE))
    ww = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, ww), (hh, ww), (hh, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    hh = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (hh, 0), (hh, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


def _build_overlap_rectangle() -> list[tuple[int, int]]:
    ww = int(np.random.randint(*OVERLAP_RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, ww), (1, ww), (1, 0)])


_SHAPE_BUILDERS = [
    _build_line,
    _build_triangle,
    _build_rectangle,
    _build_diamond,
    _build_pentagon,
    _build_hexagon,
    _build_overlap_rectangle,
]


def _make_vertices() -> list[tuple[int, int]]:
    if np.random.rand() < OVERLAP_BIAS:
        return _build_overlap_rectangle()
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _translate_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _collinear_with_any(offset: tuple[int, int], directions: list[tuple[int, int]]) -> bool:
    return any(ds_r * offset[1] - ds_c * offset[0] == 0 for ds_r, ds_c in directions)


def _adjacent_points(
    pos: tuple[int, int],
    points: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    return {
        pt for pt in points
        if pt != pos and max(abs(pos[0] - pt[0]), abs(pos[1] - pt[1])) == 1
    }


def _layer_visibility(
    layer_cells: list[set[tuple[int, int]]],
) -> tuple[int, int, list[int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for cells in layer_cells:
        for cell in cells:
            counts[cell] += 1
    hole_cells = sum(1 for cnt in counts.values() if cnt >= 2)
    visible = [sum(1 for cell in cells if counts[cell] == 1) for cells in layer_cells]
    base_overlap = sum(1 for cell in layer_cells[0] if counts[cell] >= 2)
    return hole_cells, base_overlap, visible


def _choose_shadow_steps(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    layer_count: int,
) -> list[tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)

    for _ in range(CHAIN_RETRIES):
        steps: list[tuple[int, int]] = []
        cumulative: list[tuple[int, int]] = []
        used_dirs: set[tuple[int, int]] = set()
        anchor_points = {anchor}
        prev_pos = anchor
        cum = (0, 0)
        ok = True

        for _layer in range(layer_count):
            candidates: list[tuple[int, int]] = []
            for off in DIR8:
                if off in used_dirs:
                    continue
                new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
                if new_pos in shape_cells or new_pos in anchor_points:
                    continue
                neighbors = _adjacent_points(new_pos, set(vertices) | anchor_points)
                if neighbors != {prev_pos}:
                    continue
                new_cum = (cum[0] + off[0], cum[1] + off[1])
                if new_cum in cumulative:
                    continue
                if _collinear_with_any(new_cum, directions):
                    continue
                candidates.append(off)

            if not candidates:
                ok = False
                break

            off = candidates[int(np.random.randint(len(candidates)))]
            prev_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
            cum = (cum[0] + off[0], cum[1] + off[1])
            steps.append(off)
            cumulative.append(cum)
            used_dirs.add(off)
            anchor_points.add(prev_pos)

        if not ok:
            continue

        layer_cells = [shape_cells] + [_translate_cells(shape_cells, off) for off in cumulative]
        hole_cells, base_overlap, visible = _layer_visibility(layer_cells)
        if hole_cells < MIN_HOLE_CELLS:
            continue
        if base_overlap < MIN_BASE_OVERLAP:
            continue
        if any(v < MIN_VISIBLE_PER_LAYER for v in visible):
            continue
        return steps

    return None


def _build_shape_patch(
    shape_color: int,
    shadow_colors: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    shape_cells = _shape_cells(vertices)
    steps = _choose_shadow_steps(vertices, shape_cells, len(shadow_colors))
    if steps is None:
        return None

    anchor = min(vertices)
    cumulative_offsets: list[tuple[int, int]] = []
    anchor_positions: list[tuple[int, int]] = []
    touched = set(shape_cells)
    cum = (0, 0)
    cur = anchor
    for off in steps:
        cur = (cur[0] + off[0], cur[1] + off[1])
        cum = (cum[0] + off[0], cum[1] + off[1])
        anchor_positions.append(cur)
        cumulative_offsets.append(cum)
        touched |= _translate_cells(shape_cells, cum)

    coords = list(touched) + vertices + anchor_positions
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    hh, ww = r1 - r0 + 1, c1 - c0 + 1

    inp = np.full((hh, ww), BG, dtype=int)
    occ = np.full((hh, ww), BG, dtype=int)

    for r, c in vertices:
        inp[r - r0, c - c0] = int(shape_color)
    for (r, c), col in zip(anchor_positions, shadow_colors):
        rr = r - r0
        cc = c - c0
        if inp[rr, cc] != BG:
            return None
        inp[rr, cc] = int(col)
    for r, c in touched:
        occ[r - r0, c - c0] = OCC

    out = main(inp)
    shape_wins = _solve(inp, erase_overlaps=False)

    recovered = _extract_shadow_chain(inp, (anchor[0] - r0, anchor[1] - c0), int(shape_color), BG, max_layers=3)
    if len(recovered) != len(shadow_colors):
        return None
    if len(colors_present(inp, exclude=[BG])) != 1 + len(shadow_colors):
        return None
    if int(np.count_nonzero(out != shape_wins)) < MIN_DIFF_VS_SHAPE_WINS:
        return None
    if int(np.count_nonzero(occ != BG)) <= int(np.count_nonzero(out != BG)):
        return None
    for col in [shape_color] + shadow_colors:
        if int(np.count_nonzero(out == int(col))) < MIN_VISIBLE_PER_LAYER:
            return None

    return inp, out, occ


def _sample_distractor_coords(hh: int, ww: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(hh)), int(np.random.randint(ww)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Keep distractors to a single pixel so every repeated color in the input
    # unambiguously belongs to a shadow-bearing shape rather than inert noise.
    patch = np.full((1, 1), int(color), dtype=int)
    occ = np.full((1, 1), OCC, dtype=int)
    return patch.copy(), patch.copy(), occ


def _sample_canvas_size() -> tuple[int, int]:
    hh = int(np.random.randint(*CANVAS_H_RANGE))
    ww = int(np.clip(hh + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return hh, ww


def _sample_shape_plan() -> tuple[list[int], bool]:
    weights = SHAPE_PLAN_WEIGHTS / SHAPE_PLAN_WEIGHTS.sum()
    idx = int(np.random.choice(len(SHAPE_PLANS), p=weights))
    shape_count, layer_counts = SHAPE_PLANS[idx]
    has_distractor = sum(layer_counts) + shape_count <= len(COLORS) - 1 and bool(np.random.rand() < DISTRACTOR_PROB)
    if shape_count == 2 and layer_counts != [2, 2]:
        has_distractor = False
    return list(layer_counts), has_distractor


def _sample_palette(
    layer_counts: list[int],
    has_distractor: bool,
) -> tuple[list[tuple[int, list[int]]], int | None] | None:
    total = len(layer_counts) + sum(layer_counts) + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    ptr = 0
    plans: list[tuple[int, list[int]]] = []
    for count in layer_counts:
        base = int(palette[ptr])
        ptr += 1
        shadows = [int(v) for v in palette[ptr:ptr + count]]
        ptr += count
        plans.append((base, shadows))
    distractor = int(palette[ptr]) if has_distractor else None
    return plans, distractor


def _try_place(
    inp: np.ndarray,
    out: np.ndarray,
    occ: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    patch_occ: np.ndarray,
) -> bool:
    try:
        loc = random_free_location(
            occ,
            patch_occ,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(occ, patch_occ, loc[0], loc[1], transparent=BG)
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    hh, ww = _sample_canvas_size()
    inp = np.full((hh, ww), BG, dtype=int)
    out = np.full((hh, ww), BG, dtype=int)
    occ = np.full((hh, ww), BG, dtype=int)

    layer_counts, has_distractor = _sample_shape_plan()
    palette = _sample_palette(layer_counts, has_distractor)
    if palette is None:
        return None
    shape_plans, distractor_color = palette
    total_colors = len(layer_counts) + sum(layer_counts) + int(has_distractor)

    for shape_color, shadow_colors in shape_plans:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _build_shape_patch(int(shape_color), shadow_colors)
            if patch is None:
                continue
            if _try_place(inp, out, occ, patch[0], patch[1], patch[2]):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        distractor = _make_distractor_patch(distractor_color)
        if not _try_place(inp, out, occ, distractor[0], distractor[1], distractor[2]):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    present_colors = [int(c) for c in colors_present(inp, exclude=[BG])]
    if len(present_colors) != total_colors:
        return None
    repeated_colors = sorted(int(c) for c in present_colors if int(np.count_nonzero(inp == int(c))) >= 2)
    expected_shape_colors = sorted(int(shape_color) for shape_color, _shadow_colors in shape_plans)
    if repeated_colors != expected_shape_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if int(np.count_nonzero(out != _solve(inp, erase_overlaps=False))) < MIN_DIFF_VS_SHAPE_WINS:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
