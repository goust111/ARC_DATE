# Variant of 35ab12c3
# Dimension: D5 — 阴影/平行偏移机制 (path)
# Change: parallel_offset——起始顶点的8邻居中的非背景非同色像素作为阴影锚点，其相对偏移被应用到整条形状路径上，生成平行的阴影副本
#         → 多层嵌套阴影：每个形状强制带 2-3 层递归阴影，每层颜色不同、方向不同，形成链式/同心式多重轮廓
"""
Design analysis for the D5 variant
==================================

[a] 具体化与行为追踪
- 原题输入里，一个 shape 颜色只给顶点；如果某个起始顶点旁边有一个异色像素，就把整条 shape 路径按这个单一偏移复制一次。
- 新变体把这个“单个邻居触发一次复制”改成“邻居链递归触发 2-3 次复制”：
  1. 基色顶点集记为 S0，取带阴影链的起始顶点 p0。
  2. 第一层颜色 c1 在输入中只出现一个像素 p1，且 p1 与 p0 8 邻接，offset1 = p1 - p0。
  3. 第二层颜色 c2 在输入中只出现一个像素 p2，且 p2 与 p1 8 邻接，offset2 = p2 - p1。
  4. 若有第三层，同理 p3 与 p2 8 邻接。
  5. 输出先画 S0，再画 S0 平移 offset1 的整条轮廓为 c1，再画 S0 平移 (offset1+offset2) 的整条轮廓为 c2，第三层类似。
- 小例子：
  原逻辑：顶点 A 在 (5,5)，邻居 7 在 (5,6)，整条图形只复制到 “整体右移 1” 的位置。
  新逻辑：再放一个 8 在 (6,7)，则第二层不是再从 A 读，而是从 7 的位置继续读出 offset=(+1,+1)，于是外层轮廓位于 “累计右 1 下 1” 的位置。
  差异：阴影不再是单层独立副本，而是有顺序的递归层级；输入里会出现一条靠近起始顶点的短彩色链，输出里出现 2-3 道不同颜色的嵌套平行轮廓。

[b] 复杂度校准
- 如果仍保留原题 2-4 个 shape 且每个都强制 2-3 层阴影，会超出 {1,2,3,4,6,7,8} 的 7 色预算。
- 因此把 shape 数量调成 1-2 个；两形状时只允许层数组合 [2,2] 或 [2,3]，单形状时允许 2 或 3 层，并可选 distractor。
- 每层偏移方向必须互异，且累计偏移不能与任何主边方向共线；否则外层会退化成沿原线滑动的重叠，而不是可见的平行轮廓。

[c] 原 solver 的隐式假设审查
- `seed_idxs` 这段默认“有阴影时，相关异色像素都在某个 base 顶点周围”。新变体保留第一层仍贴着 base 顶点，从而原始起点选择逻辑依然成立。
- 原 `carried` 只从起始顶点收集一圈异色邻居，并把它们全都当成“直接平移 shape 的单层影子”。这在多层链里不成立，因为第二/三层锚点不再直接贴着 base 顶点，而是贴着上一层锚点。
- 原 `for (r0,c0),(r1,c1) in drawable_edges: for carried ...` 默认每个阴影层都使用同一个单步偏移。新变体必须改成“按链顺序累计偏移”。
- 原 `if len(pts_rc) < 2: continue` 对 shadow color 是无害的，因为单层 shadow 只作为附属标记。新变体仍保留这个性质：阴影颜色在输入里依然只出现 1 个像素，不被单独当作 shape 处理，而是在父 shape 的求解过程中递归消费。
- 原 `_clear_in_original(... allow=set())` 对单层可行；新变体仍可沿用，但 generator 必须保证更外层轮廓的输入锚点不会落在更内层轮廓的边内部，否则合法边会被错误阻断。

[d] 放置合法性
- generator 为每个 shape 先在局部 patch 中挑选一条锚点链，再验证：
  1. 每个新锚点只与链前驱 8 邻接，不与其他顶点/旧锚点邻接，避免 solver 歧义。
  2. 每一层累计平移后的整条轮廓与已存在层完全不重叠，保证每层都可见。
  3. 累计平移后的轮廓 bbox 都在 patch 内；全局放置时再用 `random_free_location` 加 padding 隔开不同对象。

[e] 视觉可观察性
- 规则只依赖“链式邻接”和“轮廓层数/方向”，不依赖颜色 id 排序。
- 2-3 层会在输入中形成短彩色链，在输出中形成 2-3 道同形平行轮廓；这比原题的“可能有、也可能没有的一层影子”更稳定也更显眼。

[f] 可视化整洁度
- 不允许层与层重叠；全局 padding 提高到 2，避免不同 shape 的外层轮廓互相贴脸。
- 取消“两形状都三层”这种颜色不够且画面过满的情况；避免输出拥挤到难辨层级。

[g] 颜色变化可推断性
- 每层颜色都由输入里的那一个链上锚点直接给出；输出不做隐藏映射。
- 同一 shape 的多层颜色在输入和输出一一对应，人类可以从“哪颗单像素锚点触发哪一道外轮廓”直接归纳规则。

新增 solver 假设与 generator 保证
-----------------------------------
- solver 假设：阴影锚点形成一条简单链；链上中间节点除前驱/后继外没有别的非背景邻居。
- generator 保证：锚点选取时显式检查这个链式度数条件。
- solver 假设：每层累计平移后的轮廓不会与之前层重叠到把整层吃掉。
- generator 保证：每次增一层前检查平移后的 shape cell 集与已有层不相交。
- solver 假设：只有第一层锚点贴着 base 起始顶点，因此起始顶点选择稳定。
- generator 保证：只在 `min(vertices)` 附近放链，且不让别的顶点碰到这条链。
"""

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

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


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _extract_shadow_chain(g: np.ndarray, start_rc: tuple[int, int], base_col: int, bg: int,
                          max_layers: int = 3) -> list[tuple[int, int, int]]:
    """Read a simple chain of 2-3 shadow anchors from the original input."""
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


def main(grid: np.ndarray) -> np.ndarray:
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

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        chain = _extract_shadow_chain(g, cycle[0], col, bg, max_layers=3)
        cum_r = 0
        cum_c = 0
        for drn, dcn, ncol in chain:
            cum_r += drn
            cum_c += dcn
            for (r0, c0), (r1, c1) in drawable_edges:
                rr0 = r0 + cum_r
                cc0 = c0 + cum_c
                rr1 = r1 + cum_r
                cc1 = c1 + cum_c
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

SHAPE_PLANS = [
    (2, [2, 2]),
    (2, [2, 3]),
    (2, [3, 2]),
    (1, [2]),
    (1, [3]),
]
SHAPE_PLAN_WEIGHTS = np.array([0.26, 0.22, 0.14, 0.18, 0.20], dtype=float)
DISTRACTOR_PROB = 0.35
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
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

CHAIN_RETRIES = 80
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
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
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


def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _choose_shadow_steps(vertices: list[tuple[int, int]], shape_cells: set[tuple[int, int]],
                         layer_count: int) -> list[tuple[int, int]] | None:
    """Choose 2-3 distinct step offsets that form an unambiguous anchor chain."""
    anchor = min(vertices)
    directions = _edge_dirs(vertices)

    for _ in range(CHAIN_RETRIES):
        used_dirs: set[tuple[int, int]] = set()
        existing_points = set(vertices)
        occupied = set(shape_cells)
        steps: list[tuple[int, int]] = []
        prev_pos = anchor
        cum = (0, 0)
        ok = True

        for _layer in range(layer_count):
            candidates: list[tuple[int, int]] = []
            for off in DIR8:
                if off in used_dirs:
                    continue
                new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
                if new_pos in existing_points or new_pos in shape_cells:
                    continue
                if any(_is_adjacent(new_pos, pt) for pt in existing_points if pt != prev_pos):
                    continue
                new_cum = (cum[0] + off[0], cum[1] + off[1])
                if _collinear_with_any(new_cum, directions):
                    continue
                shifted = _translate_cells(shape_cells, new_cum)
                if shifted & occupied:
                    continue
                candidates.append(off)

            if not candidates:
                ok = False
                break

            off = candidates[int(np.random.randint(len(candidates)))]
            new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
            cum = (cum[0] + off[0], cum[1] + off[1])
            used_dirs.add(off)
            existing_points.add(new_pos)
            occupied |= _translate_cells(shape_cells, cum)
            steps.append(off)
            prev_pos = new_pos

        if ok:
            return steps

    return None


def _make_shape_patch(shape_color: int, shadow_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    shape_cells = _shape_cells(vertices)
    steps = _choose_shadow_steps(vertices, shape_cells, len(shadow_colors))
    if steps is None:
        return None

    anchor = min(vertices)
    cumulative_offsets: list[tuple[int, int]] = []
    anchor_positions: list[tuple[int, int]] = []
    all_cells = set(shape_cells)
    cum = (0, 0)
    pos = anchor
    for off in steps:
        cum = (cum[0] + off[0], cum[1] + off[1])
        pos = (pos[0] + off[0], pos[1] + off[1])
        cumulative_offsets.append(cum)
        anchor_positions.append(pos)
        all_cells |= _translate_cells(shape_cells, cum)

    r0, c0, r1, c1 = bbox_coords(np.array(sorted(all_cells), dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    for r, c in vertices:
        inp[r - r0, c - c0] = int(shape_color)
    for (r, c), col in zip(anchor_positions, shadow_colors):
        rr = r - r0
        cc = c - c0
        if not (0 <= rr < h and 0 <= cc < w) or inp[rr, cc] != BG:
            return None
        inp[rr, cc] = int(col)

    out = main(inp)

    if len(_extract_shadow_chain(inp, (anchor[0] - r0, anchor[1] - c0), int(shape_color), BG, max_layers=3)) != len(shadow_colors):
        return None
    if len(colors_present(inp, exclude=[BG])) != 1 + len(shadow_colors):
        return None
    for col in shadow_colors:
        if int(np.count_nonzero(out == int(col))) <= 1:
            return None
    if int(np.count_nonzero(out == int(shape_color))) <= len(vertices):
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


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _sample_shape_plan() -> tuple[list[int], bool]:
    weights = SHAPE_PLAN_WEIGHTS / SHAPE_PLAN_WEIGHTS.sum()
    idx = int(np.random.choice(len(SHAPE_PLANS), p=weights))
    shape_count, layer_counts = SHAPE_PLANS[idx]
    has_distractor = shape_count == 1 and bool(np.random.rand() < DISTRACTOR_PROB)
    return list(layer_counts), has_distractor


def _sample_palette(layer_counts: list[int], has_distractor: bool) -> tuple[list[tuple[int, list[int]]], int | None] | None:
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


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    layer_counts, has_distractor = _sample_shape_plan()
    palette = _sample_palette(layer_counts, has_distractor)
    if palette is None:
        return None
    shape_plans, distractor_color = palette
    total_colors = len(layer_counts) + sum(layer_counts) + int(has_distractor)

    for shape_color, shadow_colors in shape_plans:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_colors)
            if patch is None:
                continue
            if _try_place(inp, out, patch[0], patch[1]):
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
